import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup_instrumented import ACCUPInstrumented
from scripts.perturbation_analysis_utils import (
    amplitude_drift,
    gaussian_noise_view,
    magnitude_warp_view,
    pgd_entropy_attack,
    signal_energy_ratios,
    time_warp_view,
    total_variation,
)
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
)


VIEW_BUILDERS = {
    "gaussian_noise": lambda model, x, args: gaussian_noise_view(x),
    "magnitude_warp": lambda model, x, args: magnitude_warp_view(x),
    "time_warp": lambda model, x, args: time_warp_view(x),
    "pgd_entropy": lambda model, x, args: pgd_entropy_attack(model, x, eps=args.pgd_eps, steps=args.pgd_steps),
    "amplitude_drift_ref": lambda model, x, args: amplitude_drift(x, "moderate"),
    "ssaw": lambda tta_model, x, args: tta_model.get_adversarial_view(x, tta_model.model),
}

DEFAULT_VIEW_ORDER = [
    "gaussian_noise",
    "magnitude_warp",
    "pgd_entropy",
    "ssaw",
]


def parse_view_list(raw_text):
    if not raw_text:
        return list(DEFAULT_VIEW_ORDER)
    views = [item.strip() for item in str(raw_text).split(",") if item.strip()]
    invalid = [view for view in views if view not in VIEW_BUILDERS]
    if invalid:
        raise ValueError(f"Unsupported views: {invalid}")
    return views


def collect_samples(loader, limit):
    x_chunks, y_chunks = [], []
    total = 0
    for data, labels, _ in loader:
        primary = extract_primary_tensor(data)
        x_chunks.append(primary)
        y_chunks.append(labels)
        total += primary.size(0)
        if total >= limit:
            break
    return torch.cat(x_chunks, dim=0)[:limit], torch.cat(y_chunks, dim=0)[:limit]


def extract_features(model, x):
    feats, _ = model.feature_extractor(x)
    return feats


def reduce_features(features, method="tsne"):
    features_np = features.detach().cpu().numpy()
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(features_np)
        except Exception:
            method = "tsne"
    perplexity = min(30, max(5, (features_np.shape[0] - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    return reducer.fit_transform(features_np)


def centroid_cosine_distance(feats, ref_feats):
    centroid = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)
    ref_centroid = F.normalize(ref_feats.mean(dim=0, keepdim=True), dim=1)
    return float((1.0 - F.cosine_similarity(centroid, ref_centroid, dim=1)).item())


def _rbf_kernel(x, y, gamma):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).transpose(0, 1)
    sq_dist = (x_norm + y_norm - 2.0 * x @ y.transpose(0, 1)).clamp_min(0.0)
    return torch.exp(-gamma * sq_dist)


def rbf_mmd(feats, ref_feats):
    combined = torch.cat([feats, ref_feats], dim=0)
    pairwise = torch.cdist(combined, combined, p=2)
    median = pairwise.median().clamp_min(1e-6)
    gamma = float(1.0 / (2.0 * median.item() ** 2))
    k_xx = _rbf_kernel(feats, feats, gamma).mean()
    k_yy = _rbf_kernel(ref_feats, ref_feats, gamma).mean()
    k_xy = _rbf_kernel(feats, ref_feats, gamma).mean()
    return float((k_xx + k_yy - 2.0 * k_xy).item())


def summarize_signal(signal):
    low, high = signal_energy_ratios(signal)
    return {
        "low_freq_ratio_mean": float(np.mean(low)),
        "high_freq_ratio_mean": float(np.mean(high)),
        "total_variation_mean": float(np.mean(total_variation(signal))),
    }


def add_shift_delta_columns(summary_df):
    source_row = summary_df[summary_df["view"] == "source_clean"].iloc[0]
    target_row = summary_df[summary_df["view"] == "target_clean"].iloc[0]
    source_to_target = {
        "low": float(target_row["low_freq_ratio_mean"] - source_row["low_freq_ratio_mean"]),
        "high": float(target_row["high_freq_ratio_mean"] - source_row["high_freq_ratio_mean"]),
        "tv": float(target_row["total_variation_mean"] - source_row["total_variation_mean"]),
    }

    def _align(delta_view, delta_target):
        scale = abs(delta_target) + 1e-8
        return 1.0 - abs(delta_view - delta_target) / scale

    out = summary_df.copy()
    out["low_freq_shift_from_source"] = out["low_freq_ratio_mean"] - float(source_row["low_freq_ratio_mean"])
    out["high_freq_shift_from_source"] = out["high_freq_ratio_mean"] - float(source_row["high_freq_ratio_mean"])
    out["total_variation_shift_from_source"] = out["total_variation_mean"] - float(source_row["total_variation_mean"])
    out["low_freq_shift_alignment"] = out["low_freq_shift_from_source"].map(
        lambda val: _align(val, source_to_target["low"])
    )
    out["high_freq_shift_alignment"] = out["high_freq_shift_from_source"].map(
        lambda val: _align(val, source_to_target["high"])
    )
    out["total_variation_shift_alignment"] = out["total_variation_shift_from_source"].map(
        lambda val: _align(val, source_to_target["tv"])
    )
    return out


def build_embedding_panel(embedding_map, output_path, title, view_order):
    n_panels = 1 + len(view_order)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    base = embedding_map["source_clean"]
    target = embedding_map["target_clean"]
    x_all = np.concatenate([values[:, 0] for values in embedding_map.values()], axis=0)
    y_all = np.concatenate([values[:, 1] for values in embedding_map.values()], axis=0)
    x_margin = 0.05 * (x_all.max() - x_all.min() + 1e-6)
    y_margin = 0.05 * (y_all.max() - y_all.min() + 1e-6)
    x_lim = (x_all.min() - x_margin, x_all.max() + x_margin)
    y_lim = (y_all.min() - y_margin, y_all.max() + y_margin)

    panel_specs = [("dataset_shift", "Dataset Shift: source vs target")] + [
        (view_name, f"Shift Mimicry: {view_name}") for view_name in view_order
    ]

    for ax, (panel_key, panel_title) in zip(axes, panel_specs):
        ax.scatter(base[:, 0], base[:, 1], s=18, alpha=0.28, color="#9aa0a6", label="source clean")
        ax.scatter(target[:, 0], target[:, 1], s=18, alpha=0.38, color="#1f77b4", label="target clean")
        if panel_key != "dataset_shift":
            view_points = embedding_map[panel_key]
            ax.scatter(view_points[:, 0], view_points[:, 1], s=18, alpha=0.42, color="#e45756", label=panel_key)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(alpha=0.15)
        handles, labels = ax.get_legend_handles_labels()
        keep = {}
        for handle, label in zip(handles, labels):
            keep.setdefault(label, handle)
        ax.legend(keep.values(), keep.keys(), fontsize=8, frameon=False, loc="best")

    for idx in range(len(panel_specs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_target_gap_bars(summary_df, output_path):
    metrics = [
        ("feature_centroid_cosine_to_target", "Centroid Cosine Dist to Target"),
        ("rbf_mmd_to_target", "RBF-MMD to Target"),
        ("low_freq_gap_to_target", "Low-Freq Gap to Target"),
        ("high_freq_gap_to_target", "High-Freq Gap to Target"),
        ("total_variation_gap_to_target", "TV Gap to Target"),
    ]
    plot_df = summary_df[~summary_df["view"].isin(["source_clean", "target_clean"])].copy()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    x = np.arange(len(plot_df))
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(x, plot_df[column], color="#e45756")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["view"], rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_shift_profile(summary_df, output_path):
    metrics = [
        ("low_freq_ratio_mean", "Low-Freq Ratio"),
        ("high_freq_ratio_mean", "High-Freq Ratio"),
        ("total_variation_mean", "Total Variation"),
    ]
    x = np.arange(len(summary_df))
    labels = summary_df["view"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(x, summary_df[column], color=["#9aa0a6" if name == "source_clean" else "#1f77b4" if name == "target_clean" else "#e45756" for name in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_frequency_impact(summary_df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    labels = summary_df["view"].tolist()
    x = np.arange(len(labels))
    colors = [
        "#9aa0a6" if name == "source_clean" else "#1f77b4" if name == "target_clean" else "#e45756"
        for name in labels
    ]

    metric_specs = [
        ("low_freq_ratio_mean", "Low-Frequency Ratio"),
        ("high_freq_ratio_mean", "High-Frequency Ratio"),
        ("low_freq_shift_from_source", "Delta Low-Freq Ratio vs Source"),
        ("high_freq_shift_from_source", "Delta High-Freq Ratio vs Source"),
    ]
    for ax, (column, title) in zip(axes, metric_specs):
        ax.bar(x, summary_df[column], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if "shift_from_source" in column:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_shift_alignment(summary_df, output_path):
    plot_df = summary_df[~summary_df["view"].isin(["source_clean", "target_clean"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(plot_df))
    specs = [
        ("low_freq_shift_alignment", "Low-Freq Shift Alignment"),
        ("high_freq_shift_alignment", "High-Freq Shift Alignment"),
        ("total_variation_shift_alignment", "TV Shift Alignment"),
    ]
    for ax, (column, title) in zip(axes, specs):
        ax.bar(x, plot_df[column], color="#e45756")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["view"], rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--scenario", required=True, help="src->trg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--views", default=",".join(DEFAULT_VIEW_ORDER))
    parser.add_argument("--embedding", default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--pgd_eps", type=float, default=0.1)
    parser.add_argument("--pgd_steps", type=int, default=10)
    args = parser.parse_args()

    view_order = parse_view_list(args.views)
    output_dir = ensure_dir(
        RESULTS_ROOT / "shift_mimicry" / f"{args.dataset}_{args.scenario.replace('->', 'to')}_{'_'.join(view_order)}"
    )
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=args.dataset,
        da_method="ACCUP",
        backbone=args.backbone,
        exp_name="shift_mimicry",
        seed=args.seed,
        tta_model_class=ACCUPInstrumented,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = args.scenario.split("->", 1)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=args.seed)

        source_x, source_labels = collect_samples(trainer.src_test_dl, args.max_samples)
        target_x, target_labels = collect_samples(trainer.trg_whole_dl, args.max_samples)
        source_x = source_x.to(trainer.device)
        target_x = target_x.to(trainer.device)

        view_bank = {
            "source_clean": source_x,
            "target_clean": target_x,
        }
        for view_name in view_order:
            if view_name == "ssaw":
                view_bank[view_name] = VIEW_BUILDERS[view_name](tta_model, source_x, args)
            else:
                view_bank[view_name] = VIEW_BUILDERS[view_name](tta_model.model, source_x, args)

        with torch.no_grad():
            feature_bank = {name: extract_features(tta_model.model, signal) for name, signal in view_bank.items()}

        stacked_features = torch.cat([feature_bank[name] for name in view_bank], dim=0)
        embedding = reduce_features(stacked_features, method=args.embedding)

        embedding_map = {}
        cursor = 0
        for name in view_bank:
            count = feature_bank[name].size(0)
            embedding_map[name] = embedding[cursor : cursor + count]
            cursor += count

        rows = []
        target_summary = summarize_signal(view_bank["target_clean"])
        for name in view_bank:
            signal_summary = summarize_signal(view_bank[name])
            rows.append(
                {
                    "dataset": args.dataset,
                    "scenario": args.scenario,
                    "view": name,
                    "num_samples": int(view_bank[name].size(0)),
                    "feature_centroid_cosine_to_target": centroid_cosine_distance(
                        feature_bank[name], feature_bank["target_clean"]
                    ),
                    "rbf_mmd_to_target": rbf_mmd(feature_bank[name], feature_bank["target_clean"]),
                    "low_freq_ratio_mean": signal_summary["low_freq_ratio_mean"],
                    "high_freq_ratio_mean": signal_summary["high_freq_ratio_mean"],
                    "total_variation_mean": signal_summary["total_variation_mean"],
                    "low_freq_gap_to_target": abs(signal_summary["low_freq_ratio_mean"] - target_summary["low_freq_ratio_mean"]),
                    "high_freq_gap_to_target": abs(signal_summary["high_freq_ratio_mean"] - target_summary["high_freq_ratio_mean"]),
                    "total_variation_gap_to_target": abs(signal_summary["total_variation_mean"] - target_summary["total_variation_mean"]),
                }
            )
        summary_df = add_shift_delta_columns(pd.DataFrame(rows))
        summary_df.to_csv(output_dir / "shift_mimicry_summary.csv", index=False)

        embedding_df = []
        for name, coords in embedding_map.items():
            labels = source_labels if name != "target_clean" else target_labels
            labels_np = labels.detach().cpu().numpy()
            embedding_df.append(
                pd.DataFrame(
                    {
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "view": name,
                        "class_id": labels_np,
                    }
                )
            )
        pd.concat(embedding_df, ignore_index=True).to_csv(output_dir / "embedding_points.csv", index=False)

        build_embedding_panel(
            embedding_map,
            output_dir / f"{args.embedding}_shift_mimicry_panel.pdf",
            title=f"{args.dataset} {args.scenario} | source-target shift mimicry",
            view_order=view_order,
        )
        plot_target_gap_bars(summary_df, output_dir / "target_gap_summary.pdf")
        plot_shift_profile(summary_df, output_dir / "shift_profile_summary.pdf")
        plot_frequency_impact(summary_df, output_dir / "frequency_impact_summary.pdf")
        plot_shift_alignment(summary_df, output_dir / "shift_alignment_summary.pdf")
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)

    print(f"Shift mimicry visualization completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
