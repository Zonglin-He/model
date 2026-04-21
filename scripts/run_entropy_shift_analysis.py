import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    extract_primary_tensor,
)
from utils.utils import softmax_entropy_from_logits


DATASET_LABELS = {
    "EEG": "Sleep-EDF (EEG)",
    "HAR": "UCI-HAR",
    "FD": "MFD",
}


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def _upsample_controls(search, flat_controls, target_len):
    if hasattr(search, "_natural_cubic_spline_upsample"):
        return search._natural_cubic_spline_upsample(flat_controls, target_len)
    if hasattr(search, "_spline_upsample"):
        return search._spline_upsample(flat_controls, target_len)
    raise AttributeError("Active search does not expose a spline upsampling helper.")


@torch.no_grad()
def _batch_entropy_triplet(model, search, x, sigma):
    batch_size, channels, target_len = x.shape

    raw_feats = search._extract_features(model, x)
    raw_logits = model.classifier(raw_feats)
    raw_entropy = softmax_entropy_from_logits(raw_logits)

    if (
        search is None
        or getattr(search, "num_candidates", 0) <= 0
        or sigma <= 0.0
    ):
        return raw_entropy, raw_entropy, raw_entropy

    controls = search._sample_control_points(batch_size, x.device, x.dtype, sigma)
    flat_controls = controls.reshape(batch_size * search.num_candidates, search.num_control_points)
    upsampled = _upsample_controls(search, flat_controls, target_len)
    warps = upsampled.reshape(batch_size, search.num_candidates, target_len)
    warped_x = x.unsqueeze(1) * warps.unsqueeze(2)
    flat_x = warped_x.reshape(batch_size * search.num_candidates, channels, target_len)

    feats = search._extract_features(model, flat_x)
    logits = model.classifier(feats)
    entropies = softmax_entropy_from_logits(logits).reshape(batch_size, search.num_candidates)

    rand_idx = torch.randint(search.num_candidates, (batch_size,), device=x.device)
    adv_idx = entropies.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=x.device)
    rand_entropy = entropies[batch_indices, rand_idx]
    adv_entropy = entropies[batch_indices, adv_idx]
    return raw_entropy, rand_entropy, adv_entropy


def collect_entropy_shift(data_path, device, dataset, seed, backbone, src_id, trg_id):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        exp_name="entropy_shift_analysis",
        seed=seed,
        backbone=backbone,
    )
    tta_model = None
    pre_trained_model = None
    try:
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        model = tta_model.model
        model.eval()
        search = getattr(tta_model, "active_search", None)
        sigma = float(getattr(tta_model, "adv_sigma", 0.0))

        raw_values = []
        rand_values = []
        adv_values = []

        for batch in trainer.trg_whole_dl:
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = extract_primary_tensor(data)
            x = x.float().to(trainer.device)
            raw_entropy, rand_entropy, adv_entropy = _batch_entropy_triplet(model, search, x, sigma)
            raw_values.append(raw_entropy.detach().cpu())
            rand_values.append(rand_entropy.detach().cpu())
            adv_values.append(adv_entropy.detach().cpu())

        raw_tensor = torch.cat(raw_values)
        rand_tensor = torch.cat(rand_values)
        adv_tensor = torch.cat(adv_values)

        return {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "seed": seed,
            "h_raw_mean": float(raw_tensor.mean().item()),
            "h_rand_mean": float(rand_tensor.mean().item()),
            "h_adv_mean": float(adv_tensor.mean().item()),
            "delta_mean": float((adv_tensor - rand_tensor).mean().item()),
            "raw_values": raw_tensor.numpy(),
            "rand_values": rand_tensor.numpy(),
            "adv_values": adv_tensor.numpy(),
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def plot_dataset_distribution(dataset, summary_df, arrays_by_scenario, output_dir):
    rows = summary_df[summary_df["dataset"] == dataset].copy()
    scenarios = rows["scenario"].tolist()
    if not scenarios:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        arrays = arrays_by_scenario[(dataset, scenario)]
        raw_vals = arrays["raw"]
        rand_vals = arrays["rand"]
        adv_vals = arrays["adv"]

        bins = 30
        ax.hist(raw_vals, bins=bins, density=True, alpha=0.35, color="#4c78a8", label="H(raw)")
        ax.hist(rand_vals, bins=bins, density=True, alpha=0.35, color="#72b7b2", label="H(rand)")
        ax.hist(adv_vals, bins=bins, density=True, alpha=0.35, color="#e45756", label="H(adv)")

        raw_mean = float(np.mean(raw_vals))
        rand_mean = float(np.mean(rand_vals))
        adv_mean = float(np.mean(adv_vals))
        delta_mean = adv_mean - rand_mean

        ax.axvline(raw_mean, color="#4c78a8", linestyle="--", linewidth=1.0)
        ax.axvline(rand_mean, color="#72b7b2", linestyle="--", linewidth=1.0)
        ax.axvline(adv_mean, color="#e45756", linestyle="--", linewidth=1.0)
        ax.set_title(f"Entropy Distribution ({scenario.replace('->', ' to ')})", fontsize=10)
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Density")
        ax.text(
            0.98,
            0.94,
            f"dmean={delta_mean:+.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        if idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in axes[len(scenarios):]:
        ax.axis("off")

    fig.suptitle(
        f"Entropy distributions | {DATASET_LABELS.get(dataset, dataset)}",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / f"entropy_shift_{dataset.lower()}.pdf")
    fig.savefig(output_dir / f"entropy_shift_{dataset.lower()}.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "entropy_shift_analysis")
    seeds = parse_seed_list(args.seeds)
    datasets = [item.strip().upper() for item in str(args.datasets).split(",") if item.strip()]

    raw_rows = []
    pooled_arrays = {}

    for dataset in datasets:
        trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method="ACCUP",
            exp_name="entropy_shift_probe",
            seed=seeds[0],
            backbone=args.backbone,
        )
        scenarios = dataset_scenarios(trainer)
        cleanup_trainer(trainer, close_summary=True)

        for src_id, trg_id in scenarios:
            scenario = f"{src_id}->{trg_id}"
            pooled_arrays[(dataset, scenario)] = {"raw": [], "rand": [], "adv": []}
            for seed in seeds:
                print(
                    f"[Run] dataset={dataset} scenario={scenario} seed={seed}",
                    flush=True,
                )
                row = collect_entropy_shift(
                    data_path=args.data_path,
                    device=args.device,
                    dataset=dataset,
                    seed=seed,
                    backbone=args.backbone,
                    src_id=src_id,
                    trg_id=trg_id,
                )
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "seed": seed,
                        "h_raw_mean": row["h_raw_mean"],
                        "h_rand_mean": row["h_rand_mean"],
                        "h_adv_mean": row["h_adv_mean"],
                        "delta_mean": row["delta_mean"],
                    }
                )
                pooled_arrays[(dataset, scenario)]["raw"].append(row["raw_values"])
                pooled_arrays[(dataset, scenario)]["rand"].append(row["rand_values"])
                pooled_arrays[(dataset, scenario)]["adv"].append(row["adv_values"])

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "entropy_shift_seed_summary.csv", index=False)

    summary_rows = []
    for (dataset, scenario), group in raw_df.groupby(["dataset", "scenario"]):
        summary_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "h_raw_mean": float(group["h_raw_mean"].mean()),
                "h_raw_std": float(group["h_raw_mean"].std(ddof=0)),
                "h_rand_mean": float(group["h_rand_mean"].mean()),
                "h_rand_std": float(group["h_rand_mean"].std(ddof=0)),
                "h_adv_mean": float(group["h_adv_mean"].mean()),
                "h_adv_std": float(group["h_adv_mean"].std(ddof=0)),
                "delta_mean": float(group["delta_mean"].mean()),
                "delta_std": float(group["delta_mean"].std(ddof=0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "scenario"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "entropy_shift_summary.csv", index=False)

    dataset_avg_rows = []
    for dataset, group in summary_df.groupby("dataset"):
        dataset_avg_rows.append(
            {
                "dataset": dataset,
                "h_raw_mean": float(group["h_raw_mean"].mean()),
                "h_rand_mean": float(group["h_rand_mean"].mean()),
                "h_adv_mean": float(group["h_adv_mean"].mean()),
                "delta_mean": float(group["delta_mean"].mean()),
            }
        )
    pd.DataFrame(dataset_avg_rows).to_csv(output_dir / "entropy_shift_dataset_avg.csv", index=False)

    for key, arrays in pooled_arrays.items():
        arrays["raw"] = np.concatenate(arrays["raw"], axis=0)
        arrays["rand"] = np.concatenate(arrays["rand"], axis=0)
        arrays["adv"] = np.concatenate(arrays["adv"], axis=0)

    for dataset in datasets:
        plot_dataset_distribution(dataset, summary_df, pooled_arrays, output_dir)

    print(f"Entropy shift analysis completed: {output_dir}")


if __name__ == "__main__":
    main()
