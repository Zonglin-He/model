import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
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

from scripts.supplementary_utils import (
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
    prepare_scenario,
)
from utils.utils import fix_randomness, softmax_entropy_from_logits


DEFAULT_EEG_ORDER = ["16->1", "12->5", "9->14", "0->11", "7->18"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "case_study"))
    return parser.parse_args()


def scenario_slug(text: str) -> str:
    return text.replace("->", "_to_")


def find_eeg_ssaw_ablation(results_root: Path):
    files = sorted(
        results_root.rglob("ablation_delta_vs_full.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "ablation" not in df.columns:
            continue
        if "w/o_SSAW" not in df["ablation"].astype(str).tolist():
            continue
        scenario_cols = [col for col in df.columns if "->" in col]
        if not scenario_cols:
            continue
        row = df[df["ablation"].astype(str) == "w/o_SSAW"].iloc[0]
        ranked = []
        for scenario in scenario_cols:
            value = float(row[scenario])
            ranked.append(
                {
                    "scenario": scenario,
                    "delta_value": value,
                    "drop_points": max(0.0, -value * 100.0),
                    "source_file": str(path),
                }
            )
        ranked.sort(key=lambda item: item["drop_points"], reverse=True)
        return ranked
    return []


def build_scenario_candidates():
    ranked = find_eeg_ssaw_ablation(ROOT / "results" / "tta_experiments_logs")
    if ranked:
        seen = {item["scenario"] for item in ranked}
        for scenario in DEFAULT_EEG_ORDER:
            if scenario not in seen:
                ranked.append(
                    {
                        "scenario": scenario,
                        "delta_value": None,
                        "drop_points": None,
                        "source_file": None,
                    }
                )
        return ranked
    return [
        {
            "scenario": scenario,
            "delta_value": None,
            "drop_points": None,
            "source_file": None,
        }
        for scenario in DEFAULT_EEG_ORDER
    ]


def random_warp(x: torch.Tensor, epsilon: float, active_search) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")
    batch_size, _, target_len = x.shape
    if epsilon <= 0.0:
        return x
    controls = torch.empty(
        batch_size,
        active_search.num_control_points,
        device=x.device,
        dtype=x.dtype,
    ).uniform_(1.0 - epsilon, 1.0 + epsilon)
    curves = active_search._natural_cubic_spline_upsample(controls, target_len)
    return x * curves.unsqueeze(1)


@torch.no_grad()
def collect_views(tta_model, x: torch.Tensor):
    model = tta_model.model
    model.eval()

    raw_x = x
    rand_x = random_warp(raw_x, float(tta_model.adv_sigma), tta_model.active_search)
    adv_x = tta_model.get_adversarial_view(raw_x, model)

    raw_feats = tta_model.active_search._extract_features(model, raw_x)
    rand_feats = tta_model.active_search._extract_features(model, rand_x)
    adv_feats = tta_model.active_search._extract_features(model, adv_x)

    raw_logits = model.classifier(raw_feats)
    rand_logits = model.classifier(rand_feats)
    adv_logits = model.classifier(adv_feats)

    raw_probs = F.softmax(raw_logits, dim=1)
    rand_probs = F.softmax(rand_logits, dim=1)
    adv_probs = F.softmax(adv_logits, dim=1)

    return {
        "raw_feats": raw_feats.detach().cpu(),
        "rand_feats": rand_feats.detach().cpu(),
        "adv_feats": adv_feats.detach().cpu(),
        "raw_pred": raw_probs.argmax(dim=1).detach().cpu(),
        "rand_pred": rand_probs.argmax(dim=1).detach().cpu(),
        "adv_pred": adv_probs.argmax(dim=1).detach().cpu(),
        "rand_entropy": softmax_entropy_from_logits(rand_logits).detach().cpu(),
        "adv_entropy": softmax_entropy_from_logits(adv_logits).detach().cpu(),
    }


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return 1.0 - (a_norm * b_norm).sum(dim=1)


def build_prototypes(raw_feats: torch.Tensor, raw_pred: torch.Tensor):
    prototypes = {}
    for cls_idx in torch.unique(raw_pred).tolist():
        cls_mask = raw_pred == cls_idx
        cls_feats = raw_feats[cls_mask]
        if cls_feats.numel() == 0:
            continue
        prototypes[int(cls_idx)] = cls_feats.mean(dim=0)
    return prototypes


def dist_to_prototype(feats: torch.Tensor, labels: torch.Tensor, prototypes):
    rows = []
    classes = []
    for idx in range(feats.size(0)):
        cls_idx = int(labels[idx].item())
        if cls_idx not in prototypes:
            continue
        rows.append(cosine_distance(feats[idx:idx + 1], prototypes[cls_idx].unsqueeze(0))[0].item())
        classes.append(cls_idx)
    return np.asarray(rows, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def _tsne_worker(features: np.ndarray):
    perplexity = min(30, max(5, (features.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def run_tsne_with_timeout(features: np.ndarray, timeout_sec: int = 180):
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_tsne_worker, features)
        return future.result(timeout=timeout_sec)


def make_tsne_plot(tsne_payload, scenario: str, metrics: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.scatter(
        tsne_payload["raw"][:, 0],
        tsne_payload["raw"][:, 1],
        s=12,
        alpha=0.4,
        c="#4c78a8",
        label="raw",
    )
    ax.scatter(
        tsne_payload["rand"][:, 0],
        tsne_payload["rand"][:, 1],
        s=12,
        alpha=0.4,
        c="#72b7b2",
        label="rand",
    )
    ax.scatter(
        tsne_payload["adv"][:, 0],
        tsne_payload["adv"][:, 1],
        s=12,
        alpha=0.4,
        c="#e45756",
        label="adv",
    )
    ax.scatter(
        tsne_payload["proto"][:, 0],
        tsne_payload["proto"][:, 1],
        s=200,
        c="gold",
        edgecolors="black",
        linewidths=0.8,
        marker="*",
        label="prototype",
    )
    text = (
        f"raw->rand: {metrics['cosine_raw_rand']:.4f}\n"
        f"raw->adv: {metrics['cosine_raw_adv']:.4f}\n"
        f"proto rand: {metrics['proto_dist_rand']:.4f}\n"
        f"proto adv: {metrics['proto_dist_adv']:.4f}\n"
        f"entropy gap: {metrics['entropy_gap']:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_title(f"Feature Space: Raw vs Random vs Adversarial Views ({scenario})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_proto_hist_plot(rand_dist, adv_dist, scenario: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.hist(rand_dist, bins=40, alpha=0.5, color="#72b7b2", density=True, label="rand")
    ax.hist(adv_dist, bins=40, alpha=0.5, color="#e45756", density=True, label="adv")
    rand_mean = float(np.mean(rand_dist))
    adv_mean = float(np.mean(adv_dist))
    delta = adv_mean - rand_mean
    ax.axvline(rand_mean, color="#72b7b2", linestyle="--", linewidth=1.2)
    ax.axvline(adv_mean, color="#e45756", linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        0.94,
        f"delta={delta:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel("Cosine distance to assigned class prototype")
    ax.set_ylabel("Density")
    ax.set_title(f"Distance to Class Prototype ({scenario})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def select_scenario_with_enough_samples(args):
    candidates = build_scenario_candidates()
    for item in candidates:
        scenario = item["scenario"]
        src_id, trg_id = scenario.split("->")
        trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=args.dataset,
            da_method="ACCUP",
            exp_name="feature_case_study_probe",
            seed=args.seed,
            backbone=args.backbone,
        )
        try:
            prepare_scenario(trainer, src_id, trg_id, run_seed=args.seed)
            sample_count = len(trainer.trg_whole_dl.dataset)
        finally:
            cleanup_trainer(trainer, close_summary=True)
        if sample_count >= 100:
            item["sample_count"] = sample_count
            return item
    fallback = candidates[0]
    fallback["sample_count"] = sample_count
    return fallback


def run_case_study(args, selected):
    fix_randomness(args.seed)
    src_id, trg_id = selected["scenario"].split("->")
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=args.dataset,
        da_method="ACCUP",
        exp_name="feature_case_study",
        seed=args.seed,
        backbone=args.backbone,
    )
    tta_model = None
    pre_trained_model = None
    try:
        prepare_scenario(trainer, src_id, trg_id, run_seed=args.seed)
        cache_path = trainer._pretrain_cache_path()
        if not cache_path or not Path(cache_path).exists():
            print(f"Checkpoint not found. Attempted path: {cache_path}")
            raise SystemExit(1)

        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=args.seed)
        tta_model.model.eval()

        raw_feats_list = []
        rand_feats_list = []
        adv_feats_list = []
        raw_pred_list = []
        rand_pred_list = []
        adv_pred_list = []
        labels_list = []
        rand_entropy_list = []
        adv_entropy_list = []

        seen = 0
        with torch.no_grad():
            for batch in trainer.trg_whole_dl:
                if seen >= args.max_samples:
                    break
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    data = batch
                    labels = None

                x = extract_primary_tensor(data).float().to(trainer.device)
                take = min(x.size(0), args.max_samples - seen)
                x = x[:take]
                if labels is not None:
                    labels = labels[:take]

                payload = collect_views(tta_model, x)
                raw_feats_list.append(payload["raw_feats"])
                rand_feats_list.append(payload["rand_feats"])
                adv_feats_list.append(payload["adv_feats"])
                raw_pred_list.append(payload["raw_pred"])
                rand_pred_list.append(payload["rand_pred"])
                adv_pred_list.append(payload["adv_pred"])
                rand_entropy_list.append(payload["rand_entropy"])
                adv_entropy_list.append(payload["adv_entropy"])
                if labels is not None:
                    labels_list.append(labels.detach().cpu())
                seen += take

        raw_feats = torch.cat(raw_feats_list, dim=0)
        rand_feats = torch.cat(rand_feats_list, dim=0)
        adv_feats = torch.cat(adv_feats_list, dim=0)
        raw_pred = torch.cat(raw_pred_list, dim=0)
        rand_pred = torch.cat(rand_pred_list, dim=0)
        adv_pred = torch.cat(adv_pred_list, dim=0)
        rand_entropy = torch.cat(rand_entropy_list, dim=0)
        adv_entropy = torch.cat(adv_entropy_list, dim=0)
        labels = torch.cat(labels_list, dim=0) if labels_list else None

        prototypes = build_prototypes(raw_feats, raw_pred)
        proto_dist_rand, proto_classes_rand = dist_to_prototype(rand_feats, raw_pred, prototypes)
        proto_dist_adv, proto_classes_adv = dist_to_prototype(adv_feats, raw_pred, prototypes)

        cosine_raw_rand = cosine_distance(raw_feats, rand_feats).mean().item()
        cosine_raw_adv = cosine_distance(raw_feats, adv_feats).mean().item()
        proto_mean_rand = float(np.mean(proto_dist_rand))
        proto_mean_adv = float(np.mean(proto_dist_adv))
        entropy_rand_mean = float(rand_entropy.mean().item())
        entropy_adv_mean = float(adv_entropy.mean().item())

        classwise = {}
        for cls_idx in sorted(prototypes.keys()):
            rand_mask = proto_classes_rand == cls_idx
            adv_mask = proto_classes_adv == cls_idx
            classwise[str(cls_idx)] = {
                "proto_dist_rand": float(np.mean(proto_dist_rand[rand_mask])) if np.any(rand_mask) else None,
                "proto_dist_adv": float(np.mean(proto_dist_adv[adv_mask])) if np.any(adv_mask) else None,
                "count": int((raw_pred.numpy() == cls_idx).sum()),
            }

        metrics = {
            "cosine_raw_rand": float(cosine_raw_rand),
            "cosine_raw_adv": float(cosine_raw_adv),
            "cosine_ratio_adv_over_rand": float(cosine_raw_adv / cosine_raw_rand) if cosine_raw_rand > 0 else None,
            "proto_dist_rand": proto_mean_rand,
            "proto_dist_adv": proto_mean_adv,
            "proto_delta_adv_minus_rand": float(proto_mean_adv - proto_mean_rand),
            "entropy_rand_mean": entropy_rand_mean,
            "entropy_adv_mean": entropy_adv_mean,
            "entropy_gap": float(entropy_adv_mean - entropy_rand_mean),
        }

        warnings = []
        for key, value in list(metrics.items()):
            if value is None:
                continue
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                warnings.append(f"NaN/Inf detected in metric: {key}")
                metrics[key] = None

        raw_np = raw_feats.numpy()
        rand_np = rand_feats.numpy()
        adv_np = adv_feats.numpy()
        proto_np = np.stack([prototypes[key].numpy() for key in sorted(prototypes.keys())], axis=0)

        tsne_retry_note = None
        sample_idx = np.arange(raw_np.shape[0])
        combined = np.concatenate([raw_np, rand_np, adv_np, proto_np], axis=0)
        try:
            embedding = run_tsne_with_timeout(combined, timeout_sec=180)
        except TimeoutError:
            tsne_retry_note = "t-SNE exceeded 3 minutes at full sample count; retried with 500 samples."
            keep = min(500, raw_np.shape[0])
            sample_idx = np.random.default_rng(42).choice(raw_np.shape[0], size=keep, replace=False)
            sample_idx.sort()
            reduced = np.concatenate(
                [raw_np[sample_idx], rand_np[sample_idx], adv_np[sample_idx], proto_np],
                axis=0,
            )
            embedding = run_tsne_with_timeout(reduced, timeout_sec=180)

        n_view = sample_idx.shape[0]
        proto_offset = 3 * n_view
        tsne_payload = {
            "raw": embedding[:n_view],
            "rand": embedding[n_view:2 * n_view],
            "adv": embedding[2 * n_view:3 * n_view],
            "proto": embedding[proto_offset:proto_offset + proto_np.shape[0]],
        }

        output_dir = ensure_dir(Path(args.output_dir))
        tsne_path = output_dir / f"tsne_{scenario_slug(selected['scenario'])}.pdf"
        proto_hist_path = output_dir / f"proto_dist_{scenario_slug(selected['scenario'])}.pdf"
        make_tsne_plot(tsne_payload, selected["scenario"], metrics, tsne_path)
        make_proto_hist_plot(proto_dist_rand, proto_dist_adv, selected["scenario"], proto_hist_path)

        summary = {
            "scenario": selected["scenario"],
            "ssaw_ablation_drop_points": selected.get("drop_points"),
            "ablation_source_file": selected.get("source_file"),
            "samples_analyzed": int(raw_np.shape[0]),
            "checkpoint_path": cache_path,
            "hparams": {
                "adv_sigma": float(getattr(tta_model, "adv_sigma", 0.0)),
                "adv_num_candidates": int(getattr(tta_model, "adv_num_candidates", 0)),
                "adv_ctrl_points": int(getattr(tta_model, "adv_ctrl_points", 0)),
            },
            "metrics": metrics,
            "classwise_proto_dist": classwise,
            "tsne_retry_note": tsne_retry_note,
            "warnings": warnings,
            "label_available": labels is not None,
            "figure_paths": {
                "tsne": str(tsne_path),
                "proto_hist": str(proto_hist_path),
            },
        }
        summary_path = output_dir / f"summary_{scenario_slug(selected['scenario'])}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def print_case_summary(summary):
    metrics = summary["metrics"]
    drop = summary.get("ssaw_ablation_drop_points")
    drop_text = "unknown" if drop is None else f"{drop:.4f}"
    print(f"=== CASE STUDY: {summary['scenario']} ===")
    print(f"Scenario selected because: SSAW ablation drop = {drop_text} F1 points")
    print(f"Samples analyzed: {summary['samples_analyzed']}")
    print()
    print("Feature displacement:")
    print(f"  Cosine dist raw->rand: {metrics['cosine_raw_rand']:.4f}")
    print(f"  Cosine dist raw->adv:  {metrics['cosine_raw_adv']:.4f}")
    print(f"  Ratio (adv/rand):      {metrics['cosine_ratio_adv_over_rand']:.2f}x")
    print()
    print("Prototype proximity:")
    print(f"  Proto dist rand: {metrics['proto_dist_rand']:.4f}")
    print(f"  Proto dist adv:  {metrics['proto_dist_adv']:.4f}")
    print(f"  Delta (adv-rand): {metrics['proto_delta_adv_minus_rand']:.4f}  [positive = adv pushes toward boundary]")
    print()
    print("Entropy:")
    print(f"  H(rand): {metrics['entropy_rand_mean']:.4f}")
    print(f"  H(adv):  {metrics['entropy_adv_mean']:.4f}")
    print(f"  Gap:     {metrics['entropy_gap']:.4f}")
    print()
    print(f"Figures: {Path(summary['figure_paths']['tsne']).parent}")
    if summary.get("tsne_retry_note"):
        print(summary["tsne_retry_note"])


def main():
    args = parse_args()
    selected = select_scenario_with_enough_samples(args)
    summary = run_case_study(args, selected)
    print_case_summary(summary)


if __name__ == "__main__":
    main()
