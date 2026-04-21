import argparse
import ast
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

from algorithms.accup_instrumented import ACCUPInstrumented
from algorithms.eata_instrumented import EATAInstrumented
from algorithms.sar_instrumented import SARInstrumented
from algorithms.tent_instrumented import TentInstrumented
from dataloader.corruption_transforms import CORRUPTION_REGISTRY
from scripts.supplementary_utils import (
    BatchTransformLoader,
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    move_data_to_device,
)


METHOD_REGISTRY = {
    "ACCUP": ("ACCUP", ACCUPInstrumented),
    "NUSTAR": ("ACCUP", ACCUPInstrumented),
    "EATA": ("EATA", EATAInstrumented),
    "TENT": ("Tent", TentInstrumented),
    "SAR": ("SAR", SARInstrumented),
}


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def parse_method_list(method_text):
    return [item.strip().upper() for item in str(method_text).split(",") if item.strip()]


def parse_override_value(raw_value):
    text = str(raw_value).strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_overrides(entries):
    overrides = {}
    for entry in entries or []:
        if "=" not in str(entry):
            raise ValueError(f"Invalid --override value: {entry}")
        key, value = str(entry).split("=", 1)
        overrides[key.strip()] = parse_override_value(value)
    return overrides


def override_tag(overrides):
    if not overrides:
        return ""
    parts = []
    for key in sorted(overrides):
        value = str(overrides[key]).replace(".", "p").replace("/", "_")
        parts.append(f"{key}_{value}")
    return "_ovr_" + "_".join(parts[:4])


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def build_payload(data, labels, corruption_type, severity):
    return {
        "data": data,
        "labels": labels,
        "meta": {
            "corruption_phase": "corrupted" if corruption_type else "clean",
            "corruption_type": corruption_type,
            "severity": severity,
        },
    }


def extract_selected_mask(tta_model, batch_size):
    gate_log = getattr(tta_model, "_last_gate_log", {}) or {}
    for key in ("selected_mask", "active_mask"):
        mask = gate_log.get(key)
        if isinstance(mask, torch.Tensor) and mask.numel() == batch_size:
            return mask.to(dtype=torch.bool)
    return torch.zeros(batch_size, dtype=torch.bool)


def run_once(args, method_name, seed):
    da_method, tta_model_class = METHOD_REGISTRY[method_name]
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=args.dataset,
        da_method=da_method,
        backbone=args.backbone,
        exp_name="wrong_class_confidence",
        seed=seed,
        tta_model_class=tta_model_class,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = args.scenario.split("->", 1)
        if args.overrides:
            trainer._train_params.update(args.overrides)
            trainer.hparams.update(args.overrides)
            merged_override = dict(trainer.get_scenario_override(src_id, trg_id))
            merged_override.update(args.overrides)
            trainer.store_scenario_override(src_id, trg_id, merged_override)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        loader = trainer.trg_whole_dl
        if args.corruption_type:
            loader = BatchTransformLoader(loader, CORRUPTION_REGISTRY[args.corruption_type], args.severity)

        sample_rows = []
        step_rows = []
        global_step = 0
        for batch_idx, (data, labels, indices) in enumerate(loader):
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            payload = build_payload(data, labels, args.corruption_type, args.severity)
            logits = tta_model(payload)
            probs = softmax_probs(logits)
            preds = probs.argmax(dim=1)
            max_conf, _ = probs.max(dim=1)
            wrong_mask = preds.ne(labels)
            wrong_class_conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            selected_mask = extract_selected_mask(tta_model, labels.size(0)).to(labels.device)
            high_conf_wrong = wrong_mask & (wrong_class_conf >= args.high_conf_threshold)

            wrong_vals = wrong_class_conf[wrong_mask]
            step_rows.append(
                {
                    "dataset": args.dataset,
                    "scenario": args.scenario,
                    "method": method_name,
                    "seed": seed,
                    "batch_idx": batch_idx,
                    "samples_seen": global_step + labels.size(0),
                    "wrong_count": int(wrong_mask.sum().item()),
                    "selected_count": int(selected_mask.sum().item()),
                    "mean_wrong_class_conf": float(wrong_vals.mean().item()) if wrong_vals.numel() > 0 else float("nan"),
                    "median_wrong_class_conf": float(wrong_vals.median().item()) if wrong_vals.numel() > 0 else float("nan"),
                    "high_conf_wrong_ratio": float(high_conf_wrong[wrong_mask].float().mean().item()) if wrong_vals.numel() > 0 else float("nan"),
                    "selected_wrong_ratio": float(selected_mask[wrong_mask].float().mean().item()) if wrong_vals.numel() > 0 else float("nan"),
                    "selected_high_conf_wrong_ratio": float((selected_mask & high_conf_wrong)[wrong_mask].float().mean().item()) if wrong_vals.numel() > 0 else float("nan"),
                }
            )

            indices_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
            for row_idx in range(labels.size(0)):
                sample_rows.append(
                    {
                        "dataset": args.dataset,
                        "scenario": args.scenario,
                        "method": method_name,
                        "seed": seed,
                        "batch_idx": batch_idx,
                        "stream_index": global_step + row_idx,
                        "sample_index": int(indices_list[row_idx]),
                        "y_true": int(labels[row_idx].item()),
                        "y_pred": int(preds[row_idx].item()),
                        "max_confidence": float(max_conf[row_idx].item()),
                        "wrong_class_confidence": float(wrong_class_conf[row_idx].item()),
                        "is_wrong": bool(wrong_mask[row_idx].item()),
                        "is_high_conf_wrong": bool(high_conf_wrong[row_idx].item()),
                        "selected_for_update": bool(selected_mask[row_idx].item()),
                    }
                )
            global_step += labels.size(0)

        return pd.DataFrame(sample_rows), pd.DataFrame(step_rows)
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def plot_wrong_conf_hist(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub_df in df.groupby("method"):
        values = sub_df.loc[sub_df["is_wrong"], "wrong_class_confidence"].dropna().to_numpy()
        if len(values) == 0:
            continue
        ax.hist(values, bins=30, density=True, alpha=0.4, label=method)
        ax.axvline(values.mean(), linestyle="--", linewidth=1.0)
    ax.set_xlabel("wrong-class confidence")
    ax.set_ylabel("Density")
    ax.set_title("Wrong-Class Confidence Distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_summary_bar(summary_df, column, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df[column], color=["#4c78a8", "#e45756", "#72b7b2"][: len(summary_df)])
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"].tolist())
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_step_curve(step_df, column, ylabel, output_path):
    agg = step_df.groupby(["method", "batch_idx"], as_index=False)[column].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub_df in agg.groupby("method"):
        ax.plot(sub_df["batch_idx"], sub_df[column], marker="o", linewidth=1.5, label=method)
    ax.set_xlabel("streaming batch index (adaptation step)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_xticks(sorted(agg["batch_idx"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--scenario", required=True, help="src->trg, e.g. 16->1")
    parser.add_argument("--methods", default="EATA,ACCUP")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--corruption_type", default=None, choices=sorted(CORRUPTION_REGISTRY.keys()))
    parser.add_argument("--severity", default="moderate")
    parser.add_argument("--high_conf_threshold", type=float, default=0.9)
    parser.add_argument("--override", action="append", default=None, help="Repeat key=value overrides for the active scenario.")
    args = parser.parse_args()
    args.overrides = parse_overrides(args.override)

    output_tag = f"{args.dataset}_{args.scenario.replace('->', 'to')}"
    if args.corruption_type:
        output_tag += f"_{args.corruption_type}_{args.severity}"
    output_tag += override_tag(args.overrides)
    output_dir = ensure_dir(RESULTS_ROOT / "wrong_class_confidence" / output_tag)

    sample_frames = []
    step_frames = []
    methods = parse_method_list(args.methods)
    seeds = parse_seed_list(args.seeds)
    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            raise ValueError(f"Unsupported method: {method_name}")
        for seed in seeds:
            print(f"[Run] method={method_name} seed={seed} scenario={args.scenario}", flush=True)
            sample_df, step_df = run_once(args, method_name, seed)
            sample_frames.append(sample_df)
            step_frames.append(step_df)

    samples_df = pd.concat(sample_frames, ignore_index=True)
    steps_df = pd.concat(step_frames, ignore_index=True)
    samples_df.to_csv(output_dir / "sample_level.csv", index=False)
    steps_df.to_csv(output_dir / "step_level.csv", index=False)

    wrong_df = samples_df[samples_df["is_wrong"]].copy()
    summary_rows = []
    for method, sub_df in wrong_df.groupby("method"):
        values = sub_df["wrong_class_confidence"].dropna().to_numpy()
        summary_rows.append(
            {
                "method": method,
                "wrong_count": int(len(values)),
                "mean_wrong_class_conf": float(np.mean(values)) if len(values) else float("nan"),
                "median_wrong_class_conf": float(np.median(values)) if len(values) else float("nan"),
                "p25_wrong_class_conf": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
                "p75_wrong_class_conf": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
                "p90_wrong_class_conf": float(np.quantile(values, 0.90)) if len(values) else float("nan"),
                "high_conf_wrong_ratio": float(sub_df["is_high_conf_wrong"].mean()) if len(values) else float("nan"),
                "selected_high_conf_wrong_ratio": float(sub_df.loc[sub_df["is_high_conf_wrong"], "selected_for_update"].mean()) if sub_df["is_high_conf_wrong"].any() else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_wrong_class_conf", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    plot_wrong_conf_hist(samples_df, output_dir / "wrong_class_conf_hist.pdf")
    plot_summary_bar(summary_df, "mean_wrong_class_conf", "Mean Wrong-Class Confidence", output_dir / "summary_mean_wrong_conf.pdf")
    plot_step_curve(steps_df, "mean_wrong_class_conf", "Mean Wrong-Class Confidence vs Step", output_dir / "step_mean_wrong_conf.pdf")
    plot_step_curve(steps_df, "high_conf_wrong_ratio", "High-Confidence Wrong Ratio vs Step", output_dir / "step_high_conf_wrong_ratio.pdf")
    plot_step_curve(steps_df, "selected_high_conf_wrong_ratio", "Selected High-Confidence Wrong Ratio vs Step", output_dir / "step_selected_high_conf_wrong_ratio.pdf")

    print(f"Wrong-class confidence analysis completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
