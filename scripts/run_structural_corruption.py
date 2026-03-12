import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.corruption_transforms import CORRUPTION_REGISTRY
from scripts.supplementary_utils import (
    BatchTransformLoader,
    RESULTS_ROOT,
    build_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
)


DATASETS = ["EEG", "HAR", "FD"]
CORRUPTION_TYPES = [
    "signal_freeze",
    "channel_dropout",
    "amplitude_drift",
    "piecewise_scaling",
    "burst_noise",
    "sensor_disconnect",
]
SEVERITIES = ["mild", "moderate", "severe"]
DA_METHODS = ["ACCUP", "NoAdap"]


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def evaluate_once(data_path, device, dataset, method, corruption_type, severity, seed, backbone):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method=method,
        backbone=backbone,
        exp_name="structural_corruption",
        seed=seed,
    )
    rows = []
    try:
        for src_id, trg_id in dataset_scenarios(trainer):
            tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
            trainer.trg_whole_dl = BatchTransformLoader(
                trainer.trg_whole_dl,
                CORRUPTION_REGISTRY[corruption_type],
                severity,
            )
            metrics = trainer.calculate_metrics(tta_model)
            rows.append(
                {
                    "dataset": dataset,
                    "corruption_type": corruption_type,
                    "severity": severity,
                    "method": method,
                    "seed": seed,
                    "scenario": f"{src_id}->{trg_id}",
                    "f1": float(metrics[1]),
                }
            )
    finally:
        trainer.summary_f1_scores.close()
    return rows


def plot_severity_curves(raw_df, output_dir):
    severity_order = {name: idx for idx, name in enumerate(SEVERITIES)}
    agg = (
        raw_df.groupby(["dataset", "method", "corruption_type", "severity"], as_index=False)["f1"]
        .mean()
        .assign(severity_order=lambda df: df["severity"].map(severity_order))
        .sort_values("severity_order")
    )

    for (dataset, method), group in agg.groupby(["dataset", "method"]):
        fig, ax = plt.subplots(figsize=(8, 5))
        for corruption_type, sub_df in group.groupby("corruption_type"):
            ax.plot(sub_df["severity"], sub_df["f1"], marker="o", label=corruption_type)
        ax.set_title(f"{dataset} | {method}")
        ax.set_xlabel("severity")
        ax.set_ylabel("Macro-F1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"severity_curve_{dataset}_{method}.pdf")
        plt.close(fig)


def plot_method_compare(raw_df, output_dir):
    severity_order = {name: idx for idx, name in enumerate(SEVERITIES)}
    agg = (
        raw_df.groupby(["dataset", "corruption_type", "method", "severity"], as_index=False)["f1"]
        .mean()
        .assign(severity_order=lambda df: df["severity"].map(severity_order))
        .sort_values("severity_order")
    )

    for (dataset, corruption_type), group in agg.groupby(["dataset", "corruption_type"]):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, sub_df in group.groupby("method"):
            ax.plot(sub_df["severity"], sub_df["f1"], marker="o", label=method)
        ax.set_title(f"{dataset} | {corruption_type}")
        ax.set_xlabel("severity")
        ax.set_ylabel("Macro-F1")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"method_compare_{dataset}_{corruption_type}.pdf")
        plt.close(fig)


def build_summary_table(raw_df):
    stats = (
        raw_df.groupby(["dataset", "method", "corruption_type", "severity"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    rows = []
    for (dataset, method, corruption_type), group in stats.groupby(["dataset", "method", "corruption_type"]):
        row = {"dataset": dataset, "method": method, "corruption_type": corruption_type}
        for severity in SEVERITIES:
            matched = group[group["severity"] == severity]
            row[f"{severity}_f1_mean"] = float(matched["mean"].iloc[0]) if not matched.empty else float("nan")
            row[f"{severity}_f1_std"] = float(matched["std"].iloc[0]) if not matched.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "structural_corruption")
    all_rows = []
    seeds = parse_seed_list(args.seeds)

    for dataset in DATASETS:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        for corruption_type in CORRUPTION_TYPES:
            for severity in SEVERITIES:
                for method in DA_METHODS:
                    for seed in seeds:
                        all_rows.extend(
                            evaluate_once(
                                data_path=args.data_path,
                                device=args.device,
                                dataset=dataset,
                                method=method,
                                corruption_type=corruption_type,
                                severity=severity,
                                seed=seed,
                                backbone=args.backbone,
                            )
                        )

    raw_df = pd.DataFrame(all_rows)
    raw_path = output_dir / "raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = build_summary_table(raw_df)
    summary_path = output_dir / "corruption_results_table.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_severity_curves(raw_df, output_dir)
    plot_method_compare(raw_df, output_dir)

    print("Structural corruption experiments completed.")
    print(f"Raw results: {raw_path}")
    print(f"Summary table: {summary_path}")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
