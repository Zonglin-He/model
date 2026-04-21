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

from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
)


REPRESENTATIVE_SCENARIOS = {
    "EEG": ("12", "5"),
    "HAR": ("12", "16"),
    "FD": ("0", "1"),
}

HYPERPARAMETER_GRIDS = {
    "adv_sigma": [0.05, 0.10, 0.15, 0.30, 0.50],
    "adv_num_candidates": [8, 20, 24, 32, 56],
    "sem_thresh": [0.05, 0.10, 0.20, 0.25, 0.40],
    "cons_thresh": [0.01, 0.05, 0.10, 0.20, 0.31],
}

PARAM_LABELS = {
    "adv_sigma": "SSAW sigma",
    "adv_num_candidates": "Num candidates",
    "sem_thresh": "Semantic threshold",
    "cons_thresh": "Consistency threshold",
}


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def evaluate_setting(data_path, device, dataset, seed, backbone, src_id, trg_id, param_name, param_value):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        exp_name="hparam_sensitivity",
        seed=seed,
        backbone=backbone,
    )
    tta_model = None
    pre_trained_model = None
    try:
        baseline = trainer.get_scenario_override(src_id, trg_id)
        trainer.store_scenario_override(src_id, trg_id, {param_name: param_value})
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        f1 = float(trainer.calculate_metrics(tta_model)[1])
        return {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "seed": seed,
            "hyperparameter": param_name,
            "value": param_value,
            "f1": f1,
            "baseline_value": baseline.get(param_name),
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def style_axis(ax, title):
    ax.set_title(title)
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "hparam_sensitivity")
    seeds = parse_seed_list(args.seeds)
    raw_rows = []

    for dataset, (src_id, trg_id) in REPRESENTATIVE_SCENARIOS.items():
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found", flush=True)
            continue
        for hyperparameter, values in HYPERPARAMETER_GRIDS.items():
            for value in values:
                for seed in seeds:
                    print(
                        f"[Run] dataset={dataset} scenario={src_id}->{trg_id} "
                        f"param={hyperparameter} value={value} seed={seed}",
                        flush=True,
                    )
                    raw_rows.append(
                        evaluate_setting(
                            data_path=args.data_path,
                            device=args.device,
                            dataset=dataset,
                            seed=seed,
                            backbone=args.backbone,
                            src_id=src_id,
                            trg_id=trg_id,
                            param_name=hyperparameter,
                            param_value=value,
                        )
                    )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "hparam_sensitivity_raw.csv", index=False)

    summary_rows = []
    for (dataset, scenario, hyperparameter, value, baseline_value), group in raw_df.groupby(
        ["dataset", "scenario", "hyperparameter", "value", "baseline_value"]
    ):
        vals = group["f1"].astype(float)
        summary_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "hyperparameter": hyperparameter,
                "value": value,
                "baseline_value": baseline_value,
                "mean_f1": float(vals.mean()),
                "std_f1": float(vals.std(ddof=0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "hparam_sensitivity_summary.csv", index=False)

    for hyperparameter, values in HYPERPARAMETER_GRIDS.items():
        fig, ax = plt.subplots(figsize=(7, 4.8))
        sub = summary_df[summary_df["hyperparameter"] == hyperparameter].copy()
        for dataset in ("EEG", "HAR", "FD"):
            ds = sub[sub["dataset"] == dataset].sort_values("value")
            if ds.empty:
                continue
            x = ds["value"].astype(float).tolist()
            y = (ds["mean_f1"] * 100.0).tolist()
            yerr = (ds["std_f1"] * 100.0).tolist()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=f"{dataset} ({ds.iloc[0]['scenario']})",
            )
        ax.set_xlabel(PARAM_LABELS[hyperparameter])
        style_axis(ax, f"Hyperparameter Sensitivity | {PARAM_LABELS[hyperparameter]}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{hyperparameter}_sensitivity.pdf")
        fig.savefig(output_dir / f"{hyperparameter}_sensitivity.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()
    for ax, hyperparameter in zip(axes_flat, HYPERPARAMETER_GRIDS):
        sub = summary_df[summary_df["hyperparameter"] == hyperparameter].copy()
        for dataset in ("EEG", "HAR", "FD"):
            ds = sub[sub["dataset"] == dataset].sort_values("value")
            if ds.empty:
                continue
            ax.errorbar(
                ds["value"].astype(float).tolist(),
                (ds["mean_f1"] * 100.0).tolist(),
                yerr=(ds["std_f1"] * 100.0).tolist(),
                marker="o",
                linewidth=1.6,
                capsize=3,
                label=dataset,
            )
        ax.set_xlabel(PARAM_LABELS[hyperparameter])
        style_axis(ax, PARAM_LABELS[hyperparameter])
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "hparam_sensitivity_overview.pdf")
    fig.savefig(output_dir / "hparam_sensitivity_overview.png", dpi=200)
    plt.close(fig)

    print("Hyperparameter sensitivity experiments completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
