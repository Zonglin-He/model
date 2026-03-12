import argparse
import glob
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
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
)


ABLATION_CONFIGS = {
    "Full_NuSTAR": {},
    "w/o_semantic_gate": {"sem_thresh": -999.0},
    "w/o_consistency_gate": {"cons_thresh": 999.0},
    "w/o_fisher": {"lambda_reg": 0.0, "fisher_alpha": 0.0},
    "w/o_SSAW": {"adv_sigma": 0.0, "adv_sigmas": [0.0], "enable_piecewise_adv": False},
    "w/o_all_gates": {"sem_thresh": -999.0, "cons_thresh": 999.0, "entropy_quantile": 1.0},
}
DATASETS = ["HAR", "FD"]


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def find_existing_eeg_table():
    patterns = [
        str(RESULTS_ROOT / "**" / "*ablation*.csv"),
        str(ROOT / "results" / "**" / "*ablation*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if "crossdataset_ablation" in match:
                continue
            try:
                df = pd.read_csv(match)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "crossdataset_ablation")
    raw_rows = []
    seeds = parse_seed_list(args.seeds)

    for dataset in DATASETS:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        for seed in seeds:
            base_trainer = build_trainer(
                data_path=args.data_path,
                device=args.device,
                dataset=dataset,
                da_method="ACCUP",
                exp_name="crossdataset_ablation_probe",
                seed=seed,
                backbone=args.backbone,
            )
            scenarios = dataset_scenarios(base_trainer)
            base_trainer.summary_f1_scores.close()
            for ablation_name, ablation_override in ABLATION_CONFIGS.items():
                trainer = build_trainer(
                    data_path=args.data_path,
                    device=args.device,
                    dataset=dataset,
                    da_method="ACCUP",
                    exp_name="crossdataset_ablation",
                    seed=seed,
                    backbone=args.backbone,
                )
                try:
                    for src_id, trg_id in scenarios:
                        trainer.store_scenario_override(src_id, trg_id, ablation_override)
                        tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
                        f1 = float(trainer.calculate_metrics(tta_model)[1])
                        raw_rows.append(
                            {
                                "dataset": dataset,
                                "scenario": f"{src_id}->{trg_id}",
                                "ablation": ablation_name,
                                "seed": seed,
                                "f1": f1,
                            }
                        )
                finally:
                    trainer.summary_f1_scores.close()

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "ablation_raw_results.csv", index=False)

    pivot = (
        raw_df.groupby(["ablation", "dataset", "scenario"])["f1"]
        .mean()
        .reset_index()
        .assign(dataset_scenario=lambda df: df["dataset"] + "_" + df["scenario"].str.replace("->", "to", regex=False))
        .pivot(index="ablation", columns="dataset_scenario", values="f1")
    )

    eeg_df = find_existing_eeg_table()
    if eeg_df is not None and "ablation" in eeg_df.columns:
        eeg_numeric = eeg_df.set_index("ablation")
        for column in eeg_numeric.columns:
            if column not in pivot.columns:
                pivot[column] = eeg_numeric[column]

    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_index(axis=1)
    pivot.to_csv(output_dir / "ablation_table.csv")

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), 5))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Cross-dataset ablation heatmap")
    fig.colorbar(im, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_heatmap.pdf")
    plt.close(fig)

    print("Cross-dataset ablation completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
