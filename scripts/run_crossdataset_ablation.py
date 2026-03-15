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
    dataset_scenarios,
    ensure_dir,
)


ABLATION_CONFIGS = {
    "Full_NuSTAR": {},
    "w/o_semantic_gate": {"enable_semantic_gate": False},
    "w/o_consistency_gate": {"enable_consistency_gate": False},
    "w/o_fisher": {"lambda_reg": 0.0, "fisher_alpha": 0.0, "online_fisher": False},
    # A fair SSAW ablation must also remove the SSAW-driven consistency branch.
    # Otherwise x_adv collapses to x, KL becomes identically zero, and the
    # objective turns into easier raw-view entropy minimization.
    "w/o_SSAW": {
        "enable_ssaw": False,
        "enable_consistency_gate": False,
        "adv_sigma": 0.0,
        "adv_sigmas": [0.0],
        "adv_num_candidates": 0,
    },
    "w/o_all_gates": {
        "enable_stat_gate": False,
        "enable_semantic_gate": False,
        "enable_consistency_gate": False,
    },
}
DATASETS = ["EEG", "HAR", "FD"]


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def parse_dataset_list(dataset_text):
    values = [dataset.strip().upper() for dataset in str(dataset_text).split(",") if dataset.strip()]
    if not values:
        raise ValueError("At least one dataset must be provided.")
    return values


def evaluate_ablation_scenario(data_path, device, dataset, seed, backbone, ablation_name, ablation_override, src_id, trg_id):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        exp_name="crossdataset_ablation",
        seed=seed,
        backbone=backbone,
    )
    trainer.store_scenario_override(src_id, trg_id, ablation_override)
    tta_model = None
    pre_trained_model = None
    try:
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        f1 = float(trainer.calculate_metrics(tta_model)[1])
        active_hparams = {
            "enable_ssaw": bool(trainer.hparams.get("enable_ssaw", True)),
            "enable_stat_gate": bool(trainer.hparams.get("enable_stat_gate", True)),
            "enable_semantic_gate": bool(trainer.hparams.get("enable_semantic_gate", True)),
            "enable_consistency_gate": bool(trainer.hparams.get("enable_consistency_gate", True)),
            "lambda_reg": float(trainer.hparams.get("lambda_reg", trainer.hparams.get("fisher_alpha", 0.0))),
            "fisher_alpha": float(trainer.hparams.get("fisher_alpha", 0.0)),
            "online_fisher": bool(trainer.hparams.get("online_fisher", True)),
            "adv_sigma": float(trainer.hparams.get("adv_sigma", 0.0)),
            "adv_num_candidates": int(trainer.hparams.get("adv_num_candidates", 0)),
        }
        return {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "ablation": ablation_name,
            "seed": seed,
            "f1": f1,
            **active_hparams,
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "crossdataset_ablation")
    raw_rows = []
    seeds = parse_seed_list(args.seeds)
    datasets = parse_dataset_list(args.datasets)

    for dataset in datasets:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        probe_trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method="ACCUP",
            exp_name="crossdataset_ablation_probe",
            seed=seeds[0],
            backbone=args.backbone,
        )
        scenarios = dataset_scenarios(probe_trainer)
        cleanup_trainer(probe_trainer, close_summary=True)

        for seed in seeds:
            for ablation_name, ablation_override in ABLATION_CONFIGS.items():
                print(f"[Run] dataset={dataset} seed={seed} ablation={ablation_name}", flush=True)
                for src_id, trg_id in scenarios:
                    raw_rows.append(
                        evaluate_ablation_scenario(
                            data_path=args.data_path,
                            device=args.device,
                            dataset=dataset,
                            seed=seed,
                            backbone=args.backbone,
                            ablation_name=ablation_name,
                            ablation_override=ablation_override,
                            src_id=src_id,
                            trg_id=trg_id,
                        )
                    )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "ablation_raw_results.csv", index=False)

    scenario_pivot = (
        raw_df.groupby(["ablation", "dataset", "scenario"])["f1"]
        .mean()
        .reset_index()
        .assign(dataset_scenario=lambda df: df["dataset"] + "_" + df["scenario"].str.replace("->", "to", regex=False))
        .pivot(index="ablation", columns="dataset_scenario", values="f1")
    )

    scenario_pivot = scenario_pivot.sort_index(axis=1)
    scenario_pivot.to_csv(output_dir / "ablation_table.csv")

    dataset_pivot = (
        raw_df.groupby(["ablation", "dataset"])["f1"]
        .mean()
        .reset_index()
        .pivot(index="ablation", columns="dataset", values="f1")
        .sort_index(axis=1)
    )
    dataset_pivot.to_csv(output_dir / "ablation_dataset_table.csv")

    if "Full_NuSTAR" in dataset_pivot.index:
        full_row = dataset_pivot.loc["Full_NuSTAR"]
        dataset_delta = dataset_pivot.subtract(full_row, axis=1)
        dataset_delta.to_csv(output_dir / "ablation_dataset_delta_vs_full.csv")

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(scenario_pivot.columns)), 5))
    im = ax.imshow(scenario_pivot.fillna(0).values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(scenario_pivot.columns)))
    ax.set_xticklabels(scenario_pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(scenario_pivot.index)))
    ax.set_yticklabels(scenario_pivot.index)
    ax.set_title("Cross-dataset ablation heatmap")
    fig.colorbar(im, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_heatmap.pdf")
    plt.close(fig)

    print("Cross-dataset ablation completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
