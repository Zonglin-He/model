import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    prepare_scenario,
)


DATASETS = ["EEG", "HAR", "FD"]
PRETRAIN_SEEDS = [10, 20, 30]
TTA_SEEDS = [41, 42, 43]
DA_METHODS = ["ACCUP", "NoAdap"]


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def pretrain_checkpoint_path(dataset, src_id, pretrain_seed):
    path = ROOT / "results" / "pretrain_cache" / "sensitivity"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"checkpoint_{dataset}_{src_id}_{pretrain_seed}.pt"


def ensure_pretrained_checkpoint(data_path, device, dataset, src_id, trg_id, pretrain_seed, backbone):
    ckpt_path = pretrain_checkpoint_path(dataset, src_id, pretrain_seed)
    if ckpt_path.exists():
        return ckpt_path
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        exp_name="source_sensitivity_pretrain",
        seed=pretrain_seed,
        backbone=backbone,
    )
    try:
        prepare_scenario(trainer, src_id, trg_id, run_seed=pretrain_seed, run_id=0)
        non_adapted, pre_trained_model = trainer.pre_train()
        torch.save({"non_adapted": non_adapted, "model_state": pre_trained_model.state_dict()}, ckpt_path)
    finally:
        trainer.summary_f1_scores.close()
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "source_sensitivity")
    raw_rows = []
    tta_seeds = parse_seed_list(args.seeds)

    for dataset in DATASETS:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        probe_trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method="ACCUP",
            exp_name="source_sensitivity_probe",
            seed=PRETRAIN_SEEDS[0],
            backbone=args.backbone,
        )
        scenarios = dataset_scenarios(probe_trainer)
        probe_trainer.summary_f1_scores.close()

        for src_id, trg_id in scenarios:
            checkpoints = {
                pretrain_seed: ensure_pretrained_checkpoint(
                    args.data_path, args.device, dataset, src_id, trg_id, pretrain_seed, args.backbone
                )
                for pretrain_seed in PRETRAIN_SEEDS
            }
            for method in DA_METHODS:
                for pretrain_seed, ckpt_path in checkpoints.items():
                    for tta_seed in tta_seeds:
                        trainer = build_trainer(
                            data_path=args.data_path,
                            device=args.device,
                            dataset=dataset,
                            da_method=method,
                            exp_name="source_sensitivity",
                            seed=tta_seed,
                            pretrained_checkpoint=str(ckpt_path),
                            backbone=args.backbone,
                        )
                        try:
                            tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=tta_seed)
                            f1 = float(trainer.calculate_metrics(tta_model)[1])
                            raw_rows.append(
                                {
                                    "dataset": dataset,
                                    "scenario": f"{src_id}->{trg_id}",
                                    "method": method,
                                    "pretrain_seed": pretrain_seed,
                                    "tta_seed": tta_seed,
                                    "f1": f1,
                                }
                            )
                        finally:
                            trainer.summary_f1_scores.close()

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "source_sensitivity_raw.csv", index=False)

    stats_rows = []
    grouped = raw_df.groupby(["dataset", "scenario", "method"])
    for (dataset, scenario, method), group in grouped:
        source_means = group.groupby("pretrain_seed")["f1"].mean()
        algo_means = group.groupby("tta_seed")["f1"].mean()
        stats_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "source_sensitivity_variance": float(source_means.var()),
                "algorithm_variance": float(algo_means.var()),
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_dir / "source_sensitivity_results.csv", index=False)

    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8, 5))
        dataset_df = raw_df[raw_df["dataset"] == dataset]
        box_data = []
        labels = []
        for method in DA_METHODS:
            method_df = dataset_df[dataset_df["method"] == method]
            per_pretrain = method_df.groupby("pretrain_seed")["f1"].mean().reindex(PRETRAIN_SEEDS)
            box_data.append(per_pretrain.values)
            labels.append(method)
        ax.boxplot(box_data, labels=labels)
        ax.set_title(f"Source sensitivity | {dataset}")
        ax.set_ylabel("Macro-F1")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"source_sensitivity_boxplot_{dataset}.pdf")
        plt.close(fig)

    print("Source sensitivity experiments completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
