import argparse
import random
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
from dataloader.corruption_transforms import burst_noise
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    apply_corruption_to_data,
    build_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    move_data_to_device,
)


DATASETS = ["EEG", "HAR", "FD"]
GATES = ["stat", "sem", "cons", "all"]


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def sample_mask(indices):
    idx = torch.as_tensor(indices)
    return (idx % 2) == 0


def mask_for_gate(gate_name, gate_log):
    if gate_name == "stat":
        return np.asarray(gate_log["stat_indices"], dtype=bool)
    if gate_name == "sem":
        return np.asarray(gate_log["sem_indices"], dtype=bool)
    if gate_name == "cons":
        return np.asarray(gate_log["cons_indices"], dtype=bool)
    return np.asarray(gate_log["active_indices"], dtype=bool)


def run_condition(data_path, device, dataset, seed, condition, backbone):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        tta_model_class=ACCUPInstrumented,
        exp_name="gate_diagnostics",
        seed=seed,
        backbone=backbone,
    )
    pass_rows, overlap_rows, pr_rows = [], [], []
    false_cases = []

    try:
        for src_id, trg_id in dataset_scenarios(trainer):
            tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
            for batch_idx, (data, labels, indices) in enumerate(trainer.trg_whole_dl):
                data_cpu = data
                corruption_mask = torch.zeros(labels.size(0), dtype=torch.bool)
                if condition == "corrupted":
                    corruption_mask = sample_mask(indices)
                    data_cpu = apply_corruption_to_data(data_cpu, burst_noise, "severe", sample_mask=corruption_mask)

                data_dev = move_data_to_device(data_cpu, trainer.device)
                labels_dev = labels.view(-1).long().to(trainer.device)
                payload = {
                    "data": data_dev,
                    "labels": labels_dev,
                    "meta": {"corruption_phase": condition},
                }
                _ = tta_model(payload)
                gate_log = dict(tta_model.gate_log[-1])
                batch_size = int(gate_log["B"])

                for gate in GATES:
                    gate_mask = mask_for_gate(gate, gate_log)
                    pass_rows.append(
                        {
                            "dataset": dataset,
                            "scenario": f"{src_id}->{trg_id}",
                            "seed": seed,
                            "condition": condition,
                            "gate": gate,
                            "batch_idx": batch_idx,
                            "pass_rate": float(gate_mask.mean()),
                        }
                    )

                stat_mask = mask_for_gate("stat", gate_log)
                sem_mask = mask_for_gate("sem", gate_log)
                cons_mask = mask_for_gate("cons", gate_log)
                overlap_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": f"{src_id}->{trg_id}",
                        "seed": seed,
                        "condition": condition,
                        "batch_idx": batch_idx,
                        "stat_sem_overlap": float((stat_mask & sem_mask).sum() / batch_size),
                        "stat_cons_overlap": float((stat_mask & cons_mask).sum() / batch_size),
                        "sem_cons_overlap": float((sem_mask & cons_mask).sum() / batch_size),
                    }
                )

                if condition == "corrupted":
                    clean_mask = (~corruption_mask).numpy()
                    corrupted_np = corruption_mask.numpy()
                    for gate in GATES:
                        gate_mask = mask_for_gate(gate, gate_log)
                        tp = float((gate_mask & clean_mask).sum())
                        passed = float(gate_mask.sum())
                        positives = float(clean_mask.sum())
                        precision = tp / passed if passed > 0 else np.nan
                        recall = tp / positives if positives > 0 else np.nan
                        pr_rows.append(
                            {
                                "dataset": dataset,
                                "scenario": f"{src_id}->{trg_id}",
                                "seed": seed,
                                "gate": gate,
                                "batch_idx": batch_idx,
                                "precision": precision,
                                "recall": recall,
                            }
                        )

                        false_accept = np.where(gate_mask & corrupted_np)[0]
                        false_reject = np.where((~gate_mask) & clean_mask)[0]
                        for sample_idx in false_accept[:2]:
                            false_cases.append(
                                {
                                    "dataset": dataset,
                                    "case_type": "false_accept",
                                    "gate": gate,
                                    "signal": data_cpu[0][sample_idx].detach().cpu().numpy() if isinstance(data_cpu, (list, tuple)) else data_cpu[sample_idx].detach().cpu().numpy(),
                                }
                            )
                        for sample_idx in false_reject[:2]:
                            false_cases.append(
                                {
                                    "dataset": dataset,
                                    "case_type": "false_reject",
                                    "gate": gate,
                                    "signal": data_cpu[0][sample_idx].detach().cpu().numpy() if isinstance(data_cpu, (list, tuple)) else data_cpu[sample_idx].detach().cpu().numpy(),
                                }
                            )
    finally:
        trainer.summary_f1_scores.close()

    return (
        pd.DataFrame(pass_rows),
        pd.DataFrame(overlap_rows),
        pd.DataFrame(pr_rows),
        false_cases,
    )


def plot_false_cases(false_cases, dataset, output_dir):
    dataset_cases = [case for case in false_cases if case["dataset"] == dataset]
    if not dataset_cases:
        return
    random.shuffle(dataset_cases)
    selected = dataset_cases[:10]
    fig, axes = plt.subplots(len(selected), 1, figsize=(9, 2 * len(selected)))
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, selected):
        signal = case["signal"]
        if signal.ndim > 1:
            signal = signal[0]
        ax.plot(signal, linewidth=1.0)
        ax.set_title(f'{case["case_type"]} | {case["gate"]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"gate_false_cases_{dataset}.pdf")
    plt.close(fig)


def plot_summary(pass_df, pr_df, output_dir):
    pass_summary = (
        pass_df.groupby(["condition", "gate"])["pass_rate"]
        .mean()
        .reset_index()
        .pivot(index="gate", columns="condition", values="pass_rate")
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    pass_summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Pass rate")
    ax.set_title("Gate pass rates")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "gate_pass_rate_comparison.pdf")
    plt.close(fig)

    if not pr_df.empty:
        pr_summary = pr_df.groupby("gate")[["precision", "recall"]].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        pr_summary.plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Gate precision / recall")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "gate_pr_comparison.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "gate_diagnostics")
    pass_frames, overlap_frames, pr_frames, false_cases = [], [], [], []
    seeds = parse_seed_list(args.seeds)

    for dataset in DATASETS:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        for seed in seeds:
            for condition in ["clean", "corrupted"]:
                pass_df, overlap_df, pr_df, cases = run_condition(
                    args.data_path, args.device, dataset, seed, condition, args.backbone
                )
                pass_frames.append(pass_df)
                overlap_frames.append(overlap_df)
                if not pr_df.empty:
                    pr_frames.append(pr_df)
                false_cases.extend(cases)

    pass_df = pd.concat(pass_frames, ignore_index=True)
    overlap_df = pd.concat(overlap_frames, ignore_index=True)
    pr_df = pd.concat(pr_frames, ignore_index=True) if pr_frames else pd.DataFrame()

    pass_stats = pass_df.groupby(["dataset", "scenario", "seed", "condition", "gate"])["pass_rate"].agg(["mean", "std"]).reset_index()
    pass_stats.columns = ["dataset", "scenario", "seed", "condition", "gate", "pass_rate_mean", "pass_rate_std"]
    pass_stats.to_csv(output_dir / "gate_pass_rates.csv", index=False)

    overlap_df.to_csv(output_dir / "gate_overlap.csv", index=False)
    pr_df.to_csv(output_dir / "gate_precision_recall.csv", index=False)

    for dataset in DATASETS:
        plot_false_cases(false_cases, dataset, output_dir)
    plot_summary(pass_df, pr_df, output_dir)

    print("Gate diagnostics completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
