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

from algorithms.accup_instrumented import ACCUPInstrumented
from dataloader.corruption_transforms import CORRUPTION_REGISTRY
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    apply_corruption_to_data,
    build_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    move_data_to_device,
    rolling_macro_f1,
)


DATASETS = ["EEG", "HAR", "FD"]
ROLLING_WINDOW = 50


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def get_corruption_for_step(step, total_steps):
    if step < 0.3 * total_steps:
        return None
    if step < 0.4 * total_steps:
        return ("burst_noise", "severe", "burst_noise_severe")
    return ("amplitude_drift", "moderate", "amplitude_drift_moderate")


def flatten_window(items):
    y_true, y_pred = [], []
    for true_part, pred_part in items:
        y_true.extend(true_part)
        y_pred.extend(pred_part)
    return y_true, y_pred


def run_streaming_once(data_path, device, dataset, seed, output_dir, backbone):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="ACCUP",
        tta_model_class=ACCUPInstrumented,
        exp_name="online_streaming",
        seed=seed,
        backbone=backbone,
    )
    scenario_rows = []
    onset_rows = []

    try:
        for src_id, trg_id in dataset_scenarios(trainer):
            tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
            per_batch_history = []
            cumulative_true, cumulative_pred = [], []
            loader = trainer.trg_whole_dl
            total_steps = len(loader)
            onset_rows.append({"dataset": dataset, "scenario": f"{src_id}->{trg_id}", "seed": seed, "onset": int(0.3 * total_steps)})

            for batch_idx, (data, labels, _) in enumerate(loader):
                schedule = get_corruption_for_step(batch_idx, total_steps)
                phase_name = "clean"
                data = move_data_to_device(data, trainer.device)
                labels = labels.view(-1).long().to(trainer.device)
                if schedule is not None:
                    corruption_name, severity, phase_name = schedule
                    data = apply_corruption_to_data(data, CORRUPTION_REGISTRY[corruption_name], severity)

                payload = {
                    "data": data,
                    "labels": labels,
                    "meta": {
                        "corruption_phase": phase_name,
                        "corruption_type": None if schedule is None else schedule[0],
                        "severity": None if schedule is None else schedule[1],
                    },
                }
                logits = tta_model(payload)
                preds = logits.argmax(dim=1)
                true_np = labels.detach().cpu().tolist()
                pred_np = preds.detach().cpu().tolist()
                cumulative_true.extend(true_np)
                cumulative_pred.extend(pred_np)
                per_batch_history.append((true_np, pred_np))
                window_true, window_pred = flatten_window(per_batch_history[-ROLLING_WINDOW:])
                rolling_f1 = rolling_macro_f1(window_true, window_pred, len(window_true))[-1]
                cumulative_f1 = rolling_macro_f1(cumulative_true, cumulative_pred, len(cumulative_true))[-1]
                last_log = dict(tta_model.stream_log[-1])
                scenario_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": f"{src_id}->{trg_id}",
                        "seed": seed,
                        "batch_idx": batch_idx,
                        "samples_seen": last_log["samples_seen"],
                        "rolling_f1": rolling_f1,
                        "cumulative_f1": cumulative_f1,
                        "gate_acceptance_rate": last_log["gate_acceptance_rate"],
                        "proto_drift_norm": last_log["proto_drift_norm"],
                        "fisher_reg_value": last_log["fisher_reg_value"],
                        "corruption_phase": last_log["corruption_phase"],
                    }
                )

            scenario_df = pd.DataFrame([row for row in scenario_rows if row["scenario"] == f"{src_id}->{trg_id}" and row["seed"] == seed])
            scenario_path = output_dir / f"{dataset}_{src_id}to{trg_id}_seed{seed}.csv"
            scenario_df.to_csv(scenario_path, index=False)
    finally:
        trainer.summary_f1_scores.close()

    return pd.DataFrame(scenario_rows), pd.DataFrame(onset_rows)


def plot_metric(df, onset_df, dataset, metric, output_name, output_dir):
    dataset_df = df[df["dataset"] == dataset]
    agg = dataset_df.groupby("batch_idx")[metric].agg(["mean", "std"]).reset_index()
    onset = int(round(onset_df[onset_df["dataset"] == dataset]["onset"].mean()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agg["batch_idx"], agg["mean"], color="tab:blue")
    ax.fill_between(agg["batch_idx"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), alpha=0.2)
    ax.axvline(onset, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_title(f"{dataset} | {metric}")
    ax.set_xlabel("batch_idx")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / output_name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "online_streaming")
    all_frames = []
    onset_frames = []
    seeds = parse_seed_list(args.seeds)

    for dataset in DATASETS:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        for seed in seeds:
            run_df, onset_df = run_streaming_once(args.data_path, args.device, dataset, seed, output_dir, args.backbone)
            all_frames.append(run_df)
            onset_frames.append(onset_df)

    full_df = pd.concat(all_frames, ignore_index=True)
    onset_df = pd.concat(onset_frames, ignore_index=True)
    full_df.to_csv(output_dir / "streaming_all_results.csv", index=False)

    for dataset in DATASETS:
        plot_metric(full_df, onset_df, dataset, "rolling_f1", f"streaming_f1_{dataset}.pdf", output_dir)
        plot_metric(full_df, onset_df, dataset, "gate_acceptance_rate", f"streaming_gate_{dataset}.pdf", output_dir)
        plot_metric(full_df, onset_df, dataset, "proto_drift_norm", f"streaming_proto_drift_{dataset}.pdf", output_dir)
        plot_metric(full_df, onset_df, dataset, "fisher_reg_value", f"streaming_fisher_{dataset}.pdf", output_dir)

    print("Online streaming analysis completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
