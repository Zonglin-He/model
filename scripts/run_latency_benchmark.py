import argparse
import statistics
import sys
import time
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
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    move_data_to_device,
)


DA_METHODS = ["ACCUP", "NoAdap"]
DATASETS = ["EEG", "HAR", "FD"]
NUM_WARMUP_BATCHES = 10
NUM_MEASURE_BATCHES = 50
SEED = 42
NCAND_VALUES = [1, 4, 8, 16, 32, 64]
BATCH_SIZES = [16, 32, 64, 128, 256]


def sync_if_needed(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_batches(loader):
    return list(loader)


def measure_method(data_path, device, dataset, method, backbone, override=None, need_f1=True):
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method=method,
        tta_model_class=ACCUPInstrumented if method == "ACCUP" else None,
        exp_name="latency",
        seed=SEED,
        backbone=backbone,
    )
    override = override or {}
    try:
        src_id, trg_id = dataset_scenarios(trainer)[0]
        if override:
            trainer.store_scenario_override(src_id, trg_id, override)
        tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=SEED)
        batches = collect_batches(trainer.trg_whole_dl)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=trainer.device)

        timings = []
        total_needed = NUM_WARMUP_BATCHES + NUM_MEASURE_BATCHES
        for batch_idx in range(total_needed):
            data, _, _ = batches[batch_idx % len(batches)]
            data = move_data_to_device(data, trainer.device)
            sync_if_needed(device)
            start = time.perf_counter()
            _ = tta_model(data)
            sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if batch_idx >= NUM_WARMUP_BATCHES:
                timings.append(elapsed_ms)

        peak_gpu_mb = float("nan")
        if str(device).startswith("cuda") and torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated(device=trainer.device) / (1024 ** 2)

        f1_value = float("nan")
        if need_f1:
            fresh_trainer = build_trainer(
                data_path=data_path,
                device=device,
                dataset=dataset,
                da_method=method,
                tta_model_class=ACCUPInstrumented if method == "ACCUP" else None,
                exp_name="latency_f1",
                seed=SEED,
                backbone=backbone,
            )
            try:
                if override:
                    fresh_trainer.store_scenario_override(src_id, trg_id, override)
                fresh_model, _ = create_tta_model(fresh_trainer, src_id, trg_id, run_seed=SEED)
                f1_value = float(fresh_trainer.calculate_metrics(fresh_model)[1])
            finally:
                fresh_trainer.summary_f1_scores.close()

        return {
            "method": method,
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "latency_mean_ms": float(statistics.mean(timings)),
            "latency_std_ms": float(statistics.pstdev(timings)),
            "peak_gpu_mb": peak_gpu_mb,
            "f1": f1_value,
        }
    finally:
        trainer.summary_f1_scores.close()


def plot_pareto(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(row["latency_mean_ms"], row["f1"], s=80)
        ax.text(row["latency_mean_ms"], row["f1"], f'{row["method"]}-{row["dataset"]}', fontsize=8)
    ax.set_xlabel("latency_mean_ms")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy-Cost Pareto")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_latency_comparison(df, output_path):
    pivot = df.pivot(index="dataset", columns="method", values="latency_mean_ms")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("Latency comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ncand_curve(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df["adv_num_candidates"], df["latency_mean_ms"], yerr=df["latency_std_ms"], marker="o")
    ax.set_xlabel("Ncand")
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("Ncand cost curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--seeds", default="41,42,43")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "latency")

    latency_rows = []
    for dataset in DATASETS:
        if not (Path(args.data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        for method in DA_METHODS:
            latency_rows.append(measure_method(args.data_path, args.device, dataset, method, args.backbone))
    latency_df = pd.DataFrame(latency_rows)
    latency_df.to_csv(output_dir / "latency_results.csv", index=False)

    ncand_rows = []
    for ncand in NCAND_VALUES:
        ncand_rows.append(
                {
                    **measure_method(
                        args.data_path,
                        args.device,
                        "EEG",
                        "ACCUP",
                        args.backbone,
                        override={
                            "adv_num_candidates": ncand,
                            "enable_piecewise_adv": True,
                        "adv_sigma": 0.1,
                    },
                    need_f1=False,
                ),
                "adv_num_candidates": ncand,
            }
        )
    ncand_df = pd.DataFrame(ncand_rows)
    ncand_df.to_csv(output_dir / "ncand_cost_curve.csv", index=False)

    batch_rows = []
    for batch_size in BATCH_SIZES:
        for method in DA_METHODS:
            batch_rows.append(
                {
                    **measure_method(
                        args.data_path,
                        args.device,
                        "EEG",
                        method,
                        args.backbone,
                        override={"batch_size": batch_size},
                        need_f1=True,
                    ),
                    "batch_size": batch_size,
                }
            )
    batch_df = pd.DataFrame(batch_rows)
    batch_df.to_csv(output_dir / "batchsize_cost_curve.csv", index=False)

    plot_latency_comparison(latency_df, output_dir / "latency_comparison.pdf")
    plot_ncand_curve(ncand_df, output_dir / "ncand_cost_curve.pdf")
    plot_pareto(latency_df, output_dir / "pareto_accuracy_cost.pdf")

    print("Latency benchmark completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
