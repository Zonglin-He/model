import argparse
import csv
import gc
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainers.tta_trainer import TTATrainer


SUMMARY_PATHS = {
    "EEG": ROOT / "results" / "tta_experiments_logs" / "eeg_stepwise_summary" / "summary.csv",
    "HAR": ROOT / "results" / "tta_experiments_logs" / "har_stepwise_summary" / "summary.csv",
    "FD": ROOT / "results" / "tta_experiments_logs" / "fd_stepwise_summary" / "summary.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-run the current config and compare results against stepwise summary.csv files."
    )
    parser.add_argument(
        "--datasets",
        default="EEG,HAR,FD",
        help="Comma-separated datasets to verify.",
    )
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "data" / "Dataset"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--seeds",
        default="41,42,43",
    )
    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "config_sync_verify_runs"),
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "config_sync_verify_summary"),
    )
    parser.add_argument(
        "--f1-tol",
        type=float,
        default=1e-4,
        help="Allowed absolute difference in F1 mean/std (percentage units).",
    )
    parser.add_argument(
        "--metric-tol",
        type=float,
        default=1e-4,
        help="Allowed absolute difference for general percentage metrics.",
    )
    return parser.parse_args()


def parse_dataset_list(text):
    values = [item.strip().upper() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dataset is required.")
    unsupported = [item for item in values if item not in SUMMARY_PATHS]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    return values


def build_trainer_args(args, dataset):
    seed_values = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    return Namespace(
        save_dir=str(Path(args.save_dir).resolve()),
        exp_name=f"verify_{dataset.lower()}",
        da_method="ACCUP",
        data_path=str(Path(args.data_path).resolve()),
        dataset=dataset,
        backbone="CNN",
        num_runs=max(1, len(seed_values)),
        device=args.device,
        seed=seed_values[0],
        seeds=",".join(str(seed) for seed in seed_values),
        pretrain_cache_dir=str(Path(args.pretrain_cache_dir).resolve()),
        disable_pretrain_cache=False,
        scenario=None,
        override=None,
    )


def load_expected_rows(summary_path):
    with summary_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {}
    for row in rows:
        expected[row["scenario"]] = row
    return expected


def compare_row(dataset, scenario, expected_row, actual_metrics, tol_f1, tol_metric):
    expected_f1 = float(expected_row["final_f1_mean"])
    expected_std = float(expected_row["final_f1_std"])
    actual_f1 = round(float(actual_metrics["f1_mean"]) * 100.0, 4)
    actual_std = round(float(actual_metrics["f1_std"]) * 100.0, 4)
    actual_acc = round(float(actual_metrics["acc_mean"]) * 100.0, 4)
    actual_auroc = round(float(actual_metrics["auroc_mean"]) * 100.0, 4)

    f1_diff = abs(actual_f1 - expected_f1)
    std_diff = abs(actual_std - expected_std)
    ok = (f1_diff <= tol_f1) and (std_diff <= tol_f1)

    return {
        "dataset": dataset,
        "scenario": scenario,
        "expected_f1_mean": expected_f1,
        "actual_f1_mean": actual_f1,
        "f1_diff": round(f1_diff, 6),
        "expected_f1_std": expected_std,
        "actual_f1_std": actual_std,
        "f1_std_diff": round(std_diff, 6),
        "actual_acc_mean": actual_acc,
        "actual_auroc_mean": actual_auroc,
        "target_status": expected_row["target_status"],
        "match": ok,
    }


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        fields = []
        for key in headers:
            value = str(row[key])
            if "," in value or "\"" in value:
                value = "\"" + value.replace("\"", "\"\"") + "\""
            fields.append(value)
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verification_rows = []
    dataset_summaries = {}

    for dataset in datasets:
        trainer = TTATrainer(build_trainer_args(args, dataset))
        trainer.test_time_adaptation()
        expected_rows = load_expected_rows(SUMMARY_PATHS[dataset])

        scenario_rows = []
        for (src_id, trg_id), metrics in sorted(trainer.scenario_metrics.items()):
            scenario = f"{src_id}->{trg_id}"
            expected = expected_rows[scenario]
            row = compare_row(
                dataset,
                scenario,
                expected,
                metrics,
                args.f1_tol,
                args.metric_tol,
            )
            verification_rows.append(row)
            scenario_rows.append(row)

        dataset_summaries[dataset] = {
            "num_scenarios": len(scenario_rows),
            "all_match": all(row["match"] for row in scenario_rows),
            "rows": scenario_rows,
            "exp_log_dir": str(trainer.exp_log_dir),
        }

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    overall = {
        "timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "all_match": all(row["match"] for row in verification_rows),
        "rows": verification_rows,
        "dataset_summaries": dataset_summaries,
    }

    (output_dir / "verification.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(output_dir / "verification.csv", verification_rows)

    print(f"Verification complete. all_match={overall['all_match']}")
    print(f"JSON: {output_dir / 'verification.json'}")
    print(f"CSV: {output_dir / 'verification.csv'}")


if __name__ == "__main__":
    main()
