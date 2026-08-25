"""Add DuSafe components cumulatively using dedicated runner classes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optuna_stepwise import parse_csv
from scripts.run_structural_ssaw_matrix import (
    PAIR_COLUMNS,
    RUNNER_SCRIPTS,
    collect_results,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir


ADDITION_STAGES = (
    ("addition_raw_entropy", "raw pseudo-label entropy minimization"),
    ("addition_confidence", "source-calibrated confidence admission"),
    ("addition_source_semantic", "raw-view source-semantic admission"),
    (
        "addition_full_ssaw",
        "physical view, label qualification, and invariance",
    ),
)


def addition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        previous_name = None
        for stage_index, (runner, added_component) in enumerate(
            ADDITION_STAGES
        ):
            current = dataset_frame[dataset_frame["runner"].eq(runner)]
            row = {
                "dataset": dataset,
                "stage_index": stage_index,
                "runner": runner,
                "added_component": added_component,
                "jobs": int(len(current)),
                "f1_mean": float(current["f1"].mean()),
                "paired_cells_vs_previous": 0,
                "incremental_f1_gain": float("nan"),
                "increment_helped_cells": 0,
                "increment_tied_cells": 0,
                "increment_hurt_cells": 0,
            }
            if previous_name is not None:
                previous = dataset_frame[
                    dataset_frame["runner"].eq(previous_name)
                ][[*PAIR_COLUMNS, "f1"]].rename(
                    columns={"f1": "previous_f1"}
                )
                paired = current.merge(
                    previous,
                    on=list(PAIR_COLUMNS),
                    how="inner",
                    validate="one_to_one",
                )
                gain = paired["f1"] - paired["previous_f1"]
                row.update(
                    {
                        "paired_cells_vs_previous": int(len(paired)),
                        "incremental_f1_gain": float(gain.mean()),
                        "increment_helped_cells": int((gain > 1e-12).sum()),
                        "increment_tied_cells": int(
                            (gain.abs() <= 1e-12).sum()
                        ),
                        "increment_hurt_cells": int((gain < -1e-12).sum()),
                    }
                )
            rows.append(row)
            previous_name = runner
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tuning-dir", required=True)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR")
    parser.add_argument(
        "--test-time-seeds",
        default=None,
        help="Optional seed subset forwarded to every dedicated runner.",
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--max-jobs-per-runner", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--control-points", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--invariance-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    datasets = [name.upper() for name in parse_csv(args.datasets)]
    if not args.aggregate_only:
        for runner, _ in ADDITION_STAGES:
            command = [
                sys.executable,
                str(ROOT / "scripts" / RUNNER_SCRIPTS[runner]),
                "--output-dir",
                str(output_root / runner),
                "--tuning-dir",
                str(args.tuning_dir),
                "--data-path",
                str(args.data_path),
                "--device",
                str(args.device),
                "--backbone",
                str(args.backbone),
                "--datasets",
                ",".join(datasets),
                "--pretrain-cache-dir",
                str(args.pretrain_cache_dir),
            ]
            optional_arguments = {
                "--test-time-seeds": args.test_time_seeds,
                "--max-jobs": args.max_jobs_per_runner,
                "--num-candidates": args.num_candidates,
                "--sigma": args.sigma,
                "--control-points": args.control_points,
                "--strength": args.strength,
                "--invariance-weight": args.invariance_weight,
                "--learning-rate": args.learning_rate,
                "--steps": args.steps,
                "--batch-size": args.batch_size,
            }
            for option, value in optional_arguments.items():
                if value is not None:
                    command.extend((option, str(value)))
            print(f"[Addition] launching {runner}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Addition runner {runner} exited with "
                    f"code {completed.returncode}"
                )
    stage_names = [name for name, _ in ADDITION_STAGES]
    frame = collect_results(output_root, datasets, stage_names)
    atomic_write_csv(frame, output_root / "addition_raw.csv", index=False)
    summary = addition_summary(frame)
    atomic_write_csv(
        summary, output_root / "addition_summary.csv", index=False
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
