"""Validate the one-view random SSAW production path on all paper domains."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optuna_stepwise import (
    acquire_run_lock,
    atomic_write_json,
    parse_csv,
    run_tta_job,
    scenario_label,
    scenario_pairs,
    utc_now,
)
from scripts.run_ssaw_internal_ablation import (
    SAFETY_METRICS,
    sanitized_tta_config,
    validate_state,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir


KEY_COLUMNS = ("dataset", "scenario", "source_seed", "test_time_seed")
LEGACY_VARIANTS = {
    "full": "legacy_ranked_source_full",
    "random_smooth_warp": "legacy_random_only",
    "no_source_supported_selection": "legacy_no_source_support_only",
    "random_no_source_support": "legacy_strict_joint",
    "no_ssaw": "legacy_no_ssaw",
}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_key(row: Mapping) -> tuple[str, str, int, int]:
    return (
        str(row["dataset"]).upper(),
        str(row["scenario"]),
        int(row["source_seed"]),
        int(row["test_time_seed"]),
    )


def validate_rows(
    rows: list[dict],
    *,
    dataset: str,
    scenario_names: set[str],
    source_seed: int,
    test_time_seeds: set[int],
) -> list[dict]:
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate simplified SSAW result row: {key}")
        seen.add(key)
        if key[0] != dataset or key[1] not in scenario_names:
            raise ValueError(f"{dataset}: foreign result row: {key}")
        if key[2] != source_seed or key[3] not in test_time_seeds:
            raise ValueError(f"{dataset}: seed mismatch in result row: {key}")
        if str(row.get("ablation", "full")) != "full":
            raise ValueError(f"{dataset}: validation must run production Full")
    return rows


def publish_dataset(rows: list[dict], dataset_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    atomic_write_csv(frame, dataset_dir / "raw.csv", index=False)


def _variant_frame(frame: pd.DataFrame, ablation: str, prefix: str) -> pd.DataFrame:
    available_metrics = [
        metric for metric in ("f1", *SAFETY_METRICS) if metric in frame.columns
    ]
    selected = frame[frame["ablation"].eq(ablation)][
        [*KEY_COLUMNS, *available_metrics]
    ].copy()
    if len(selected) != len(selected.drop_duplicates(list(KEY_COLUMNS))):
        raise ValueError(f"Legacy variant {ablation} has duplicate cells")
    return selected.rename(
        columns={metric: f"{prefix}_{metric}" for metric in available_metrics}
    )


def build_comparison(
    *,
    output_dir: Path,
    legacy_dir: Path,
    datasets: Iterable[str],
) -> pd.DataFrame:
    comparisons = []
    for dataset in datasets:
        current = pd.read_csv(output_dir / dataset / "raw.csv")
        current = current[[
            *KEY_COLUMNS,
            "f1",
            *[metric for metric in SAFETY_METRICS if metric in current.columns],
        ]].rename(
            columns={
                "f1": "simplified_f1",
                **{
                    metric: f"simplified_{metric}"
                    for metric in SAFETY_METRICS
                    if metric in current.columns
                },
            }
        )
        legacy = pd.read_csv(legacy_dir / dataset / "raw.csv")
        merged = current
        for ablation, prefix in LEGACY_VARIANTS.items():
            merged = merged.merge(
                _variant_frame(legacy, ablation, prefix),
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
        if len(merged) != len(current):
            raise ValueError(f"{dataset}: incomplete historical comparison")
        for prefix in LEGACY_VARIANTS.values():
            merged[f"delta_f1_vs_{prefix}"] = (
                merged["simplified_f1"] - merged[f"{prefix}_f1"]
            )
        comparisons.append(merged)
    result = pd.concat(comparisons, ignore_index=True)
    result = result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    atomic_write_csv(result, output_dir / "paired_comparison.csv", index=False)
    return result


def comparison_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(dataset, group) for dataset, group in comparison.groupby("dataset")]
    groups.append(("ALL", comparison))
    for dataset, group in groups:
        for prefix in LEGACY_VARIANTS.values():
            delta = group["simplified_f1"] - group[f"{prefix}_f1"]
            rows.append(
                {
                    "dataset": dataset,
                    "reference": prefix,
                    "paired_cells": int(len(group)),
                    "scenarios": int(group["scenario"].nunique()),
                    "simplified_f1_mean": float(group["simplified_f1"].mean()),
                    "reference_f1_mean": float(group[f"{prefix}_f1"].mean()),
                    "paired_f1_delta": float(delta.mean()),
                    "paired_f1_delta_std": float(delta.std(ddof=0)),
                    "paired_wins": int((delta > 1e-12).sum()),
                    "paired_ties": int((delta.abs() <= 1e-12).sum()),
                    "paired_losses": int((delta < -1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def scenario_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in comparison.columns
        if column == "simplified_f1"
        or column.endswith("_f1")
        or column.startswith("delta_f1_vs_")
    ]
    return (
        comparison.groupby(["dataset", "scenario"], as_index=False)[numeric]
        .mean()
        .sort_values(["dataset", "scenario"])
        .reset_index(drop=True)
    )


def safety_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in comparison.groupby("dataset"):
        for metric in SAFETY_METRICS:
            current = f"simplified_{metric}"
            if current not in group:
                continue
            row = {
                "dataset": dataset,
                "metric": metric,
                "simplified_mean": float(group[current].mean()),
            }
            for prefix in LEGACY_VARIANTS.values():
                reference = f"{prefix}_{metric}"
                if reference in group:
                    row[f"{prefix}_mean"] = float(group[reference].mean())
                    row[f"delta_vs_{prefix}"] = float(
                        (group[current] - group[reference]).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--tuning-dir",
        default=str(ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"),
    )
    parser.add_argument(
        "--legacy-dir",
        default=str(
            ROOT / "results" / "ablation" / "ssaw_internal_signal_removed_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "ablation" / "simplified_random_ssaw_v1"
        ),
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR,EEG,FD")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    return args


def main() -> int:
    args = parse_args()
    tuning_dir = Path(args.tuning_dir).resolve()
    legacy_dir = Path(args.legacy_dir).resolve()
    output_dir = ensure_dir(args.output_dir)
    lock = acquire_run_lock(output_dir)
    new_jobs = 0
    try:
        for dataset in args.datasets:
            scenarios = scenario_pairs(dataset)
            if len(scenarios) != 5:
                raise ValueError(f"{dataset}: expected exactly five scenarios")
            state = load_json(tuning_dir / dataset / "state.json")
            source_seed, test_time_seeds = validate_state(
                state, dataset=dataset, scenarios=scenarios
            )
            dataset_dir = ensure_dir(output_dir / dataset)
            raw_path = dataset_dir / "raw.csv"
            rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
            rows = validate_rows(
                rows,
                dataset=dataset,
                scenario_names={scenario_label(pair) for pair in scenarios},
                source_seed=source_seed,
                test_time_seeds=set(test_time_seeds),
            )
            completed = {row_key(row) for row in rows}
            tta_config = sanitized_tta_config(state)
            for scenario in scenarios:
                for test_time_seed in test_time_seeds:
                    key = (
                        dataset,
                        scenario_label(scenario),
                        source_seed,
                        int(test_time_seed),
                    )
                    if key in completed:
                        continue
                    print(
                        f"[Simplified SSAW] {dataset} {key[1]} "
                        f"seed={test_time_seed}",
                        flush=True,
                    )
                    row = run_tta_job(
                        dataset=dataset,
                        scenario=scenario,
                        source_seed=source_seed,
                        test_time_seed=int(test_time_seed),
                        source_config=state["source_config"],
                        tta_config=tta_config,
                        ablation="full",
                        data_path=args.data_path,
                        device=args.device,
                        backbone=args.backbone,
                        pretrain_cache_dir=args.pretrain_cache_dir,
                        include_batch_diagnostics=True,
                    )
                    rows.append(row)
                    completed.add(key)
                    new_jobs += 1
                    publish_dataset(rows, dataset_dir)
                    if args.max_jobs is not None and new_jobs >= args.max_jobs:
                        print("Reached --max-jobs; progress is saved.", flush=True)
                        return 0
            publish_dataset(rows, dataset_dir)

        comparison = build_comparison(
            output_dir=output_dir,
            legacy_dir=legacy_dir,
            datasets=args.datasets,
        )
        summary = comparison_summary(comparison)
        atomic_write_csv(summary, output_dir / "summary.csv", index=False)
        atomic_write_csv(
            scenario_summary(comparison),
            output_dir / "scenario_summary.csv",
            index=False,
        )
        atomic_write_csv(
            safety_summary(comparison),
            output_dir / "safety_summary.csv",
            index=False,
        )
        manifest = {
            "completed_at": utc_now(),
            "git_commit": git_commit(),
            "datasets": args.datasets,
            "jobs": int(len(comparison)),
            "scenarios_per_dataset": 5,
            "source_seed": 1,
            "test_time_seeds": [1, 2, 3],
            "production_variant": (
                "one random physical SSAW view + label-preserving invariance; "
                "no entropy ranking; no source-supported view selection"
            ),
            "legacy_references": LEGACY_VARIANTS,
            "selection_objective": "post_adaptation_macro_f1",
            "target_labels_used_for_original_tuning": True,
            "independent_source_seed_replication": False,
            "all_f1_values_finite": bool(
                comparison.filter(regex=r"_f1$").map(math.isfinite).all().all()
            ),
        }
        atomic_write_json(manifest, output_dir / "manifest.json")
        print(f"Simplified SSAW validation complete: {output_dir}", flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
