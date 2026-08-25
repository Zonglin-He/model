"""Resume-safe whole-branch SSAW ablation over all benchmark domains.

All non-SSAW DuSafe components remain fixed. The paired comparison removes or
adds physical view generation, pseudo-label-preserving view selection, and
the consistency objective together as one atomic SSAW branch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.dusafe_ablation import ssaw_cumulative_ablation_stages
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
    KEY_COLUMNS,
    PAIR_COLUMNS,
    SAFETY_METRICS,
    load_json,
    row_key,
    sanitized_tta_config,
    validate_rows,
    validate_state,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir


CUMULATIVE_STAGES = ssaw_cumulative_ablation_stages()
CUMULATIVE_ABLATIONS = tuple(stage["name"] for stage in CUMULATIVE_STAGES)


def required_jobs(
    scenarios: Iterable[tuple[str, str]],
    test_time_seeds: Iterable[int],
    ablations: Iterable[str] = CUMULATIVE_ABLATIONS,
) -> list[tuple[tuple[str, str], int, str]]:
    """Keep every paired stream together and execute stages in order."""
    return [
        (scenario, int(test_time_seed), ablation)
        for scenario in scenarios
        for test_time_seed in test_time_seeds
        for ablation in ablations
    ]


def _paired_metric_difference(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
) -> pd.Series:
    if metric not in current or metric not in reference:
        return pd.Series(dtype=float)
    left = current[[*PAIR_COLUMNS, metric]].rename(
        columns={metric: "current_value"}
    )
    right = reference[[*PAIR_COLUMNS, metric]].rename(
        columns={metric: "reference_value"}
    )
    paired = left.merge(
        right,
        on=list(PAIR_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    current_values = pd.to_numeric(paired["current_value"], errors="coerce")
    reference_values = pd.to_numeric(
        paired["reference_value"], errors="coerce"
    )
    return current_values - reference_values


def _mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if len(numeric) else math.nan


def _slice_summary(
    frame: pd.DataFrame,
    *,
    stages: Iterable[Mapping],
    identity: Mapping[str, object],
) -> list[dict]:
    stages = tuple(stages)
    baseline = frame[frame["ablation"].eq(stages[0]["name"])]
    numeric_columns = [
        column
        for column in frame.columns
        if column in SAFETY_METRICS or column.startswith("diag_")
    ]
    rows = []
    previous = None
    for stage in stages:
        group = frame[frame["ablation"].eq(stage["name"])]
        f1_previous = (
            pd.Series(dtype=float)
            if previous is None
            else _paired_metric_difference(group, previous, "f1")
        )
        f1_baseline = _paired_metric_difference(group, baseline, "f1")
        row = {
            **dict(identity),
            "stage_index": int(stage["stage_index"]),
            "ablation": stage["name"],
            "added_module": stage["added_module"],
            "jobs": int(len(group)),
            "scenarios": int(group["scenario"].nunique()),
            "test_time_seeds": int(group["test_time_seed"].nunique()),
            "f1_mean": _mean_or_nan(group["f1"]),
            "f1_std": float(
                pd.to_numeric(group["f1"], errors="coerce").std(ddof=0)
            ),
            "f1_min": float(
                pd.to_numeric(group["f1"], errors="coerce").min()
            ),
            "paired_f1_delta_vs_previous": _mean_or_nan(f1_previous),
            "paired_f1_delta_vs_fixed_source": _mean_or_nan(f1_baseline),
            "paired_cells_vs_previous": int(len(f1_previous)),
            "paired_wins_vs_previous": int((f1_previous > 1e-12).sum()),
            "paired_ties_vs_previous": int(
                (f1_previous.abs() <= 1e-12).sum()
            ),
            "paired_losses_vs_previous": int((f1_previous < -1e-12).sum()),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = _mean_or_nan(group[column])
            if previous is not None:
                difference = _paired_metric_difference(
                    group, previous, column
                )
                row[f"paired_{column}_delta_vs_previous"] = _mean_or_nan(
                    difference
                )
        rows.append(row)
        previous = group
    return rows


def cumulative_summary(
    frame: pd.DataFrame,
    stages: Iterable[Mapping] = CUMULATIVE_STAGES,
) -> pd.DataFrame:
    """Aggregate cellwise increments instead of comparing only with Full."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in frame.groupby("dataset", sort=False):
        rows.extend(
            _slice_summary(
                group,
                stages=stages,
                identity={"dataset": dataset},
            )
        )
    return pd.DataFrame(rows)


def cumulative_scenario_summary(
    frame: pd.DataFrame,
    stages: Iterable[Mapping] = CUMULATIVE_STAGES,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, scenario), group in frame.groupby(
        ["dataset", "scenario"], sort=False
    ):
        rows.extend(
            _slice_summary(
                group,
                stages=stages,
                identity={"dataset": dataset, "scenario": scenario},
            )
        )
    return pd.DataFrame(rows)


def publish_dataset(rows: list[dict], dataset_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    atomic_write_csv(frame, dataset_dir / "raw.csv", index=False)
    atomic_write_csv(
        cumulative_summary(frame), dataset_dir / "summary.csv", index=False
    )
    atomic_write_csv(
        cumulative_scenario_summary(frame),
        dataset_dir / "scenario_summary.csv",
        index=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--tuning-dir",
        default=str(
            ROOT
            / "results"
            / "optuna"
            / "simplified_ssaw_four_param_final_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "ablation" / "cumulative_ssaw_components_v1"
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
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Run at most this many missing cells, then exit cleanly.",
    )
    args = parser.parse_args()
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    return args


def main() -> int:
    args = parse_args()
    tuning_dir = Path(args.tuning_dir).resolve()
    output_dir = ensure_dir(args.output_dir)
    lock = acquire_run_lock(output_dir)
    new_jobs = 0
    protocol = {}
    try:
        for dataset in args.datasets:
            scenarios = scenario_pairs(dataset)
            if len(scenarios) != 5:
                raise ValueError(f"{dataset}: expected exactly five scenarios")
            state = load_json(tuning_dir / dataset / "state.json")
            source_seed, test_time_seeds = validate_state(
                state, dataset=dataset, scenarios=scenarios
            )
            protocol[dataset] = {
                "source_seed": source_seed,
                "test_time_seeds": test_time_seeds,
                "scenarios": [scenario_label(pair) for pair in scenarios],
            }
            dataset_dir = ensure_dir(output_dir / dataset)
            raw_path = dataset_dir / "raw.csv"
            rows = (
                pd.read_csv(raw_path).to_dict("records")
                if raw_path.exists()
                else []
            )
            rows = validate_rows(
                rows,
                dataset=dataset,
                scenario_names={scenario_label(pair) for pair in scenarios},
                source_seed=source_seed,
                test_time_seeds=set(test_time_seeds),
                allowed_ablations=CUMULATIVE_ABLATIONS,
            )
            completed = {row_key(row) for row in rows}
            tta_config = sanitized_tta_config(state)
            atomic_write_json(
                {
                    "dataset": dataset,
                    "source_config": state["source_config"],
                    "tta_config": tta_config,
                    "stages": list(CUMULATIVE_STAGES),
                },
                dataset_dir / "experiment_config.json",
            )
            for scenario, test_time_seed, ablation in required_jobs(
                scenarios, test_time_seeds
            ):
                key = (
                    dataset,
                    scenario_label(scenario),
                    source_seed,
                    test_time_seed,
                    ablation,
                )
                if key in completed:
                    continue
                print(
                    f"[Cumulative] {dataset} {key[1]} "
                    f"seed={test_time_seed} {ablation}",
                    flush=True,
                )
                row = run_tta_job(
                    dataset=dataset,
                    scenario=scenario,
                    source_seed=source_seed,
                    test_time_seed=test_time_seed,
                    source_config=state["source_config"],
                    tta_config=tta_config,
                    ablation=ablation,
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

        summaries = [
            pd.read_csv(output_dir / dataset / "summary.csv")
            for dataset in args.datasets
        ]
        scenario_summaries = [
            pd.read_csv(output_dir / dataset / "scenario_summary.csv")
            for dataset in args.datasets
        ]
        atomic_write_csv(
            pd.concat(summaries, ignore_index=True),
            output_dir / "summary.csv",
            index=False,
        )
        atomic_write_csv(
            pd.concat(scenario_summaries, ignore_index=True),
            output_dir / "scenario_summary.csv",
            index=False,
        )
        atomic_write_json(
            {
                "completed_at": utc_now(),
                "datasets": args.datasets,
                "protocol": protocol,
                "stages": list(CUMULATIVE_STAGES),
                "jobs_per_dataset": {
                    dataset: (
                        len(protocol[dataset]["scenarios"])
                        * len(protocol[dataset]["test_time_seeds"])
                        * len(CUMULATIVE_STAGES)
                    )
                    for dataset in args.datasets
                },
                "selection_objective": "post_adaptation_macro_f1",
                "target_labels_used_for_original_tuning": True,
                "ssaw_is_atomic_branch": True,
                "ssaw_atomic_components": [
                    "physical_view_generation",
                    "pseudo_label_preserving_view_selection",
                    "prediction_consistency",
                ],
            },
            output_dir / "manifest.json",
        )
        print(f"SSAW branch ablation complete: {output_dir}", flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
