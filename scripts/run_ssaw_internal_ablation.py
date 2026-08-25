"""Resume-safe paired ablation of the complete SSAW branch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optuna_stepwise import (
    SSAW_BRANCH_ABLATIONS,
    acquire_run_lock,
    atomic_write_json,
    parse_csv,
    run_tta_job,
    scenario_label,
    scenario_pairs,
    utc_now,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir


KEY_COLUMNS = (
    "dataset",
    "scenario",
    "source_seed",
    "test_time_seed",
    "ablation",
)
PAIR_COLUMNS = KEY_COLUMNS[:-1]
SAFETY_METRICS = (
    "coverage",
    "accepted_pseudo_label_accuracy",
    "unsafe_update_rate",
    "wrong_rejection_recall",
    "correct_false_rejection_rate",
)
OBSOLETE_TTA_KEYS = {
    "signal_anomaly_quantile",
    "ssaw_num_candidates",
    "ssaw_selection_rule",
    "ssaw_require_source_support",
    "ssaw_invariance_ratio",
    "ssaw_enable_physical_warp",
    "ssaw_require_label_preservation",
}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sanitized_tta_config(state: Mapping) -> dict:
    """Use tuned numeric values except parameters removed from SSAW."""
    return {
        key: value
        for key, value in dict(state["tta_config"]).items()
        if key not in OBSOLETE_TTA_KEYS
    }


def validate_state(
    state: Mapping,
    *,
    dataset: str,
    scenarios: Iterable[tuple[str, str]],
) -> tuple[int, list[int]]:
    signature = dict(state.get("signature", {}))
    if not state.get("completed"):
        raise ValueError(f"{dataset}: tuning state is incomplete")
    if str(signature.get("dataset", "")).upper() != dataset:
        raise ValueError(f"{dataset}: tuning-state dataset mismatch")
    expected = [scenario_label(pair) for pair in scenarios]
    if list(signature.get("scenarios", [])) != expected:
        raise ValueError(f"{dataset}: tuning-state scenario mismatch")
    source_seed = int(signature.get("source_seed", 1))
    test_time_seeds = [
        int(seed) for seed in signature.get("test_time_seeds", [1, 2, 3])
    ]
    if len(test_time_seeds) != len(set(test_time_seeds)):
        raise ValueError(f"{dataset}: duplicate test-time seeds")
    return source_seed, test_time_seeds


def row_key(row: Mapping) -> tuple[str, str, int, int, str]:
    return (
        str(row["dataset"]).upper(),
        str(row["scenario"]),
        int(row["source_seed"]),
        int(row["test_time_seed"]),
        str(row["ablation"]),
    )


def required_jobs(
    scenarios: Iterable[tuple[str, str]],
    test_time_seeds: Iterable[int],
    ablations: Iterable[str] = SSAW_BRANCH_ABLATIONS,
) -> list[tuple[tuple[str, str], int, str]]:
    return [
        (scenario, int(test_time_seed), ablation)
        for scenario in scenarios
        for test_time_seed in test_time_seeds
        for ablation in ablations
    ]


def validate_rows(
    rows: list[dict],
    *,
    dataset: str,
    scenario_names: set[str],
    source_seed: int,
    test_time_seeds: set[int],
    allowed_ablations: Iterable[str] = SSAW_BRANCH_ABLATIONS,
) -> list[dict]:
    allowed_ablations = set(allowed_ablations)
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate SSAW ablation row: {key}")
        seen.add(key)
        if key[0] != dataset or key[1] not in scenario_names:
            raise ValueError(f"{dataset}: foreign result row: {key}")
        if key[2] != source_seed or key[3] not in test_time_seeds:
            raise ValueError(f"{dataset}: seed mismatch in result row: {key}")
        if key[4] not in allowed_ablations:
            raise ValueError(f"{dataset}: unsupported SSAW ablation: {key}")
    return rows


def paired_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    full = frame[frame["ablation"].eq("full")][
        [*PAIR_COLUMNS, "f1"]
    ].rename(columns={"f1": "full_f1"})
    numeric_columns = [
        column
        for column in frame.columns
        if column in SAFETY_METRICS or column.startswith("diag_")
    ]
    rows = []
    for (dataset, ablation), group in frame.groupby(
        ["dataset", "ablation"], sort=False
    ):
        paired = group.merge(
            full, on=list(PAIR_COLUMNS), how="inner", validate="one_to_one"
        )
        differences = paired["f1"] - paired["full_f1"]
        row = {
            "dataset": dataset,
            "ablation": ablation,
            "jobs": int(len(group)),
            "scenarios": int(group["scenario"].nunique()),
            "test_time_seeds": int(group["test_time_seed"].nunique()),
            "f1_mean": float(group["f1"].mean()),
            "f1_std": float(group["f1"].std(ddof=0)),
            "f1_min": float(group["f1"].min()),
            "paired_f1_delta_vs_full": (
                float(differences.mean()) if len(differences) else math.nan
            ),
            "paired_wins_vs_full": int((differences > 1e-12).sum()),
            "paired_ties_vs_full": int((differences.abs() <= 1e-12).sum()),
            "paired_losses_vs_full": int((differences < -1e-12).sum()),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def publish_dataset(rows: list[dict], dataset_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    atomic_write_csv(frame, dataset_dir / "raw.csv", index=False)
    atomic_write_csv(
        paired_summary(frame), dataset_dir / "summary.csv", index=False
    )


def legacy_equivalence(
    *,
    tuning_dir: Path,
    output_dir: Path,
    datasets: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        current = pd.read_csv(output_dir / dataset / "raw.csv")
        current = current[current["ablation"].eq("full")][
            [*PAIR_COLUMNS, "f1"]
        ].rename(columns={"f1": "new_full_f1"})
        legacy = pd.read_csv(
            tuning_dir / dataset / "all_scenarios_component_ablation.csv"
        )
        legacy = legacy[legacy["ablation"].eq("no_signal_integrity")][
            [*PAIR_COLUMNS, "f1"]
        ].rename(columns={"f1": "legacy_no_signal_f1"})
        paired = current.merge(
            legacy,
            on=list(PAIR_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(current):
            raise ValueError(f"{dataset}: incomplete legacy equivalence pairs")
        paired["difference"] = (
            paired["new_full_f1"] - paired["legacy_no_signal_f1"]
        )
        rows.extend(paired.to_dict("records"))
    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, output_dir / "legacy_equivalence.csv", index=False)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--tuning-dir",
        default=str(
            ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "results"
            / "ablation"
            / "ssaw_internal_signal_removed_v1"
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
    parser.add_argument(
        "--ablations",
        default=",".join(SSAW_BRANCH_ABLATIONS),
        help="Comma-separated whole-branch variants; Full is required.",
    )
    parser.add_argument(
        "--skip-legacy-equivalence",
        action="store_true",
        help="Do not compare against the obsolete signal-gate audit table.",
    )
    args = parser.parse_args()
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    args.ablations = parse_csv(args.ablations)
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    if not args.ablations or "full" not in args.ablations:
        parser.error("--ablations must contain full")
    if len(args.ablations) != len(set(args.ablations)):
        parser.error("--ablations must not contain duplicates")
    unknown = [
        name for name in args.ablations if name not in SSAW_BRANCH_ABLATIONS
    ]
    if unknown:
        parser.error("unknown --ablations: " + ",".join(unknown))
    return args


def main() -> int:
    args = parse_args()
    tuning_dir = Path(args.tuning_dir).resolve()
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
                allowed_ablations=args.ablations,
            )
            completed = {row_key(row) for row in rows}
            tta_config = sanitized_tta_config(state)
            for scenario, test_time_seed, ablation in required_jobs(
                scenarios, test_time_seeds, args.ablations
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
                    f"[SSAW] {dataset} {key[1]} seed={test_time_seed} "
                    f"{ablation}",
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
        atomic_write_csv(
            pd.concat(summaries, ignore_index=True),
            output_dir / "summary.csv",
            index=False,
        )
        equivalence = None
        if not args.skip_legacy_equivalence:
            equivalence = legacy_equivalence(
                tuning_dir=tuning_dir,
                output_dir=output_dir,
                datasets=args.datasets,
            )
        manifest = {
            "completed_at": utc_now(),
            "datasets": args.datasets,
            "ablations": list(args.ablations),
            "jobs_per_dataset": 5 * 3 * len(args.ablations),
            "source_seed": 1,
            "test_time_seeds": [1, 2, 3],
            "signal_gate_removed": True,
            "selection_objective": "post_adaptation_macro_f1",
            "target_labels_used_for_original_tuning": True,
            "max_abs_full_difference_from_legacy_no_signal": (
                None
                if equivalence is None
                else float(equivalence["difference"].abs().max())
            ),
        }
        atomic_write_json(manifest, output_dir / "manifest.json")
        print(f"SSAW branch ablation complete: {output_dir}", flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
