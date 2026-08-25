"""Aggregate and validate the frozen FD q=.95 transfer panel.

The final panel is deliberately evaluated after source-only selection.  This
script only reads per-cell outputs; it has no parameter-selection logic and
never reads target labels.  Its expected grain is five transfer flows x three
source checkpoint seeds x two conditions, with one paired stream seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import atomic_write_csv


FLOWS = ("0->1", "1->2", "3->1", "1->0", "2->3")
CONDITIONS = ("clean", "signal_freeze_moderate")
CONDITION_FRACTIONS = {"clean": 0.0, "signal_freeze_moderate": 0.5}
DEFAULT_SOURCE_SEEDS = (1, 2, 3)
DEFAULT_STREAM_SEED = 42

PRIMARY_METRICS = (
    "f1",
    "coverage",
    "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
)

GATE_METRICS = (
    "confidence_rejected_count",
    "semantic_rejected_count",
    "ssaw_veto_candidate_count",
    "commit_guard_rejected_count",
    "confidence_rejected_rate",
    "semantic_rejected_rate",
    "ssaw_veto_candidate_rate",
    "commit_guard_rejected_rate",
    "confidence_rejected_clean_correct_fpr",
    "semantic_rejected_clean_correct_fpr",
    "ssaw_veto_candidate_clean_correct_fpr",
    "commit_guard_rejected_clean_correct_fpr",
    "confidence_rejected_corrupted_recall",
    "semantic_rejected_corrupted_recall",
    "ssaw_veto_candidate_corrupted_recall",
    "commit_guard_rejected_corrupted_recall",
)


def _parse_int_list(text, name):
    values = [int(value.strip()) for value in str(text).split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")
    return tuple(values)


def _cell_dir(input_dir, flow, condition, source_seed):
    source, target = flow.split("->", 1)
    return (
        Path(input_dir)
        / f"flow_{source}_to_{target}"
        / condition
        / f"source_seed_{source_seed}"
    )


def _cell_key(flow, condition, source_seed, stream_seed, corruption_seed):
    return (
        str(flow),
        str(condition),
        int(source_seed),
        int(stream_seed),
        int(corruption_seed),
    )


def _validate_cell_metadata(
    cell_dir, flow, condition, source_seed, stream_seed, corruption_seed
):
    metadata_path = Path(cell_dir) / "cell_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cell metadata: {metadata_path}") from exc
    expected = {
        "flow": str(flow),
        "condition": str(condition),
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
        "confidence_keep_fraction": 0.95,
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_selection": False,
    }
    for key, value in expected.items():
        observed = metadata.get(key)
        if isinstance(value, float):
            if abs(float(observed) - value) > 1e-12:
                raise ValueError(
                    f"Metadata mismatch in {metadata_path}: {key}={observed!r}"
                )
        elif observed != value:
            raise ValueError(
                f"Metadata mismatch in {metadata_path}: {key}={observed!r}"
            )
    fraction = float(metadata.get("corruption_fraction"))
    if abs(fraction - CONDITION_FRACTIONS[condition]) > 1e-12:
        raise ValueError(
            f"Metadata condition fraction mismatch in {metadata_path}: "
            f"{fraction!r}"
        )


def _read_manifest(input_dir):
    manifest_path = Path(input_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "FD final transfer safety q=.95 v2":
        raise ValueError(
            f"Unexpected final-panel protocol in {manifest_path}: "
            f"{manifest.get('protocol')!r}"
        )
    if manifest.get("target_labels_used_for_selection") is not False:
        raise ValueError("Final-panel manifest does not certify target-label exclusion")
    if manifest.get("stream_seed_is_paired_control") is not True:
        raise ValueError("Final-panel manifest does not certify paired stream control")
    return manifest


def _read_status(input_dir, expected_keys):
    status_path = Path(input_dir) / "cell_status.csv"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    frame = pd.read_csv(status_path)
    required = {
        "flow", "condition", "source_seed", "stream_seed",
        "corruption_seed", "status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing status columns: {missing}")
    rows = {}
    for row in frame.to_dict("records"):
        key = _cell_key(
            row["flow"], row["condition"], row["source_seed"],
            row["stream_seed"], row["corruption_seed"],
        )
        rows[key] = row
    if set(rows) != set(expected_keys):
        missing_keys = sorted(set(expected_keys) - set(rows))
        extra_keys = sorted(set(rows) - set(expected_keys))
        raise ValueError(
            f"cell_status.csv key mismatch; missing={missing_keys}, extra={extra_keys}"
        )
    incomplete = [key for key, row in rows.items() if row["status"] != "completed"]
    if incomplete:
        raise ValueError(f"Final panel has incomplete cells: {incomplete}")
    return rows


def collect_cells(input_dir, source_seeds, stream_seed, corruption_seed):
    rows = []
    expected_keys = []
    required_columns = {
        "dataset", "scenario", "method", "variant", "corruption", "severity",
        "source_seed", "stream_seed", "corruption_seed", *PRIMARY_METRICS,
        *GATE_METRICS,
    }
    for flow in FLOWS:
        for condition in CONDITIONS:
            for source_seed in source_seeds:
                key = _cell_key(
                    flow, condition, source_seed, stream_seed, corruption_seed
                )
                expected_keys.append(key)
                cell_dir = _cell_dir(input_dir, flow, condition, source_seed)
                _validate_cell_metadata(
                    cell_dir, flow, condition, source_seed, stream_seed,
                    corruption_seed,
                )
                summary_path = cell_dir / "summary_raw.csv"
                if not summary_path.exists():
                    raise FileNotFoundError(summary_path)
                frame = pd.read_csv(summary_path)
                if len(frame) != 1:
                    raise ValueError(
                        f"Expected exactly one row in {summary_path}, found {len(frame)}"
                    )
                missing = sorted(required_columns - set(frame.columns))
                if missing:
                    raise ValueError(f"Missing columns in {summary_path}: {missing}")
                row = frame.iloc[0].to_dict()
                expected = {
                    "dataset": "FD",
                    "scenario": flow,
                    "method": "DuSafe",
                    "variant": "full",
                    "corruption": "signal_freeze",
                    "severity": "moderate",
                    "source_seed": source_seed,
                    "stream_seed": stream_seed,
                    "corruption_seed": corruption_seed,
                }
                for column, value in expected.items():
                    observed = row[column]
                    if column in {"source_seed", "stream_seed", "corruption_seed"}:
                        if int(observed) != int(value):
                            raise ValueError(
                                f"Key mismatch in {summary_path}: {column}="
                                f"{observed!r}, expected {value!r}"
                            )
                    elif str(observed) != str(value):
                        raise ValueError(
                            f"Key mismatch in {summary_path}: {column}="
                            f"{observed!r}, expected {value!r}"
                        )
                row.update(
                    {
                        "flow": flow,
                        "condition": condition,
                        "cell_output_dir": str(cell_dir),
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.duplicated(
        ["flow", "condition", "source_seed", "stream_seed", "corruption_seed"]
    ).any():
        raise ValueError("Duplicate final-panel cell keys")
    _read_status(input_dir, expected_keys)
    return frame, expected_keys


def _aggregate(frame, group_columns, metrics):
    aggregations = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    result = frame.groupby(group_columns, as_index=False).agg(**aggregations)
    counts = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            n_cells=("source_seed", "size"),
            n_source_seeds=("source_seed", "nunique"),
        )
    )
    return result.merge(counts, on=group_columns, how="left", validate="one_to_one")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--source_seeds", default=",".join(map(str, DEFAULT_SOURCE_SEEDS)))
    parser.add_argument("--stream_seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument("--corruption_seed", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_seeds = _parse_int_list(args.source_seeds, "--source_seeds")
    manifest = _read_manifest(input_dir)
    if tuple(manifest.get("source_seeds", ())) != source_seeds:
        raise ValueError("Manifest source_seeds do not match aggregation request")
    if int(manifest.get("stream_seed")) != int(args.stream_seed):
        raise ValueError("Manifest stream_seed does not match aggregation request")
    expected_count = len(FLOWS) * len(CONDITIONS) * len(source_seeds)
    if int(manifest.get("cell_count_expected", -1)) != expected_count:
        raise ValueError("Manifest cell_count_expected is inconsistent with protocol")

    frame, expected_keys = collect_cells(
        input_dir, source_seeds, args.stream_seed, args.corruption_seed
    )
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(frame)}")

    atomic_write_csv(frame, output_dir / "summary_raw.csv", index=False)
    by_flow_source = frame.sort_values(
        ["flow", "condition", "source_seed"]
    ).reset_index(drop=True)
    atomic_write_csv(by_flow_source, output_dir / "summary_by_flow_source.csv", index=False)
    by_flow = _aggregate(frame, ["flow", "condition"], PRIMARY_METRICS)
    atomic_write_csv(by_flow, output_dir / "summary_aggregate.csv", index=False)
    overall = _aggregate(frame, ["condition"], PRIMARY_METRICS)
    atomic_write_csv(overall, output_dir / "summary_overall.csv", index=False)
    gate_by_flow = _aggregate(frame, ["flow", "condition"], GATE_METRICS)
    atomic_write_csv(gate_by_flow, output_dir / "gate_decomposition.csv", index=False)
    gate_overall = _aggregate(frame, ["condition"], GATE_METRICS)
    atomic_write_csv(gate_overall, output_dir / "gate_decomposition_overall.csv", index=False)

    report_manifest = {
        "protocol": "FD final transfer safety q=.95 v2 aggregate",
        "input_dir": str(input_dir),
        "source_seeds": list(source_seeds),
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "corruption_seed": int(args.corruption_seed),
        "flows": list(FLOWS),
        "conditions": list(CONDITIONS),
        "cell_count": int(len(frame)),
        "expected_cell_count": int(expected_count),
        "unique_cell_keys": int(len(set(expected_keys))),
        "confidence_keep_fraction": 0.95,
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_selection": False,
        "selection_source": "results/calibration/fd_source_only_gate_v1",
        "excluded_invalid_run": (
            "results/diagnostics/fd_gate_q095_final_v1/INVALID_PROTOCOL.json"
        ),
        "metric_grain": "one summary row per flow, condition, source_seed, stream_seed",
        "primary_metrics": list(PRIMARY_METRICS),
        "gate_metrics": list(GATE_METRICS),
        "outputs": [
            "summary_raw.csv",
            "summary_by_flow_source.csv",
            "summary_aggregate.csv",
            "summary_overall.csv",
            "gate_decomposition.csv",
            "gate_decomposition_overall.csv",
        ],
    }
    (output_dir / "aggregate_manifest.json").write_text(
        json.dumps(report_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
