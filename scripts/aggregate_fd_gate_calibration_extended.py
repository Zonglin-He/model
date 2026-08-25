"""Strictly aggregate the extended FD source-only confidence calibration.

This module never launches a benchmark.  It validates the runner manifest,
the atomic cell status, same-domain flow metadata, and the expected 96-cell
cartesian product before computing source-domain/source-seed means.  Selection
uses only the resulting source-only rows and therefore cannot inspect target
transfer labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fd_gate_calibration_extended import (
    CANDIDATE_KEEP_FRACTIONS,
    CANDIDATE_LABELS,
    CANDIDATE_LABEL_TO_VALUE,
    CONDITIONS,
    CONDITION_FRACTIONS,
    CORRUPTION_FRACTION,
    CORRUPTION_SEED,
    PROTOCOL,
    SOURCE_DOMAINS,
    SOURCE_SEEDS,
    STREAM_SEED,
    SUMMARY_REQUIRED_COLUMNS,
    atomic_write_csv,
    candidate_label,
    cell_dir,
    cell_key,
    expected_cell_keys,
    summary_matches,
)


F1_TOLERANCE = 0.002
AGGREGATE_METRICS = (
    "f1",
    "coverage",
    "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
)


def _float_equal(left, right, tolerance=1e-12):
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _condition_alias(value: str) -> str:
    """Normalize the legacy ``corrupt`` spelling for selection-only helpers."""
    value = str(value)
    if value == "corrupt":
        return "signal_freeze_moderate"
    if value not in CONDITIONS:
        raise ValueError(f"Unknown calibration condition: {value!r}")
    return value


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


def _read_manifest(input_dir: Path) -> dict:
    manifest_path = Path(input_dir) / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Unexpected calibration protocol: {manifest.get('protocol')!r}"
        )
    if manifest.get("target_labels_used_for_selection") is not False:
        raise ValueError("Calibration manifest does not certify target-label exclusion")
    if manifest.get("target_transfer_flows_excluded") is not True:
        raise ValueError("Calibration manifest includes transfer flows")
    if manifest.get("source_seed_is_independent_unit") is not True:
        raise ValueError("Calibration manifest does not certify independent source seeds")
    if manifest.get("stream_seed_is_paired_control") is not True:
        raise ValueError("Calibration manifest does not certify paired stream control")
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Calibration runner is not complete: {manifest.get('status')!r}"
        )
    if tuple(int(value) for value in manifest.get("source_domains", ())) != SOURCE_DOMAINS:
        raise ValueError("Manifest source domains do not match the fixed protocol")
    if tuple(int(value) for value in manifest.get("source_seeds", ())) != SOURCE_SEEDS:
        raise ValueError("Manifest source seeds do not match the fixed protocol")
    if int(manifest.get("stream_seed", -1)) != STREAM_SEED:
        raise ValueError("Manifest stream seed does not match the fixed protocol")
    if int(manifest.get("corruption_seed", -1)) != CORRUPTION_SEED:
        raise ValueError("Manifest corruption seed does not match the fixed protocol")
    try:
        fraction = float(
            manifest["conditions"]["signal_freeze_moderate"]["fraction"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest has no fixed moderate corruption condition") from exc
    if not _float_equal(fraction, CORRUPTION_FRACTION):
        raise ValueError("Manifest corruption fraction does not match the fixed protocol")
    candidates = manifest.get("candidates", ())
    observed_candidates = tuple(
        (str(item.get("candidate_label")), float(item.get("confidence_keep_fraction")))
        for item in candidates
    )
    expected_candidates = tuple(
        (label, float(value))
        for label, value in zip(CANDIDATE_LABELS, CANDIDATE_KEEP_FRACTIONS)
    )
    if observed_candidates != expected_candidates:
        raise ValueError(
            f"Manifest candidate grid does not match: {observed_candidates!r}"
        )
    expected_count = len(SOURCE_DOMAINS) * len(SOURCE_SEEDS) * len(CONDITIONS) * len(
        CANDIDATE_KEEP_FRACTIONS
    )
    if int(manifest.get("expected_cells", -1)) != expected_count:
        raise ValueError("Manifest expected_cells is inconsistent with the protocol")
    if int(manifest.get("completed_cells", -1)) != expected_count:
        raise ValueError("Manifest completed_cells is inconsistent with the protocol")
    return manifest


STATUS_REQUIRED_COLUMNS = {
    "candidate_label",
    "confidence_keep_fraction",
    "source_domain",
    "calibration_flow",
    "condition",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    "output_dir",
    "status",
}


def _read_status(input_dir: Path, expected_keys):
    status_path = Path(input_dir) / "cell_status.csv"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    try:
        # Candidate labels are identifiers, not numbers.  Without an explicit
        # dtype pandas turns ``095`` into integer ``95`` and makes a completed
        # calibration impossible to resume/aggregate.
        frame = pd.read_csv(status_path, dtype={"candidate_label": "string"})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid cell status file: {status_path}") from exc
    missing = sorted(STATUS_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing cell status columns: {missing}")
    rows = {}
    for row in frame.to_dict(orient="records"):
        try:
            label = candidate_label(row["candidate_label"])
            fraction_label = candidate_label(row["confidence_keep_fraction"])
            if label != fraction_label:
                raise ValueError(
                    "candidate_label and confidence_keep_fraction disagree: "
                    f"{label!r} != {fraction_label!r}"
                )
            key = cell_key(
                label,
                row["source_domain"],
                row["condition"],
                row["source_seed"],
                row["stream_seed"],
                row["corruption_seed"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cell status key: {row!r}") from exc
        if key in rows:
            raise ValueError(f"Duplicate cell status key: {key}")
        rows[key] = row
    expected = set(expected_keys)
    if set(rows) != expected:
        missing_keys = sorted(expected - set(rows))
        extra_keys = sorted(set(rows) - expected)
        raise ValueError(
            f"cell_status.csv key mismatch; missing={missing_keys}, extra={extra_keys}"
        )
    incomplete = [key for key, row in rows.items() if str(row["status"]) != "completed"]
    if incomplete:
        raise ValueError(f"Calibration has incomplete cells: {incomplete}")
    return rows


def collect_cells(input_dir: Path):
    """Validate and collect all 96 source-only cell summaries."""
    input_dir = Path(input_dir)
    _read_manifest(input_dir)
    expected_keys = expected_cell_keys()
    status = _read_status(input_dir, expected_keys)
    rows = []
    for candidate in CANDIDATE_KEEP_FRACTIONS:
        label = candidate_label(candidate)
        for source_domain in SOURCE_DOMAINS:
            for source_seed in SOURCE_SEEDS:
                for condition in CONDITIONS:
                    key = cell_key(
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                        STREAM_SEED,
                        CORRUPTION_SEED,
                    )
                    cell_path = cell_dir(
                        input_dir,
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                    )
                    if not summary_matches(
                        cell_path,
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                        STREAM_SEED,
                        CORRUPTION_SEED,
                    ):
                        raise ValueError(f"Invalid or incomplete calibration cell: {cell_path}")
                    summary_path = cell_path / "summary_raw.csv"
                    frame = pd.read_csv(summary_path)
                    if len(frame) != 1:
                        raise ValueError(f"Expected one summary row: {summary_path}")
                    missing = sorted(set(SUMMARY_REQUIRED_COLUMNS) - set(frame.columns))
                    if missing:
                        raise ValueError(f"Missing columns in {summary_path}: {missing}")
                    row = frame.iloc[0].to_dict()
                    status_row = status[key]
                    if str(status_row.get("status")) != "completed":
                        raise ValueError(f"Cell status is not completed: {key}")
                    row.update(
                        {
                            "candidate_label": label,
                            "confidence_keep_fraction": float(candidate),
                            "source_domain": int(source_domain),
                            "calibration_flow": f"{source_domain}->{source_domain}",
                            "condition": condition,
                            "source_seed": int(source_seed),
                            "stream_seed": STREAM_SEED,
                            "corruption_seed": CORRUPTION_SEED,
                            "cell_output_dir": str(cell_path),
                        }
                    )
                    rows.append(row)
    result = pd.DataFrame(rows)
    key_columns = [
        "candidate_label",
        "source_domain",
        "condition",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    if len(result) != len(expected_keys):
        raise ValueError(f"Expected {len(expected_keys)} cells, found {len(result)}")
    if result.duplicated(key_columns).any():
        raise ValueError("Duplicate calibration cell keys")
    return result, expected_keys


def _aggregate(frame: pd.DataFrame, group_columns=None, metrics=AGGREGATE_METRICS):
    """Aggregate metrics while retaining exact cell/domain/seed counts."""
    if group_columns is None:
        group_columns = [
            "candidate_label",
            "confidence_keep_fraction",
            "condition",
        ]
    aggregations = {}
    for metric in metrics:
        if metric not in frame.columns:
            continue
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    if not aggregations:
        raise ValueError("No aggregate metrics are present")
    result = frame.groupby(group_columns, as_index=False).agg(**aggregations)
    counts = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            n_cells=("source_seed", "size"),
            n_source_domains=("source_domain", "nunique"),
            n_source_seeds=("source_seed", "nunique"),
        )
    )
    return result.merge(counts, on=group_columns, how="left", validate="one_to_one")


def _summary_from_raw(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["condition"] = normalized["condition"].map(_condition_alias)
    normalized["confidence_keep_fraction"] = normalized[
        "confidence_keep_fraction"
    ].map(float)
    expected = set(CANDIDATE_KEEP_FRACTIONS)
    if set(normalized["confidence_keep_fraction"]) != expected:
        raise ValueError("Raw calibration rows do not contain all candidates")
    if set(normalized["condition"]) != set(CONDITIONS):
        raise ValueError("Raw calibration rows do not contain both conditions")
    return _aggregate(normalized)


def candidate_selection_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create one row per candidate with both condition metrics."""
    required = {
        "confidence_keep_fraction",
        "condition",
        "f1_mean",
        "clean_correct_false_rejection_rate_mean",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Selection summary is missing columns: {missing}")
    rows = []
    for value in CANDIDATE_KEEP_FRACTIONS:
        candidate_rows = summary[
            summary["confidence_keep_fraction"].map(
                lambda observed: _float_equal(observed, value)
            )
        ].copy()
        candidate_rows["condition"] = candidate_rows["condition"].map(_condition_alias)
        if set(candidate_rows["condition"]) != set(CONDITIONS):
            raise ValueError(f"Candidate q={value} does not have both conditions")
        by_condition = candidate_rows.set_index("condition")
        clean = by_condition.loc["clean"]
        corrupt = by_condition.loc["signal_freeze_moderate"]
        rows.append(
            {
                "candidate_label": candidate_label(value),
                "confidence_keep_fraction": float(value),
                "clean_f1_mean": float(clean["f1_mean"]),
                "corrupt_f1_mean": float(corrupt["f1_mean"]),
                "clean_f1_std": float(clean.get("f1_std", math.nan)),
                "corrupt_f1_std": float(corrupt.get("f1_std", math.nan)),
                "clean_correct_fpr_mean": float(
                    clean["clean_correct_false_rejection_rate_mean"]
                ),
                "corrupt_correct_fpr_mean": float(
                    corrupt["clean_correct_false_rejection_rate_mean"]
                ),
                "clean_correct_fpr_std": float(
                    clean.get("clean_correct_false_rejection_rate_std", math.nan)
                ),
                "corrupt_correct_fpr_std": float(
                    corrupt.get("clean_correct_false_rejection_rate_std", math.nan)
                ),
                "clean_coverage_mean": float(clean.get("coverage_mean", math.nan)),
                "corrupt_coverage_mean": float(
                    corrupt.get("coverage_mean", math.nan)
                ),
                "n_clean_cells": int(clean["n_cells"]),
                "n_corrupt_cells": int(corrupt["n_cells"]),
                "n_clean_source_domains": int(clean["n_source_domains"]),
                "n_corrupt_source_domains": int(corrupt["n_source_domains"]),
                "n_clean_source_seeds": int(clean["n_source_seeds"]),
                "n_corrupt_source_seeds": int(corrupt["n_source_seeds"]),
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(frame: pd.DataFrame, tolerance: float = F1_TOLERANCE):
    """Select q using the preregistered dual-F1-floor and FPR priorities.

    ``frame`` may be the 96-row raw cell table or the 8-row condition summary.
    The returned audit is JSON-serializable and explicitly records that target
    labels were not consulted.
    """
    if tolerance < 0:
        raise ValueError("F1 tolerance must be non-negative")
    if {"f1", "clean_correct_false_rejection_rate"}.issubset(frame.columns):
        summary = _summary_from_raw(frame)
    else:
        summary = frame.copy()
    candidates = candidate_selection_table(summary)
    baseline = candidates[
        candidates["confidence_keep_fraction"].map(
            lambda value: _float_equal(value, 0.95)
        )
    ].iloc[0]
    baseline_clean = float(baseline["clean_f1_mean"])
    baseline_corrupt = float(baseline["corrupt_f1_mean"])
    candidates["eligible"] = (
        candidates["clean_f1_mean"].ge(baseline_clean - float(tolerance))
        & candidates["corrupt_f1_mean"].ge(baseline_corrupt - float(tolerance))
    )
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No confidence candidate satisfies the dual F1 floor")
    # The candidate registration order is retained as the final deterministic
    # tie-break.  No unregistered metric is allowed to influence selection.
    rank = {label: index for index, label in enumerate(CANDIDATE_LABELS)}
    eligible["_registration_rank"] = eligible["candidate_label"].map(rank)
    selected = eligible.sort_values(
        [
            "clean_correct_fpr_mean",
            "corrupt_correct_fpr_mean",
            "corrupt_f1_mean",
            "_registration_rank",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).iloc[0]
    candidates = candidates.drop(columns=["_registration_rank"], errors="ignore")
    audit = {
        "selection_rule": (
            "dual_f1_floor_then_min_clean_correct_fpr_then_"
            "min_corrupt_correct_fpr_then_max_corrupt_f1"
        ),
        "baseline_candidate_label": "095",
        "baseline_confidence_keep_fraction": 0.95,
        "f1_tolerance_absolute": float(tolerance),
        "baseline_clean_f1_mean": baseline_clean,
        "baseline_corrupt_f1_mean": baseline_corrupt,
        "eligible_candidates": candidates[
            candidates["eligible"]
        ].to_dict(orient="records"),
        "selected_candidate_label": str(selected["candidate_label"]),
        "selected_confidence_keep_fraction": float(
            selected["confidence_keep_fraction"]
        ),
        "selected_clean_correct_fpr_mean": float(
            selected["clean_correct_fpr_mean"]
        ),
        "selected_corrupt_correct_fpr_mean": float(
            selected["corrupt_correct_fpr_mean"]
        ),
        "selected_corrupt_f1_mean": float(selected["corrupt_f1_mean"]),
        "target_labels_used_for_selection": False,
    }
    return float(selected["confidence_keep_fraction"]), audit, candidates


def _select_quantile(frame: pd.DataFrame, tolerance: float = F1_TOLERANCE):
    """Backward-friendly alias returning ``(selected_q, audit)``."""
    selected, audit, _ = _select_candidate(frame, tolerance)
    return selected, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    parser.add_argument("--f1-tolerance", "--f1_tolerance", dest="f1_tolerance", type=float, default=F1_TOLERANCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, expected_keys = collect_cells(input_dir)
    summary = _aggregate(frame)
    selected, audit, candidate_table = _select_candidate(
        frame, tolerance=float(args.f1_tolerance)
    )

    atomic_write_csv(frame, output_dir / "source_only_calibration_raw.csv")
    atomic_write_csv(summary, output_dir / "source_only_calibration_summary.csv")
    atomic_write_csv(candidate_table, output_dir / "candidate_selection.csv")
    selection_payload = dict(audit)
    selection_payload["selected_confidence_keep_fraction"] = float(selected)
    selection_payload["protocol"] = PROTOCOL
    from scripts.run_fd_gate_calibration_extended import atomic_write_json

    atomic_write_json(output_dir / "selected_candidate.json", selection_payload)
    aggregate_manifest = {
        "protocol": f"{PROTOCOL} aggregate",
        "input_dir": str(input_dir),
        "source_domains": list(SOURCE_DOMAINS),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_fraction": CORRUPTION_FRACTION,
        "conditions": list(CONDITIONS),
        "candidates": [
            {
                "candidate_label": label,
                "confidence_keep_fraction": float(value),
                "baseline": label == "095",
            }
            for label, value in zip(CANDIDATE_LABELS, CANDIDATE_KEEP_FRACTIONS)
        ],
        "cell_count": int(len(frame)),
        "expected_cell_count": int(len(expected_keys)),
        "unique_cell_keys": int(len(set(expected_keys))),
        "confidence_keep_fraction": float(selected),
        "selection": audit,
        "target_labels_used_for_selection": False,
        "target_transfer_flows_excluded": True,
        "metric_grain": (
            "raw rows are one source-domain/source-seed/candidate/condition cell; "
            "selection means are over 4 domains x 3 source seeds"
        ),
        "outputs": [
            "source_only_calibration_raw.csv",
            "source_only_calibration_summary.csv",
            "candidate_selection.csv",
            "selected_candidate.json",
        ],
    }
    atomic_write_json(output_dir / "aggregate_manifest.json", aggregate_manifest)
    print(json.dumps(aggregate_manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
