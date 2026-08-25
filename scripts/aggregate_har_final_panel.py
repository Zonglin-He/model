"""Strictly aggregate the frozen HAR Full-vs-no-SSAW safety panel.

This module treats the source checkpoint seed as the independent unit.  It
requires exactly 5 flows x 2 conditions x 3 source seeds x 2 variants = 60
summary rows, then computes Full-minus-no-SSAW differences within each paired
cell before any flow or overall aggregation.  It has no tuning or target-label
selection logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_har_final_panel import (
    CONDITIONS,
    CONDITION_CORRUPTION,
    CONDITION_FRACTIONS,
    CONDITION_SEVERITY,
    DATASET,
    FLOWS,
    FROZEN_HAR_PROFILE_ID,
    FROZEN_HAR_TTA_PARAMS,
    PROTOCOL,
    REQUIRED_SAFETY_METRICS,
    RESUME_KEY_FIELDS,
    VARIANTS,
    cell_dir,
    cell_key,
    resume_key,
)
from scripts.supplementary_utils import atomic_write_csv


DEFAULT_SOURCE_SEEDS = (1, 2, 3)
DEFAULT_STREAM_SEED = 42
DEFAULT_CORRUPTION_SEED = 1
PRIMARY_METRICS = ("f1", *REQUIRED_SAFETY_METRICS)
CELL_KEY_FIELDS = (
    "flow",
    "condition",
    "source_seed",
    "stream_seed",
    "corruption_seed",
)
VARIANT_REQUIRED_COLUMNS = {
    "dataset",
    "scenario",
    "method",
    "variant",
    "corruption",
    "severity",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    *PRIMARY_METRICS,
}


def _parse_int_list(text: str, name: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")
    return values


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _validate_manifest(
    input_dir: Path,
    source_seeds: tuple[int, ...],
    stream_seed: int,
    corruption_seed: int,
) -> dict:
    path = Path(input_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = _read_json(path)
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Unexpected HAR final-panel protocol: {manifest.get('protocol')!r}"
        )
    if manifest.get("dataset") != DATASET:
        raise ValueError("Final-panel manifest dataset is not HAR")
    if tuple(manifest.get("flows", ())) != tuple(FLOWS):
        raise ValueError("Final-panel manifest flows do not match scenario_pairs('HAR')")
    if tuple(manifest.get("variants", ())) != VARIANTS:
        raise ValueError("Final-panel manifest variants must be exactly full,no_ssaw")
    if tuple(manifest.get("conditions", ())) != CONDITIONS:
        raise ValueError("Final-panel manifest conditions do not match the protocol")
    if tuple(int(value) for value in manifest.get("source_seeds", ())) != source_seeds:
        raise ValueError("Final-panel manifest source_seeds do not match aggregation request")
    if int(manifest.get("stream_seed", -1)) != int(stream_seed):
        raise ValueError("Final-panel manifest stream_seed does not match aggregation request")
    if int(manifest.get("corruption_seed", -1)) != int(corruption_seed):
        raise ValueError("Final-panel manifest corruption_seed does not match aggregation request")
    if manifest.get("condition_fractions") != {
        key: float(value) for key, value in CONDITION_FRACTIONS.items()
    }:
        raise ValueError("Final-panel manifest condition fractions are not frozen")
    if manifest.get("condition_corruption") != CONDITION_CORRUPTION:
        raise ValueError("Final-panel manifest corruption mapping is not frozen")
    if manifest.get("condition_severity") != CONDITION_SEVERITY:
        raise ValueError("Final-panel manifest severity mapping is not frozen")
    if manifest.get("frozen_har_profile_id") != FROZEN_HAR_PROFILE_ID:
        raise ValueError("Final-panel manifest HAR profile ID is not frozen")
    if manifest.get("frozen_har_tta_hparams") != FROZEN_HAR_TTA_PARAMS:
        raise ValueError("Final-panel manifest HAR hyperparameters are not frozen")
    for key in (
        "source_seed_is_independent_unit",
        "stream_seed_is_paired_control",
        "selection_completed_before_evaluation",
    ):
        if manifest.get(key) is not True:
            raise ValueError(f"Final-panel manifest does not certify {key}")
    for key in (
        "target_labels_used_for_tuning",
        "target_labels_used_for_selection",
    ):
        if manifest.get(key) is not True:
            raise ValueError(
                f"Final-panel manifest does not record {key}=True"
            )
    if (
        "target_labels_used_online" in manifest
        and manifest.get("target_labels_used_online") is not False
    ):
        raise ValueError("Final-panel manifest does not certify target_labels_used_online=False")
    expected_cells = len(FLOWS) * len(CONDITIONS) * len(source_seeds)
    if int(manifest.get("cell_count_expected", -1)) != expected_cells:
        raise ValueError("Final-panel manifest cell_count_expected is inconsistent")
    if int(manifest.get("expected_job_count", -1)) != expected_cells:
        raise ValueError("Final-panel manifest expected_job_count is inconsistent")
    if int(manifest.get("expected_row_count", -1)) != expected_cells * len(VARIANTS):
        raise ValueError("Final-panel manifest expected_row_count is inconsistent")
    if tuple(manifest.get("resume_key_fields", ())) != RESUME_KEY_FIELDS:
        raise ValueError("Final-panel manifest resume key fields are inconsistent")
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Final panel is not complete; manifest status={manifest.get('status')!r}"
        )
    return manifest


def _read_status(
    input_dir: Path,
    expected_keys: list[tuple[str, str, int, int, int]],
) -> dict[tuple, dict]:
    path = Path(input_dir) / "cell_status.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid status CSV: {path}") from exc
    required = {
        *RESUME_KEY_FIELDS,
        "resume_key",
        "status",
        "returncode",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing status columns: {missing}")
    rows = {}
    for row in frame.to_dict("records"):
        try:
            key = cell_key(
                row["flow"],
                row["condition"],
                int(row["source_seed"]),
                int(row["stream_seed"]),
                int(row["corruption_seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid status key in {path}: {row!r}") from exc
        if key in rows:
            raise ValueError(f"Duplicate cell status key: {key}")
        if row.get("resume_key") != resume_key(*key):
            raise ValueError(f"Status resume_key mismatch for {key}")
        if row.get("status") != "completed":
            raise ValueError(f"Final panel has incomplete cell: {key}")
        try:
            if int(row["returncode"]) != 0:
                raise ValueError(f"Final panel cell returncode is not zero: {key}")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Final panel cell"):
                raise
            raise ValueError(f"Invalid status returncode for {key}") from exc
        rows[key] = row
    expected = set(expected_keys)
    observed = set(rows)
    if observed != expected:
        raise ValueError(
            "cell_status.csv key mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if len(frame) != len(expected_keys):
        raise ValueError(
            f"Expected exactly {len(expected_keys)} status rows, found {len(frame)}"
        )
    return rows


def _validate_metadata(
    metadata_path: Path,
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
    runtime_hparam_overrides: dict | None = None,
) -> dict:
    metadata = _read_json(metadata_path)
    expected = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "flow": str(flow),
        "condition": str(condition),
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
        "corruption": CONDITION_CORRUPTION[condition],
        "severity": CONDITION_SEVERITY[condition],
        "variants": list(VARIANTS),
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_tuning": True,
        "target_labels_used_for_selection": True,
        "frozen_har_profile_id": FROZEN_HAR_PROFILE_ID,
        "frozen_har_tta_hparams": dict(FROZEN_HAR_TTA_PARAMS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Metadata mismatch in {metadata_path}: {key}="
                f"{metadata.get(key)!r}, expected {value!r}"
            )
    if metadata.get("runtime_hparam_overrides", {}) != dict(
        runtime_hparam_overrides or {}
    ):
        raise ValueError(
            f"Metadata runtime override mismatch in {metadata_path}"
        )
    try:
        observed_fraction = float(metadata["corruption_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid corruption_fraction in {metadata_path}") from exc
    if not math.isclose(
        observed_fraction,
        float(CONDITION_FRACTIONS[condition]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Metadata condition fraction mismatch in {metadata_path}")
    if metadata.get("resume_key") != resume_key(
        flow, condition, source_seed, stream_seed, corruption_seed
    ):
        raise ValueError(f"Metadata resume_key mismatch in {metadata_path}")
    return metadata


def _validate_summary(
    summary_path: Path,
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(summary_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid summary CSV: {summary_path}") from exc
    missing = sorted(VARIANT_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {missing}")
    if len(frame) != len(VARIANTS):
        raise ValueError(
            f"Expected exactly {len(VARIANTS)} rows in {summary_path}, found {len(frame)}"
        )
    if set(frame["variant"].astype(str)) != set(VARIANTS):
        raise ValueError(f"Variant set mismatch in {summary_path}")
    expected = {
        "dataset": DATASET,
        "scenario": str(flow),
        "method": "DuSafe",
        "corruption": CONDITION_CORRUPTION[condition],
        "severity": CONDITION_SEVERITY[condition],
    }
    for column, value in expected.items():
        if not frame[column].astype(str).eq(str(value)).all():
            raise ValueError(
                f"Summary key mismatch in {summary_path}: {column}"
            )
    for column, value in {
        "source_seed": source_seed,
        "stream_seed": stream_seed,
        "corruption_seed": corruption_seed,
    }.items():
        try:
            values = frame[column].map(int)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer column {column} in {summary_path}") from exc
        if not values.eq(int(value)).all():
            raise ValueError(f"Summary key mismatch in {summary_path}: {column}")
    duplicate_columns = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError(f"Duplicate variant rows in {summary_path}")
    for metric in PRIMARY_METRICS:
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        if metric == "corruption_rejection_recall" and condition == "clean":
            if not numeric.isna().all():
                raise ValueError(
                    f"Expected undefined clean-condition {metric} in {summary_path}"
                )
        elif numeric.isna().any() or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Non-finite {metric} in {summary_path}")
        frame[metric] = numeric.astype(float)
    frame["source_seed"] = frame["source_seed"].map(int)
    frame["stream_seed"] = frame["stream_seed"].map(int)
    frame["corruption_seed"] = frame["corruption_seed"].map(int)
    return frame


def collect_cells(
    input_dir: Path,
    source_seeds: tuple[int, ...] = DEFAULT_SOURCE_SEEDS,
    stream_seed: int = DEFAULT_STREAM_SEED,
    corruption_seed: int = DEFAULT_CORRUPTION_SEED,
) -> tuple[pd.DataFrame, list[tuple[str, str, int, int, int]]]:
    """Read and strictly validate every expected cell and variant row."""

    input_dir = Path(input_dir)
    source_seeds = tuple(int(value) for value in source_seeds)
    expected_keys = [
        cell_key(flow, condition, source_seed, stream_seed, corruption_seed)
        for flow in FLOWS
        for condition in CONDITIONS
        for source_seed in source_seeds
    ]
    manifest = _validate_manifest(
        input_dir, source_seeds, stream_seed, corruption_seed
    )
    runtime_hparam_overrides = dict(
        manifest.get("runtime_hparam_overrides", {}) or {}
    )
    _read_status(input_dir, expected_keys)
    rows = []
    for flow, condition, source_seed, cell_stream_seed, cell_corruption_seed in expected_keys:
        output_cell_dir = cell_dir(input_dir, flow, condition, source_seed)
        _validate_metadata(
            output_cell_dir / "cell_metadata.json",
            flow,
            condition,
            source_seed,
            cell_stream_seed,
            cell_corruption_seed,
            runtime_hparam_overrides,
        )
        frame = _validate_summary(
            output_cell_dir / "summary_raw.csv",
            flow,
            condition,
            source_seed,
            cell_stream_seed,
            cell_corruption_seed,
        )
        frame = frame.copy()
        frame["flow"] = flow
        frame["condition"] = condition
        frame["cell_output_dir"] = str(output_cell_dir)
        rows.extend(frame.to_dict("records"))
    result = pd.DataFrame(rows)
    if len(result) != len(expected_keys) * len(VARIANTS):
        raise ValueError(
            f"Expected {len(expected_keys) * len(VARIANTS)} rows, found {len(result)}"
        )
    row_keys = [
        (*cell_key(row["flow"], row["condition"], row["source_seed"], row["stream_seed"], row["corruption_seed"]), row["variant"])
        for row in result.to_dict("records")
    ]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("Duplicate final-panel variant row keys")
    return result, expected_keys


def paired_cell_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Full-minus-no-SSAW values within every shared cell."""

    keys = list(CELL_KEY_FIELDS)
    left = frame[frame["variant"].eq("full")].set_index(keys)
    right = frame[frame["variant"].eq("no_ssaw")].set_index(keys)
    if set(left.index) != set(right.index):
        raise ValueError("Full and no_ssaw cells are not exactly paired")
    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError("Duplicate variant cell keys prevent paired aggregation")
    paired = left[list(PRIMARY_METRICS)].join(
        right[list(PRIMARY_METRICS)],
        lsuffix="_full",
        rsuffix="_no_ssaw",
        how="inner",
        validate="one_to_one",
    ).reset_index()
    for metric in PRIMARY_METRICS:
        paired[f"paired_{metric}_delta"] = (
            paired[f"{metric}_full"] - paired[f"{metric}_no_ssaw"]
        )
        paired[f"full_minus_no_ssaw_{metric}"] = paired[f"paired_{metric}_delta"]
    paired["paired_f1_win"] = paired["paired_f1_delta"] > 1e-12
    paired["paired_f1_tie"] = paired["paired_f1_delta"].abs() <= 1e-12
    paired["paired_f1_loss"] = paired["paired_f1_delta"] < -1e-12
    return paired.sort_values(keys).reset_index(drop=True)


def _aggregate_paired(
    paired: pd.DataFrame,
    group_columns: list[str],
    aggregation_level: str,
) -> pd.DataFrame:
    rows = []
    grouped = paired.groupby(group_columns, sort=True, dropna=False)
    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {
            "aggregation_level": aggregation_level,
            **dict(zip(group_columns, group_values)),
            "n_cells": int(len(group)),
            "n_source_seeds": int(group["source_seed"].nunique()),
            "n_flows": int(group["flow"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            values = group[f"paired_{metric}_delta"].to_numpy(dtype=float)
            row[f"paired_{metric}_delta_mean"] = float(np.mean(values))
            row[f"paired_{metric}_delta_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        row.update(
            {
                "paired_f1_wins": int(group["paired_f1_win"].sum()),
                "paired_f1_ties": int(group["paired_f1_tie"].sum()),
                "paired_f1_losses": int(group["paired_f1_loss"].sum()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if aggregation_level == "flow":
        return result.sort_values(["flow", "condition"]).reset_index(drop=True)
    return result.sort_values(["condition"]).reset_index(drop=True)


def aggregate_paired(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cellwise, by-flow/condition, and overall paired summaries."""

    paired = paired_cell_differences(frame)
    by_flow = _aggregate_paired(paired, ["flow", "condition"], "flow")
    overall = _aggregate_paired(paired, ["condition"], "overall")
    return paired, by_flow, overall


def _exact_sign_flip_pvalue(values: np.ndarray) -> float:
    """Two-sided exact paired randomization p-value for seed-level deltas."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    exceedances = 0
    total = 1 << int(values.size)
    for bits in range(total):
        signs = np.array(
            [1.0 if bits & (1 << index) else -1.0 for index in range(values.size)]
        )
        statistic = abs(float(np.mean(signs * values)))
        if statistic >= observed - 1e-15:
            exceedances += 1
    return float(exceedances / total)


def _holm_adjust(values: np.ndarray) -> np.ndarray:
    """Holm-adjust a vector while preserving NaN positions."""

    values = np.asarray(values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if valid_indices.size == 0:
        return adjusted
    order = valid_indices[np.argsort(values[valid_indices], kind="stable")]
    running_maximum = 0.0
    total = int(order.size)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (total - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def source_seed_inference(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average fixed flows within seed, then infer across source seeds.

    The five transfer flows are fixed benchmark blocks.  Treating their
    15 flow-by-seed cells as IID would overstate the effective sample size.
    This table first averages the five paired deltas within each source seed;
    uncertainty is then computed over the independent source checkpoints.
    """

    seed_aggregations = {
        f"paired_{metric}_delta_mean": (f"paired_{metric}_delta", "mean")
        for metric in PRIMARY_METRICS
    }
    seed_summary = (
        paired.groupby(["condition", "source_seed"], as_index=False)
        .agg(n_flows=("flow", "nunique"), **seed_aggregations)
        .sort_values(["condition", "source_seed"])
        .reset_index(drop=True)
    )

    inference_rows = []
    for condition, group in seed_summary.groupby("condition", sort=True):
        for metric in PRIMARY_METRICS:
            column = f"paired_{metric}_delta_mean"
            values = group[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            count = int(values.size)
            if count == 0:
                mean = std = standard_error = float("nan")
                ci_low = ci_high = ttest_pvalue = sign_flip_pvalue = float("nan")
            else:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if count > 1 else float("nan")
                standard_error = std / math.sqrt(count) if count > 1 else float("nan")
                if count > 1 and np.isfinite(standard_error):
                    critical = float(stats.t.ppf(0.975, df=count - 1))
                    ci_low = mean - critical * standard_error
                    ci_high = mean + critical * standard_error
                    if standard_error == 0.0:
                        ttest_pvalue = 1.0 if abs(mean) <= 1e-15 else 0.0
                    else:
                        ttest_pvalue = float(
                            stats.ttest_1samp(values, popmean=0.0).pvalue
                        )
                else:
                    ci_low = ci_high = ttest_pvalue = float("nan")
                sign_flip_pvalue = _exact_sign_flip_pvalue(values)
            inference_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "inference_unit": "source_seed_after_averaging_fixed_flows",
                    "n_source_seeds": count,
                    "n_flows_per_seed_min": int(group["n_flows"].min()),
                    "n_flows_per_seed_max": int(group["n_flows"].max()),
                    "mean_delta": mean,
                    "std_across_source_seeds": std,
                    "standard_error": standard_error,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_ttest_pvalue": ttest_pvalue,
                    "exact_sign_flip_pvalue": sign_flip_pvalue,
                }
            )
    inference = pd.DataFrame(inference_rows)
    for column in ("paired_ttest_pvalue", "exact_sign_flip_pvalue"):
        inference[f"{column}_holm"] = _holm_adjust(
            inference[column].to_numpy(dtype=float)
        )
    return seed_summary, inference


def _variant_aggregate(frame: pd.DataFrame, group_columns: list[str], level: str) -> pd.DataFrame:
    aggregations = {}
    for metric in PRIMARY_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    result = (
        frame.groupby([*group_columns, "variant"], as_index=False)
        .agg(**aggregations)
    )
    result.insert(0, "aggregation_level", level)
    counts = (
        frame.groupby([*group_columns, "variant"], as_index=False)
        .agg(n_cells=("source_seed", "size"))
    )
    return result.merge(
        counts,
        on=[*group_columns, "variant"],
        how="left",
        validate="one_to_one",
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default=None)
    parser.add_argument("--source_seeds", "--source-seeds", dest="source_seeds", default="1,2,3")
    parser.add_argument("--stream_seed", "--stream-seed", dest="stream_seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument("--corruption_seed", "--corruption-seed", dest="corruption_seed", type=int, default=DEFAULT_CORRUPTION_SEED)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source_seeds = _parse_int_list(args.source_seeds, "--source_seeds")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _validate_manifest(
        input_dir,
        source_seeds,
        args.stream_seed,
        args.corruption_seed,
    )
    frame, expected_keys = collect_cells(
        input_dir,
        source_seeds,
        args.stream_seed,
        args.corruption_seed,
    )
    expected_rows = len(expected_keys) * len(VARIANTS)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, found {len(frame)}")
    paired, by_flow, overall = aggregate_paired(frame)
    seed_summary, inference = source_seed_inference(paired)
    if len(paired) != len(expected_keys):
        raise ValueError(f"Expected {len(expected_keys)} paired cells, found {len(paired)}")

    ordered = frame.sort_values(
        ["flow", "condition", "source_seed", "variant"]
    ).reset_index(drop=True)
    atomic_write_csv(ordered, output_dir / "summary_raw.csv", index=False)
    atomic_write_csv(ordered, output_dir / "summary_by_flow_condition_source.csv", index=False)
    atomic_write_csv(paired, output_dir / "paired_cell_differences.csv", index=False)
    atomic_write_csv(by_flow, output_dir / "paired_by_flow_condition.csv", index=False)
    atomic_write_csv(overall, output_dir / "paired_overall.csv", index=False)
    atomic_write_csv(
        seed_summary,
        output_dir / "paired_by_condition_source_seed.csv",
        index=False,
    )
    atomic_write_csv(
        inference,
        output_dir / "paired_inference_by_condition.csv",
        index=False,
    )
    combined = pd.concat([by_flow, overall], ignore_index=True, sort=False)
    atomic_write_csv(combined, output_dir / "paired_summary.csv", index=False)
    variant_by_flow = _variant_aggregate(frame, ["flow", "condition"], "flow")
    variant_overall = _variant_aggregate(frame, ["condition"], "overall")
    atomic_write_csv(
        pd.concat([variant_by_flow, variant_overall], ignore_index=True, sort=False),
        output_dir / "variant_summary.csv",
        index=False,
    )

    report_manifest = {
        "protocol": f"{PROTOCOL} aggregate",
        "input_dir": str(input_dir),
        "source_seeds": list(source_seeds),
        "source_seed_is_independent_unit": True,
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "corruption_seed": int(args.corruption_seed),
        "flows": list(FLOWS),
        "conditions": list(CONDITIONS),
        "variants": list(VARIANTS),
        "row_count": int(len(frame)),
        "expected_row_count": int(expected_rows),
        "paired_cell_count": int(len(paired)),
        "expected_paired_cell_count": int(len(expected_keys)),
        "unique_cell_keys": int(len(set(expected_keys))),
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_tuning": True,
        "target_labels_used_for_selection": True,
        "target_labels_used_online": False,
        "frozen_har_profile_id": FROZEN_HAR_PROFILE_ID,
        "frozen_har_tta_hparams": dict(FROZEN_HAR_TTA_PARAMS),
        "runtime_hparam_overrides": dict(
            manifest.get("runtime_hparam_overrides", {}) or {}
        ),
        "metric_grain": (
            "Full-minus-no-SSAW differences are computed per flow, condition, "
            "source_seed, stream_seed, and corruption_seed before aggregation"
        ),
        "inference_grain": (
            "The five flows are fixed benchmark blocks. Their paired deltas are "
            "averaged within each source seed before confidence intervals and "
            "hypothesis tests; n_source_seeds is the inferential sample size."
        ),
        "paired_metrics": list(PRIMARY_METRICS),
        "outputs": [
            "summary_raw.csv",
            "summary_by_flow_condition_source.csv",
            "paired_cell_differences.csv",
            "paired_by_flow_condition.csv",
            "paired_overall.csv",
            "paired_by_condition_source_seed.csv",
            "paired_inference_by_condition.csv",
            "paired_summary.csv",
            "variant_summary.csv",
        ],
        "source_manifest_protocol": manifest.get("protocol"),
    }
    _atomic_json(output_dir / "aggregate_manifest.json", report_manifest)
    print(json.dumps(report_manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
