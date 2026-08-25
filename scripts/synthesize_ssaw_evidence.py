"""Fail-closed synthesis of the A--F SSAW evidence panels.

This module only reads finalized CPU/GPU-run artifacts.  It does not run a
trainer, infer missing rows, or turn safety/calibration metrics into a
replacement for macro-F1.  Missing, stale, malformed, or incorrectly
partitioned evidence is recorded as ``inconclusive`` and cannot produce a
positive method-selection decision.

The current formal protocol has five flows per dataset.  HHAR uses the same
five flows used for its dataset-level parameter selection, so every formal
row is target-selected descriptive evidence.  There is deliberately no
confirmatory partition in this ledger.  Supporting probability, safety, and
operator evidence never substitutes for the primary F1 endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from configs.formal_evaluation_protocol import (
    HHAR_REPORTED_FLOWS,
    formal_scenario_pairs,
)
from configs.ssaw_evidence_ledger_protocol import (
    EXPECTED_COUNTS,
    FORMAL_CONFIRMATORY,
    FORMAL_EVALUATION_PARTITION,
    PROTOCOL_VERSION,
)
from configs.ssaw_evaluation_protocol import (
    PRIMARY_CORRUPTIONS,
    PROTOCOL_VERSION as PHYSICAL_PROTOCOL_VERSION,
)
from scripts.analyze_hhar_coupling_factorial import (
    ENDPOINTS as COUPLING_ENDPOINTS,
    HOLDOUT_FLOWS as HHAR_FORMAL_FLOWS,
    RUNNERS as COUPLING_RUNNERS,
    SOURCE_SEEDS,
)
from scripts.analyze_heldout_ssaw_panel import ENDPOINTS as HELDOUT_ENDPOINTS
from scripts.analyze_full_no_ssaw_horizon_queue import (
    HORIZONS,
    SCOPES as HORIZON_SCOPES,
)
from scripts.finalize_baseline_physical_reference_panel import (
    ALL_METHODS as BASELINE_ALL_METHODS,
    BASELINE_METHODS,
    CORRUPTION_SEED,
    DATASETS,
    SEVERITIES as BASELINE_SEVERITIES,
    SOURCE_SEEDS as BASELINE_SOURCE_SEEDS,
    STREAM_SEED as BASELINE_STREAM_SEED,
    VARIANT as BASELINE_VARIANT,
    expected_keys as baseline_expected_keys,
)

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Formal A--F registry.  Do not derive this from the complete HHAR ten-flow
# data-model registry: the reported HHAR subset is intentionally the same
# five-flow subset used by the dataset-level tuner.
FORMAL_DATASETS = tuple(DATASETS)
FORMAL_FLOW_PAIRS = {
    dataset: tuple(formal_scenario_pairs(dataset)) for dataset in FORMAL_DATASETS
}
FORMAL_FLOW_COUNT = sum(len(pairs) for pairs in FORMAL_FLOW_PAIRS.values())
EXPECTED_PARTITION_KEYS = frozenset(
    (dataset, "target_selected_evaluation") for dataset in FORMAL_DATASETS
)
TARGET_SELECTED_PARTITION = FORMAL_EVALUATION_PARTITION
DESCRIPTIVE_STATUS = "descriptive_target_selected"

# Compatibility aliases for callers that imported the old names.  These are
# flow names only; neither alias denotes an untouched/confirmatory partition.
HHAR_DEVELOPMENT_FLOWS = frozenset(HHAR_REPORTED_FLOWS)
HHAR_HOLDOUT_FLOWS = frozenset(HHAR_FORMAL_FLOWS)
PHYSICAL_PARTITIONS = {
    dataset: TARGET_SELECTED_PARTITION for dataset in FORMAL_DATASETS
}
PHYSICAL_F1_ENDPOINTS = (
    "clean_full_minus_no_ssaw_f1",
    "mean_physical_full_minus_no_ssaw_f1",
    "mean_full_minus_no_ssaw_physical_auc",
)
PROBABILITY_ENDPOINTS = {
    "clean_nll",
    "clean_brier",
    "clean_aurc",
    "physical_nll",
    "physical_brier",
    "physical_aurc",
}
HORIZON_F1_ENDPOINT = "future_macro_f1"
BASELINE_F1_ENDPOINTS = {"f1", "corrupted_f1"}
PRIMARY_DECISION_ENDPOINTS = {
    "physical_mean_f1",
    "physical_auc_f1",
    "heldout_f1",
    "future_horizon_f1",
}


class EvidenceError(ValueError):
    """A finalized evidence component violates its declared protocol."""


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise EvidenceError(f"missing or empty manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"manifest is not a JSON object: {path}")
    return payload


def _reject_legacy_confirmatory_claims(
    payload: Mapping[str, Any], source: Path
) -> None:
    """Reject artifacts from the retired HHAR holdout protocol.

    A stale artifact that still describes ``untouched_holdout`` is not
    interchangeable with the current target-selected five-flow protocol.
    Failing closed here prevents an old confirmatory claim from silently
    entering the new descriptive ledger.
    """

    serialized = json.dumps(payload, sort_keys=True, default=str)
    legacy_tokens = ("untouched_holdout", "confirmatory_untouched_holdout")
    if any(token in serialized for token in legacy_tokens):
        raise EvidenceError(
            f"{source} contains retired confirmatory/untouched-holdout claims"
        )


def _read_csv(path: Path, *, required: Sequence[str] = ()) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise EvidenceError(f"missing or empty evidence CSV: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise EvidenceError(f"cannot read evidence CSV: {path}") from exc
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise EvidenceError(f"{path} lacks columns: {missing}")
    return frame


def _require_manifest(
    path: Path,
    *,
    status: str = "complete",
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _reject_legacy_confirmatory_claims(payload, path)
    if status is not None and payload.get("status") != status:
        raise EvidenceError(f"{path} status is {payload.get('status')!r}, expected {status!r}")
    for key, value in (expected or {}).items():
        if payload.get(key) != value:
            raise EvidenceError(
                f"{path} has {key}={payload.get(key)!r}, expected {value!r}"
            )
    return payload


def _finite(frame: pd.DataFrame, columns: Sequence[str], *, bounds: Mapping[str, tuple[float, float]] = ()) -> None:
    for column in columns:
        if column not in frame:
            raise EvidenceError(f"evidence lacks numeric column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise EvidenceError(f"non-finite evidence values in {column}")
        if column in bounds:
            low, high = bounds[column]
            if not values.between(low, high).all():
                raise EvidenceError(f"{column} lies outside [{low}, {high}]")


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise EvidenceError(f"invalid boolean value: {value!r}")


def _partition_keys(frame: pd.DataFrame, source: Path) -> set[tuple[str, str]]:
    required = {"dataset", "evaluation_partition"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EvidenceError(f"{source} lacks partition columns: {missing}")
    keys = {
        (str(dataset).upper(), str(partition))
        for dataset, partition in zip(
            frame["dataset"], frame["evaluation_partition"], strict=True
        )
    }
    if len(keys) != len(frame):
        raise EvidenceError(f"{source} contains duplicate dataset/partition rows")
    if keys != set(EXPECTED_PARTITION_KEYS):
        raise EvidenceError(
            f"{source} partition grid mismatch: "
            f"missing={sorted(set(EXPECTED_PARTITION_KEYS) - keys)}, "
            f"unexpected={sorted(keys - set(EXPECTED_PARTITION_KEYS))}"
        )
    return keys


def _partition_membership(frame: pd.DataFrame, source: Path) -> set[tuple[str, str]]:
    required = {"dataset", "evaluation_partition"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EvidenceError(f"{source} lacks partition columns: {missing}")
    keys = {
        (str(dataset).upper(), str(partition))
        for dataset, partition in zip(
            frame["dataset"], frame["evaluation_partition"], strict=True
        )
    }
    if not keys.issubset(EXPECTED_PARTITION_KEYS):
        raise EvidenceError(
            f"{source} has unregistered dataset/partition rows: "
            f"{sorted(keys - set(EXPECTED_PARTITION_KEYS))}"
        )
    return keys


def _validate_holm(
    frame: pd.DataFrame,
    *,
    raw: str,
    holm: str,
    source: Path,
) -> None:
    _finite(frame, [raw, holm], bounds={raw: (0.0, 1.0), holm: (0.0, 1.0)})
    raw_values = pd.to_numeric(frame[raw], errors="coerce").to_numpy(dtype=float)
    holm_values = pd.to_numeric(frame[holm], errors="coerce").to_numpy(dtype=float)
    if np.any(holm_values + 1e-12 < raw_values):
        raise EvidenceError(f"{source} has Holm-adjusted p-values below raw p-values")


def _validate_effect_definition(
    frame: pd.DataFrame,
    *,
    column: str,
    expected: Mapping[str, str],
    source: Path,
) -> None:
    if column not in frame:
        raise EvidenceError(f"{source} lacks effect-direction column {column}")
    for endpoint, values in frame.groupby("endpoint", sort=False):
        expected_value = expected.get(str(endpoint))
        if expected_value is None:
            raise EvidenceError(f"{source} has unregistered endpoint {endpoint!r}")
        observed = values[column].astype(str).str.strip()
        if observed.ne(expected_value).any():
            raise EvidenceError(
                f"{source} has an incorrect effect direction for {endpoint!r}"
            )


def _ledger_row(
    *,
    component: str,
    role: str,
    dataset: str,
    partition: str,
    endpoint: str,
    metric: str,
    estimate: float,
    source_path: Path,
    confirmatory: bool,
    ci_low: float | None = None,
    ci_high: float | None = None,
    p_raw: float | None = None,
    p_holm: float | None = None,
    direction: str = "higher_is_better",
    status: str = "complete",
) -> dict[str, Any]:
    values = {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high, "p_raw": p_raw, "p_holm": p_holm}
    for name, value in values.items():
        if value is not None and not math.isfinite(float(value)):
            raise EvidenceError(f"non-finite ledger value {name} for {component}/{endpoint}")
    return {
        "component": component,
        "role": role,
        "dataset": str(dataset),
        "evaluation_partition": str(partition),
        "endpoint": endpoint,
        "metric": metric,
        "direction": direction,
        "estimate": float(estimate),
        "ci_low": None if ci_low is None else float(ci_low),
        "ci_high": None if ci_high is None else float(ci_high),
        "p_raw": None if p_raw is None else float(p_raw),
        "p_holm": None if p_holm is None else float(p_holm),
        "confirmatory": bool(confirmatory),
        "status": status,
        "source_path": str(Path(source_path).resolve()),
        "f1_is_primary": bool(metric == "macro_f1"),
    }


def _partition_confirmatory(dataset: str, partition: str) -> bool:
    """Return the current formal status; all rows are descriptive."""

    if str(partition) != TARGET_SELECTED_PARTITION:
        raise EvidenceError(
            f"retired/non-formal partition for {dataset}: {partition}"
        )
    return bool(FORMAL_CONFIRMATORY)


def _check_partition_rows(frame: pd.DataFrame, *, expected_rows: int, source: Path) -> None:
    if len(frame) != int(expected_rows):
        raise EvidenceError(f"{source} has {len(frame)} rows; expected {expected_rows}")
    if "evaluation_partition" not in frame:
        raise EvidenceError(f"{source} lacks evaluation_partition")
    if frame["evaluation_partition"].isna().any():
        raise EvidenceError(f"{source} has missing evaluation partitions")
    _partition_keys(frame, source)
    # Every formal dataset now uses the same target-selected descriptive
    # partition.  Retired development/untouched-holdout labels are rejected.
    for row in frame.itertuples(index=False):
        dataset = str(row.dataset).upper()
        partition = str(row.evaluation_partition)
        if dataset not in FORMAL_DATASETS or partition != TARGET_SELECTED_PARTITION:
            raise EvidenceError(f"HHAR row has invalid evaluation partition: {partition}")


def _physical_component(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    manifest = _require_manifest(
        root / "manifest.json",
        status=None,
        expected={
            "protocol_version": PHYSICAL_PROTOCOL_VERSION,
            "expected_cells": EXPECTED_COUNTS["physical_cells"],
            "validated_cells": EXPECTED_COUNTS["physical_cells"],
            "online_target_labels_used": False,
            "source_seeds": [1, 2, 3],
            "stream_seed": 42,
            "variants": ["full", "no_ssaw"],
        },
    )
    analysis_root = root / "physical_analysis"
    analysis_manifest = _require_manifest(
        analysis_root / "manifest.json",
        status=None,
        expected={
            "paired_cell_count": EXPECTED_COUNTS["physical_paired_cells"],
            "paired_auc_count": EXPECTED_COUNTS["physical_auc_pairs"],
        },
    )
    if analysis_manifest.get("dependence_cluster") != "source_model_sha256":
        raise EvidenceError("physical analysis does not cluster by source checkpoint")
    if "checkpoint_cluster" not in str(analysis_manifest.get("paired_test", "")).lower():
        raise EvidenceError("physical analysis does not declare checkpoint-cluster inference")
    if "holm" not in str(analysis_manifest.get("multiple_comparison_correction", "")).lower():
        raise EvidenceError("physical analysis does not declare Holm correction")
    paired_cells = _read_csv(analysis_root / "paired_physical_cells.csv")
    if len(paired_cells) != EXPECTED_COUNTS["physical_paired_cells"]:
        raise EvidenceError(
            "physical paired-cell panel has an unexpected row count: "
            f"{len(paired_cells)}"
        )
    paired_auc = _read_csv(analysis_root / "paired_physical_auc.csv")
    if len(paired_auc) != EXPECTED_COUNTS["physical_auc_pairs"]:
        raise EvidenceError(
            "physical paired-AUC panel has an unexpected row count: "
            f"{len(paired_auc)}"
        )
    summary_path = analysis_root / "physical_panel_summary_by_partition.csv"
    summary = _read_csv(
        summary_path,
        required=[
            "dataset",
            "evaluation_partition",
            "confirmatory_status",
            *PHYSICAL_F1_ENDPOINTS,
            "physical_cluster_ci95_low",
            "physical_cluster_ci95_high",
            "physical_cluster_signflip_p_raw",
            "physical_cluster_signflip_p_holm",
            "clean_cluster_signflip_p_raw",
            "clean_cluster_signflip_p_holm",
            "auc_cluster_ci95_low",
            "auc_cluster_ci95_high",
            "auc_cluster_signflip_p_raw",
            "auc_cluster_signflip_p_holm",
        ],
    )
    _check_partition_rows(summary, expected_rows=len(EXPECTED_PARTITION_KEYS), source=summary_path)
    _partition_keys(summary, summary_path)
    for row in summary.itertuples(index=False):
        if str(row.confirmatory_status) != DESCRIPTIVE_STATUS:
            raise EvidenceError("physical summary must be descriptive target-selected evidence")
    _finite(
        summary,
        [
            *PHYSICAL_F1_ENDPOINTS,
            "physical_cluster_ci95_low",
            "physical_cluster_ci95_high",
            "physical_cluster_signflip_p_raw",
            "physical_cluster_signflip_p_holm",
            "clean_cluster_signflip_p_raw",
            "clean_cluster_signflip_p_holm",
            "auc_cluster_ci95_low",
            "auc_cluster_ci95_high",
            "auc_cluster_signflip_p_raw",
            "auc_cluster_signflip_p_holm",
        ],
    )
    for raw, holm in (
        ("physical_cluster_signflip_p_raw", "physical_cluster_signflip_p_holm"),
        ("clean_cluster_signflip_p_raw", "clean_cluster_signflip_p_holm"),
        ("auc_cluster_signflip_p_raw", "auc_cluster_signflip_p_holm"),
    ):
        _validate_holm(summary, raw=raw, holm=holm, source=summary_path)
    rows: list[dict[str, Any]] = []
    for item in summary.itertuples(index=False):
        dataset = str(item.dataset)
        partition = str(item.evaluation_partition)
        confirmatory = _partition_confirmatory(dataset, partition)
        rows.extend(
            [
                _ledger_row(
                    component="A_physical_f1",
                    role="primary_f1",
                    dataset=dataset,
                    partition=partition,
                    endpoint="physical_mean_f1",
                    metric="macro_f1",
                    estimate=float(item.mean_physical_full_minus_no_ssaw_f1),
                    ci_low=float(item.physical_cluster_ci95_low),
                    ci_high=float(item.physical_cluster_ci95_high),
                    p_raw=float(item.physical_cluster_signflip_p_raw),
                    p_holm=float(item.physical_cluster_signflip_p_holm),
                    source_path=summary_path,
                    confirmatory=confirmatory,
                ),
                _ledger_row(
                    component="A_physical_f1",
                    role="primary_f1",
                    dataset=dataset,
                    partition=partition,
                    endpoint="physical_auc_f1",
                    metric="macro_f1",
                    estimate=float(item.mean_full_minus_no_ssaw_physical_auc),
                    ci_low=float(item.auc_cluster_ci95_low),
                    ci_high=float(item.auc_cluster_ci95_high),
                    p_raw=float(item.auc_cluster_signflip_p_raw),
                    p_holm=float(item.auc_cluster_signflip_p_holm),
                    source_path=summary_path,
                    confirmatory=confirmatory,
                ),
            ]
        )

    probability_path = root / "probability_effect_summary_by_partition.csv"
    probability = _read_csv(
        probability_path,
        required=[
            "dataset",
            "evaluation_partition",
            "confirmatory_status",
            "endpoint",
            "full_improvement_mean",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm",
        ],
    )
    if len(probability) != len(EXPECTED_PARTITION_KEYS) * len(PROBABILITY_ENDPOINTS):
        raise EvidenceError(
            f"{probability_path} has {len(probability)} rows; expected "
            f"{len(EXPECTED_PARTITION_KEYS) * len(PROBABILITY_ENDPOINTS)}"
        )
    probability_keys = {
        (str(item.dataset).upper(), str(item.evaluation_partition), str(item.endpoint))
        for item in probability.itertuples(index=False)
    }
    expected_probability_keys = {
        (dataset, partition, endpoint)
        for dataset, partition in EXPECTED_PARTITION_KEYS
        for endpoint in PROBABILITY_ENDPOINTS
    }
    if probability_keys != expected_probability_keys or len(probability_keys) != len(probability):
        raise EvidenceError("physical probability endpoint grid drifted")
    for row in probability.itertuples(index=False):
        if str(row.confirmatory_status) != DESCRIPTIVE_STATUS:
            raise EvidenceError("physical probability rows must be descriptive target-selected evidence")
    _finite(
        probability,
        [
            "full_improvement_mean",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm",
        ],
    )
    _validate_holm(
        probability,
        raw="cluster_signflip_p_raw",
        holm="cluster_signflip_p_holm",
        source=probability_path,
    )
    for item in probability.itertuples(index=False):
        dataset = str(item.dataset)
        partition = str(item.evaluation_partition)
        rows.append(
            _ledger_row(
                component="B_probability_risk",
                role="supporting_probability",
                dataset=dataset,
                partition=partition,
                endpoint=str(item.endpoint),
                metric="probability_metric",
                estimate=float(item.full_improvement_mean),
                ci_low=float(item.cluster_ci95_low),
                ci_high=float(item.cluster_ci95_high),
                p_raw=float(item.cluster_signflip_p_raw),
                p_holm=float(item.cluster_signflip_p_holm),
                direction="lower_is_better_for_source_metric",
                source_path=probability_path,
                confirmatory=_partition_confirmatory(dataset, partition),
            )
        )
    # Safety aggregate is descriptive/diagnostic. It must exist and have the
    # declared standard metrics, but it is never used in F1 decision logic.
    safety_path = root / "safety_metrics_aggregate.csv"
    safety = _read_csv(safety_path)
    if safety.empty:
        raise EvidenceError(f"{safety_path} contains no safety rows")
    required_safety = {
        "coverage_mean",
        "accepted_accuracy_mean",
        "corruption_recall_mean",
        "clean_correct_false_rejection_mean",
        "unsafe_update_rate_mean",
    }
    if not required_safety.issubset(safety.columns):
        raise EvidenceError(f"{safety_path} lacks declared safety metrics")
    _finite(safety, sorted(required_safety))
    metadata = {
        "manifest": manifest,
        "analysis_manifest": analysis_manifest,
        "physical_paired_cells": int(len(paired_cells)),
        "physical_paired_auc": int(len(paired_auc)),
        "physical_summary_rows": int(len(summary)),
        "probability_summary_rows": int(len(probability)),
        "safety_rows": int(len(safety)),
    }
    return rows, metadata


def _heldout_component(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    manifest = _require_manifest(
        root / "manifest.json",
        status=None,
        expected={
            "protocol_version": "ssaw_heldout_clustered_analysis_v2_five_formal_flows",
            "paired_units": EXPECTED_COUNTS["heldout_paired_units"],
            "expected_paired_units": EXPECTED_COUNTS["heldout_paired_units"],
            "datasets": list(DATASETS),
            "source_seeds": [1, 2, 3],
            "checkpoint_is_independent_cluster": True,
            "confirmatory_partition": None,
            "hhar_formal_flow_policy": "five target-selected flows; no confirmatory subset",
            "target_selected_partitions_are_confirmatory": False,
            "holm_global_family_size": 24,
            "holm_confirmatory_family_size": 0,
            "ground_truth_lpr_observed": False,
            "operator_metrics_are_algorithm_effects": False,
        },
    )
    path = root / "confirmatory_inference.csv"
    paired_units = _read_csv(root / "paired_units.csv")
    if len(paired_units) != EXPECTED_COUNTS["heldout_paired_units"]:
        raise EvidenceError(
            "held-out paired-unit panel has an unexpected row count: "
            f"{len(paired_units)}"
        )
    frame = _read_csv(
        path,
        required=[
            "dataset",
            "evaluation_partition",
            "confirmatory",
            "endpoint",
            "benefit_mean",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm_global",
        ],
    )
    expected_rows = len(EXPECTED_PARTITION_KEYS) * len(HELDOUT_ENDPOINTS)
    if len(frame) != expected_rows:
        raise EvidenceError(f"{path} has {len(frame)} rows; expected {expected_rows}")
    _partition_membership(frame, path)
    observed_keys = {
        (str(item.dataset).upper(), str(item.evaluation_partition), str(item.endpoint))
        for item in frame.itertuples(index=False)
    }
    expected_keys = {
        (dataset, partition, endpoint)
        for dataset, partition in EXPECTED_PARTITION_KEYS
        for endpoint in HELDOUT_ENDPOINTS
    }
    if observed_keys != expected_keys or len(observed_keys) != len(frame):
        raise EvidenceError("held-out mechanism endpoint grid drifted")
    confirmatory = frame["confirmatory"].map(_bool)
    if confirmatory.any():
        raise EvidenceError("held-out mechanism contains confirmatory rows")
    _finite(frame, ["benefit_mean", "cluster_ci95_low", "cluster_ci95_high", "cluster_signflip_p_raw", "cluster_signflip_p_holm_global"])
    _validate_holm(
        frame,
        raw="cluster_signflip_p_raw",
        holm="cluster_signflip_p_holm_global",
        source=path,
    )
    _validate_effect_definition(
        frame,
        column="benefit_direction",
        expected={
            "clean_f1": "Full-minus-noSSAW",
            "heldout_f1": "Full-minus-noSSAW",
            "heldout_js": "noSSAW-minus-Full",
            "heldout_flip": "noSSAW-minus-Full",
            "heldout_margin_degradation": "noSSAW-minus-Full",
            "heldout_feature_distance": "noSSAW-minus-Full",
        },
        source=path,
    )
    rows = []
    for item in frame.itertuples(index=False):
        dataset = str(item.dataset)
        partition = str(item.evaluation_partition)
        metric = "macro_f1" if str(item.endpoint) in {"clean_f1", "heldout_f1"} else "mechanism_metric"
        rows.append(
            _ledger_row(
                component="C_heldout_mechanism",
                role="primary_f1" if metric == "macro_f1" else "supporting_mechanism",
                dataset=dataset,
                partition=partition,
                endpoint=str(item.endpoint),
                metric=metric,
                estimate=float(item.benefit_mean),
                ci_low=float(item.cluster_ci95_low),
                ci_high=float(item.cluster_ci95_high),
                p_raw=float(item.cluster_signflip_p_raw),
                p_holm=float(item.cluster_signflip_p_holm_global),
                direction="higher_is_better_after_endpoint_direction",
                source_path=path,
                confirmatory=_bool(item.confirmatory),
            )
        )
    return rows, {
        "manifest": manifest,
        "paired_units": int(len(paired_units)),
        "inference_rows": int(len(frame)),
    }


def _horizon_component(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    manifest = _require_manifest(
        root / "manifest.json",
        status=None,
        expected={
            "protocol_version": "full_no_ssaw_horizon_clustered_analysis_v2_five_formal_flows",
            "stream_cells": EXPECTED_COUNTS["horizon_stream_cells"],
            "expected_horizon_endpoint_cells": EXPECTED_COUNTS["horizon_endpoint_cells"],
            "horizons_share_exact_online_trajectory": True,
            "source_checkpoint_is_independent_cluster": True,
            "target_labels_used_for_updates": False,
            "confirmatory_partition": None,
            "hhar_formal_flow_policy": "five target-selected flows; no confirmatory subset",
            "target_selected_partitions_are_confirmatory": False,
            "holm_global_family_size": 96,
            "holm_confirmatory_family_size": 0,
        },
    )
    endpoint_path = root / "paired_horizon_endpoints.csv"
    endpoints = _read_csv(endpoint_path, required=["endpoint_key", "horizon", "condition"])
    if len(endpoints) != EXPECTED_COUNTS["horizon_endpoint_cells"] or endpoints["endpoint_key"].astype(str).duplicated().any():
        raise EvidenceError(
            "horizon endpoint panel is not exactly "
            f"{EXPECTED_COUNTS['horizon_endpoint_cells']} unique cells"
        )
    _finite(endpoints, ["horizon"], bounds={"horizon": (1.0, 5.0)})
    if set(pd.to_numeric(endpoints["horizon"], errors="raise").astype(int)) != set(HORIZONS):
        raise EvidenceError("horizon endpoint panel has an unregistered horizon")
    path = root / "clustered_inference.csv"
    frame = _read_csv(
        path,
        required=[
            "dataset",
            "evaluation_partition",
            "confirmatory",
            "horizon",
            "condition_scope",
            "endpoint",
            "cluster_mean",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm_global",
        ],
    )
    expected_rows = len(EXPECTED_PARTITION_KEYS) * len(HORIZONS) * len(HORIZON_SCOPES) * 2
    if len(frame) != expected_rows:
        raise EvidenceError(f"{path} has {len(frame)} rows; expected {expected_rows}")
    _partition_membership(frame, path)
    if set(frame["endpoint"].astype(str)) != {"future_macro_f1", "future_true_label_nll"}:
        raise EvidenceError("horizon endpoint family drifted")
    expected_inference_keys = {
        (dataset, partition, int(horizon), str(scope), endpoint)
        for dataset, partition in EXPECTED_PARTITION_KEYS
        for horizon in HORIZONS
        for scope in HORIZON_SCOPES
        for endpoint in ("future_macro_f1", "future_true_label_nll")
    }
    observed_inference_keys = {
        (
            str(item.dataset).upper(), str(item.evaluation_partition), int(item.horizon),
            str(item.condition_scope), str(item.endpoint),
        )
        for item in frame.itertuples(index=False)
    }
    if observed_inference_keys != expected_inference_keys or len(observed_inference_keys) != len(frame):
        raise EvidenceError("horizon inference grid is incomplete or duplicated")
    if set(frame["condition_scope"].astype(str)) != set(HORIZON_SCOPES):
        raise EvidenceError("horizon condition-scope grid drifted")
    confirmatory = frame["confirmatory"].map(_bool)
    if confirmatory.any():
        raise EvidenceError("horizon panel contains confirmatory rows")
    _finite(frame, ["cluster_mean", "cluster_ci95_low", "cluster_ci95_high", "cluster_signflip_p_raw", "cluster_signflip_p_holm_global"])
    _validate_holm(
        frame,
        raw="cluster_signflip_p_raw",
        holm="cluster_signflip_p_holm_global",
        source=path,
    )
    _validate_effect_definition(
        frame,
        column="effect_definition",
        expected={
            "future_macro_f1": "Full-minus-noSSAW",
            "future_true_label_nll": "noSSAW-NLL minus Full-NLL",
        },
        source=path,
    )
    rows = []
    for item in frame.itertuples(index=False):
        is_f1 = str(item.endpoint) == HORIZON_F1_ENDPOINT
        rows.append(
            _ledger_row(
                component="D_future_horizon",
                role="primary_f1" if is_f1 else "supporting_nll",
                dataset=str(item.dataset),
                partition=str(item.evaluation_partition),
                endpoint=f"{item.endpoint}_h{int(item.horizon)}_{item.condition_scope}",
                metric="macro_f1" if is_f1 else "true_label_nll",
                estimate=float(item.cluster_mean),
                ci_low=float(item.cluster_ci95_low),
                ci_high=float(item.cluster_ci95_high),
                p_raw=float(item.cluster_signflip_p_raw),
                p_holm=float(item.cluster_signflip_p_holm_global),
                direction="higher_is_better" if is_f1 else "higher_improvement_is_better",
                source_path=path,
                confirmatory=_bool(item.confirmatory),
            )
        )
    return rows, {"manifest": manifest, "endpoint_rows": int(len(endpoints)), "inference_rows": int(len(frame))}


def _baseline_component(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    manifest = _require_manifest(
        root / "manifest.json",
        status=None,
        expected={
            "protocol": "baseline_physical_reference_s3_s6_v2_five_flow",
            "status": "complete",
            "datasets": list(DATASETS),
            "methods": list(BASELINE_ALL_METHODS),
            "baseline_methods": list(BASELINE_METHODS),
            "variant": BASELINE_VARIANT,
            "corruptions": list(PRIMARY_CORRUPTIONS),
            "severities": list(BASELINE_SEVERITIES),
            "baseline_expected_cells": EXPECTED_COUNTS["baseline_cells"],
            "dusafe_expected_cells": EXPECTED_COUNTS["dusafe_baseline_cells"],
            "expected_cells": EXPECTED_COUNTS["baseline_merged_cells"],
            "validated_cells": EXPECTED_COUNTS["baseline_merged_cells"],
            "online_target_labels_used": False,
            "corruption_seed": CORRUPTION_SEED,
            "hhar_partition_policy": {
                "reported_flows": list(HHAR_REPORTED_FLOWS),
                "reported_status": DESCRIPTIVE_STATUS,
                "parameter_selection_data_overlap": True,
                "confirmatory_results": "none",
            },
        },
    )
    if manifest.get("source_seeds") != list(BASELINE_SOURCE_SEEDS):
        raise EvidenceError("baseline source-seed protocol drifted")
    if manifest.get("stream_seed") != BASELINE_STREAM_SEED:
        raise EvidenceError("baseline stream seed protocol drifted")
    inference_meta = manifest.get("inference")
    if not isinstance(inference_meta, Mapping):
        raise EvidenceError("baseline manifest lacks inference provenance")
    if inference_meta.get("cluster") != "source_model_sha256":
        raise EvidenceError("baseline inference does not cluster by source checkpoint")
    if "holm" not in str(inference_meta.get("multiple_comparison_correction", "")).lower():
        raise EvidenceError("baseline inference does not declare Holm correction")
    panel_path = root / "panel_raw.csv"
    panel = _read_csv(panel_path)
    if len(panel) != EXPECTED_COUNTS["baseline_merged_cells"]:
        raise EvidenceError(
            "baseline panel_raw is not exactly "
            f"{EXPECTED_COUNTS['baseline_merged_cells']} rows"
        )
    panel_required = {
        "dataset", "scenario", "method", "variant", "corruption", "severity",
        "source_seed", "stream_seed", "corruption_seed", "source_model_sha256",
    }
    if not panel_required.issubset(panel.columns):
        raise EvidenceError(
            f"{panel_path} lacks source/checkpoint grid columns: "
            f"{sorted(panel_required - set(panel.columns))}"
        )
    panel_keys = [
        (
            str(row.dataset), str(row.scenario), str(row.method), str(row.variant),
            str(row.corruption), str(row.severity), int(row.source_seed),
            int(row.stream_seed), int(row.corruption_seed),
        )
        for row in panel.itertuples(index=False)
    ]
    expected_panel_keys = baseline_expected_keys()
    if len(set(panel_keys)) != len(panel_keys) or set(panel_keys) != expected_panel_keys:
        raise EvidenceError(f"{panel_path} key grid is incomplete or duplicated")
    hashes = panel["source_model_sha256"].astype(str).str.strip()
    if hashes.eq("").any() or hashes.eq("nan").any() or not hashes.str.fullmatch(SHA256_RE).all():
        raise EvidenceError(f"{panel_path} lacks source checkpoint hashes")
    # Every method must use the same source checkpoint for an exact cell, and
    # each source-domain/seed must map to one independent checkpoint cluster.
    cell_columns = [
        "dataset", "scenario", "corruption", "severity", "source_seed",
        "stream_seed", "corruption_seed",
    ]
    if not panel.groupby(cell_columns, dropna=False)["source_model_sha256"].nunique().eq(1).all():
        raise EvidenceError("baseline methods do not share source checkpoint hashes")
    source_unit = panel.assign(
        _source_domain=panel["scenario"].astype(str).str.split("->", n=1).str[0],
        _source_seed=pd.to_numeric(panel["source_seed"], errors="raise").astype(int),
    )
    if not source_unit.groupby(["dataset", "_source_domain", "_source_seed"])["source_model_sha256"].nunique().eq(1).all():
        raise EvidenceError("baseline source-domain/seed maps to multiple checkpoints")
    aggregate = _read_csv(root / "panel_aggregate.csv")
    if len(aggregate) != EXPECTED_COUNTS["baseline_aggregate_rows"]:
        raise EvidenceError(
            "baseline panel aggregate is not exactly "
            f"{EXPECTED_COUNTS['baseline_aggregate_rows']} rows"
        )
    aggregate_keys = [
        "dataset", "scenario", "method", "variant", "corruption", "severity"
    ]
    if not set(aggregate_keys).issubset(aggregate.columns):
        raise EvidenceError(f"{root / 'panel_aggregate.csv'} lacks aggregate key columns")
    observed_aggregate_keys = {
        tuple(str(getattr(row, column)) for column in aggregate_keys)
        for row in aggregate.itertuples(index=False)
    }
    expected_aggregate_keys = {
        key[:6] for key in expected_panel_keys
    }
    if len(observed_aggregate_keys) != len(aggregate) or observed_aggregate_keys != expected_aggregate_keys:
        raise EvidenceError("baseline aggregate key grid is incomplete or duplicated")
    path = root / "dusafe_vs_baseline_paired_inference.csv"
    frame = _read_csv(
        path,
        required=[
            "dataset",
            "evaluation_partition",
            "confirmatory_status",
            "baseline_method",
            "endpoint",
            "paired_improvement_mean",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm",
        ],
    )
    expected_rows = len(EXPECTED_PARTITION_KEYS) * len(BASELINE_METHODS) * 13
    if len(frame) != expected_rows:
        raise EvidenceError(f"{path} has {len(frame)} rows; expected {expected_rows}")
    if set(frame["endpoint"].astype(str)) != {
        "f1",
        "corrupted_f1",
        "coverage",
        "accepted_accuracy",
        "rejection_recall",
        "false_rejection",
        "unsafe_update",
        "nll",
        "brier",
        "aurc",
        "corrupted_nll",
        "corrupted_brier",
        "corrupted_aurc",
    }:
        raise EvidenceError("baseline comparison endpoint grid drifted")
    inference_keys = {
        (
            str(row.dataset), str(row.evaluation_partition),
            str(row.baseline_method), str(row.endpoint),
        )
        for row in frame.itertuples(index=False)
    }
    expected_inference_keys = {
        (dataset, partition, method, endpoint)
        for dataset, partition in EXPECTED_PARTITION_KEYS
        for method in BASELINE_METHODS
        for endpoint in BASELINE_F1_ENDPOINTS | {
            "coverage", "accepted_accuracy", "rejection_recall", "false_rejection",
            "unsafe_update", "nll", "brier", "aurc", "corrupted_nll",
            "corrupted_brier", "corrupted_aurc",
        }
    }
    if len(inference_keys) != len(frame) or inference_keys != expected_inference_keys:
        raise EvidenceError("baseline inference key grid is incomplete or duplicated")
    if not set(frame["direction"].astype(str)).issubset({"higher", "lower"}):
        raise EvidenceError("baseline inference has an invalid endpoint direction")
    expected_directions = {
        "f1": "higher", "corrupted_f1": "higher", "coverage": "higher",
        "accepted_accuracy": "higher", "rejection_recall": "higher",
        "false_rejection": "lower", "unsafe_update": "lower", "nll": "lower",
        "brier": "lower", "aurc": "lower", "corrupted_nll": "lower",
        "corrupted_brier": "lower", "corrupted_aurc": "lower",
    }
    _validate_effect_definition(
        frame, column="direction", expected=expected_directions, source=path
    )
    for row in frame.itertuples(index=False):
        expected_statuses = {
            DESCRIPTIVE_STATUS,
            "registered_non_hhar_reference",
        }
        if str(row.confirmatory_status) not in expected_statuses:
            raise EvidenceError("baseline partition status is not descriptive target-selected evidence")
    _finite(frame, ["paired_improvement_mean", "cluster_ci95_low", "cluster_ci95_high", "cluster_signflip_p_raw", "cluster_signflip_p_holm"])
    _validate_holm(
        frame,
        raw="cluster_signflip_p_raw",
        holm="cluster_signflip_p_holm",
        source=path,
    )
    rows = []
    for item in frame.itertuples(index=False):
        endpoint = str(item.endpoint)
        is_f1 = endpoint in BASELINE_F1_ENDPOINTS
        rows.append(
            _ledger_row(
                component="E_baseline_reference",
                role="primary_f1" if is_f1 else "supporting_safety_or_probability",
                dataset=str(item.dataset),
                partition=str(item.evaluation_partition),
                endpoint=f"DuSafe_vs_{item.baseline_method}_{endpoint}",
                metric="macro_f1" if is_f1 else endpoint,
                estimate=float(item.paired_improvement_mean),
                ci_low=float(item.cluster_ci95_low),
                ci_high=float(item.cluster_ci95_high),
                p_raw=float(item.cluster_signflip_p_raw),
                p_holm=float(item.cluster_signflip_p_holm),
                direction="higher_is_better_after_declared_direction",
                source_path=path,
                confirmatory=False,
            )
        )
    return rows, {"manifest": manifest, "panel_rows": int(len(panel)), "aggregate_rows": int(len(aggregate)), "inference_rows": int(len(frame))}


def _coupling_component(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    manifest = _require_manifest(
        root / "manifest.json",
        status=None,
        expected={
            "protocol_version": "hhar_coupling_factorial_clustered_analysis_v2_single_flow",
            "validated_cells": EXPECTED_COUNTS["coupling_cells"],
            "expected_cells": EXPECTED_COUNTS["coupling_cells"],
            "paired_flow_seed_units": EXPECTED_COUNTS["coupling_flow_seed_units"],
            "evaluation_partition": TARGET_SELECTED_PARTITION,
            "confirmatory": False,
            "target_labels_used_for_parameter_selection": True,
            "parameter_selection_data_overlap": True,
            "holm_family_size": len(COUPLING_ENDPOINTS),
        },
    )
    cells = _read_csv(root / "validated_cells.csv")
    if len(cells) != EXPECTED_COUNTS["coupling_cells"]:
        raise EvidenceError(
            "coupling validated-cell count is not "
            f"{EXPECTED_COUNTS['coupling_cells']}"
        )
    cell_required = {
        "dataset", "scenario", "source_seed", "stream_seed", "runner",
        "source_model_sha256", "target_labels_used_for_parameter_selection",
        "parameter_selection_data_overlap", "evaluation_partition", "confirmatory",
    }
    if not cell_required.issubset(cells.columns):
        raise EvidenceError(
            f"coupling validated cells lack provenance columns: "
            f"{sorted(cell_required - set(cells.columns))}"
        )
    cell_keys = {
        (str(row.dataset).upper(), str(row.scenario), int(row.source_seed),
         int(row.stream_seed), str(row.runner))
        for row in cells.itertuples(index=False)
    }
    expected_cell_keys = {
        ("HHAR", str(flow), int(seed), 42, str(runner))
        for flow in HHAR_HOLDOUT_FLOWS
        for seed in SOURCE_SEEDS
        for runner in COUPLING_RUNNERS
    }
    if len(cell_keys) != len(cells) or cell_keys != expected_cell_keys:
        raise EvidenceError("coupling validated-cell grid is incomplete or duplicated")
    cell_hashes = cells["source_model_sha256"].astype(str).str.strip()
    if not cell_hashes.str.fullmatch(SHA256_RE).all():
        raise EvidenceError("coupling cells require SHA-256 source checkpoint hashes")
    if cells["target_labels_used_for_parameter_selection"].map(_bool).ne(True).any():
        raise EvidenceError("coupling target-label selection provenance is incomplete")
    if cells["parameter_selection_data_overlap"].map(_bool).ne(True).any():
        raise EvidenceError("coupling rows must declare target-selected overlap")
    if cells["evaluation_partition"].astype(str).ne(TARGET_SELECTED_PARTITION).any() or cells[
        "confirmatory"
    ].map(_bool).ne(False).any():
        raise EvidenceError("coupling cells are not descriptive target-selected rows")
    group_cols = ["scenario", "source_seed", "stream_seed"]
    if not cells.groupby(group_cols, dropna=False)["source_model_sha256"].nunique().eq(1).all():
        raise EvidenceError("coupling runners do not share source checkpoints")
    source_unit = cells.assign(
        _source_domain=cells["scenario"].astype(str).str.split("->", n=1).str[0],
        _source_seed=pd.to_numeric(cells["source_seed"], errors="raise").astype(int),
    )
    if not source_unit.groupby(["_source_domain", "_source_seed"])["source_model_sha256"].nunique().eq(1).all():
        raise EvidenceError("coupling source-domain/seed maps to multiple checkpoints")
    effects = _read_csv(root / "paired_effects.csv")
    if len(effects) != EXPECTED_COUNTS["coupling_effect_rows"]:
        raise EvidenceError("coupling paired-effect count is incorrect")
    effect_required = {"scenario", "source_seed", "stream_seed", "endpoint", "effect", "source_model_sha256"}
    if not effect_required.issubset(effects.columns):
        raise EvidenceError("coupling paired effects lack cluster provenance")
    effect_keys = {
        (str(row.scenario), int(row.source_seed), int(row.stream_seed), str(row.endpoint))
        for row in effects.itertuples(index=False)
    }
    expected_effect_keys = {
        (str(flow), int(seed), 42, str(endpoint))
        for flow in HHAR_HOLDOUT_FLOWS
        for seed in SOURCE_SEEDS
        for endpoint in COUPLING_ENDPOINTS
    }
    if len(effect_keys) != len(effects) or effect_keys != expected_effect_keys:
        raise EvidenceError("coupling paired-effect grid is incomplete or duplicated")
    if not effects["source_model_sha256"].astype(str).str.fullmatch(SHA256_RE).all():
        raise EvidenceError("coupling paired effects lack SHA-256 source hashes")
    cell_hash_map = {
        (str(row.scenario), int(row.source_seed), int(row.stream_seed)): str(row.source_model_sha256)
        for row in cells.itertuples(index=False)
    }
    for row in effects.itertuples(index=False):
        cell_key = (str(row.scenario), int(row.source_seed), int(row.stream_seed))
        if str(row.source_model_sha256) != cell_hash_map.get(cell_key):
            raise EvidenceError("coupling effect hash does not match its validated cell")
    path = root / "clustered_inference.csv"
    frame = _read_csv(
        path,
        required=[
            "endpoint",
            "effect_mean",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "cluster_signflip_p_raw",
            "cluster_signflip_p_holm",
            "paired_flow_seed_units",
            "confirmatory",
        ],
    )
    if len(frame) != len(COUPLING_ENDPOINTS):
        raise EvidenceError("coupling inference endpoint count is incorrect")
    if set(frame["endpoint"].astype(str)) != set(COUPLING_ENDPOINTS):
        raise EvidenceError("coupling endpoint grid drifted")
    if not frame["paired_flow_seed_units"].astype(int).eq(
        EXPECTED_COUNTS["coupling_flow_seed_units"]
    ).all() or frame["confirmatory"].map(_bool).any():
        raise EvidenceError("coupling inference lacks exact descriptive clustered units")
    _finite(frame, ["effect_mean", "cluster_ci95_low", "cluster_ci95_high", "cluster_signflip_p_raw", "cluster_signflip_p_holm"])
    _validate_holm(
        frame,
        raw="cluster_signflip_p_raw",
        holm="cluster_signflip_p_holm",
        source=path,
    )
    rows = []
    for item in frame.itertuples(index=False):
        rows.append(
            _ledger_row(
                component="F_coupling_factorial",
                role="primary_f1" if str(item.endpoint) != "ssaw_x_dual_gate_interaction" else "supporting_synergy",
                dataset="HHAR",
                partition=TARGET_SELECTED_PARTITION,
                endpoint=str(item.endpoint),
                metric="macro_f1",
                estimate=float(item.effect_mean),
                ci_low=float(item.cluster_ci95_low),
                ci_high=float(item.cluster_ci95_high),
                p_raw=float(item.cluster_signflip_p_raw),
                p_holm=float(item.cluster_signflip_p_holm),
                source_path=path,
                confirmatory=False,
            )
        )
    return rows, {"manifest": manifest, "cells": int(len(cells)), "effects": int(len(effects)), "inference_rows": int(len(frame))}


def _decision(ledger: pd.DataFrame, component_errors: Mapping[str, str]) -> dict[str, Any]:
    """Summarize descriptive evidence without making a confirmatory decision.

    The current protocol intentionally has no independent confirmatory set.
    Therefore even a complete, positive F1 panel cannot receive a ``keep``
    recommendation.  Missing components are distinguished from a complete
    descriptive-only ledger so downstream validators cannot mistake either
    state for a confirmatory result.
    """

    required = ledger[ledger["role"].eq("primary_f1")].copy()
    # Probability, safety, and operator metrics remain supporting evidence;
    # they cannot replace the primary F1 endpoints.
    required_names = {
        "physical_mean_f1",
        "physical_auc_f1",
        "heldout_f1",
        "future_macro_f1_h1_clean",
        "future_macro_f1_h3_clean",
        "future_macro_f1_h5_clean",
        "future_macro_f1_h1_physical_all",
        "future_macro_f1_h3_physical_all",
        "future_macro_f1_h5_physical_all",
    }
    observed = required[required["endpoint"].isin(required_names)]
    missing = sorted(required_names - set(observed["endpoint"]))
    duplicate = sorted(
        observed.loc[observed["endpoint"].duplicated(keep=False), "endpoint"].unique()
    )
    if component_errors or missing:
        return {
            "recommendation": "inconclusive_due_to_no_independent_confirmatory_set",
            "reason": (
                "formal evidence is incomplete and the protocol has no "
                "independent confirmatory set"
            ),
            "missing_descriptive_primary_f1_endpoints": missing,
            "duplicate_descriptive_primary_f1_endpoints": duplicate,
            "component_errors": dict(component_errors),
            "confirmatory_evidence_present": False,
        }
    return {
        "recommendation": "descriptive_only",
        "reason": (
            "all required descriptive F1 endpoints are present, but every "
            "formal row is target-selected and no independent confirmatory "
            "set exists"
        ),
        "required_descriptive_f1_endpoints": sorted(required_names),
        "observed_descriptive_f1_endpoint_rows": int(len(observed)),
        "confirmatory_evidence_present": False,
        "independent_confirmatory_set_required_for_keep_change_drop": True,
        "supporting_metrics_cannot_replace_f1": True,
    }


def synthesize(
    *,
    physical_dir: Path,
    heldout_dir: Path,
    horizon_dir: Path,
    baseline_dir: Path,
    coupling_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    components = {
        "A_physical_f1": lambda: _physical_component(physical_dir),
        "C_heldout_mechanism": lambda: _heldout_component(heldout_dir),
        "D_future_horizon": lambda: _horizon_component(horizon_dir),
        "E_baseline_reference": lambda: _baseline_component(baseline_dir),
        "F_coupling_factorial": lambda: _coupling_component(coupling_dir),
    }
    all_rows: list[dict[str, Any]] = []
    component_status: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, loader in components.items():
        try:
            rows, metadata = loader()
            all_rows.extend(rows)
            component_status[name] = {"status": "complete", **metadata}
            if name == "A_physical_f1":
                component_status["B_probability_risk"] = {
                    "status": "complete",
                    "rows": int(metadata.get("probability_summary_rows", 0)),
                    "safety_rows": int(metadata.get("safety_rows", 0)),
                    "supporting_only": True,
                }
        except (EvidenceError, OSError, KeyError, ValueError, TypeError) as exc:
            errors[name] = str(exc)
            component_status[name] = {"status": "inconclusive", "error": str(exc)}
            if name == "A_physical_f1":
                errors["B_probability_risk"] = str(exc)
                component_status["B_probability_risk"] = {
                    "status": "inconclusive",
                    "error": str(exc),
                    "supporting_only": True,
                }
    ledger = pd.DataFrame(all_rows)
    if ledger.empty:
        ledger = pd.DataFrame(
            columns=[
                "component", "role", "dataset", "evaluation_partition", "endpoint",
                "metric", "direction", "estimate", "ci_low", "ci_high", "p_raw",
                "p_holm", "confirmatory", "status", "source_path", "f1_is_primary",
            ]
        )
    else:
        ledger = ledger.sort_values(
            ["component", "dataset", "evaluation_partition", "endpoint"],
            kind="stable",
        ).reset_index(drop=True)
    _atomic_csv(ledger, Path(output_dir) / "evidence_ledger.csv")
    decision = _decision(ledger, errors)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete" if not errors else "inconclusive",
        "component_status": component_status,
        "component_errors": errors,
        "ledger_rows": int(len(ledger)),
        "formal_flow_policy": (
            "five flows per dataset; HHAR uses the same five flows as tuning"
        ),
        "formal_datasets": list(FORMAL_DATASETS),
        "formal_flow_counts": {
            dataset: len(FORMAL_FLOW_PAIRS[dataset]) for dataset in FORMAL_DATASETS
        },
        "evaluation_partition": TARGET_SELECTED_PARTITION,
        "confirmatory_partition": None,
        "confirmatory_results": "none",
        "confirmatory_rows": 0,
        "target_selected_partitions_are_descriptive": True,
        "parameter_selection_data_overlap": True,
        "supporting_metrics_cannot_replace_f1": True,
        "decision": decision,
        "files": {"evidence_ledger": "evidence_ledger.csv"},
    }
    _atomic_json(manifest, Path(output_dir) / "manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-dir", default="results/ssaw_evidence_v1/physical_panel/final")
    parser.add_argument("--heldout-dir", default="results/ssaw_heldout_mechanism_v1/analysis")
    parser.add_argument("--horizon-dir", default="results/full_no_ssaw_horizon_queue/analysis")
    parser.add_argument("--baseline-dir", default="results/ssaw_evidence_v1/baseline_physical_reference/final_panel")
    parser.add_argument(
        "--coupling-dir",
        default=(
            "results/optuna/hhar_ssaw_f1_delta_v1/"
            "coupling_factorial_single_flow/analysis"
        ),
    )
    parser.add_argument("--output-dir", default="results/ssaw_evidence_v1/evidence_ledger")
    args = parser.parse_args(argv)
    manifest = synthesize(
        physical_dir=Path(args.physical_dir),
        heldout_dir=Path(args.heldout_dir),
        horizon_dir=Path(args.horizon_dir),
        baseline_dir=Path(args.baseline_dir),
        coupling_dir=Path(args.coupling_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
