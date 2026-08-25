"""Fail-closed CPU finalization for the paper_evidence_v2 evidence bundle.

The finalizer consumes completed artifacts only.  It never starts a trainer,
touches CUDA, re-tunes a profile, or replaces a missing cell with an
aggregate.  A protocol/count/hash/non-finite-value error writes a failed
manifest and suppresses all compact evidence tables.

The causal horizon is read from each manifest's declared role-A horizon.  The
finalizer requires that declaration, the plan, and the raw rows agree; it does
not silently replace the current bundle's horizon-one estimand with a
hard-coded horizon-five claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "paper_evidence_v2"
DEFAULT_OUTPUT = DEFAULT_ROOT / "finalizer_v2"
CAUSAL_PROTOCOL = "paper_representative_causal_ablation_v2_joint_reference"
HELDOUT_PAIR_PROTOCOL = "ssaw_full_no_ssaw_paired_summary_v1"
HELDOUT_QUEUE_PROTOCOL_PREFIX = "ssaw_full_no_ssaw_heldout_queue_v3_spline_direction_bank"
SAFETY_SIGNATURE_PREFIX = "controlled_safety_known_mask_v5_canonical_source_hash:"
# Paper-evidence v2 is a frozen historical artifact and must keep validating
# the v3 efficiency table it was built from. New efficiency reruns use the v4
# runner/output and are finalized by the current evidence pipeline.
EFFICIENCY_PROTOCOL = "compute_overhead_formal_v3"
CAUSAL_VARIANTS = (
    "accept_all_raw",
    "confidence_only",
    "matched_raw_duplicate",
    "random_eligible_spline",
    "hard_ssaw",
)
PANEL_B_VARIANTS = (
    "confidence_only",
    "matched_raw_duplicate",
    "random_eligible_spline",
    "hard_ssaw",
)
SAFETY_METHODS = ("DuSafe", "EATA", "SAR", "ACCUPOfficial")
SAFETY_CORRUPTIONS = ("signal_freeze", "blackout")
SAFETY_SEVERITIES = ("s3", "s6")
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
CORRUPTION_SEED = 1
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class FinalizationError(ValueError):
    """Raised when evidence cannot be safely summarized."""


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"JSON artifact is not an object: {path}")
    return value


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FinalizationError(f"missing CSV artifact: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas exposes several parser exception types
        raise FinalizationError(f"invalid CSV artifact: {path}: {exc}") from exc
    if frame.empty:
        raise FinalizationError(f"empty CSV artifact: {path}")
    return frame


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FinalizationError(f"{label} missing columns: {missing}")


def _finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    columns = tuple(columns)
    _require_columns(frame, columns, label)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            bad = int(values.isna().sum() + (~values.map(math.isfinite)).sum())
            raise FinalizationError(
                f"{label} has non-finite required metric {column}: {bad} rows"
            )


def _unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if frame.duplicated(list(columns), keep=False).any():
        raise FinalizationError(f"{label} contains duplicate keys {list(columns)}")


def _sha(value: Any, label: str) -> str:
    text = str(value).strip()
    if not SHA256_RE.fullmatch(text):
        raise FinalizationError(f"{label} is not a SHA-256 digest: {text!r}")
    return text.lower()


def _scenario(value: Any) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[^\s>-]+->[^\s>-]+", text):
        raise FinalizationError(f"invalid transfer scenario: {value!r}")
    return text


def _profile(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        profile = dict(value)
    else:
        text = str(value)
        try:
            profile = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                profile = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise FinalizationError(f"{label} is not a profile mapping") from exc
    if not isinstance(profile, dict):
        raise FinalizationError(f"{label} is not a profile mapping")
    return profile


def _normalize_manifest_scenarios(value: Any) -> list[str]:
    """Normalize the known safety JSON encoding without accepting ambiguity."""

    if isinstance(value, Mapping):
        values = value.get("HAR")
    else:
        values = value
    if isinstance(values, str):
        return [_scenario(values)]
    if isinstance(values, (list, tuple)):
        if len(values) == 1 and "->" in str(values[0]):
            return [_scenario(values[0])]
        if len(values) == 2 and "->" not in str(values[0]) and "->" not in str(values[1]):
            # The safety runner serializes its tuple scenario as [source,target].
            return [_scenario(f"{values[0]}->{values[1]}")]
        return [_scenario(item) for item in values]
    raise FinalizationError(f"cannot normalize manifest scenarios: {value!r}")


def _check_v2_profile_path(value: Any, label: str) -> None:
    text = str(value).replace("/", "\\").lower()
    if "paper_flow_profiles_v2.json" not in text:
        raise FinalizationError(f"{label} is not paper_flow_profiles_v2.json: {value!r}")


def _parse_runtime_hparams(frame: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    _require_columns(frame, ("runtime_hparams",), label)
    profiles = []
    for index, value in frame["runtime_hparams"].items():
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise FinalizationError(f"{label} runtime_hparams row {index} is invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise FinalizationError(f"{label} runtime_hparams row {index} is not an object")
        profiles.append(parsed)
    return profiles


def validate_safety(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / "controlled_safety_har_12_16"
    manifest = _json(directory / "manifest.json")
    raw = _csv(directory / "summary_raw.csv")
    aggregate = _csv(directory / "summary_aggregate.csv")
    if manifest.get("algorithm_registry") != "benchmark":
        raise FinalizationError("safety registry must be benchmark")
    if manifest.get("effective_method_registry", {}).get("DuSafe") != "production":
        raise FinalizationError("safety DuSafe is not routed to production")
    if manifest.get("methods_completed") != list(sorted(SAFETY_METHODS)):
        raise FinalizationError("safety methods_completed does not cover all four methods")
    if int(manifest.get("requested_job_count", -1)) != 48:
        raise FinalizationError("safety requested_job_count is not 48")
    if int(manifest.get("requested_completed_job_count", -1)) != 48:
        raise FinalizationError("safety completed-job count is not 48")
    if int(manifest.get("requested_missing_job_count", -1)) != 0 or int(manifest.get("failure_count", -1)) != 0:
        raise FinalizationError("safety manifest contains missing or failed jobs")
    _check_v2_profile_path(manifest.get("paper_flow_profile_json"), "safety profile")
    if not bool(manifest.get("physical_protocol")):
        raise FinalizationError("safety physical protocol is not enabled")
    _require_columns(
        raw,
        (
            "dataset", "scenario", "method", "variant", "corruption", "severity",
            "source_seed", "stream_seed", "corruption_seed", "protocol_signature",
            "source_model_sha256", "probability_record_schema", "runtime_hparams",
            "clean_f1", "corrupted_f1", "coverage", "admitted_accuracy",
            "incorrect_admission_rate", "unsafe_admission_rate",
        ),
        "safety summary_raw",
    )
    expected_keys = {
        ("HAR", "12->16", method, "full", corruption, severity, seed, STREAM_SEED, CORRUPTION_SEED)
        for method in SAFETY_METHODS
        for corruption in SAFETY_CORRUPTIONS
        for severity in SAFETY_SEVERITIES
        for seed in SOURCE_SEEDS
    }
    observed_keys = {
        (
            str(row.dataset), _scenario(row.scenario), str(row.method), str(row.variant),
            str(row.corruption), str(row.severity), int(row.source_seed),
            int(row.stream_seed), int(row.corruption_seed),
        )
        for row in raw.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise FinalizationError(
            f"safety key mismatch: missing={len(expected_keys-observed_keys)}, "
            f"unexpected={len(observed_keys-expected_keys)}"
        )
    _unique(
        raw,
        ["dataset", "scenario", "method", "variant", "corruption", "severity", "source_seed", "stream_seed", "corruption_seed"],
        "safety summary_raw",
    )
    _finite(
        raw,
        ("clean_f1", "corrupted_f1", "coverage", "admitted_accuracy", "incorrect_admission_rate", "unsafe_admission_rate"),
        "safety summary_raw",
    )
    for index, profile in enumerate(_parse_runtime_hparams(raw, "safety summary_raw")):
        if not float(profile.get("ssaw_auxiliary_weight", 1.0)) > 0:
            raise FinalizationError(f"safety row {index} has non-positive SSAW weight")
        method = str(raw.iloc[index]["method"])
        if method == "DuSafe":
            if profile.get("dusafe_variant") != "spline_residual":
                raise FinalizationError("safety DuSafe runtime is not spline_residual")
            if profile.get("enable_ssaw") is not True:
                raise FinalizationError("safety DuSafe SSAW is not enabled")
            if profile.get("enable_source_semantic_router") is not False:
                raise FinalizationError("safety DuSafe semantic router is not disabled")
        if method == "EATA":
            if profile.get("fisher_enabled") is not True:
                raise FinalizationError("safety EATA Fisher is not enabled")
    signatures = raw.groupby(["method", "variant", "corruption", "severity"], dropna=False)["protocol_signature"].nunique()
    if not signatures.eq(1).all() or not raw["protocol_signature"].astype(str).str.startswith(SAFETY_SIGNATURE_PREFIX).all():
        raise FinalizationError("safety protocol signatures are mixed or invalid")
    for row in raw.itertuples(index=False):
        source_hash = _sha(row.source_model_sha256, "safety source_model_sha256")
        if str(row.source_checkpoint_sha256).lower() != source_hash:
            raise FinalizationError("safety source checkpoint/model hash mismatch")
        if str(row.method) == "EATA":
            if str(row.eata_fisher_status) != "validated_source_fisher":
                raise FinalizationError("safety EATA Fisher status is not validated_source_fisher")
            if str(row.eata_fisher_source_checkpoint_sha256).lower() != source_hash:
                raise FinalizationError("safety EATA Fisher is tied to a different checkpoint")
    expected_aggregate_keys = {
        ("HAR", "12->16", method, "full", corruption, severity, CORRUPTION_SEED)
        for method in SAFETY_METHODS
        for corruption in SAFETY_CORRUPTIONS
        for severity in SAFETY_SEVERITIES
    }
    _require_columns(aggregate, ("dataset", "scenario", "method", "variant", "corruption", "severity", "corruption_seed"), "safety summary_aggregate")
    aggregate_keys = {
        (str(row.dataset), _scenario(row.scenario), str(row.method), str(row.variant), str(row.corruption), str(row.severity), int(row.corruption_seed))
        for row in aggregate.itertuples(index=False)
    }
    if aggregate_keys != expected_aggregate_keys:
        raise FinalizationError("safety summary_aggregate key set mismatch")
    _unique(aggregate, ["dataset", "scenario", "method", "variant", "corruption", "severity", "corruption_seed"], "safety summary_aggregate")
    _finite(aggregate, ("clean_f1_mean", "corrupted_f1_mean", "coverage_mean", "admitted_accuracy_mean", "incorrect_admission_rate_mean", "unsafe_update_rate_mean"), "safety summary_aggregate")
    raw_group = raw.groupby(["dataset", "scenario", "method", "variant", "corruption", "severity", "corruption_seed"], as_index=False)[["clean_f1", "corrupted_f1", "coverage", "admitted_accuracy", "incorrect_admission_rate", "unsafe_admission_rate"]].mean()
    merged = aggregate.merge(raw_group, on=["dataset", "scenario", "method", "variant", "corruption", "severity", "corruption_seed"], how="left", validate="one_to_one")
    for agg_col, raw_col in {
        "clean_f1_mean": "clean_f1", "corrupted_f1_mean": "corrupted_f1", "coverage_mean": "coverage", "admitted_accuracy_mean": "admitted_accuracy", "incorrect_admission_rate_mean": "incorrect_admission_rate", "unsafe_update_rate_mean": "unsafe_admission_rate",
    }.items():
        if not (merged[agg_col].astype(float).sub(merged[raw_col].astype(float)).abs() <= 1e-10).all():
            raise FinalizationError(f"safety aggregate disagrees with raw rows: {agg_col}")
    compact = aggregate[aggregate["severity"].astype(str).eq("s6")].copy()
    compact = compact[["dataset", "scenario", "method", "variant", "corruption", "severity", "corruption_seed", "normalized_severity", "clean_f1_mean", "corrupted_f1_mean", "coverage_mean", "admitted_accuracy_mean", "incorrect_admission_rate_mean", "unsafe_update_rate_mean"]].sort_values(["method", "corruption"], kind="stable").reset_index(drop=True)
    return compact, {"raw_rows": len(raw), "aggregate_rows": len(aggregate), "s6_rows": len(compact), "expected_rows": 48}


def _causal_base_keys(frame: pd.DataFrame) -> list[str]:
    return ["dataset", "scenario", "source_seed", "stream_seed", "condition", "batch_index", "horizon"]


def validate_causal_dir(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _json(path / "manifest.json")
    plan = _json(path / "plan.json")
    raw = _csv(path / "raw.csv")
    if manifest.get("protocol") != CAUSAL_PROTOCOL or manifest.get("status") != "complete":
        raise FinalizationError(f"{label} manifest protocol/status is invalid")
    if plan.get("protocol") != CAUSAL_PROTOCOL:
        raise FinalizationError(f"{label} plan protocol is invalid")
    _check_v2_profile_path(plan.get("flow_profile_json"), f"{label} plan profile")
    roles = manifest.get("evidence_roles")
    role_a = roles.get("A") if isinstance(roles, Mapping) else None
    if not isinstance(role_a, Mapping):
        raise FinalizationError(f"{label} lacks evidence role A")
    try:
        declared_horizon = int(role_a["future_horizon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} role A has no valid future horizon") from exc
    if declared_horizon < 1:
        raise FinalizationError(f"{label} role A future horizon must be positive")
    declared_horizons = role_a.get("future_horizons", [declared_horizon])
    try:
        declared_horizons = {int(value) for value in declared_horizons}
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} role A future_horizons is invalid") from exc
    if declared_horizons != {declared_horizon}:
        raise FinalizationError(f"{label} role A horizon declaration is inconsistent")
    if roles.get("A", {}).get("variants") != ["accept_all_raw", "confidence_only"]:
        raise FinalizationError(f"{label} role A variant grid is invalid")
    if roles.get("B", {}).get("variants") != list(PANEL_B_VARIANTS):
        raise FinalizationError(f"{label} role B variant grid is invalid")
    plan_cells = plan.get("cells")
    if not isinstance(plan_cells, list) or not plan_cells:
        raise FinalizationError(f"{label} plan has no cells")
    plan_keys = set()
    plan_horizons = set()
    for cell in plan_cells:
        plan_keys.add((str(cell["dataset"]), _scenario(cell["scenario"]), int(cell["source_seed"]), int(cell["stream_seed"]), str(cell["condition"])))
        plan_horizons.update(int(v) for v in cell.get("horizons", []))
    actual_horizons = set(pd.to_numeric(raw["horizon"], errors="raise").astype(int).tolist()) if "horizon" in raw else set()
    if plan_horizons != {declared_horizon} or actual_horizons != {declared_horizon}:
        raise FinalizationError(
            f"{label} future horizon mismatch: role={declared_horizon}, plan={sorted(plan_horizons)}, raw={sorted(actual_horizons)}"
        )
    required = _causal_base_keys(raw) + [
        "variant", "source_model_sha256", "target_labels_used_for_online_updates", "target_labels_used_for_parameter_selection", "target_labels_used_for_metrics", "future_macro_f1", "coverage", "eligible_coverage", "admitted_accuracy", "incorrect_admission_rate", "update_norm", "pre_batch_state_hash", "pre_batch_model_buffer_hash", "pre_batch_optimizer_hash", "shared_reference_variant", "joint_causal_start_state", "post_update_state_hash", "future_eval_untouched", "future_eval_rng_untouched", "admission_mask_sha256", "heldout_eligible_coverage", "heldout_margin_ratio", "heldout_flip_rate", "heldout_worst_margin", "heldout_consistency", "heldout_candidate_count", "heldout_sobol_seed", "candidate_pool_sha256", "heldout_candidate_pool_sha256",
    ]
    _require_columns(raw, required, f"{label} raw")
    _unique(raw, _causal_base_keys(raw) + ["variant"], f"{label} raw")
    if set(raw["variant"].astype(str).unique()) != set(CAUSAL_VARIANTS):
        raise FinalizationError(f"{label} raw variant grid is incomplete")
    observed_plan_keys = set(zip(raw.dataset.astype(str), raw.scenario.map(_scenario), raw.source_seed.astype(int), raw.stream_seed.astype(int), raw.condition.astype(str)))
    if observed_plan_keys != plan_keys:
        raise FinalizationError(f"{label} raw/plan cell keys disagree")
    group_columns = _causal_base_keys(raw)
    for key, group in raw.groupby(group_columns, dropna=False):
        if set(group["variant"].astype(str)) != set(CAUSAL_VARIANTS):
            raise FinalizationError(f"{label} incomplete five-variant batch {key}")
        if not bool(group["joint_causal_start_state"].astype(bool).all()):
            raise FinalizationError(f"{label} has a non-joint causal start state at {key}")
        if set(group["shared_reference_variant"].astype(str)) != {"confidence_only"}:
            raise FinalizationError(f"{label} shared reference is not confidence_only at {key}")
        panel_b = group[group["variant"].isin(PANEL_B_VARIANTS)]
        for column in ("admission_mask_sha256", "candidate_pool_sha256", "heldout_candidate_pool_sha256", "heldout_sobol_seed", "heldout_candidate_count", "pre_batch_state_hash", "pre_batch_model_buffer_hash", "pre_batch_optimizer_hash"):
            values = panel_b[column].dropna().astype(str).unique()
            if len(values) != 1:
                raise FinalizationError(f"{label} Panel B {column} is not shared at {key}")
        if int(float(panel_b["heldout_candidate_count"].iloc[0])) != 24:
            raise FinalizationError(f"{label} held-out candidate budget is not 24 at {key}")
        heldout_seed = int(float(panel_b["heldout_sobol_seed"].iloc[0]))
        source_seed = int(key[2])
        if heldout_seed in {source_seed, 1729, 42}:
            raise FinalizationError(f"{label} held-out Sobol seed overlaps a reserved seed at {key}")
    _finite(raw, ("future_macro_f1", "coverage", "eligible_coverage", "admitted_accuracy", "incorrect_admission_rate", "update_norm", "heldout_eligible_coverage", "heldout_margin_ratio", "heldout_flip_rate", "heldout_worst_margin", "heldout_consistency"), f"{label} raw")
    for column, expected in (("target_labels_used_for_online_updates", False), ("target_labels_used_for_parameter_selection", True), ("target_labels_used_for_metrics", True), ("future_eval_untouched", True), ("future_eval_rng_untouched", True)):
        if not raw[column].map(lambda value: bool(value) is expected).all():
            raise FinalizationError(f"{label} provenance flag {column} is invalid")
    if not bool(raw["source_model_sha256"].map(lambda value: bool(SHA256_RE.fullmatch(str(value).strip()))).all()):
        raise FinalizationError(f"{label} source checkpoint hash is invalid")
    # A and B are retained as batch-level means; inactive rows remain present
    # and are not dropped from the overall panel.
    metrics = ["future_macro_f1", "coverage", "eligible_coverage", "admitted_accuracy", "incorrect_admission_rate", "heldout_eligible_coverage", "heldout_margin_ratio", "heldout_flip_rate", "heldout_worst_margin", "heldout_consistency", "update_norm"]
    overall = raw.groupby(["dataset", "scenario"], as_index=False)[metrics].mean()
    return raw, {"raw_rows": len(raw), "plan_cells": len(plan_cells), "batch_groups": int(raw.groupby(group_columns).ngroups), "horizons": sorted(actual_horizons), "declared_horizon": declared_horizon, "flows": sorted({f"{d}:{s}" for d, s in zip(raw.dataset, raw.scenario)})}


def _wide_causal_panel(raw: pd.DataFrame, *, active_only: bool = False) -> pd.DataFrame:
    base = _causal_base_keys(raw)
    work = raw.copy()
    if active_only:
        hard = work[work["variant"].eq("hard_ssaw")][base + ["eligible_coverage"]].rename(columns={"eligible_coverage": "_hard_eligible"})
        work = work.merge(hard, on=base, how="left", validate="many_to_one")
        work = work[work["_hard_eligible"] >= 0.25].drop(columns=["_hard_eligible"])
    metrics = ["future_macro_f1", "coverage", "admitted_accuracy", "incorrect_admission_rate", "heldout_eligible_coverage", "heldout_margin_ratio", "heldout_flip_rate", "heldout_worst_margin", "heldout_consistency", "update_norm"]
    # Rebuild the wide columns from the filtered raw rows, retaining a compact
    # per-flow row and the number of contributing batches.
    rows = []
    for (dataset, scenario, horizon), group in work.groupby(["dataset", "scenario", "horizon"], sort=True):
        row: dict[str, Any] = {"dataset": dataset, "scenario": scenario, "horizon": int(horizon), "segment": "active" if active_only else "overall", "batch_count": int(group.groupby(base).ngroups), "source_seed_count": int(group["source_seed"].nunique())}
        for variant in PANEL_B_VARIANTS:
            subset = group[group["variant"].eq(variant)]
            for metric in metrics:
                row[f"{metric}__{variant}"] = float(subset[metric].mean())
        for variant in PANEL_B_VARIANTS[1:]:
            for metric in metrics:
                row[f"delta_{metric}_{variant}_vs_confidence_only"] = row[f"{metric}__{variant}"] - row[f"{metric}__confidence_only"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "scenario", "horizon", "segment"], kind="stable").reset_index(drop=True)


def build_confidence_panel(raw_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(raw_frames, ignore_index=True)
    subset = raw[raw["variant"].isin(("accept_all_raw", "confidence_only"))].copy()
    metrics = ["future_macro_f1", "coverage", "admitted_accuracy", "incorrect_admission_rate"]
    rows = []
    for (dataset, scenario, horizon), group in subset.groupby(["dataset", "scenario", "horizon"], sort=True):
        row: dict[str, Any] = {"dataset": dataset, "scenario": scenario, "condition": "clean", "horizon": int(horizon), "batch_count": int(group.groupby(_causal_base_keys(group)).ngroups), "source_seed_count": int(group["source_seed"].nunique())}
        for variant in ("accept_all_raw", "confidence_only"):
            part = group[group["variant"].eq(variant)]
            for metric in metrics:
                row[f"{metric}__{variant}"] = float(part[metric].mean())
        for metric in metrics:
            row[f"delta_{metric}_confidence_only_vs_accept_all_raw"] = row[f"{metric}__confidence_only"] - row[f"{metric}__accept_all_raw"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "scenario", "horizon"], kind="stable").reset_index(drop=True)


def validate_heldout_dir(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _json(path / "manifest.json")
    paired_payload = _json(path / "paired_summary.json")
    if not str(manifest.get("protocol_version", "")).startswith(HELDOUT_QUEUE_PROTOCOL_PREFIX) or manifest.get("status") != "complete":
        raise FinalizationError(f"{label} manifest protocol/status is invalid")
    if paired_payload.get("protocol_version") != HELDOUT_PAIR_PROTOCOL:
        raise FinalizationError(f"{label} paired_summary protocol is invalid")
    if paired_payload.get("ground_truth_lpr_observed") is not False or paired_payload.get("independent_reannotation_available") is not False:
        raise FinalizationError(f"{label} paired summary makes a forbidden LPR claim")
    rows = paired_payload.get("paired_rows")
    if not isinstance(rows, list) or not rows:
        raise FinalizationError(f"{label} paired summary is empty")
    frame = pd.DataFrame(rows)
    _require_columns(frame, ("dataset", "scenario", "source_seed", "training_view_seed", "test_seed", "heldout_test_seed", "variants_paired", "source_checkpoint_sha256", "current_training_view_family", "heldout_direction_family", "heldout_direction_seed", "heldout_sobol_seed_overlap", "heldout_candidate_count", "heldout_direction_candidate_count", "heldout_direction_bank_sha256", "full_clean_f1", "no_ssaw_clean_f1", "full_heldout_f1", "no_ssaw_heldout_f1", "full_eligible_coverage", "no_ssaw_eligible_coverage", "full_margin_ratio", "no_ssaw_margin_ratio", "full_heldout_flip_rate", "no_ssaw_heldout_flip_rate", "full_heldout_worst_margin", "no_ssaw_heldout_worst_margin", "full_heldout_consistency", "no_ssaw_heldout_consistency"), f"{label} paired_summary")
    _unique(frame, ["dataset", "scenario", "source_seed", "training_view_seed", "test_seed"], f"{label} paired_summary")
    if set(frame["variants_paired"].astype(str)) != {"Full,no_ssaw"}:
        raise FinalizationError(f"{label} paired variants are incomplete")
    _finite(frame, ("full_clean_f1", "no_ssaw_clean_f1", "full_heldout_f1", "no_ssaw_heldout_f1", "full_eligible_coverage", "no_ssaw_eligible_coverage", "full_margin_ratio", "no_ssaw_margin_ratio", "full_heldout_flip_rate", "no_ssaw_heldout_flip_rate", "full_heldout_worst_margin", "no_ssaw_heldout_worst_margin", "full_heldout_consistency", "no_ssaw_heldout_consistency"), f"{label} paired_summary")
    if set(frame["current_training_view_family"].astype(str)) != {"spline_residual_sobol_direction"} or set(frame["heldout_direction_family"].astype(str)) != {"unseen_spline_residual_sobol_direction"}:
        raise FinalizationError(f"{label} mechanism direction family is not production spline")
    if not (frame["heldout_sobol_seed_overlap"].astype(bool) == False).all():
        raise FinalizationError(f"{label} held-out Sobol seed overlap was reported")
    if not (pd.to_numeric(frame["heldout_candidate_count"], errors="coerce") == 24).all() or not (pd.to_numeric(frame["heldout_direction_candidate_count"], errors="coerce") == 24).all():
        raise FinalizationError(f"{label} held-out candidate budget is not 24")
    for row in frame.itertuples(index=False):
        _sha(row.source_checkpoint_sha256, f"{label} source checkpoint")
        _sha(row.heldout_direction_bank_sha256, f"{label} direction bank")
        if int(row.heldout_direction_seed) in {int(row.source_seed), int(row.training_view_seed), int(row.test_seed)}:
            raise FinalizationError(f"{label} held-out direction seed overlaps cell roles")
    expected_cells = int(manifest.get("expected_cells", -1))
    if expected_cells != len(rows) * 2 or int(manifest.get("completed_cells", -1)) != expected_cells:
        raise FinalizationError(f"{label} manifest/paired cell counts disagree")
    cell_rows = []
    for cell_path in sorted((path / "cells").glob("*.json")):
        payload = _json(cell_path)
        if payload.get("completed") is True and isinstance(payload.get("row"), Mapping):
            cell_rows.append(dict(payload["row"]))
    if len(cell_rows) != expected_cells:
        raise FinalizationError(f"{label} cell artifacts do not match manifest count")
    cell_frame = pd.DataFrame(cell_rows)
    _require_columns(cell_frame, ("dataset", "scenario", "source_seed", "variant", "source_checkpoint_sha256", "heldout_direction_bank_sha256"), f"{label} cell rows")
    for _, group in cell_frame.groupby(["dataset", "scenario", "source_seed"], dropna=False):
        if set(group["variant"].astype(str)) != {"Full", "no_ssaw"}:
            raise FinalizationError(f"{label} cell rows lack Full/no_ssaw pairing")
        if group["source_checkpoint_sha256"].astype(str).nunique() != 1 or group["heldout_direction_bank_sha256"].astype(str).nunique() != 1:
            raise FinalizationError(f"{label} Full/no_ssaw hashes disagree")
    return frame, {"paired_rows": len(frame), "cell_rows": len(cell_rows), "datasets": sorted(frame.dataset.unique().tolist()), "flows": sorted(set(zip(frame.dataset, frame.scenario)))}


HELDOUT_METRICS = (
    "clean_f1", "heldout_f1", "eligible_coverage", "margin_ratio",
    "heldout_flip_rate", "heldout_worst_margin", "heldout_consistency",
)


def build_a3_flow_hparams(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize v2 flow overrides together with production DuSafe defaults.

    The checked-in A3 table predates the unified spline constants and therefore
    leaves those fields blank.  This table is generated from the authoritative
    v2 profile JSON plus ``configs.tta_hparams_new``; it never infers a value
    from a result row.
    """

    profile_path = ROOT / "configs" / "paper_flow_profiles_v2.json"
    payload = _json(profile_path)
    if payload.get("protocol") != "paper_flow_profiles_v2_target_selected_descriptive":
        raise FinalizationError("A3 profile JSON protocol is not v2 target-selected descriptive")
    if payload.get("source_training_overridden") is not False:
        raise FinalizationError("A3 profile JSON overrides source training")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise FinalizationError("A3 profile JSON has no profiles")
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from configs.tta_hparams_new import get_hparams_class
    except Exception as exc:  # pragma: no cover - import failure is actionable
        raise FinalizationError(f"cannot import production DuSafe defaults: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for profile_key, override in sorted(profiles.items()):
        if not isinstance(override, Mapping):
            raise FinalizationError(f"A3 override is not a mapping: {profile_key}")
        try:
            dataset, scenario = str(profile_key).split(":", 1)
            scenario = _scenario(scenario)
            defaults_obj = get_hparams_class(dataset)()
            defaults = {
                **dict(defaults_obj.alg_hparams["DuSafe"]),
                **dict(defaults_obj.train_params),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalizationError(f"A3 profile key/defaults are invalid: {profile_key}") from exc
        required_override = ("batch_size", "learning_rate", "steps", "ssaw_auxiliary_weight")
        if any(key not in override for key in required_override):
            raise FinalizationError(f"A3 profile override is incomplete: {profile_key}")
        merged = {**defaults, **dict(override)}
        radii = merged.get("spline_radius_levels")
        if not isinstance(radii, (list, tuple)) or not radii:
            raise FinalizationError(f"A3 radii are missing: {profile_key}")
        try:
            numeric = {
                "batch_size": float(merged["batch_size"]),
                "learning_rate": float(merged["learning_rate"]),
                "steps": float(merged["steps"]),
                "q": float(merged["confidence_keep_fraction"]),
                "lambda": float(merged["ssaw_auxiliary_weight"]),
                "spline_log_strength": float(merged["spline_log_strength"]),
                "num_dirs": float(merged["spline_num_directions"]),
                "control_points": float(merged["spline_control_points"]),
                "grad_clip": float(merged["grad_clip"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalizationError(f"A3 defaults/override are non-numeric: {profile_key}") from exc
        if any(not math.isfinite(value) for value in numeric.values()):
            raise FinalizationError(f"A3 defaults/override are non-finite: {profile_key}")
        if numeric["batch_size"] <= 0 or numeric["learning_rate"] <= 0 or numeric["steps"] <= 0:
            raise FinalizationError(f"A3 runtime override is non-positive: {profile_key}")
        if not 0.0 < numeric["q"] <= 1.0 or numeric["lambda"] <= 0.0:
            raise FinalizationError(f"A3 q/lambda violates protocol: {profile_key}")
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in radii):
            raise FinalizationError(f"A3 radii are invalid: {profile_key}")
        row = {
            "dataset": dataset,
            "flow": scenario,
            "batch_size": int(numeric["batch_size"]),
            "learning_rate": numeric["learning_rate"],
            "LR": numeric["learning_rate"],
            "steps": int(numeric["steps"]),
            "q": numeric["q"],
            "confidence_keep_fraction": numeric["q"],
            "lambda": numeric["lambda"],
            "ssaw_auxiliary_weight": numeric["lambda"],
            "spline_log_strength": numeric["spline_log_strength"],
            "num_dirs": int(numeric["num_dirs"]),
            "spline_num_directions": int(numeric["num_dirs"]),
            "radii": json.dumps([float(value) for value in radii], separators=(",", ":")),
            "spline_radius_levels": json.dumps([float(value) for value in radii], separators=(",", ":")),
            "control_points": int(numeric["control_points"]),
            "spline_control_points": int(numeric["control_points"]),
            "grad_clip": numeric["grad_clip"],
            "profile_source": str(profile_path.resolve()),
            "algorithm_variant": str(merged.get("dusafe_variant", "")),
        }
        if row["algorithm_variant"] != "spline_residual":
            raise FinalizationError(f"A3 DuSafe variant is not spline_residual: {profile_key}")
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["dataset", "flow"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["dataset", "flow"], keep=False).any():
        raise FinalizationError("A3 flow profile keys are not unique")
    _finite(frame, ("batch_size", "learning_rate", "LR", "steps", "q", "confidence_keep_fraction", "lambda", "ssaw_auxiliary_weight", "spline_log_strength", "num_dirs", "spline_num_directions", "control_points", "spline_control_points", "grad_clip"), "A3 flow hparams")
    return frame, {"rows": len(frame), "profile_json": str(profile_path.resolve()), "protocol": payload["protocol"]}


def build_heldout_panel(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    rows = []
    for scope, group in [(str(dataset), frame[frame["dataset"].eq(dataset)]) for dataset in sorted(frame["dataset"].unique())] + [("pooled", frame)]:
        row: dict[str, Any] = {"scope": scope, "flow_count": int(group[["dataset", "scenario"]].drop_duplicates().shape[0]), "paired_cell_count": int(len(group)), "source_seed_count": int(group["source_seed"].nunique())}
        for variant in ("full", "no_ssaw"):
            for metric in HELDOUT_METRICS:
                column = f"{variant}_{metric}"
                row[f"{column}_mean"] = float(pd.to_numeric(group[column], errors="raise").mean())
        for metric in HELDOUT_METRICS:
            row[f"full_minus_no_ssaw_{metric}_mean"] = row[f"full_{metric}_mean"] - row[f"no_ssaw_{metric}_mean"]
        row["direction_bank_hash_count"] = int(group["heldout_direction_bank_sha256"].nunique())
        row["heldout_seed_overlap_any"] = bool(group["heldout_sobol_seed_overlap"].astype(bool).any())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scope", key=lambda values: values.map({"EEG": 0, "HAR": 1, "HHAR": 2, "pooled": 3})).reset_index(drop=True)


def validate_efficiency(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / "efficiency_har_12to16"
    manifest = _json(directory / "manifest.json")
    finalization = _json(directory / "finalization.json")
    frame = _csv(directory / "method_overhead.csv")
    if manifest.get("protocol") != EFFICIENCY_PROTOCOL or finalization.get("status") != "complete":
        raise FinalizationError("efficiency protocol/finalization is not complete")
    if int(manifest.get("expected_cell_count", -1)) != 2 or int(finalization.get("expected_cells", -1)) != 2 or finalization.get("missing_cells") != [] or finalization.get("errors") != []:
        raise FinalizationError("efficiency cell finalization is incomplete")
    _check_v2_profile_path(manifest.get("paper_flow_profile_json"), "efficiency profile")
    _require_columns(frame, ("status", "dataset", "scenario", "method", "variant", "profile", "source_checkpoint_sha256", "latency_mean_ms", "throughput_samples_per_second", "peak_vram_mb", "runtime_hparams", "effective_method_registry"), "efficiency rows")
    if set(zip(frame.dataset.astype(str), frame.scenario.astype(str), frame.method.astype(str), frame.variant.astype(str), frame.profile.astype(str))) != {("HAR", "12->16", "DuSafe", "full", "default"), ("HAR", "12->16", "DuSafe", "no_ssaw", "default")}:
        raise FinalizationError("efficiency cell key set mismatch")
    _unique(frame, ["dataset", "scenario", "method", "variant", "profile"], "efficiency rows")
    if set(frame.status.astype(str)) != {"ok"} or set(frame.effective_method_registry.astype(str)) != {"production"}:
        raise FinalizationError("efficiency rows are not successful production DuSafe cells")
    _finite(frame, ("latency_mean_ms", "throughput_samples_per_second", "peak_vram_mb"), "efficiency rows")
    if (frame[["latency_mean_ms", "throughput_samples_per_second", "peak_vram_mb"]].astype(float) <= 0).any().any():
        raise FinalizationError("efficiency metrics must be positive")
    if frame.source_checkpoint_sha256.map(lambda value: _sha(value, "efficiency source checkpoint")).nunique() != 1:
        raise FinalizationError("efficiency Full/no_ssaw source checkpoint hashes disagree")
    for row in frame.itertuples(index=False):
        profile = _profile(row.runtime_hparams, "efficiency runtime_hparams")
        expected_variant = "spline_residual" if row.variant == "full" else "confidence_raw"
        if profile.get("dusafe_variant") != expected_variant or profile.get("enable_source_semantic_router") is not False:
            raise FinalizationError(f"efficiency {row.variant} runtime variant/router is invalid")
    output = frame[["dataset", "scenario", "method", "variant", "profile", "source_seed", "stream_seed", "latency_mean_ms", "throughput_samples_per_second", "peak_vram_mb", "source_checkpoint_sha256"]].copy()
    output = output.sort_values("variant", key=lambda values: values.map({"full": 0, "no_ssaw": 1})).reset_index(drop=True)
    full = output.loc[output["variant"].eq("full")].iloc[0]
    output["latency_ratio_vs_full"] = output["latency_mean_ms"].astype(float) / float(full["latency_mean_ms"])
    output["throughput_ratio_vs_full"] = output["throughput_samples_per_second"].astype(float) / float(full["throughput_samples_per_second"])
    output["peak_vram_ratio_vs_full"] = output["peak_vram_mb"].astype(float) / float(full["peak_vram_mb"])
    hardware = None
    if "hardware" in frame.columns:
        hardware_values = frame["hardware"].dropna().astype(str).unique().tolist()
        if len(hardware_values) > 1:
            raise FinalizationError("efficiency hardware differs between Full/no_ssaw cells")
        hardware = hardware_values[0] if hardware_values else None
    return output, {"rows": len(output), "hardware": hardware}


def finalize(
    *,
    root: str | Path = DEFAULT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    causal_dirs: Sequence[str | Path] | None = None,
    heldout_dirs: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    output_dir = Path(output_dir)
    causal_dirs = tuple(causal_dirs or (
        root / "causal_ablation_primary_eeg_har",
        root / "causal_ablation_secondary_eeg_har",
        root / "causal_ablation_primary_hhar",
        root / "causal_ablation_secondary_hhar",
    ))
    heldout_dirs = tuple(heldout_dirs or (
        root / "heldout_mechanism_eeg_har",
        root / "heldout_mechanism_hhar",
    ))
    checks: dict[str, Any] = {}
    errors: list[str] = []
    safety = None
    causal_frames: list[pd.DataFrame] = []
    heldout_frames: list[pd.DataFrame] = []
    efficiency = None
    a3 = None
    main_table = root / "final_main_table" / "tables" / "main_table.csv"
    try:
        main_table_frame = _csv(main_table)
        checks["main_table"] = {"rows": int(len(main_table_frame)), "columns": int(len(main_table_frame.columns))}
    except Exception as exc:
        errors.append(f"main_table: {exc}")
    for name, thunk in (
        ("safety", lambda: validate_safety(root)),
        ("efficiency", lambda: validate_efficiency(root)),
        ("a3_flow_hparams", lambda: build_a3_flow_hparams(root)),
    ):
        try:
            value, check = thunk()
            if name == "safety": safety = value
            elif name == "efficiency": efficiency = value
            else: a3 = value
            checks[name] = check
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for index, directory in enumerate(causal_dirs):
        label = f"causal[{index}]::{Path(directory).name}"
        try:
            frame, check = validate_causal_dir(Path(directory), label)
            causal_frames.append(frame)
            checks[label] = check
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    for index, directory in enumerate(heldout_dirs):
        label = f"heldout[{index}]::{Path(directory).name}"
        try:
            frame, check = validate_heldout_dir(Path(directory), label)
            heldout_frames.append(frame)
            checks[label] = check
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    causal_horizon_values = {
        int(check["declared_horizon"])
        for label, check in checks.items()
        if label.startswith("causal[") and "declared_horizon" in check
    }
    if len(causal_horizon_values) > 1:
        errors.append(
            "causal evidence mixes future horizons: "
            + repr(sorted(causal_horizon_values))
        )
    manifest: dict[str, Any] = {
        "protocol": "paper_evidence_v2_finalizer_v1",
        "status": "failed" if errors else "complete",
        "cpu_only": True,
        "cuda_started": False,
        "negative_results_preserved": True,
        "root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "inputs": {
            "safety": str((root / "controlled_safety_har_12_16").resolve()),
            "causal": [str(Path(value).resolve()) for value in causal_dirs],
            "heldout": [str(Path(value).resolve()) for value in heldout_dirs],
            "efficiency": str((root / "efficiency_har_12to16").resolve()),
            "main_table": str((root / "final_main_table" / "tables" / "main_table.csv").resolve()),
        },
        "causal_horizon_policy": (
            "each causal manifest role-A future_horizon must match its plan and "
            "raw rows; all validated causal directories must share one horizon"
        ),
        "causal_future_horizons": sorted(causal_horizon_values),
        "checks": checks,
        "errors": errors,
        "outputs": {},
    }
    if errors:
        _write_json(output_dir / "manifest.json", manifest)
        raise FinalizationError("paper evidence finalization failed: " + " | ".join(errors))
    if not causal_frames or not heldout_frames or safety is None or efficiency is None or a3 is None:
        raise FinalizationError("finalization passed no validated artifact groups")
    confidence = build_confidence_panel(causal_frames)
    ssaw_overall = pd.concat([_wide_causal_panel(frame, active_only=False) for frame in causal_frames], ignore_index=True)
    ssaw_active = pd.concat([_wide_causal_panel(frame, active_only=True) for frame in causal_frames], ignore_index=True)
    ssaw = pd.concat([ssaw_overall, ssaw_active], ignore_index=True).sort_values(["dataset", "scenario", "horizon", "segment"], kind="stable").reset_index(drop=True)
    heldout = build_heldout_panel(heldout_frames)
    outputs = {
        "confidence_panel": "confidence_panel.csv",
        "ssaw_causal_panel": "ssaw_causal_panel.csv",
        "heldout_panel": "heldout_panel.csv",
        "safety_s6": "safety_s6.csv",
        "efficiency_a2": "efficiency_a2.csv",
        "a3_paper_flow_hparams": "a3_paper_flow_hparams.csv",
    }
    _write_csv(output_dir / outputs["confidence_panel"], confidence)
    _write_csv(output_dir / outputs["ssaw_causal_panel"], ssaw)
    _write_csv(output_dir / outputs["heldout_panel"], heldout)
    _write_csv(output_dir / outputs["safety_s6"], safety)
    _write_csv(output_dir / outputs["efficiency_a2"], efficiency)
    _write_csv(output_dir / outputs["a3_paper_flow_hparams"], a3)
    manifest["outputs"] = {key: str((output_dir / filename).resolve()) for key, filename in outputs.items()}
    manifest["row_counts"] = {key: int(pd.read_csv(output_dir / filename).shape[0]) for key, filename in outputs.items()}
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = finalize(root=args.root, output_dir=args.output_dir)
    except FinalizationError as exc:
        print(str(exc))
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAUSAL_PROTOCOL",
    "FinalizationError",
    "build_a3_flow_hparams",
    "build_confidence_panel",
    "build_heldout_panel",
    "finalize",
    "validate_causal_dir",
    "validate_efficiency",
    "validate_heldout_dir",
    "validate_safety",
]
