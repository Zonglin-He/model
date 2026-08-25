"""Finalize the fixed-source four-dataset clean main table.

This module is deliberately CPU-only and imports no trainer, model, or CUDA
code.  It treats the per-cell CSV files as the source of truth and validates
the complete formal key set before writing any aggregate result.  The legacy
three-dataset manifest in ``results/reviewer_queue_v2`` is known to describe
only its last invocation; it is audited, but never used to select rows.

The formal HHAR protocol is the five-flow target-selected protocol from
``configs.formal_evaluation_protocol``.  EEG/HAR/FD baselines are read from
the legacy merged CSV, HHAR baselines are read from ``cells/**`` and old
DuSafe rows are discarded.  The current DuSafe panel is supplied separately
with ``--dusafe-input`` (or the completed default paper-evidence directory).
Consequently the final panel is descriptive for every dataset:
dataset-level target labels were used when selecting the frozen profiles.
No output from this script is labelled confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import (  # noqa: E402
    evaluation_partition_metadata,
    formal_scenario_pairs,
)


PROTOCOL_VERSION = "fixed_source_main_table_v2_current_dusafe_five_flows_descriptive"
DATASETS = ("EEG", "HAR", "FD", "HHAR")
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
METHODS = (
    "NoAdap",
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
    "DuSafe",
)
REFERENCE_METHOD = "DuSafe"
BASELINE_METHODS = tuple(method for method in METHODS if method != REFERENCE_METHOD)
KEY_COLUMNS = ("dataset", "scenario", "method", "source_seed", "stream_seed")
CORE_REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    "src_id",
    "trg_id",
    "status",
    "f1",
    "source_model_sha256",
)
# These columns are present in the production benchmark runners, but a
# current DuSafe evidence export may intentionally omit checkpoint-container
# metadata.  Keep them optional at the merger boundary; the canonical model
# SHA is the fixed-source identity used by this protocol.
OPTIONAL_COLUMNS = (
    "source_checkpoint_file_sha256",
    "source_checkpoint_path",
    "runtime_hparams",
    "source_checkpoint_protocol",
)
REQUIRED_COLUMNS = (*CORE_REQUIRED_COLUMNS,)
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
DEFAULT_LEGACY_RAW = ROOT / "results" / "reviewer_queue_v2" / "main_table_source_calibrated"
DEFAULT_HHAR_RAW = ROOT / "results" / "hhar_five_flow_main_table_v2"
DEFAULT_DUSAFE_RAW = ROOT / "results" / "paper_evidence_v1" / "final_main_table" / "current_full_no_ssaw"
DEFAULT_FLOW_PROFILE_JSON = ROOT / "configs" / "paper_flow_profiles_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "reviewer_queue_v2" / "four_dataset_main_table_final"

# Keys that define the effective TTA profile.  Runtime exports also contain
# diagnostics, source identity, and process bookkeeping; those fields must
# not make three source seeds appear to use different TTA profiles.
EFFECTIVE_TTA_KEYS = frozenset(
    {
        "adapt_parameter_scope",
        "batch_size",
        "bn_statistics",
        "confidence_keep_fraction",
        "confidence_reference_samples",
        "enable_adaptation",
        "enable_confidence_gate",
        "enable_source_semantic_gate",
        "enable_source_semantic_router",
        "grad_clip",
        "grad_clip_value",
        "learning_rate",
        "normalization_reference",
        "optim_method",
        "record_optimizer_diagnostics",
        "source_semantic_reference_samples",
        "ssaw_antithetic",
        "ssaw_antithetic_pairs",
        "ssaw_auxiliary_weight",
        "ssaw_control_points",
        "ssaw_kl_scale",
        "ssaw_risk_temperature",
        "ssaw_sigma",
        "ssaw_sobol_seed",
        "ssaw_strength",
        "ssaw_temporal_mode",
        "spline_control_points",
        "spline_log_strength",
        "spline_num_directions",
        "spline_radius_levels",
        "steps",
        "weight_decay",
        # Variant markers are retained for pair auditing and dropped only
        # when comparing Full against NoSSAW's atomic switch.
        "algorithm_variant",
        "dusafe_variant",
        "dusafe_logging_mode",
        "enable_ssaw",
        "ssaw_enabled",
        "variant",
    }
)
ATOMIC_VARIANT_KEYS = frozenset(
    {"algorithm_variant", "dusafe_variant", "enable_ssaw", "ssaw_enabled", "variant"}
)


class FinalizationError(ValueError):
    """Raised when an input or output violates the fixed-source contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _bool_value(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _domain_value(value: Any) -> str:
    """Canonicalize pandas' integer/float CSV representation of domain IDs."""

    if _is_missing(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _key_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["dataset"]).strip().upper(),
        str(row["scenario"]).strip(),
        str(row["method"]).strip(),
        int(row["source_seed"]),
        int(row["stream_seed"]),
    )


def expected_flows(dataset: str) -> tuple[str, ...]:
    return tuple(f"{source}->{target}" for source, target in formal_scenario_pairs(dataset))


def expected_key_set(
    dataset: str,
    methods: Sequence[str] = METHODS,
) -> set[tuple[str, str, str, int, int]]:
    name = str(dataset).upper()
    return {
        (name, scenario, method, seed, STREAM_SEED)
        for scenario in expected_flows(name)
        for method in methods
        for seed in SOURCE_SEEDS
    }


def _parse_mapping(value: Any, *, row_label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if _is_missing(value):
        raise FinalizationError(f"{row_label}: runtime_hparams is missing")
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{row_label}: runtime_hparams is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise FinalizationError(f"{row_label}: runtime_hparams is not a JSON object")
    return dict(parsed)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(dict(value)), sort_keys=True, separators=(",", ":"))


def _load_flow_profiles(
    path: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], Path | None, list[str]]:
    """Load the target-selected per-flow TTA profile map when available."""

    candidate = Path(path).expanduser().resolve() if path is not None else DEFAULT_FLOW_PROFILE_JSON
    if not candidate.is_file():
        return {}, None, [f"flow profile JSON is missing: {candidate}"]
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {}, candidate, [f"flow profile JSON is unreadable: {exc}"]
    raw_profiles = payload.get("profiles", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(raw_profiles, Mapping):
        return {}, candidate, ["flow profile JSON profiles is not an object"]
    profiles: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for key, value in raw_profiles.items():
        if not isinstance(value, Mapping):
            issues.append(f"profile {key!r} is not an object")
            continue
        canonical_key = str(key).strip().upper()
        profiles[canonical_key] = {
            str(name): _json_safe(item)
            for name, item in value.items()
            if str(name) in EFFECTIVE_TTA_KEYS
        }
    return profiles, candidate, issues


def _profile_key(dataset: Any, scenario: Any) -> str:
    return f"{str(dataset).strip().upper()}:{str(scenario).strip()}"


def _row_runtime_mapping(row: Mapping[str, Any], *, row_label: str) -> dict[str, Any]:
    value = row.get("runtime_hparams")
    if _is_missing(value):
        return {}
    return _parse_mapping(value, row_label=row_label)


def _effective_dusafe_config(
    row: Mapping[str, Any],
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    row_label: str = "DuSafe",
) -> dict[str, Any]:
    """Return only effective TTA keys, overlaying measured row config on profile defaults."""

    config: dict[str, Any] = {}
    if profiles:
        config.update(dict(profiles.get(_profile_key(row.get("dataset"), row.get("scenario")), {})))
    runtime = _row_runtime_mapping(row, row_label=row_label)
    config.update(
        {
            str(key): _json_safe(value)
            for key, value in runtime.items()
            if str(key) in EFFECTIVE_TTA_KEYS
        }
    )
    for key in EFFECTIVE_TTA_KEYS:
        if key in row and not _is_missing(row[key]):
            config[key] = _json_safe(row[key])
    # Some runners encode the atomic switch only in the variant column.
    if "variant" not in config and not _is_missing(row.get("variant")):
        config["variant"] = str(row["variant"]).strip()
    return config


def _without_atomic_variant(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(value)
        for key, value in config.items()
        if str(key) not in ATOMIC_VARIANT_KEYS
    }


def _audit_legacy_manifest(input_dir: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Audit, but do not trust, the known legacy manifest."""

    path = input_dir / "manifest.json"
    audit: dict[str, Any] = {
        "path": str(path.resolve()),
        "present": path.is_file(),
        "authoritative": False,
        "consistent_with_raw": False,
        "issues": [],
    }
    if not path.is_file():
        audit["issues"] = ["manifest.json is missing"]
        return audit
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        audit["issues"] = [f"manifest is unreadable: {exc}"]
        return audit
    issues: list[str] = []
    declared_rows = payload.get("raw_rows")
    if declared_rows is not None and int(declared_rows) != len(frame):
        issues.append(f"raw_rows={declared_rows} but raw CSV has {len(frame)} rows")
    observed_datasets = sorted(frame["dataset"].astype(str).str.upper().unique().tolist())
    declared_datasets = sorted(str(value).upper() for value in payload.get("datasets", ()))
    if declared_datasets != observed_datasets:
        issues.append(f"datasets={declared_datasets} but raw has {observed_datasets}")
    observed_methods = sorted(frame["method"].astype(str).unique().tolist())
    declared_methods = sorted(str(value) for value in payload.get("methods", ()))
    if declared_methods != observed_methods:
        issues.append(f"methods={declared_methods} but raw has {observed_methods}")
    if payload.get("successful_rows") is not None and int(payload["successful_rows"]) != len(frame):
        issues.append("successful_rows does not match raw CSV")
    audit["declared"] = {
        "datasets": payload.get("datasets"),
        "methods": payload.get("methods"),
        "raw_rows": payload.get("raw_rows"),
        "successful_rows": payload.get("successful_rows"),
    }
    audit["issues"] = issues
    audit["consistent_with_raw"] = not issues
    return audit


def _audit_hhar_queue_manifest(
    input_dir: Path,
    frame: pd.DataFrame,
    *,
    strict_completion: bool = True,
) -> dict[str, Any]:
    """Audit HHAR queue metadata.

    The historical queue manifest can be ``failed`` solely because its old
    DuSafe rows disagree on checkpoint-container hashes.  For the new
    baseline path, recursive cell validation is authoritative after old
    DuSafe rows are removed, so that metadata is recorded as a warning rather
    than used to block the ten-method baseline panel.
    """

    manifest_path = input_dir / "manifest.json"
    status_path = input_dir / "status.json"
    audit: dict[str, Any] = {
        "manifest_path": str(manifest_path.resolve()),
        "status_path": str(status_path.resolve()),
        "manifest_present": manifest_path.is_file(),
        "status_present": status_path.is_file(),
        "issues": [],
    }
    issues: list[str] = []
    payload: Mapping[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("manifest is not an object")
            payload = loaded
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"manifest unreadable: {exc}")
    else:
        issues.append("HHAR queue manifest.json is missing")
    status_payload: Mapping[str, Any] = {}
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                status_payload = loaded
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"status unreadable: {exc}")
    else:
        issues.append("HHAR queue status.json is missing")
    protocol = str(payload.get("protocol_version", payload.get("protocol", "")))
    if protocol and not protocol.startswith("hhar_five_flow_main_table_queue_"):
        issues.append(f"unexpected HHAR queue protocol: {protocol}")
    for name, source in (("manifest", payload), ("status", status_payload)):
        if source and source.get("status") not in {"complete", "completed"}:
            issues.append(f"HHAR {name} status is {source.get('status')!r}, not complete")
        if source and source.get("confirmatory") is not False:
            issues.append(f"HHAR {name} confirmatory flag is not false")
        if source and source.get("selection_overlap") is not True:
            issues.append(f"HHAR {name} selection_overlap is not true")
    if payload and payload.get("raw_rows") not in {None, len(frame)}:
        issues.append("HHAR manifest raw_rows does not match CSV")
    audit["protocol"] = protocol
    audit["strict_completion"] = bool(strict_completion)
    audit["issues"] = issues
    audit["accepted"] = not issues or not strict_completion
    return audit


def _validate_source_consistency(frame: pd.DataFrame, *, label: str) -> list[str]:
    errors: list[str] = []
    grouped = frame.groupby(["dataset", "src_id", "source_seed"], dropna=False)
    for (dataset, source, seed), group in grouped:
        # Fixed-source identity is the canonical model state, not the byte
        # representation of its checkpoint container.  Re-saving the same
        # state can legitimately change the file SHA because optimizer/config
        # metadata or serialization details changed.  Every file digest is
        # still syntax-validated above and retained in the audit table.
        model_hashes = {
            str(value).strip().lower() for value in group["source_model_sha256"]
        }
        if len(model_hashes) != 1:
            errors.append(
                f"{label}: {dataset}/{source}/seed={seed} has multiple source_model_sha256 values"
            )
        # Checkpoint containers can be re-serialized independently by each
        # method runner.  Their paths/file hashes are audit metadata, not the
        # fixed-source identity.  The canonical source_model_sha256 check
        # above is the only cross-method identity requirement here.
    return errors


def _validate_key_source_hashes(frame: pd.DataFrame, *, label: str) -> list[str]:
    """Check the fixed-source hash for every non-method result key.

    A source checkpoint is shared by all methods for a particular
    ``dataset/flow/source_seed/stream_seed`` key.  Checking this after the
    baseline and DuSafe tables have been merged catches a stale DuSafe row
    even when each input table is internally self-consistent.
    """

    errors: list[str] = []
    key_columns = ["dataset", "scenario", "source_seed", "stream_seed"]
    for key, group in frame.groupby(key_columns, dropna=False, sort=True):
        hashes = {
            str(value).strip().lower()
            for value in group["source_model_sha256"]
            if not _is_missing(value)
        }
        if len(hashes) != 1:
            errors.append(
                f"{label}: key {key} has {len(hashes)} source_model_sha256 values"
            )
    return errors


def _validate_method_provenance(
    frame: pd.DataFrame,
    *,
    label: str,
    dusafe_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    forbidden_baseline_keys = {
        "enable_ssaw",
        "ssaw_antithetic",
        "ssaw_antithetic_pairs",
        "ssaw_auxiliary_weight",
        "ssaw_control_points",
        "ssaw_kl_scale",
        "ssaw_risk_temperature",
        "ssaw_sigma",
        "ssaw_sobol_seed",
        "ssaw_strength",
        "ssaw_temporal_mode",
        "enable_confidence_gate",
        "enable_source_semantic_gate",
        "enable_source_semantic_router",
        "confidence_keep_fraction",
        "confidence_reference_samples",
        "source_semantic_reference_samples",
        "spline_control_points",
        "spline_num_directions",
        "spline_log_strength",
        "spline_radius_levels",
    }
    for index, row in frame.iterrows():
        method = str(row["method"])
        row_label = f"{label} row={index} {row['dataset']}/{row['scenario']}/{method}/seed={row['source_seed']}"
        # Runtime provenance is optional for compact evidence exports.  When
        # present, retain the stricter production checks below.
        has_runtime = "runtime_hparams" in frame.columns and not _is_missing(row["runtime_hparams"])
        config = _parse_mapping(row["runtime_hparams"], row_label=row_label) if has_runtime else {}
        if method in BASELINE_METHODS:
            contaminated = sorted(forbidden_baseline_keys.intersection(config))
            if contaminated:
                errors.append(f"{row_label}: baseline has DuSafe override keys {contaminated}")
            if method == "EATA" and has_runtime:
                if config.get("fisher_enabled") is not True:
                    errors.append(f"{row_label}: EATA runtime_hparams fisher_enabled is not true")
        else:
            if method != REFERENCE_METHOD:
                errors.append(f"{row_label}: unsupported method provenance")
    # DuSafe profiles are allowed to vary by formal flow.  Within a
    # dataset/scenario, however, all source seeds must use the same effective
    # TTA profile.  This comparison intentionally ignores diagnostics and
    # source identity fields, retaining only EFFECTIVE_TTA_KEYS.
    dusafe = frame[frame["method"].eq(REFERENCE_METHOD)]
    for (dataset, scenario), group in dusafe.groupby(["dataset", "scenario"], sort=True):
        configs: dict[str, list[int]] = {}
        for index, row in group.iterrows():
            config = _effective_dusafe_config(
                row,
                profiles=dusafe_profiles,
                row_label=f"{label}/{dataset}/{scenario}/row={index}",
            )
            configs.setdefault(_canonical_json(config), []).append(index)
        if len(configs) > 1:
            errors.append(
                f"{label}: DuSafe has multiple effective TTA configs for {dataset}/{scenario}: "
                f"rows={list(configs.values())}"
            )
    return errors


def _validate_eata_fisher(frame: pd.DataFrame, *, label: str) -> tuple[list[str], int]:
    errors: list[str] = []
    eata = frame[frame["method"].eq("EATA")]
    checked_files = 0
    required = (
        "fisher_enabled",
        "fisher_cache_path",
        "fisher_cache_hash",
        "fisher_source_checkpoint_sha256",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        # Compact raw evidence exports may omit all Fisher bookkeeping.  Do
        # not invent a failure for that representation; if any Fisher column
        # is supplied, however, the complete provenance contract is required.
        if len(missing) == len(required):
            return [], checked_files
        return [f"{label}: missing EATA Fisher columns {missing}"], checked_files
    for index, row in eata.iterrows():
        row_label = f"{label} EATA row={index}"
        if _bool_value(row["fisher_enabled"]) is not True:
            errors.append(f"{row_label}: fisher_enabled is not true")
        cache_path = str(row["fisher_cache_path"]).strip()
        cache_hash = str(row["fisher_cache_hash"]).strip().lower()
        source_hash = str(row["source_model_sha256"]).strip().lower()
        fisher_source_hash = str(row["fisher_source_checkpoint_sha256"]).strip().lower()
        if not cache_path or _is_missing(row["fisher_cache_path"]):
            errors.append(f"{row_label}: fisher_cache_path is missing")
        if not HASH_RE.fullmatch(cache_hash):
            errors.append(f"{row_label}: fisher_cache_hash is not a SHA-256 digest")
        if fisher_source_hash != source_hash:
            errors.append(f"{row_label}: Fisher source checkpoint hash mismatches source hash")
        path = Path(cache_path)
        if path.is_file() and HASH_RE.fullmatch(cache_hash):
            checked_files += 1
            if _sha256(path).lower() != cache_hash:
                errors.append(f"{row_label}: fisher_cache_hash does not match cache file")
        elif not path.is_file():
            errors.append(f"{row_label}: Fisher cache file does not exist: {cache_path}")
    return errors, checked_files


def validate_dataset_frame(
    frame: pd.DataFrame,
    dataset: str,
    *,
    label: str,
    allow_missing_oom_flag: bool = False,
    methods: Sequence[str] = METHODS,
    dusafe_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate one dataset's complete formal raw table.

    ``methods`` is explicit so the merger can validate a ten-method baseline
    panel and a one-method DuSafe panel independently before joining them.
    The default preserves the historical eleven-method API.
    """

    dataset_name = str(dataset).strip().upper()
    if dataset_name not in DATASETS:
        raise FinalizationError(f"{label}: unsupported dataset {dataset_name}")
    if frame.empty:
        raise FinalizationError(f"{label}: CSV is empty")
    methods = tuple(str(method).strip() for method in methods)
    if not methods:
        raise FinalizationError(f"{label}: method set is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FinalizationError(f"{label}: missing required columns {missing}")
    result = frame.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["dataset"] = result["dataset"].astype(str).str.strip().str.upper()
    if set(result["dataset"]) != {dataset_name}:
        raise FinalizationError(
            f"{label}: dataset column is not exactly {dataset_name}: {sorted(result['dataset'].unique())}"
        )
    result["scenario"] = result["scenario"].astype(str).str.strip()
    result["method"] = result["method"].astype(str).str.strip()
    result["src_id"] = result["src_id"].map(_domain_value)
    result["trg_id"] = result["trg_id"].map(_domain_value)
    try:
        result["source_seed"] = pd.to_numeric(result["source_seed"], errors="raise").astype(int)
        result["stream_seed"] = pd.to_numeric(result["stream_seed"], errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label}: source/stream seeds are not integral") from exc
    observed_keys = []
    for index, row in result.iterrows():
        try:
            source, target = row["scenario"].split("->", 1)
        except ValueError as exc:
            raise FinalizationError(f"{label} row={index}: malformed scenario") from exc
        if row["src_id"] != _domain_value(source) or row["trg_id"] != _domain_value(target):
            raise FinalizationError(f"{label} row={index}: src_id/trg_id disagree with scenario")
        observed_keys.append(_key_tuple(row))
    expected = expected_key_set(dataset_name, methods)
    observed = set(observed_keys)
    if len(observed_keys) != len(observed):
        duplicates = sorted(key for key in observed if observed_keys.count(key) > 1)[:5]
        raise FinalizationError(f"{label}: duplicate cell keys {duplicates}")
    if observed != expected:
        missing_keys = sorted(expected - observed)[:5]
        extra_keys = sorted(observed - expected)[:5]
        raise FinalizationError(
            f"{label}: key set mismatch; missing={missing_keys}, extra={extra_keys}"
        )
    if len(result) != len(expected):
        raise FinalizationError(f"{label}: row count {len(result)} != {len(expected)}")
    if result["status"].astype(str).str.strip().ne("ok").any():
        raise FinalizationError(f"{label}: status contains non-ok rows")
    for column in ("error_type", "error", "traceback"):
        if column in result.columns and result[column].map(lambda value: not _is_missing(value)).any():
            raise FinalizationError(f"{label}: non-empty {column} in successful raw table")
    oom_warning = False
    if "is_oom" in result.columns:
        parsed_oom = result["is_oom"].map(_bool_value)
        if parsed_oom.dropna().eq(True).any() or parsed_oom.dropna().isna().any():
            raise FinalizationError(f"{label}: is_oom contains true or unparseable values")
        oom_warning = parsed_oom.isna().all()
        if oom_warning and not allow_missing_oom_flag:
            raise FinalizationError(f"{label}: is_oom is missing for every successful row")
    else:
        oom_warning = True
        if not allow_missing_oom_flag:
            raise FinalizationError(f"{label}: is_oom column is missing")
    result["f1"] = pd.to_numeric(result["f1"], errors="coerce")
    if result["f1"].isna().any() or not np.isfinite(result["f1"].to_numpy(dtype=float)).all():
        raise FinalizationError(f"{label}: f1 contains missing/non-finite values")
    for column in ("source_model_sha256", "source_checkpoint_file_sha256"):
        values = result[column]
        if column == "source_checkpoint_file_sha256":
            values = values[~values.map(_is_missing)]
        if values.map(lambda value: not HASH_RE.fullmatch(str(value).strip())).any():
            raise FinalizationError(f"{label}: invalid {column} digest")
    if set(result["method"]) != set(methods):
        raise FinalizationError(f"{label}: method set drifted: {sorted(result['method'].unique())}")
    if set(result["source_seed"]) != set(SOURCE_SEEDS):
        raise FinalizationError(f"{label}: source seed set drifted")
    if set(result["stream_seed"]) != {STREAM_SEED}:
        raise FinalizationError(f"{label}: stream seed is not exactly {STREAM_SEED}")
    if set(result["scenario"]) != set(expected_flows(dataset_name)):
        raise FinalizationError(f"{label}: formal flow set drifted")
    source_errors = _validate_source_consistency(result, label=label)
    provenance_errors = _validate_method_provenance(
        result,
        label=label,
        dusafe_profiles=dusafe_profiles,
    )
    fisher_errors, fisher_files_checked = _validate_eata_fisher(result, label=label)
    errors = source_errors + provenance_errors + fisher_errors
    if errors:
        raise FinalizationError("; ".join(errors[:20]))
    warnings = []
    if oom_warning:
        warnings.append("is_oom was absent or entirely missing; status/error fields were used as legacy success evidence")
    reserialized_units = 0
    for _, group in result.groupby(["dataset", "src_id", "source_seed"], dropna=False):
        if group["source_checkpoint_file_sha256"].astype(str).str.strip().nunique() > 1:
            reserialized_units += 1
    if reserialized_units:
        warnings.append(
            f"{reserialized_units} source units use multiple checkpoint-file SHA values but one canonical source-model SHA"
        )
    result["formal_protocol_version"] = PROTOCOL_VERSION
    result["evaluation_partition"] = "target_selected_evaluation"
    result["parameter_selection_data_overlap"] = True
    result["selection_overlap"] = True
    result["confirmatory"] = False
    result["target_labels_used_for_parameter_selection"] = True
    result["target_labels_used_for_updates"] = False
    result["target_labels_used_for_metrics"] = True
    result["formal_flow_count"] = len(expected_flows(dataset_name))
    result["source_domain"] = result["src_id"]
    result["target_domain"] = result["trg_id"]
    result["parameter_provenance"] = np.where(
        result["method"].eq(REFERENCE_METHOD),
        "dusafe_frozen_dataset_profile",
        "benchmark_default",
    )
    audit = {
        "dataset": dataset_name,
        "label": label,
        "rows": int(len(result)),
        "expected_rows": len(expected),
        "passed": True,
        "flow_count": len(expected_flows(dataset_name)),
        "flows": list(expected_flows(dataset_name)),
        "methods": list(methods),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "source_checkpoint_units": int(result[["src_id", "source_seed"]].drop_duplicates().shape[0]),
        "reserialized_checkpoint_units": int(reserialized_units),
        "eata_fisher_rows": int(len(result[result["method"].eq("EATA")])),
        "eata_fisher_cache_files_sha256_checked": int(fisher_files_checked),
        "warnings": warnings,
    }
    return result, audit


def _cluster_values(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    grouped = (
        frame.groupby(["dataset", "source_domain", "source_seed"], as_index=False)[value_column]
        .mean()
        .sort_values(["dataset", "source_domain", "source_seed"], kind="stable")
    )
    values = grouped[value_column].to_numpy(dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise FinalizationError("paired inference requires at least two finite source clusters")
    return values


def _cluster_bootstrap(values: np.ndarray, *, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
    means = values[sample_indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(values: np.ndarray, *, replicates: int, seed: int) -> float:
    observed = abs(float(values.mean()))
    n = len(values)
    if n <= 16:
        # Exact enumeration is cheap for the 15 source-domain/seed clusters
        # in each dataset and removes Monte-Carlo ambiguity from the report.
        total = 1 << n
        masks = np.arange(total, dtype=np.uint32)[:, None]
        bits = (masks >> np.arange(n, dtype=np.uint32)[None, :]) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        exceed = np.count_nonzero(np.abs((signs * values[None, :]).mean(axis=1)) >= observed - 1e-15)
        return float(int(exceed) / total)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(replicates), n))
    exceed = np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15)
    return float((int(exceed) + 1) / (int(replicates) + 1))


def _holm(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(float(value) for value in p_values), dtype=float)
    if values.size == 0:
        return values
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise FinalizationError("inference p-values are not finite in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1.0, values[order] * (len(values) - np.arange(len(values))))
    adjusted = np.maximum.accumulate(adjusted)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def dataset_method_aggregate(frame: pd.DataFrame, *, bootstrap_replicates: int, seed: int) -> pd.DataFrame:
    """Build the main dataset table using the registered two-stage mean.

    The statistical unit is the source seed.  For each dataset/method we
    first average the five formal flows within each source seed, then report
    the mean and sample standard deviation over the three seed means.  This
    avoids treating the fifteen flow/seed cells as independent replicates.
    """

    rows: list[dict[str, Any]] = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=True):
        seed_means = (
            group.groupby(["dataset", "method", "source_seed"], as_index=False)["f1"]
            .mean()
            .sort_values("source_seed", kind="stable")
        )
        values = seed_means["f1"].to_numpy(dtype=float)
        if len(values) != len(SOURCE_SEEDS):
            raise FinalizationError(
                f"{dataset}/{method}: expected {len(SOURCE_SEEDS)} source-seed means, got {len(values)}"
            )
        low, high = _cluster_bootstrap(values, replicates=bootstrap_replicates, seed=seed + len(rows))
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_successful_cells": int(len(group)),
                "n_formal_flows": int(group["scenario"].nunique()),
                "n_source_seeds": int(len(values)),
                "n_source_domain_seed_clusters": int(len(values)),
                "aggregation": "mean_over_flows_per_source_seed_then_mean_std_over_source_seeds",
                "f1_mean": mean,
                "f1_std": std,
                "mean": mean,
                "std": std,
                "mean_f1": mean,
                "std_f1": std,
                "f1_ci95_low": low,
                "f1_ci95_high": high,
                "evaluation_partition": "target_selected_evaluation",
                "selection_overlap": True,
                "confirmatory": False,
            }
        )
    return pd.DataFrame(rows)


def per_flow_macro_f1(frame: pd.DataFrame) -> pd.DataFrame:
    """Build A1's per-flow mean/std over the three source seeds."""

    result = (
        frame.groupby(["dataset", "scenario", "method"], as_index=False)
        .agg(
            n_source_seeds=("source_seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            f1_min=("f1", "min"),
            f1_max=("f1", "max"),
        )
    )
    result["evaluation_partition"] = "target_selected_evaluation"
    result["selection_overlap"] = True
    result["confirmatory"] = False
    result["mean"] = result["f1_mean"]
    result["std"] = result["f1_std"]
    result["mean_f1"] = result["f1_mean"]
    result["std_f1"] = result["f1_std"]
    result["table"] = "A1"
    return result


def _numeric_values(configs: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for config in configs:
        value = config.get(key)
        if _is_missing(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def dusafe_a3_tables(
    frame: pd.DataFrame,
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return A3's per-flow profile table and dataset range summary."""

    dusafe = frame[frame["method"].eq(REFERENCE_METHOD)].copy()
    flow_rows: list[dict[str, Any]] = []
    for (dataset, scenario), group in dusafe.groupby(["dataset", "scenario"], sort=True):
        first = group.iloc[0]
        config = _effective_dusafe_config(
            first,
            profiles=profiles,
            row_label=f"A3/{dataset}/{scenario}",
        )
        row: dict[str, Any] = {
            "dataset": str(dataset),
            "scenario": str(scenario),
            "src_id": str(first.get("src_id", "")),
            "trg_id": str(first.get("trg_id", "")),
            "method": REFERENCE_METHOD,
            "source_seed_count": int(group["source_seed"].nunique()),
            "profile_source": "flow_profile_json" if profiles and _profile_key(dataset, scenario) in profiles else "runtime_hparams_or_row_columns",
            "effective_tta_config": _canonical_json(config),
            "effective_tta_config_sha256": hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest(),
            "lambda_positive": bool(
                any(
                    value > 0
                    for value in _numeric_values((config,), "ssaw_auxiliary_weight")
                )
            ),
        }
        expected_profile = dict(profiles.get(_profile_key(dataset, scenario), {})) if profiles else {}
        for key in (
            "batch_size",
            "learning_rate",
            "steps",
            "ssaw_auxiliary_weight",
            "weight_decay",
            "grad_clip",
        ):
            row[key] = config.get(key, pd.NA)
            row[f"profile_{key}"] = expected_profile.get(key, pd.NA)
        flow_rows.append(row)
    flow_table = pd.DataFrame(flow_rows)

    summary_rows: list[dict[str, Any]] = []
    for dataset, group in flow_table.groupby("dataset", sort=True):
        configs = [json.loads(value) for value in group["effective_tta_config"]]
        row: dict[str, Any] = {
            "dataset": str(dataset),
            "flow_count": int(len(group)),
            "flows": ";".join(group["scenario"].astype(str)),
            "profile_source": ";".join(sorted(set(group["profile_source"].astype(str)))),
            "all_lambda_positive": bool(group["lambda_positive"].all()),
        }
        for key, output in (
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
            ("steps", "steps"),
            ("ssaw_auxiliary_weight", "lambda"),
            ("weight_decay", "weight_decay"),
            ("grad_clip", "grad_clip"),
        ):
            values = _numeric_values(configs, key)
            row[f"{output}_min"] = min(values) if values else pd.NA
            row[f"{output}_max"] = max(values) if values else pd.NA
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return flow_table, summary


def paired_source_seed_domain_inference(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    reference = frame[frame["method"].eq(REFERENCE_METHOD)][
        ["dataset", "scenario", "source_domain", "source_seed", "f1"]
    ].rename(columns={"f1": "reference_f1"})
    rows: list[dict[str, Any]] = []
    for method in BASELINE_METHODS:
        candidate = frame[frame["method"].eq(method)][
            ["dataset", "scenario", "source_domain", "source_seed", "f1"]
        ].rename(columns={"f1": "method_f1"})
        merged = reference.merge(
            candidate,
            on=["dataset", "scenario", "source_domain", "source_seed"],
            how="outer",
            validate="one_to_one",
        )
        if merged[["reference_f1", "method_f1"]].isna().any().any():
            raise FinalizationError(f"paired inference missing cells for {method}")
        merged["effect_method_minus_dusafe"] = merged["method_f1"] - merged["reference_f1"]
        for dataset, group in merged.groupby("dataset", sort=True):
            values = _cluster_values(
                group.rename(columns={"source_domain": "source_domain"}),
                "effect_method_minus_dusafe",
            )
            low, high = _cluster_bootstrap(
                values,
                replicates=bootstrap_replicates,
                seed=seed + len(rows),
            )
            p_value = _sign_flip_p(values, replicates=bootstrap_replicates, seed=seed + len(rows))
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "reference_method": REFERENCE_METHOD,
                    "effect_definition": "method_f1_minus_dusafe_f1",
                    "paired_cell_count": int(len(group)),
                    "source_domain_seed_cluster_count": int(len(values)),
                    "effect_mean": float(values.mean()),
                    "effect_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "cluster_signflip_p_raw": p_value,
                    "evaluation_partition": "target_selected_evaluation",
                    "selection_overlap": True,
                    "confirmatory": False,
                }
            )
    output = pd.DataFrame(rows)
    output["cluster_signflip_p_holm"] = _holm(output["cluster_signflip_p_raw"].tolist())
    output["holm_family"] = "all_dataset_x_method_comparisons"
    return output


def _load_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FinalizationError(f"{label}: missing raw CSV {path}")
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise FinalizationError(f"{label}: cannot read {path}: {exc}") from exc


def _discover_raw_csvs(
    path: Path,
    *,
    label: str,
    recursive_only: bool = False,
) -> list[Path]:
    """Resolve a raw input file or a cell directory without touching models."""

    path = Path(path).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FinalizationError(f"{label}: input path does not exist: {path}")

    # A single merged export takes precedence for ordinary inputs.  HHAR's
    # ``cells/**`` contract opts into recursive_only so a stale root CSV can
    # never shadow the ten-method cell panel.
    if not recursive_only:
        for name in ("per_source_seed_results.csv", "merged_per_source_seed_results.csv", "raw.csv"):
            direct = path / name
            if direct.is_file():
                return [direct]
    candidates = sorted(path.rglob("per_source_seed_results.csv"))
    if not candidates:
        candidates = sorted(path.rglob("merged_per_source_seed_results.csv"))
    if not candidates:
        candidates = sorted(path.rglob("raw.csv"))
    if not candidates:
        raise FinalizationError(f"{label}: no raw per-source CSV found below {path}")
    return candidates


def _load_csv_collection(
    path: Path,
    *,
    label: str,
    recursive_only: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = _discover_raw_csvs(path, label=label, recursive_only=recursive_only)
    frames: list[pd.DataFrame] = []
    for csv_path in paths:
        frame = _load_csv(csv_path, label=f"{label}/{csv_path.name}")
        if frame.empty:
            raise FinalizationError(f"{label}: raw CSV is empty: {csv_path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), paths


def _dusafe_manifest_profile_path(input_path: Path, input_paths: Sequence[Path]) -> Path | None:
    """Read a flow-profile path declared by the DuSafe evidence manifest."""

    root = input_path if input_path.is_dir() else input_path.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    declared = payload.get("flow_profile_json") if isinstance(payload, Mapping) else None
    if not declared:
        return None
    candidate = Path(str(declared)).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    return candidate if candidate.is_file() else None


def _profile_scope_summary(
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the effective profile scope represented by keyed configs.

    The input can be the flow JSON's partial overrides or the fully resolved
    configs emitted in A3.  Scope claims are made only when all four datasets
    have the complete five-flow panel; otherwise the all-five field is null.
    """

    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_key, raw_config in profiles.items():
        dataset, separator, scenario = str(raw_key).partition(":")
        if not separator or not dataset or not scenario:
            continue
        dataset = dataset.strip().upper().replace("MFD", "FD")
        scenario = scenario.strip()
        config = _without_atomic_variant(raw_config)
        by_dataset.setdefault(dataset, {})[scenario] = config

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset, scenarios in sorted(by_dataset.items()):
        canonical = {
            scenario: _canonical_json(config)
            for scenario, config in sorted(scenarios.items())
        }
        values = list(canonical.values())
        complete = set(canonical) == set(expected_flows(dataset))
        same = bool(values) and len(set(values)) == 1
        per_dataset[dataset] = {
            "flow_count": int(len(canonical)),
            "formal_five_flow_panel": bool(complete),
            "flow_specific_tta_profiles": bool(len(set(values)) > 1),
            "same_profile_for_selected_flows": bool(same),
            "same_profile_for_all_five_flows": bool(same) if complete else None,
            "effective_profiles_by_flow": {
                f"{dataset}:{scenario}": {
                    "effective_tta_config": scenarios[scenario],
                    "effective_tta_config_sha256": hashlib.sha256(
                        canonical[scenario].encode("utf-8")
                    ).hexdigest(),
                }
                for scenario in sorted(scenarios)
            },
        }

    complete_panel = bool(per_dataset) and set(per_dataset) == set(DATASETS) and all(
        bool(item["formal_five_flow_panel"]) for item in per_dataset.values()
    )
    same_selected = bool(per_dataset) and all(
        bool(item["same_profile_for_selected_flows"])
        for item in per_dataset.values()
    )
    flow_specific = bool(per_dataset) and any(
        bool(item["flow_specific_tta_profiles"]) for item in per_dataset.values()
    )
    return {
        "flow_specific_tta_profiles": bool(flow_specific) if per_dataset else None,
        "flow_specific_tta_profiles_by_dataset": {
            dataset: bool(item["flow_specific_tta_profiles"])
            for dataset, item in sorted(per_dataset.items())
        },
        "dataset_level_profiles": (
            bool(same_selected) if complete_panel else None
        ),
        "dataset_level_profiles_by_dataset": {
            dataset: bool(item["same_profile_for_selected_flows"])
            for dataset, item in sorted(per_dataset.items())
        },
        "same_profile_for_selected_flows": (
            bool(same_selected) if per_dataset else None
        ),
        "same_profile_for_all_five_flows": (
            bool(same_selected) if complete_panel else None
        ),
        "formal_five_flow_panel": bool(complete_panel),
        "by_dataset": per_dataset,
    }


def _audit_dusafe_child_manifest(
    input_path: Path,
    *,
    profile_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit child-run metadata without allowing it to control finalization.

    Older formal runner manifests claim one profile per dataset even when the
    declared flow map varies.  The main-table finalizer must retain the raw
    evidence and report that discrepancy, while deriving A3 from effective
    per-flow configs itself.  This helper is read-only and never repairs the
    child directory in place.
    """

    root = input_path if input_path.is_dir() else input_path.parent
    manifest_path = root / "manifest.json"
    audit: dict[str, Any] = {
        "path": str(manifest_path.resolve()),
        "present": bool(manifest_path.is_file()),
        "authoritative": False,
        "metadata_used_for_finalization": False,
        "child_metadata_ignored": True,
        "child_metadata_stale": False,
        "stale_fields": [],
        "missing_fields": [],
        "issues": [],
        "warnings": [],
        "expected_profile_scope": dict(profile_scope),
    }
    if not manifest_path.is_file():
        audit["issues"] = ["child manifest.json is missing"]
        audit["warnings"] = [
            "Child metadata was unavailable; finalizer uses raw rows, flow profile JSON, and A3 validation."
        ]
        return audit
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        audit["issues"] = [f"child manifest is unreadable: {exc}"]
        audit["warnings"] = [
            "Child metadata was unreadable; finalizer uses raw rows, flow profile JSON, and A3 validation."
        ]
        return audit
    if not isinstance(payload, Mapping):
        audit["issues"] = ["child manifest is not a JSON object"]
        return audit
    audit["observed_profile_scope"] = {
        field: payload.get(field)
        for field in (
            "flow_specific_tta_profiles",
            "dataset_level_profiles",
            "same_profile_for_all_five_flows",
            "same_profile_for_selected_flows",
        )
        if field in payload
    }
    comparable_fields = (
        "flow_specific_tta_profiles",
        "dataset_level_profiles",
        "same_profile_for_all_five_flows",
        "same_profile_for_selected_flows",
    )
    stale_fields: list[str] = []
    missing_fields: list[str] = []
    for field in comparable_fields:
        expected = profile_scope.get(field)
        if expected is None:
            continue
        if field not in payload:
            missing_fields.append(field)
        elif payload.get(field) != expected:
            stale_fields.append(field)
    expected_flow_specific = profile_scope.get("flow_specific_tta_profiles")
    effective_profiles = payload.get("effective_profiles")
    if expected_flow_specific is True and isinstance(effective_profiles, Mapping):
        # The stale runner stored one entry per dataset.  A truthful flow map
        # is keyed by DATASET:source->target (or uses the explicit alias).
        if effective_profiles and not any(":" in str(key) for key in effective_profiles):
            stale_fields.append("effective_profiles")
    if expected_flow_specific is True and "effective_profiles_by_flow" not in payload:
        missing_fields.append("effective_profiles_by_flow")
    audit["stale_fields"] = sorted(set(stale_fields))
    audit["missing_fields"] = sorted(set(missing_fields))
    audit["child_metadata_stale"] = bool(stale_fields or missing_fields)
    audit["metadata_comparison"] = {
        field: {
            "expected": profile_scope.get(field),
            "observed": payload.get(field),
        }
        for field in comparable_fields
        if profile_scope.get(field) is not None or field in payload
    }
    if audit["child_metadata_stale"]:
        audit["warnings"] = [
            "Child profile-scope metadata is stale or incomplete; it is non-authoritative. "
            "The finalizer uses effective per-flow configs for A3 and aggregation."
        ]
    return audit


def _canonicalize_input_frame(frame: pd.DataFrame, *, label: str, default_method: str | None = None) -> pd.DataFrame:
    """Normalize equivalent column spellings emitted by evidence runners."""

    result = frame.copy()
    aliases: dict[str, tuple[str, ...]] = {
        "dataset": ("dataset", "data_set"),
        "scenario": ("scenario", "flow"),
        "src_id": ("src_id", "source_domain"),
        "trg_id": ("trg_id", "target_domain"),
        "source_seed": ("source_seed", "seed"),
        "stream_seed": ("stream_seed", "test_time_seed"),
        "source_model_sha256": ("source_model_sha256", "source_checkpoint_sha256"),
        "source_checkpoint_path": ("source_checkpoint_path", "checkpoint_path"),
        "f1": ("f1", "macro_f1", "post_update_macro_f1", "post_final_f1"),
    }
    for target, candidates in aliases.items():
        if target in result.columns:
            continue
        for candidate in candidates:
            if candidate in result.columns:
                result[target] = result[candidate]
                break
    if "method" not in result.columns and default_method is not None:
        result["method"] = default_method
    if "status" not in result.columns:
        result["status"] = "ok"
    if "stream_seed" not in result.columns:
        result["stream_seed"] = STREAM_SEED
    if "runtime_hparams" not in result.columns:
        result["runtime_hparams"] = pd.NA
    if "source_checkpoint_protocol" not in result.columns:
        result["source_checkpoint_protocol"] = pd.NA
    if "source_checkpoint_file_sha256" not in result.columns:
        result["source_checkpoint_file_sha256"] = pd.NA
    if "source_checkpoint_path" not in result.columns:
        result["source_checkpoint_path"] = pd.NA
    required = [*CORE_REQUIRED_COLUMNS]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise FinalizationError(f"{label}: cannot canonicalize; missing columns {missing}")
    return result


def _filter_methods(frame: pd.DataFrame, methods: Sequence[str], *, label: str) -> tuple[pd.DataFrame, int]:
    """Keep only the requested methods and report excluded legacy rows."""

    requested = {str(method).strip() for method in methods}
    observed = frame["method"].astype(str).str.strip()
    unknown = sorted(set(observed) - requested - {REFERENCE_METHOD})
    if unknown:
        raise FinalizationError(f"{label}: unsupported methods {unknown}")
    excluded = int(observed.eq(REFERENCE_METHOD).sum()) if REFERENCE_METHOD not in requested else 0
    kept = frame.loc[observed.isin(requested)].copy()
    if kept.empty:
        raise FinalizationError(f"{label}: no requested methods remain after filtering")
    return kept, excluded


def _validate_dusafe_current_frame(frame: pd.DataFrame, *, label: str) -> None:
    """Reject known historical DuSafe variants in an explicit current input."""

    methods = set(frame["method"].astype(str).str.strip())
    if methods != {REFERENCE_METHOD}:
        raise FinalizationError(f"{label}: current DuSafe input contains methods {sorted(methods)}")
    for index, row in frame.iterrows():
        row_label = f"{label} row={index}"
        runtime = row.get("runtime_hparams")
        if not _is_missing(runtime):
            config = _parse_mapping(runtime, row_label=row_label)
            execution_mode = str(config.get("dusafe_execution_mode", "")).strip().lower()
            if execution_mode in {"legacy", "old", "historical"}:
                raise FinalizationError(f"{row_label}: legacy DuSafe execution mode is not accepted")
        for column in ("variant", "algorithm_variant", "protocol"):
            if column not in frame.columns or _is_missing(row.get(column)):
                continue
            value = str(row[column]).strip().lower()
            if any(marker in value for marker in ("legacy", "historical", "old_dusafe")):
                raise FinalizationError(f"{row_label}: legacy DuSafe marker in {column}")


def _select_current_dusafe_variant(frame: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, str | None]:
    """Select the single Full variant from a two-variant evidence export."""

    if "variant" not in frame.columns:
        return frame, None
    values = frame["variant"].map(lambda value: "" if _is_missing(value) else str(value).strip().lower())
    observed = {value for value in values if value}
    if len(observed) <= 1:
        return frame, next(iter(observed), None)
    if "full" in observed:
        selected = frame.loc[values.eq("full")].copy()
        if selected.empty:
            raise FinalizationError(f"{label}: Full variant is empty")
        return selected, "full"
    if "no_ssaw" in observed and observed.issubset({"no_ssaw", "current"}):
        selected = frame.loc[values.eq("no_ssaw")].copy()
        return selected, "no_ssaw"
    raise FinalizationError(f"{label}: multiple DuSafe variants are not disambiguated: {sorted(observed)}")


def _validate_dusafe_profile_contract(
    frame: pd.DataFrame,
    *,
    label: str,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate per-flow seed invariance and Full/NoSSAW pair invariance."""

    if frame.empty:
        raise FinalizationError(f"{label}: empty DuSafe profile frame")
    if "method" not in frame.columns:
        raise FinalizationError(f"{label}: method column is required for profile validation")
    errors: list[str] = []
    profile_rows: list[dict[str, Any]] = []
    for (dataset, scenario), group in frame.groupby(["dataset", "scenario"], sort=True):
        by_variant: dict[str, set[str]] = {}
        by_variant_rows: dict[str, list[int]] = {}
        for index, row in group.iterrows():
            variant = str(row.get("variant", "full")).strip().lower()
            if not variant or variant == "nan":
                variant = "full"
            config = _effective_dusafe_config(
                row,
                profiles=profiles,
                row_label=f"{label}/{dataset}/{scenario}/row={index}",
            )
            # A variant-only marker is not a profile difference.  Compare the
            # effective profile after dropping only atomic Full/NoSSAW keys.
            base_json = _canonical_json(_without_atomic_variant(config))
            by_variant.setdefault(variant, set()).add(base_json)
            by_variant_rows.setdefault(variant, []).append(int(index))
        for variant, configs in by_variant.items():
            if len(configs) > 1:
                errors.append(
                    f"{label}: multiple effective TTA configs for {dataset}/{scenario} "
                    f"variant={variant} differ across source seeds; "
                    f"rows={by_variant_rows.get(variant, [])}"
                )
        all_configs = set().union(*by_variant.values()) if by_variant else set()
        if len(all_configs) > 1:
            errors.append(
                f"{label}: {dataset}/{scenario} Full/NoSSAW effective profiles differ beyond atomic variant keys"
            )
        lambda_values = _numeric_values(
            tuple(json.loads(value) for value in all_configs),
            "ssaw_auxiliary_weight",
        )
        if lambda_values and any(value <= 0 for value in lambda_values):
            errors.append(
                f"{label}: {dataset}/{scenario} has non-positive ssaw_auxiliary_weight/lambda"
            )
        selected_config = next(iter(all_configs), "{}")
        profile_rows.append(
            {
                "dataset": str(dataset),
                "scenario": str(scenario),
                "variant_count": int(len(by_variant)),
                "variants": ";".join(sorted(by_variant)),
                "source_seed_count": int(group["source_seed"].nunique()),
                "effective_tta_config": selected_config,
                "effective_tta_config_sha256": hashlib.sha256(selected_config.encode("utf-8")).hexdigest(),
                "profile_source": "flow_profile_json" if profiles and _profile_key(dataset, scenario) in profiles else "runtime_hparams_or_row_columns",
            }
        )
    if errors:
        raise FinalizationError("; ".join(errors[:20]))
    return {
        "passed": True,
        "flow_count": int(len(profile_rows)),
        "variants": sorted(
            {
                str(value).strip().lower()
                for value in frame.get("variant", pd.Series("full", index=frame.index)).dropna()
            }
        ),
        "rows": profile_rows,
    }


def finalize(
    *,
    legacy_input_dir: str | Path = DEFAULT_LEGACY_RAW,
    hhar_input_dir: str | Path = DEFAULT_HHAR_RAW,
    dusafe_input: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT,
    bootstrap_replicates: int = 5000,
    seed: int = 20260820,
) -> dict[str, Any]:
    if int(bootstrap_replicates) < 100:
        raise FinalizationError("bootstrap_replicates must be at least 100")
    legacy_dir = Path(legacy_input_dir).expanduser().resolve()
    hhar_dir = Path(hhar_input_dir).expanduser().resolve()
    output_root = Path(output_dir).resolve()

    legacy_frame, legacy_paths = _load_csv_collection(legacy_dir, label="legacy baseline")
    legacy_frame = _canonicalize_input_frame(legacy_frame, label="legacy baseline")
    legacy_manifest_dir = legacy_dir if legacy_dir.is_dir() else legacy_dir.parent
    legacy_manifest_audit = _audit_legacy_manifest(legacy_manifest_dir, legacy_frame)
    legacy_parts: list[pd.DataFrame] = []
    input_audits: dict[str, Any] = {
        "legacy_manifest": legacy_manifest_audit,
        "legacy_baseline": {
            "paths": [str(path) for path in legacy_paths],
            "raw_rows": int(len(legacy_frame)),
        },
    }
    excluded_legacy_dusafe = 0
    for dataset in ("EEG", "HAR", "FD"):
        subset = legacy_frame[legacy_frame.get("dataset", pd.Series(dtype=object)).astype(str).str.upper().eq(dataset)]
        if subset.empty:
            raise FinalizationError(f"legacy raw table lacks dataset {dataset}")
        subset, excluded = _filter_methods(
            subset,
            BASELINE_METHODS,
            label=f"legacy/{dataset}",
        )
        excluded_legacy_dusafe += excluded
        checked, audit = validate_dataset_frame(
            subset,
            dataset,
            label=f"legacy/{dataset}",
            allow_missing_oom_flag=True,
            methods=BASELINE_METHODS,
        )
        legacy_parts.append(checked)
        input_audits[dataset] = audit

    hhar_frame, hhar_paths = _load_csv_collection(
        hhar_dir,
        label="HHAR baseline",
        recursive_only=(hhar_dir.is_dir() and (hhar_dir / "cells").is_dir()),
    )
    hhar_frame = _canonicalize_input_frame(hhar_frame, label="HHAR baseline")
    hhar_manifest_dir = hhar_dir if hhar_dir.is_dir() else hhar_dir.parent
    # Recursive cells are the new source of truth.  A direct merged CSV keeps
    # the stricter historical queue-completion gate for compatibility.
    hhar_strict_manifest = len(hhar_paths) == 1
    input_audits["HHAR_queue"] = _audit_hhar_queue_manifest(
        hhar_manifest_dir,
        hhar_frame,
        strict_completion=hhar_strict_manifest,
    )
    if hhar_strict_manifest and not input_audits["HHAR_queue"].get("accepted", False):
        raise FinalizationError(
            "HHAR queue is not complete/accepted: "
            + "; ".join(input_audits["HHAR_queue"].get("issues", ()))
        )
    hhar_frame = hhar_frame[hhar_frame["dataset"].astype(str).str.upper().eq("HHAR")].copy()
    if hhar_frame.empty:
        raise FinalizationError("HHAR baseline: no HHAR rows found")
    hhar_frame, excluded_hhar_dusafe = _filter_methods(
        hhar_frame,
        BASELINE_METHODS,
        label="HHAR baseline",
    )
    hhar_checked, hhar_audit = validate_dataset_frame(
        hhar_frame,
        "HHAR",
        label="HHAR baseline",
        allow_missing_oom_flag=True,
        methods=BASELINE_METHODS,
    )
    input_audits["HHAR"] = hhar_audit
    input_audits["HHAR_queue"]["paths"] = [str(path) for path in hhar_paths]
    input_audits["HHAR_queue"]["baseline_rows_after_legacy_dusafe_exclusion"] = int(len(hhar_frame))

    # Prefer the explicit CLI input, then the paper-evidence default when it
    # exists.  The fallback is retained solely for compatibility with the old
    # eleven-method fixture; it is clearly marked non-authoritative in the
    # manifest and is never used when a current input is supplied.
    requested_dusafe_input = dusafe_input
    if requested_dusafe_input is None and DEFAULT_DUSAFE_RAW.exists():
        # Do not let an unrelated/in-progress paper-evidence queue shadow the
        # explicit CLI input or the compatibility fixture.  A completed
        # manifest or a direct per_source export marks the default as ready.
        default_raw = DEFAULT_DUSAFE_RAW / "per_source_seed_results.csv"
        default_ready = default_raw.is_file()
        default_manifest = DEFAULT_DUSAFE_RAW / "manifest.json"
        if not default_ready and default_manifest.is_file():
            try:
                default_payload = json.loads(default_manifest.read_text(encoding="utf-8"))
                default_ready = (
                    default_payload.get("status") in {"complete", "completed"}
                    and int(default_payload.get("expected_cells", 0)) in {60, 120}
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                default_ready = False
        if default_ready:
            requested_dusafe_input = DEFAULT_DUSAFE_RAW
    dusafe_paths: list[Path] = []
    dusafe_source = "explicit_current_input"
    excluded_dusafe_legacy_rows = int(excluded_legacy_dusafe + excluded_hhar_dusafe)
    dusafe_profile_path: Path | None = None
    dusafe_profiles: dict[str, dict[str, Any]] = {}
    dusafe_profile_issues: list[str] = []
    if requested_dusafe_input is not None:
        requested_path = Path(requested_dusafe_input).expanduser().resolve()
        dusafe_frame, dusafe_paths = _load_csv_collection(
            requested_path,
            label="current DuSafe",
        )
        manifest_profile_path = _dusafe_manifest_profile_path(requested_path, dusafe_paths)
        dusafe_profiles, dusafe_profile_path, dusafe_profile_issues = _load_flow_profiles(
            manifest_profile_path or DEFAULT_FLOW_PROFILE_JSON
        )
        child_profile_scope = _profile_scope_summary(dusafe_profiles)
        input_audits["DuSafe_profiles"] = {
            "path": str(dusafe_profile_path) if dusafe_profile_path else None,
            "flow_count": int(len(dusafe_profiles)),
            "issues": list(dusafe_profile_issues),
            "scope": child_profile_scope,
        }
        input_audits["DuSafe_child_manifest"] = _audit_dusafe_child_manifest(
            requested_path,
            profile_scope=child_profile_scope,
        )
        dusafe_frame = _canonicalize_input_frame(
            dusafe_frame,
            label="current DuSafe",
            default_method=REFERENCE_METHOD,
        )
        profile_contract = _validate_dusafe_profile_contract(
            dusafe_frame,
            label="current DuSafe",
            profiles=dusafe_profiles,
        )
        input_audits["DuSafe_profile_contract"] = profile_contract
        dusafe_frame, selected_variant = _select_current_dusafe_variant(
            dusafe_frame,
            label="current DuSafe",
        )
        if selected_variant is not None:
            input_audits["DuSafe_variant"] = {
                "observed": sorted(
                    str(value).strip().lower()
                    for value in dusafe_frame["variant"].dropna().unique()
                ),
                "selected": selected_variant,
            }
        _validate_dusafe_current_frame(dusafe_frame, label="current DuSafe")
    else:
        # Legacy rows are deliberately excluded from both baseline panels.  A
        # compatibility fallback lets the pre-existing unit fixture continue
        # to exercise the full 660-cell path before current evidence lands.
        legacy_dusafe = legacy_frame[legacy_frame["method"].astype(str).str.strip().eq(REFERENCE_METHOD)]
        hhar_raw_frame = _canonicalize_input_frame(
            _load_csv_collection(
                hhar_dir,
                label="HHAR baseline fallback",
                recursive_only=(hhar_dir.is_dir() and (hhar_dir / "cells").is_dir()),
            )[0],
            label="HHAR baseline fallback",
        )
        hhar_dusafe = hhar_raw_frame[hhar_raw_frame["method"].astype(str).str.strip().eq(REFERENCE_METHOD)]
        dusafe_frame = pd.concat([legacy_dusafe, hhar_dusafe], ignore_index=True, sort=False)
        dusafe_source = "legacy_compatibility_fallback"
        if dusafe_frame.empty:
            raise FinalizationError(
                "current DuSafe input is required; pass --dusafe-input or create "
                f"{DEFAULT_DUSAFE_RAW}"
            )
        dusafe_profiles, dusafe_profile_path, dusafe_profile_issues = _load_flow_profiles()
        input_audits["DuSafe_profiles"] = {
            "path": str(dusafe_profile_path) if dusafe_profile_path else None,
            "flow_count": int(len(dusafe_profiles)),
            "issues": list(dusafe_profile_issues),
            "scope": _profile_scope_summary(dusafe_profiles),
        }
        input_audits["DuSafe_profile_contract"] = _validate_dusafe_profile_contract(
            dusafe_frame,
            label="legacy compatibility DuSafe",
            profiles=dusafe_profiles,
        )

    dusafe_parts: list[pd.DataFrame] = []
    for dataset in DATASETS:
        subset = dusafe_frame[dusafe_frame.get("dataset", pd.Series(dtype=object)).astype(str).str.upper().eq(dataset)]
        if subset.empty:
            raise FinalizationError(f"current DuSafe input lacks dataset {dataset}")
        checked, audit = validate_dataset_frame(
            subset,
            dataset,
            label=f"current DuSafe/{dataset}",
            allow_missing_oom_flag=True,
            methods=(REFERENCE_METHOD,),
            dusafe_profiles=dusafe_profiles,
        )
        dusafe_parts.append(checked)
        input_audits[f"DuSafe/{dataset}"] = audit

    merged = pd.concat([*legacy_parts, hhar_checked, *dusafe_parts], ignore_index=True, sort=False)
    expected = {
        (dataset, scenario, method, seed, STREAM_SEED)
        for dataset in DATASETS
        for scenario in expected_flows(dataset)
        for method in METHODS
        for seed in SOURCE_SEEDS
    }
    keys = [_key_tuple(row) for row in merged.to_dict("records")]
    if len(set(keys)) != len(keys):
        duplicate_keys = sorted(key for key in set(keys) if keys.count(key) > 1)[:5]
        raise FinalizationError(f"merged four-dataset raw table contains duplicate keys: {duplicate_keys}")
    if set(keys) != expected:
        missing_keys = sorted(expected - set(keys))[:5]
        extra_keys = sorted(set(keys) - expected)[:5]
        raise FinalizationError(
            "merged four-dataset raw table key set mismatch; "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    hash_errors = _validate_key_source_hashes(merged, label="merged four-dataset raw table")
    if hash_errors:
        raise FinalizationError("; ".join(hash_errors[:20]))
    aggregate = dataset_method_aggregate(merged, bootstrap_replicates=bootstrap_replicates, seed=seed)
    flow_table = per_flow_macro_f1(merged)
    a3_flow, a3_summary = dusafe_a3_tables(merged, profiles=dusafe_profiles)
    # A3 is built from the validated merged rows and supersedes any
    # dataset-level child-run summary.  Keep this scope in the authoritative
    # manifest so downstream readers can reconstruct the exact flow unit.
    a3_profile_map = {
        _profile_key(row["dataset"], row["scenario"]): json.loads(
            str(row["effective_tta_config"])
        )
        for row in a3_flow.to_dict("records")
    }
    authoritative_profile_scope = _profile_scope_summary(a3_profile_map)
    child_manifest_audit = input_audits.get("DuSafe_child_manifest")
    if isinstance(child_manifest_audit, dict):
        child_manifest_audit["authoritative_profile_scope"] = authoritative_profile_scope
        child_manifest_audit["effective_profile_source"] = (
            "validated_merged_rows_and_flow_profile_json"
        )
    inference = paired_source_seed_domain_inference(
        merged,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "merged_raw": "merged_per_source_seed_results.csv",
        "raw_merged": "merged_per_source_seed_results.csv",
        "raw_alias": "per_source_seed_results.csv",
        "dataset_method_aggregate": "dataset_method_aggregate.csv",
        "main_dataset_average": "main_dataset_average.csv",
        "main_table": "main_table.csv",
        "A1": "a1_per_flow.csv",
        "a1_per_flow": "a1_per_flow.csv",
        "per_flow_macro_f1": "per_flow_macro_f1.csv",
        "A3": "a3_dusafe_flow_hparams.csv",
        "a3_flow_hparams": "a3_dusafe_flow_hparams.csv",
        "a3_hparams_summary": "a3_dusafe_hparams_summary.csv",
        "paired_inference": "paired_source_seed_domain_inference.csv",
        "manifest": "manifest.json",
    }
    _atomic_write_csv(merged.sort_values(list(KEY_COLUMNS), kind="stable"), output_root / outputs["merged_raw"])
    _atomic_write_csv(merged.sort_values(list(KEY_COLUMNS), kind="stable"), output_root / outputs["raw_alias"])
    _atomic_write_csv(aggregate, output_root / outputs["dataset_method_aggregate"])
    _atomic_write_csv(aggregate, output_root / outputs["main_dataset_average"])
    _atomic_write_csv(aggregate, output_root / outputs["main_table"])
    _atomic_write_csv(flow_table, output_root / outputs["a1_per_flow"])
    _atomic_write_csv(flow_table, output_root / outputs["per_flow_macro_f1"])
    _atomic_write_csv(a3_flow, output_root / outputs["a3_flow_hparams"])
    _atomic_write_csv(a3_summary, output_root / outputs["a3_hparams_summary"])
    _atomic_write_csv(inference, output_root / outputs["paired_inference"])
    warnings = [
        "All results are descriptive_only because dataset-level target labels were used for frozen-profile selection.",
        "The legacy manifest is not authoritative: it omits two datasets and ten methods despite 495 valid raw rows.",
    ]
    if excluded_dusafe_legacy_rows:
        warnings.append(
            f"Excluded {excluded_dusafe_legacy_rows} legacy DuSafe rows from baseline inputs; "
            "only current DuSafe evidence is eligible for the merged table."
        )
    if dusafe_source == "legacy_compatibility_fallback":
        warnings.append(
            "No current DuSafe input was supplied; legacy DuSafe rows were used only as a compatibility fallback. "
            "This fallback is not current DuSafe evidence."
        )
    for audit in input_audits.values():
        if isinstance(audit, Mapping):
            warnings.extend(str(value) for value in audit.get("warnings", ()))
            warnings.extend(str(value) for value in audit.get("issues", ()))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "decision_status": "descriptive_only",
        "confirmatory": False,
        "confirmatory_results": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_metrics": True,
        "evaluation_partition": "target_selected_evaluation",
        "selection_overlap": True,
        "parameter_selection_data_overlap": True,
        "fixed_source": True,
        "dusafe_source": dusafe_source,
        "dusafe_input_paths": [str(path) for path in dusafe_paths],
        "dusafe_flow_profile_json": str(dusafe_profile_path) if dusafe_profile_path else None,
        "dusafe_flow_profile_count": int(len(dusafe_profiles)),
        "dusafe_flow_profile_issues": list(dusafe_profile_issues),
        # The child runner is evidence provenance only.  Its older manifest
        # may claim one profile per dataset; the finalizer's validated A3
        # flow table is authoritative for effective runtime configuration.
        "dusafe_child_manifest_authoritative": False,
        "dusafe_child_metadata_stale": bool(
            isinstance(child_manifest_audit, Mapping)
            and child_manifest_audit.get("child_metadata_stale", False)
        ),
        "dusafe_effective_profile_scope": authoritative_profile_scope,
        "flow_specific_tta_profiles": authoritative_profile_scope.get(
            "flow_specific_tta_profiles"
        ),
        "dataset_level_profiles": authoritative_profile_scope.get(
            "dataset_level_profiles"
        ),
        "same_profile_for_all_five_flows": authoritative_profile_scope.get(
            "same_profile_for_all_five_flows"
        ),
        "legacy_input_paths": [str(path) for path in legacy_paths],
        "hhar_input_paths": [str(path) for path in hhar_paths],
        "source_checkpoint_independent_unit": "dataset/source_domain/source_seed",
        "datasets": list(DATASETS),
        "flows_by_dataset": {dataset: list(expected_flows(dataset)) for dataset in DATASETS},
        "methods": list(METHODS),
        "baseline_methods": list(BASELINE_METHODS),
        "dusafe_method": REFERENCE_METHOD,
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "expected_cells": 660,
        "observed_cells": int(len(merged)),
        "baseline_rows": int(len(legacy_parts[0]) + len(legacy_parts[1]) + len(legacy_parts[2]) + len(hhar_checked)),
        "dusafe_rows": int(sum(len(part) for part in dusafe_parts)),
        "legacy_dusafe_rows_excluded": excluded_dusafe_legacy_rows,
        "source_model_sha256_validation": {
            "unit": "dataset/flow/source_seed/stream_seed across all methods",
            "passed": True,
            "checked_keys": int(len(expected) // len(METHODS)),
        },
        "aggregation": {
            "main_table": "mean five formal flows within each source seed, then mean/std over source seeds",
            "A1": "mean/std over the three source seeds for each formal flow",
            "std_ddof": 1,
        },
        "A3": {
            "flow_rows": int(len(a3_flow)),
            "dataset_summary_rows": int(len(a3_summary)),
            "profile_unit": "dataset/scenario; source-seed invariant effective TTA config",
            "full_nossaw_pair_rule": "only atomic variant keys may differ",
        },
        "input_audits": input_audits,
        "legacy_manifest_authoritative": False,
        "warnings": sorted(set(warnings)),
        "inference": {
            "unit": "source_domain x source_seed cluster; flows averaged within cluster",
            "bootstrap": "cluster bootstrap 95% CI",
            "paired_test": "two-sided exact sign-flip over source-domain/seed clusters",
            "holm_correction": "all dataset x baseline-method comparisons",
            "replicates": int(bootstrap_replicates),
            "seed": int(seed),
        },
        "outputs": outputs,
        "created_utc": _utc_now(),
    }
    _atomic_write_json(manifest, output_root / outputs["manifest"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--legacy-input-dir", default=str(DEFAULT_LEGACY_RAW))
    parser.add_argument("--hhar-input-dir", default=str(DEFAULT_HHAR_RAW))
    parser.add_argument(
        "--dusafe-input",
        default=None,
        help=(
            "Current DuSafe Full raw CSV or directory. Defaults to "
            f"{DEFAULT_DUSAFE_RAW} when it exists."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = finalize(
            legacy_input_dir=args.legacy_input_dir,
            hhar_input_dir=args.hhar_input_dir,
            dusafe_input=args.dusafe_input,
            output_dir=args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    except (FinalizationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[four-dataset main-table finalizer] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": manifest["status"], "observed_cells": manifest["observed_cells"], "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_METHODS",
    "DATASETS",
    "DEFAULT_DUSAFE_RAW",
    "DEFAULT_FLOW_PROFILE_JSON",
    "FinalizationError",
    "METHODS",
    "PROTOCOL_VERSION",
    "SOURCE_SEEDS",
    "STREAM_SEED",
    "dataset_method_aggregate",
    "dusafe_a3_tables",
    "expected_flows",
    "expected_key_set",
    "finalize",
    "paired_source_seed_domain_inference",
    "per_flow_macro_f1",
    "validate_dataset_frame",
]
