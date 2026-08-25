"""CPU-only, fail-closed finalizer for the paper-evidence v5 bundle.

This module deliberately does not import a trainer or start an experiment.  It
consumes completed cell summaries, worker specifications and CSV exports,
checks the fixed-source/logging contracts, and writes a new v5 bundle.  The
v3/v4 finalizers are historical and are not modified or called here.

The v5 contract differs from the old bundles in three ways that matter at the
merge boundary:

* source checkpoints are checked against the flow-wise source profile and the
  dataset-specific cache root (HHAR uses ``hhar_formal``);
* source metadata context is a required, canonical hash of the flowwise
  deployment context (source tensor digest, deployment batch size, confidence
  keep fraction and stream seed), never ``source_config``;
* production panels and evidence panels have separate logging contracts, with
  safety explicitly requiring graph-off execution.

All target-selected results remain descriptive.  ``confirmatory`` is always
false in the output manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "paper_evidence_protocol_v5.json"
DEFAULT_ROOT = ROOT / "results" / "paper_evidence_v5"
DEFAULT_OUTPUT = DEFAULT_ROOT / "final"
DEFAULT_OLD_V4_ROOT = ROOT / "results" / "paper_evidence_v4"
DEFAULT_EFFICIENCY_ROOT = (
    ROOT / "results" / "efficiency_optimization_v5" / "formal_all_baselines_har_12to16_v4"
)

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MAIN_VARIANTS = ("confidence_only", "hard_ssaw")
CORE_VARIANTS = (
    "accept_all_raw",
    "confidence_only",
    "random_eligible_spline",
    "hard_ssaw",
)
SAFETY_VARIANTS = ("full", "no_ssaw")
EFFICIENCY_METHODS = (
    "NoAdap", "Tent", "EATA", "SAR", "ACCUPOfficial", "CoTTA",
    "SoTTA", "RoTTA", "COME", "NOTE", "DuSafe",
)
SOURCE_SEEDS = (0, 1, 2)
STREAM_SEED = 42
MAIN_KEYS = ("dataset", "scenario", "source_seed", "runner")
SAFETY_KEYS = (
    "dataset", "scenario", "corruption", "severity", "source_seed", "stream_seed", "variant"
)


class EvidenceError(RuntimeError):
    """Raised when a v5 evidence contract is not satisfied."""


def _json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise EvidenceError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON artifact is not an object: {path}")
    return value


def _csv(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    if not Path(path).is_file():
        raise EvidenceError(f"missing CSV artifact: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise EvidenceError(f"invalid CSV artifact {path}: {exc}") from exc
    if frame.empty and not allow_empty:
        raise EvidenceError(f"empty CSV artifact: {path}")
    return frame


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def production_code_sha256() -> str:
    """Return the current production digest without importing torch/trainers."""

    files = (
        ROOT / "scripts" / "run_final_ssaw_full_no_ssaw_five_flow.py",
        ROOT / "algorithms" / "dusafe.py",
        ROOT / "algorithms" / "dusafe_spline_hard_view.py",
        ROOT / "algorithms" / "get_tta_class.py",
        ROOT / "configs" / "tta_hparams_new.py",
        ROOT / "configs" / "formal_evaluation_protocol.py",
    )
    digest = hashlib.sha256()
    for path in files:
        _require(path.is_file(), f"missing production file: {path}")
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


ABLATION_CODE_FILES = (
    ROOT / "algorithms" / "representative_causal_ablation.py",
    ROOT / "algorithms" / "dusafe_replacement_ablation.py",
    ROOT / "algorithms" / "dusafe_augmentation_controls.py",
    ROOT / "algorithms" / "dusafe_direct_ablation.py",
    ROOT / "algorithms" / "dusafe_two_factor_ablation.py",
    ROOT / "scripts" / "run_dusafe_replacement_ablation.py",
)


def ablation_code_sha256() -> str:
    """Digest the replacement/control implementation independently."""

    digest = hashlib.sha256()
    for path in ABLATION_CODE_FILES:
        _require(path.is_file(), f"missing ablation code file: {path}")
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dependency_scoped_code_sha256(files: Iterable[Any]) -> str:
    """Hash only the files that can affect one registered evidence panel."""

    digest = hashlib.sha256()
    for value in files:
        path = _norm_path(ROOT / str(value))
        _require(path.is_file(), f"missing dependency-scoped code file: {path}")
        _require(ROOT in path.parents, f"dependency-scoped code file is outside repository: {path}")
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _registered_ablation_artifact_digest(
    protocol: Mapping[str, Any], scope_name: str
) -> str:
    """Validate a current dependency scope and return its signed old digest.

    Core/augmentation artifacts were signed with an older broad digest that
    also included unrelated evidence-only files.  The migration is explicit:
    current files in the actual dependency scope are pinned independently,
    while the artifact must retain the registered legacy broad digest.  This
    does not waive a code check and cannot be used for causal evidence, whose
    current causal digest remains mandatory.
    """

    scopes = protocol.get("ablation_code_scopes", {})
    spec = scopes.get(scope_name) if isinstance(scopes, Mapping) else None
    if not isinstance(spec, Mapping):
        return ablation_code_sha256()
    observed = dependency_scoped_code_sha256(spec.get("dependency_files", ()))
    expected = str(spec.get("current_scoped_sha256", ""))
    _require(SHA256_RE.fullmatch(expected), f"{scope_name} scoped digest is not registered")
    _require(observed == expected, f"{scope_name} current dependency-scoped digest mismatch")
    if scope_name == "core":
        required_mode = str(spec.get("required_random_selection_mode", ""))
        source = (ROOT / "algorithms" / "representative_causal_ablation.py").read_text(
            encoding="utf-8"
        )
        pattern = re.compile(
            r"class\s+RepresentativeRandomEligibleSpline\b[\s\S]*?"
            r"spline_selection_mode\s*=\s*[\"']([^\"']+)[\"']",
        )
        match = pattern.search(source)
        _require(
            match is not None and match.group(1) == required_mode,
            "core random-view selection mode does not match the registered scope",
        )
    legacy = str(spec.get("accepted_legacy_broad_sha256", ""))
    _require(SHA256_RE.fullmatch(legacy), f"{scope_name} legacy artifact digest is invalid")
    return legacy


CAUSAL_EVIDENCE_CODE_FILES = (
    ROOT / "algorithms" / "representative_causal_ablation.py",
    ROOT / "scripts" / "counterfactual_horizon_common.py",
    ROOT / "scripts" / "run_representative_causal_ablation.py",
)


def causal_evidence_code_sha256() -> str:
    """Digest the causal fork/state-replay implementation independently."""

    digest = hashlib.sha256()
    for path in CAUSAL_EVIDENCE_CODE_FILES:
        _require(path.is_file(), f"missing causal evidence code file: {path}")
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _norm_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _expand_path_values(values: Iterable[Any] | None) -> list[Path]:
    """Expand repeated or comma-separated directory arguments deterministically."""

    expanded: list[Path] = []
    for value in values or ():
        for item in str(value).split(","):
            item = item.strip()
            if item:
                expanded.append(_norm_path(item))
    return expanded


def _flow_key(dataset: Any, scenario: Any) -> str:
    return f"{str(dataset).upper()}:{str(scenario).replace('→', '->')}"


def _load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    payload = _json(path)
    profiles: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        dataset = value.get("dataset")
        flow = value.get("flow")
        if dataset and isinstance(flow, (list, tuple)) and len(flow) == 2:
            scenario = f"{flow[0]}->{flow[1]}"
            profiles[_flow_key(dataset, scenario)] = value
        elif isinstance(key, str) and ":" in key:
            profiles[key.replace("→", "->")] = value
    _require(profiles, f"source profile has no usable flow entries: {path}")
    return profiles


def _load_source_reference(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load seed-specific source hashes, paths, and metadata contexts."""

    frame = _csv(path)
    required = {
        "dataset", "scenario", "source_seed", "source_model_sha256",
        "source_checkpoint_path", "source_metadata_context_sha256",
    }
    _require(
        required.issubset(frame.columns),
        f"source reference missing columns: {sorted(required - set(frame.columns))}",
    )
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.upper()
    frame["scenario"] = frame["scenario"].astype(str).str.replace("→", "->", regex=False)
    frame["source_seed"] = frame["source_seed"].astype(int)
    _require(
        frame["source_metadata_context_sha256"].astype(str).str.fullmatch(
            SHA256_RE.pattern
        ).all(),
        "source reference contains an invalid metadata context SHA-256",
    )
    _require(
        not frame.duplicated(["dataset", "scenario", "source_seed"]).any(),
        "source reference has duplicate identities",
    )
    return {
        (str(row["dataset"]), str(row["scenario"]), int(row["source_seed"])): row
        for row in frame.to_dict("records")
    }


def _profile_context(profile: Mapping[str, Any]) -> str:
    source_config = profile.get("source_config")
    _require(isinstance(source_config, Mapping), "source profile lacks source_config")
    expected = str(profile.get("source_config_sha256", ""))
    computed = _canonical_hash(source_config)
    if expected:
        _require(expected == computed, "source profile source_config_sha256 is invalid")
    return computed


def _parse_runtime_hparams(value: Any) -> dict[str, Any]:
    """Parse a worker/runtime config without importing the trainer."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _metadata_context_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the flowwise deployment metadata identity.

    ``source_config_sha256`` identifies pretraining and is deliberately not
    used here.  The flowwise runner defines the context as the source tensor
    identity plus deployment batch size, confidence keep fraction and stream
    seed.  Rebuilding this payload from the row/spec makes a copied context
    string insufficient to pass finalization.
    """

    runtime = _parse_runtime_hparams(row.get("runtime_hparams", ""))
    nested = runtime.get("tta_config")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update({key: value for key, value in runtime.items() if key != "tta_config"})
        runtime = merged
    source_hash = str(row.get("source_model_sha256", "")).strip()
    _require(SHA256_RE.fullmatch(source_hash), "metadata context source model hash is invalid")

    def first(*names: str) -> Any:
        for name in names:
            value = row.get(name)
            if value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() not in {"", "nan", "None"}:
                return value
            value = runtime.get(name)
            if value is not None and str(value).strip() not in {"", "nan", "None"}:
                return value
        return None

    batch = first("deployment_batch_size", "batch_size", "tta_batch_size")
    keep = first("confidence_keep_fraction", "confidence_fraction", "keep_fraction")
    stream_seed = first("stream_seed")
    _require(batch is not None, "metadata context lacks deployment batch size")
    _require(keep is not None, "metadata context lacks confidence keep fraction")
    _require(stream_seed is not None, "metadata context lacks stream seed")
    try:
        batch_int = int(batch)
        keep_float = float(keep)
        stream_int = int(stream_seed)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("metadata context has non-numeric deployment fields") from exc
    _require(batch_int > 0, "metadata context batch size must be positive")
    _require(math.isfinite(keep_float) and 0.0 < keep_float <= 1.0, "metadata context keep fraction is invalid")
    return {
        "source_model_sha256": source_hash,
        "deployment_batch_size": batch_int,
        "confidence_keep_fraction": keep_float,
        "stream_seed": stream_int,
    }


def _metadata_context_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_hash(_metadata_context_payload(row))


def _load_worker_specs(root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    specs: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in sorted(Path(root).rglob("worker_spec.json")):
        spec = _json(path)
        dataset = str(spec.get("dataset", "")).upper()
        scenario = str(spec.get("scenario", ""))
        flow = spec.get("flow")
        if not scenario and isinstance(flow, (list, tuple)) and len(flow) == 2:
            scenario = f"{flow[0]}->{flow[1]}"
        runner = str(spec.get("runner", spec.get("variant", "")))
        if not dataset or not scenario or not runner or "source_seed" not in spec:
            continue
        key = (dataset, scenario.replace("→", "->"), int(spec["source_seed"]), runner)
        _require(key not in specs, f"duplicate worker spec key: {key}")
        specs[key] = spec
    return specs


def _summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = _load_worker_specs(root)
    for path in sorted(Path(root).rglob("summary.json")):
        payload = _json(path)
        row = dict(payload)
        dataset = str(row.get("dataset", "")).upper()
        scenario = str(row.get("scenario", "")).replace("→", "->")
        runner = str(row.get("runner", row.get("variant", "")))
        if "source_seed" not in row:
            continue
        key = (dataset, scenario, int(row["source_seed"]), runner)
        spec = specs.get(key, {})
        _merge_worker_metadata(row, spec)
        rows.append(row)
    return rows


def _merge_worker_metadata(row: dict[str, Any], spec: Mapping[str, Any]) -> None:
    """Fill metadata absent from compact summaries using their worker spec."""

    for field in ("source_checkpoint_path", "source_model_sha256"):
        if not row.get(field):
            candidate = spec.get(
                "expected_source_model_sha256" if field == "source_model_sha256" else field
            )
            if candidate:
                row[field] = candidate
    source_config = spec.get("source_config")
    if source_config is not None:
        row.setdefault("source_config", source_config)
    context = (
        row.get("source_metadata_context_sha256")
        or spec.get("source_metadata_context_sha256")
    )
    if context:
        row.setdefault("source_metadata_context_sha256", context)
    tta = spec.get("tta_config")
    if isinstance(tta, Mapping):
        for field in (
            "dusafe_logging_mode", "candidate_cuda_graph_requested_mode",
            "candidate_cuda_graph_enabled", "candidate_cuda_graph_mode",
        ):
            if field in tta:
                row.setdefault(field, tta[field])
        if not row.get("runtime_hparams"):
            row["runtime_hparams"] = json.dumps(tta, sort_keys=True)
    # The source metadata context is not source_config_sha256.  Reconstruct
    # the flowwise deployment identity only after source hash/runtime fields
    # have been merged from the worker spec.
    try:
        derived_context = _metadata_context_sha256(row)
    except EvidenceError:
        derived_context = None
    if derived_context:
        existing = row.get("source_metadata_context_sha256")
        if existing and str(existing) not in {"nan", "None", ""}:
            # Preserve the recorded value; the validator will compare it to
            # the independently reconstructed digest and fail closed on any
            # mismatch.
            row.setdefault("recorded_source_metadata_context_sha256", str(existing))
        row["source_metadata_context_sha256"] = str(existing or derived_context)


def _read_panel(root: Path, *, raw_name: str = "raw.csv") -> pd.DataFrame:
    root = Path(root)
    raw_path = root / raw_name
    if raw_path.is_file():
        frame = _csv(raw_path)
        specs = _load_worker_specs(root)
        if specs:
            # Keep the CSV schema fixed while enriching rows.  Columns added
            # for the first row must not appear as explicit ``None`` values in
            # later row dictionaries, because that would mask worker-spec
            # values from ``setdefault`` in _merge_worker_metadata.
            base_columns = list(frame.columns)
            for index in frame.index:
                row = frame.loc[index, base_columns]
                key = (
                    str(row.get("dataset", "")).upper(),
                    str(row.get("scenario", "")).replace("→", "->"),
                    int(row.get("source_seed", -1)),
                    str(row.get("runner", row.get("variant", ""))),
                )
                augmented = dict(row.to_dict())
                _merge_worker_metadata(augmented, specs.get(key, {}))
                for field, value in augmented.items():
                    if isinstance(value, Mapping):
                        value = json.dumps(value, sort_keys=True)
                    elif isinstance(value, (list, tuple)):
                        value = json.dumps(list(value), sort_keys=True)
                    if field not in frame.columns:
                        frame[field] = pd.Series(
                            [None] * len(frame), index=frame.index, dtype=object
                        )
                    elif isinstance(value, str) and frame[field].dtype != object:
                        # Worker metadata may fill an otherwise-all-null CSV
                        # column with canonical JSON or a path/hash string.
                        frame[field] = frame[field].astype(object)
                    current = frame.at[index, field]
                    if pd.isna(current) or not str(current).strip():
                        frame.at[index, field] = value
        return frame
    rows = _summary_rows(root)
    _require(rows, f"no {raw_name} or summary.json artifacts under {root}")
    return pd.DataFrame(rows)


def _expected_keys(datasets: Iterable[str], flows: Mapping[str, list[str]], variants: Iterable[str]) -> set[tuple[Any, ...]]:
    return {
        (dataset, scenario, seed, variant)
        for dataset in datasets
        for scenario in flows[dataset]
        for seed in SOURCE_SEEDS
        for variant in variants
    }


def _check_flags(frame: pd.DataFrame, manifest: Mapping[str, Any], label: str) -> None:
    for field, expected in (
        ("target_labels_used_for_online_decision", False),
        ("confirmatory", False),
    ):
        if field in frame.columns:
            values = frame[field].astype(str).str.lower()
            _require(values.isin({str(expected).lower(), "0" if not expected else "1"}).all(), f"{label} {field} mismatch")
        if field in manifest:
            _require(bool(manifest[field]) is expected, f"{label} manifest {field} mismatch")
    if "target_labels_used_for_parameter_selection" in frame.columns:
        _require(frame["target_labels_used_for_parameter_selection"].astype(str).str.lower().isin({"true", "1"}).all(), f"{label} parameter-selection flag mismatch")
    _require(manifest.get("confirmatory") is False, f"{label} is marked confirmatory")
    _require(manifest.get("target_labels_used_for_online_decision") is not True, f"{label} uses online target labels")


def _require_target_selected_descriptive(
    manifest: Mapping[str, Any], *, label: str
) -> None:
    """Normalize the v5 descriptive marker, including a narrow legacy form.

    Early v5 manifests did not write either descriptive alias, but did record
    the equivalent parameter-selection facts.  Accept that form only when
    both flags are exact JSON booleans with the safe values; a missing,
    string-valued, or contradictory flag remains fail-closed.
    """

    if (
        manifest.get("target_selected_descriptive") is True
        or manifest.get("descriptive_target_selected_evaluation") is True
    ):
        return
    legacy_equivalent = (
        manifest.get("confirmatory") is False
        and manifest.get("target_labels_used_for_parameter_selection") is True
    )
    _require(
        legacy_equivalent,
        f"{label} is not marked target-selected descriptive",
    )


def _check_cache_root(path: str, root: Path, key: tuple[Any, ...]) -> None:
    try:
        _norm_path(path).relative_to(_norm_path(root))
    except ValueError as exc:
        raise EvidenceError(f"source checkpoint is outside registered cache root for {key}: {path}") from exc


def _validate_source_identity(
    frame: pd.DataFrame,
    *,
    label: str,
    manifest: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    source_reference: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
    cache_roots: Mapping[str, str],
    variants: set[str],
    expected_datasets: tuple[str, ...],
    flows: Mapping[str, list[str]],
    required_ablation_code_sha256: str | None = None,
) -> pd.DataFrame:
    required = {"dataset", "scenario", "source_seed", "runner", "f1", "status", "production_code_sha256", "source_model_sha256", "source_checkpoint_path"}
    _require(required.issubset(frame.columns), f"{label} missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.upper()
    frame["scenario"] = frame["scenario"].astype(str).str.replace("→", "->", regex=False)
    frame["source_seed"] = frame["source_seed"].astype(int)
    frame["runner"] = frame["runner"].astype(str)
    expected = _expected_keys(expected_datasets, flows, variants)
    observed = set(zip(frame["dataset"], frame["scenario"], frame["source_seed"], frame["runner"]))
    _require(len(frame) == len(expected), f"{label} row count mismatch: {len(frame)} != {len(expected)}")
    _require(observed == expected, f"{label} key set mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    _require(not frame.duplicated(list(MAIN_KEYS)).any(), f"{label} duplicate cell keys")
    _require(frame["status"].astype(str).eq("ok").all(), f"{label} contains failed rows")
    current_hash = production_code_sha256()
    _require(frame["production_code_sha256"].astype(str).eq(current_hash).all(), f"{label} production hash mismatch")
    if required_ablation_code_sha256 is not None:
        _require("ablation_code_sha256" in frame.columns, f"{label} missing ablation code digest")
        _require(frame["ablation_code_sha256"].astype(str).eq(required_ablation_code_sha256).all(), f"{label} ablation code digest mismatch")
    _check_flags(frame, manifest, label)
    if "dusafe_logging_mode" not in frame.columns:
        frame["dusafe_logging_mode"] = np.nan
    for index, row in frame.iterrows():
        dataset = str(row.dataset)
        scenario = str(row.scenario)
        key = (dataset, scenario, int(row.source_seed), str(row.runner))
        profile = profiles.get(_flow_key(dataset, scenario))
        _require(profile is not None, f"{label} missing source profile: {dataset}:{scenario}")
        _require(source_reference is not None, f"{label} lacks seed-specific source reference")
        reference = (source_reference or {}).get((dataset, scenario, int(row.source_seed)))
        _require(reference is not None, f"{label} missing seed-specific source reference: {key}")
        expected_hash = str(reference.get("source_model_sha256", ""))
        expected_path = str(reference.get("source_checkpoint_path", ""))
        expected_context = str(reference.get("source_metadata_context_sha256", ""))
        _require(SHA256_RE.fullmatch(expected_context), f"{label} source reference context is invalid: {key}")
        _require(SHA256_RE.fullmatch(str(row.source_model_sha256)), f"{label} invalid source model hash: {key}")
        _require(str(row.source_model_sha256) == expected_hash, f"{label} source model hash mismatch: {key}")
        _require(str(row.source_checkpoint_path) == expected_path, f"{label} source checkpoint path mismatch: {key}")
        context = str(row.get("source_metadata_context_sha256", ""))
        _require(SHA256_RE.fullmatch(context), f"{label} source metadata context is invalid: {key}")
        _require(_metadata_context_sha256(row) == context, f"{label} reconstructed source metadata context mismatch: {key}")
        _check_cache_root(str(row.source_checkpoint_path), _norm_path(cache_roots[dataset]), key)
        mode = str(row.get("dusafe_logging_mode", ""))
        _require(mode == "production", f"{label} requires production logging: {key}, got {mode!r}")
        frame.at[index, "source_metadata_context_sha256"] = context
        frame.at[index, "source_reference_metadata_context_sha256"] = expected_context
    for key, group in frame.groupby(["dataset", "scenario", "source_seed"]):
        _require(group["source_model_sha256"].astype(str).nunique() == 1, f"{label} variant source mismatch: {key}")
        _require(group["source_metadata_context_sha256"].astype(str).nunique() == 1, f"{label} variant context mismatch: {key}")
    return frame


def _validate_main_or_core(
    root: Path,
    *,
    label: str,
    manifest_name: str,
    protocol: Mapping[str, Any],
    datasets: tuple[str, ...],
    variants: tuple[str, ...],
    expected_count: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _json(Path(root) / manifest_name)
    _require(manifest.get("status") == "complete", f"{label} is incomplete")
    _require(str(manifest.get("protocol", "")).startswith("paper_evidence_v5_"), f"{label} is not a paper_evidence_v5 artifact")
    _require(manifest.get("confirmatory") is False, f"{label} confirmatory flag is not false")
    _require_target_selected_descriptive(manifest, label=label)
    _require(manifest.get("source_seeds") == list(SOURCE_SEEDS), f"{label} source seed mismatch")
    _require(str(manifest.get("production_code_sha256", "")) == production_code_sha256(), f"{label} manifest production hash mismatch")
    _require(str(manifest.get("tta_profile_json", "")).replace("\\", "/").endswith("configs/paper_flow_profiles_v1.json"), f"{label} TTA profile mismatch")
    _require(str(manifest.get("logging_mode", "production")) == "production", f"{label} manifest logging mode mismatch")
    required_ablation = (
        _registered_ablation_artifact_digest(protocol, "core")
        if label.startswith("core")
        else None
    )
    if required_ablation is not None:
        _require(str(manifest.get("ablation_code_sha256", "")) == required_ablation, f"{label} manifest ablation code digest mismatch")
    frame = _read_panel(Path(root))
    frame = _validate_source_identity(
        frame,
        label=label,
        manifest=manifest,
        profiles=protocol["_profiles"],
        source_reference=protocol.get("_source_reference"),
        cache_roots=protocol["source_cache_roots"],
        variants=set(variants),
        expected_datasets=datasets,
        flows=protocol["formal_flows"],
        required_ablation_code_sha256=required_ablation,
    )
    _require(len(frame) == expected_count, f"{label} expected {expected_count} cells")
    return frame, manifest


def _panel_datasets_for_dir(path: Path, manifest: Mapping[str, Any], all_datasets: Iterable[str]) -> tuple[str, ...]:
    declared = manifest.get("datasets")
    if isinstance(declared, list) and declared:
        return tuple(str(value).upper() for value in declared)
    name = Path(path).name.lower()
    if "hhar" in name:
        return ("HHAR",)
    if "nonhhar" in name:
        return tuple(dataset for dataset in all_datasets if dataset != "HHAR")
    return tuple(str(value).upper() for value in all_datasets)


def _validate_multi_panel(
    dirs: Iterable[Path],
    *,
    label: str,
    protocol: Mapping[str, Any],
    all_datasets: tuple[str, ...],
    variants: tuple[str, ...],
    expected_total: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = [_norm_path(path) for path in dirs]
    _require(paths, f"{label} has no input directories")
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        manifest = _json(path / "manifest.json")
        datasets = _panel_datasets_for_dir(path, manifest, all_datasets)
        expected = sum(len(protocol["formal_flows"][dataset]) * len(SOURCE_SEEDS) * len(variants) for dataset in datasets)
        frame, validated_manifest = _validate_main_or_core(
            path,
            label=f"{label}[{index}]",
            manifest_name="manifest.json",
            protocol=protocol,
            datasets=datasets,
            variants=variants,
            expected_count=expected,
        )
        frames.append(frame)
        manifests.append(validated_manifest)
    merged = pd.concat(frames, ignore_index=True)
    key_columns = list(MAIN_KEYS)
    _require(not merged.duplicated(key_columns).any(), f"{label} duplicate keys across input directories")
    expected_keys = _expected_keys(all_datasets, protocol["formal_flows"], variants)
    observed_keys = set(zip(merged["dataset"], merged["scenario"], merged["source_seed"], merged["runner"]))
    _require(len(merged) == expected_total and observed_keys == expected_keys, f"{label} combined key set/count mismatch")
    combined_manifest = dict(manifests[0])
    combined_manifest["input_manifests"] = manifests
    combined_manifest["input_dirs"] = [str(path) for path in paths]
    return merged, combined_manifest


def _validate_core_logging(frame: pd.DataFrame) -> None:
    _require(frame["dusafe_logging_mode"].astype(str).eq("production").all(), "core contains non-production logging")


def _safety_counter_contract_passes(manifest: Mapping[str, Any]) -> bool:
    """Check the signed legacy completion counters without coercing booleans."""

    def exact_int(names: tuple[str, ...], expected: int) -> bool:
        for name in names:
            if name not in manifest:
                continue
            value = manifest[name]
            if isinstance(value, bool):
                return False
            try:
                return int(value) == expected and str(value).strip() == str(expected)
            except (TypeError, ValueError):
                return False
        return False

    return (
        exact_int(("requested_job_count",), 24)
        and exact_int(("requested_completed_job_count", "completed", "completed_job_count"), 24)
        and exact_int(("requested_missing_job_count", "missing", "missing_job_count"), 0)
        and exact_int(("failure_count", "failure", "failed"), 0)
        and manifest.get("signed_sample_record_required_for_completion") is True
    )


def _safety_target_provenance_registered(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> bool:
    """Return whether the registered paper-flow selection provenance is present."""

    policy = protocol.get("online_label_policy", {})
    return (
        isinstance(policy, Mapping)
        and policy.get("confirmatory") is False
        and policy.get("target_selected_descriptive") is True
        and policy.get("target_labels_used_for_parameter_selection") is True
        and manifest.get("flowwise_source_profile_applied") is True
        and bool(str(manifest.get("flowwise_source_profile_json", "")).strip())
        and bool(str(manifest.get("source_reference_csv", "")).strip())
        and manifest.get("paper_flow_profile_overrides") is not None
    )


def _validate_safety(root: Path, protocol: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    manifest = dict(_json(root / "manifest.json"))
    legacy_status = "status" not in manifest or not str(manifest.get("status", "")).strip()
    if not legacy_status:
        _require(manifest.get("status") == "complete", "safety is incomplete")

    has_target_alias = (
        manifest.get("target_selected_descriptive") is True
        or manifest.get("descriptive_target_selected_evaluation") is True
    )
    legacy_target_flags = (
        "confirmatory" not in manifest
        or not has_target_alias
        or "target_labels_used_for_parameter_selection" not in manifest
    )
    if legacy_target_flags:
        _require(
            _safety_target_provenance_registered(manifest, protocol),
            "safety legacy target-selected provenance is missing",
        )
        if "confirmatory" in manifest:
            _require(manifest.get("confirmatory") is False, "safety is marked confirmatory")
        if "target_labels_used_for_parameter_selection" in manifest:
            _require(
                manifest.get("target_labels_used_for_parameter_selection") is True,
                "safety parameter-selection flag mismatch",
            )
        for alias in ("target_selected_descriptive", "descriptive_target_selected_evaluation"):
            if alias in manifest:
                _require(manifest.get(alias) is True, f"safety {alias} flag mismatch")
        # Normalize only after the registered legacy provenance checks above.
        manifest["confirmatory"] = False
        manifest["target_labels_used_for_parameter_selection"] = True
        manifest["target_selected_descriptive"] = True
        manifest["descriptive_target_selected_evaluation"] = True
    else:
        _require(manifest.get("confirmatory") is False, "safety is marked confirmatory")
        _require_target_selected_descriptive(manifest, label="safety")
    _require(manifest.get("source_seeds") == list(SOURCE_SEEDS), "safety source seed mismatch")
    _require(str(manifest.get("production_code_sha256", "")) == production_code_sha256(), "safety manifest production hash mismatch")
    legacy_logging = not str(manifest.get("logging_mode", "")).strip()
    if not legacy_logging:
        _require(str(manifest.get("logging_mode")) == "evidence", "safety logging mode is not evidence")
    graph_value = manifest.get("candidate_cuda_graph_mode", manifest.get("graph_mode", ""))
    legacy_graph = not str(graph_value).strip()
    if not legacy_graph:
        _require(str(graph_value).lower() in {"off", "disabled", "false"}, "safety graph-off contract missing")
    frame = _csv(root / "summary_raw.csv")
    required = {"dataset", "scenario", "corruption", "severity", "variant", "source_seed", "stream_seed", "production_code_sha256", "source_model_sha256"}
    _require(required.issubset(frame.columns), f"safety missing columns: {sorted(required-set(frame.columns))}")
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.upper()
    frame["scenario"] = frame["scenario"].astype(str).str.replace("→", "->", regex=False)
    frame["source_seed"] = frame["source_seed"].astype(int)
    frame["stream_seed"] = frame["stream_seed"].astype(int)
    # The legacy safety export recorded the checkpoint tensor hash but not
    # the path.  Reconstruct the path only from the registered seed-specific
    # source reference; a supplied path is still checked against that same
    # reference below.
    references = protocol.get("_source_reference", {})
    if "source_checkpoint_path" not in frame.columns:
        _require(
            "source_checkpoint_sha256" in frame.columns,
            "safety lacks source checkpoint path and tensor hash",
        )
        frame["source_checkpoint_path"] = frame["source_seed"].map(
            lambda seed: str(
                (references.get(("HAR", "12->16", int(seed))) or {}).get(
                    "source_checkpoint_path", ""
                )
            )
        )
        _require(frame["source_checkpoint_path"].astype(str).str.strip().ne("").all(), "safety source reference paths are missing")
    expected = {
        ("HAR", "12->16", corruption, severity, seed, STREAM_SEED, variant)
        for corruption in ("blackout", "signal_freeze")
        for severity in ("s3", "s6")
        for seed in SOURCE_SEEDS
        for variant in SAFETY_VARIANTS
    }
    observed = set(zip(*(frame[column] for column in SAFETY_KEYS)))
    _require(len(frame) == 24 and observed == expected, "safety key set/count mismatch")
    _require(not frame.duplicated(list(SAFETY_KEYS)).any(), "safety duplicate keys")
    if legacy_status:
        _require(
            _safety_counter_contract_passes(manifest),
            "safety legacy completion counters are invalid",
        )
        manifest["status"] = "complete"
    current_hash = production_code_sha256()
    _require(frame["production_code_sha256"].astype(str).eq(current_hash).all(), "safety production hash mismatch")
    _check_flags(frame, manifest, "safety")
    # Evidence rows carry runtime_hparams rather than a worker spec.  Require
    # both the evidence logger and an explicit graph-off value when present.
    for index, row in frame.iterrows():
        runtime = row.get("runtime_hparams", "")
        parsed: dict[str, Any] = {}
        if isinstance(runtime, str) and runtime.strip():
            try:
                value = json.loads(runtime)
                if isinstance(value, dict):
                    parsed = value
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"invalid safety runtime_hparams at row {index}") from exc
        mode = str(parsed.get("dusafe_logging_mode", row.get("dusafe_logging_mode", "")))
        _require(mode == "evidence", f"safety row {index} is not evidence logging")
        graph = parsed.get("candidate_cuda_graph_requested_mode", parsed.get("candidate_cuda_graph_mode", row.get("candidate_cuda_graph_requested_mode", "off")))
        enabled = parsed.get("candidate_cuda_graph_enabled", row.get("candidate_cuda_graph_enabled", False))
        _require(str(graph).lower() in {"off", "disabled", "false"}, f"safety row {index} graph is not off")
        _require(str(enabled).lower() not in {"true", "1"}, f"safety row {index} graph enabled")
    if legacy_logging:
        manifest["logging_mode"] = "evidence"
    if legacy_graph:
        manifest["candidate_cuda_graph_mode"] = "disabled"
    # Reuse the source-profile identity checker after adapting the row names.
    # Legacy safety rows did not export the deployment-context column; in
    # that narrow case reconstruct it from the signed runtime fields.  If a
    # column is present, a malformed or stale value is rejected below.
    had_context_column = "source_metadata_context_sha256" in frame.columns
    if not had_context_column:
        frame["source_metadata_context_sha256"] = pd.Series(
            [None] * len(frame), index=frame.index, dtype="object"
        )
    identity = frame.rename(columns={"variant": "runner"})
    identity["dusafe_logging_mode"] = "evidence"
    identity["source_metadata_context_sha256"] = identity.get("source_metadata_context_sha256", np.nan)
    # Source identity is checked manually because safety has two corruptions
    # per source/variant and therefore cannot use the panel key helper.
    for index, row in identity.iterrows():
        profile = protocol["_profiles"].get(_flow_key("HAR", "12->16"))
        reference = protocol.get("_source_reference", {}).get(("HAR", "12->16", int(row.source_seed)), {})
        _require(reference is not None, f"safety missing seed-specific source reference at {index}")
        expected_hash = str(reference.get("source_model_sha256", ""))
        expected_path = str(reference.get("source_checkpoint_path", ""))
        expected_context = str(reference.get("source_metadata_context_sha256", ""))
        _require(SHA256_RE.fullmatch(expected_context), f"safety source reference context invalid at {index}")
        _require(str(row.source_model_sha256) == expected_hash, f"safety source hash mismatch at {index}")
        if "source_checkpoint_sha256" in identity.columns:
            _require(
                str(row.source_checkpoint_sha256) == expected_hash,
                f"safety source checkpoint hash mismatch at {index}",
            )
        _require(str(row.source_checkpoint_path) == expected_path, f"safety source path mismatch at {index}")
        if had_context_column:
            current_context = str(row.source_metadata_context_sha256)
            _require(SHA256_RE.fullmatch(current_context), f"safety source context invalid at {index}")
            _require(_metadata_context_sha256(row) == current_context, f"safety reconstructed source context mismatch at {index}")
        else:
            current_context = _metadata_context_sha256(row)
        frame.at[index, "source_metadata_context_sha256"] = current_context
        frame.at[index, "source_reference_metadata_context_sha256"] = expected_context
        _check_cache_root(str(row.source_checkpoint_path), _norm_path(protocol["source_cache_roots"]["HAR"]), ("HAR", "12->16", int(row.source_seed)))
    for key, group in frame.groupby(["dataset", "scenario", "source_seed"]):
        _require(
            group["source_metadata_context_sha256"].astype(str).nunique() == 1,
            f"safety source context differs within {key}",
        )
    return frame, manifest


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_raw_digest(manifest: Mapping[str, Any]) -> str:
    candidates = (
        manifest.get("raw_sha256"),
        manifest.get("raw_csv_sha256"),
        manifest.get("raw_digest_sha256"),
        manifest.get("raw_digest"),
        (manifest.get("artifact_digests") or {}).get("raw.csv") if isinstance(manifest.get("artifact_digests"), Mapping) else None,
        (manifest.get("digests") or {}).get("raw.csv") if isinstance(manifest.get("digests"), Mapping) else None,
        ((manifest.get("artifacts") or {}).get("raw.csv") or {}).get("sha256") if isinstance(manifest.get("artifacts"), Mapping) and isinstance((manifest.get("artifacts") or {}).get("raw.csv"), Mapping) else None,
    )
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return ""


def _panel_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _validate_evidence_panel_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    label: str,
    spec: Mapping[str, Any],
    raw_rows: int,
    current_hash: str,
    allow_legacy_top_contract: bool = False,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    _require(manifest.get("status") == "complete", f"{label} is incomplete")
    causal_required = bool(spec.get("causal_evidence_code_required", False))
    protocol_prefix = (
        "paper_representative_causal_ablation_v5_"
        if causal_required
        else "paper_evidence_v5_"
    )
    _require(
        str(manifest.get("protocol", "")).startswith(protocol_prefix),
        f"{label} protocol does not match {protocol_prefix}",
    )
    _require(manifest.get("confirmatory") is False, f"{label} confirmatory flag is not false")
    _require(manifest.get("target_labels_used_for_online_decision") is False, f"{label} uses online target labels")
    _require(manifest.get("target_labels_used_for_parameter_selection") is True, f"{label} parameter-selection flag mismatch")
    _require_target_selected_descriptive(manifest, label=label)
    _require(str(manifest.get("production_code_sha256", "")) == current_hash, f"{label} manifest production hash mismatch")
    if spec.get("ablation_code_required", False):
        required_ablation = (
            _registered_ablation_artifact_digest(
                protocol or {}, "augmentation_controls"
            )
            if label.startswith("augmentation_controls[")
            else ablation_code_sha256()
        )
        _require(str(manifest.get("ablation_code_sha256", "")) == required_ablation, f"{label} manifest ablation code digest mismatch")
    if spec.get("causal_evidence_code_required", False):
        _require(str(manifest.get("causal_evidence_code_sha256", "")) == causal_evidence_code_sha256(), f"{label} manifest causal evidence code digest mismatch")
    top_logging = str(manifest.get("logging_mode", "")).strip()
    if allow_legacy_top_contract and not top_logging:
        # Legacy augmentation roots prove evidence logging in every worker
        # spec/row, so the normalized manifest may fill this top-level alias.
        pass
    else:
        _require(top_logging == str(spec["logging_mode"]), f"{label} manifest logging mode mismatch")
    graph = manifest.get("candidate_cuda_graph_mode", manifest.get("graph_mode", ""))
    if allow_legacy_top_contract and not str(graph).strip():
        pass
    else:
        _require(str(graph).lower() in {"off", "disabled", "false"}, f"{label} graph is not disabled")
    if causal_required:
        _require(
            manifest.get("candidate_cuda_graph_enabled") is False,
            f"{label} causal graph enabled flag is not false",
        )
        _require(
            str(manifest.get("candidate_cuda_graph_status", ""))
            == "disabled_evidence_logging",
            f"{label} causal graph status is not disabled_evidence_logging",
        )
    _require(manifest.get("source_seeds") == list(SOURCE_SEEDS), f"{label} source seed mismatch")
    _require(int(manifest.get("stream_seed", STREAM_SEED)) == STREAM_SEED, f"{label} stream seed mismatch")
    declared_raw_rows = manifest.get("raw_rows")
    if declared_raw_rows is None and allow_legacy_top_contract:
        declared_raw_rows = manifest.get(
            "completed_cells", manifest.get("expected_cells", -1)
        )
    _require(int(declared_raw_rows if declared_raw_rows is not None else -1) == int(raw_rows), f"{label} manifest raw row count mismatch")
    if "source_cells" in spec:
        declared_cells = manifest.get("source_cells", manifest.get("input_cells", manifest.get("completed_cells", -1)))
        _require(int(declared_cells) == int(spec["source_cells"]), f"{label} source-cell count mismatch")
    else:
        declared_cells = manifest.get("completed_cells", manifest.get("expected_cells", -1))
        _require(int(declared_cells) == int(raw_rows), f"{label} completed-cell count mismatch")
    digest = _manifest_raw_digest(manifest)
    if not (allow_legacy_top_contract and not digest):
        _require(SHA256_RE.fullmatch(digest), f"{label} raw artifact digest is missing or invalid")
    raw_path = root / "raw.csv"
    _require(raw_path.is_file(), f"{label} raw.csv is missing")
    if digest:
        _require(_artifact_digest(raw_path) == digest, f"{label} raw.csv digest mismatch")


def _validate_evidence_runtime_row(
    row: Mapping[str, Any], *, label: str, index: Any, graph: str,
    causal: bool = False,
) -> None:
    mode = str(row.get("dusafe_logging_mode", ""))
    _require(mode == "evidence", f"{label} row {index} is not evidence logging")
    runtime = _parse_runtime_hparams(row.get("runtime_hparams", ""))
    runtime_mode = runtime.get("dusafe_logging_mode", mode)
    _require(str(runtime_mode) == "evidence", f"{label} row {index} runtime logging mismatch")
    requested_graph = runtime.get("candidate_cuda_graph_requested_mode", runtime.get("candidate_cuda_graph_mode", row.get("candidate_cuda_graph_requested_mode", graph)))
    enabled = runtime.get("candidate_cuda_graph_enabled", row.get("candidate_cuda_graph_enabled", False))
    graph_status = runtime.get(
        "candidate_cuda_graph_status",
        row.get("candidate_cuda_graph_status", ""),
    )
    if causal:
        _require(
            str(requested_graph).lower() in {"off", "auto", "force", "disabled", "false"},
            f"{label} row {index} graph request is invalid",
        )
        _require(
            str(graph_status) == "disabled_evidence_logging",
            f"{label} row {index} graph status is not disabled_evidence_logging",
        )
    else:
        _require(str(requested_graph).lower() in {"off", "disabled", "false"}, f"{label} row {index} graph is not disabled")
    _require(not _panel_bool(enabled), f"{label} row {index} graph is enabled")
    online = row.get("target_labels_used_for_online_decision", row.get("target_labels_used_for_online_updates"))
    selection = row.get("target_labels_used_for_parameter_selection")
    _require(online is not None and not _panel_bool(online), f"{label} row {index} online target-label flag mismatch")
    _require(selection is not None and _panel_bool(selection), f"{label} row {index} parameter-selection flag mismatch")
    _require(row.get("confirmatory") is not None and not _panel_bool(row.get("confirmatory")), f"{label} row {index} confirmatory flag mismatch")


def _validate_augmentation_cell_summaries(
    root: Path,
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    label: str,
    required_ablation_code_sha256: str | None = None,
) -> None:
    """Cross-check the legacy aggregate against every cell summary/spec.

    The old augmentation root predates the top-level logging/graph/raw digest
    aliases, but its signed cell summaries and worker specs remain the source
    of truth.  This check prevents a missing aggregate digest from becoming a
    way to substitute or reorder cell results.
    """

    key_columns = ["dataset", "scenario", "source_seed", "runner"]
    raw = frame.copy()
    raw["dataset"] = raw["dataset"].astype(str).str.upper()
    raw["scenario"] = raw["scenario"].astype(str).str.replace("→", "->", regex=False)
    raw["source_seed"] = raw["source_seed"].astype(int)
    raw["runner"] = raw["runner"].astype(str)
    _require(not raw.duplicated(key_columns).any(), f"{label} duplicate aggregate cell keys")
    summary_rows = _summary_rows(root)
    _require(len(summary_rows) == expected_rows, f"{label} cell summary count mismatch")
    summaries = pd.DataFrame(summary_rows)
    summaries["dataset"] = summaries["dataset"].astype(str).str.upper()
    summaries["scenario"] = summaries["scenario"].astype(str).str.replace("→", "->", regex=False)
    summaries["source_seed"] = summaries["source_seed"].astype(int)
    summaries["runner"] = summaries["runner"].astype(str)
    _require(not summaries.duplicated(key_columns).any(), f"{label} duplicate cell summaries")
    raw_keys = set(map(tuple, raw[key_columns].to_numpy()))
    summary_keys = set(map(tuple, summaries[key_columns].to_numpy()))
    _require(raw_keys == summary_keys, f"{label} aggregate/cell summary key mismatch")

    def equal(left: Any, right: Any) -> bool:
        def missing(value: Any) -> bool:
            if value is None or isinstance(value, (Mapping, list, tuple, dict)):
                return value is None
            try:
                return bool(pd.isna(value))
            except (TypeError, ValueError):
                return False

        if missing(left) and missing(right):
            return True
        if isinstance(left, str) and isinstance(right, Mapping):
            try:
                left = json.loads(left)
            except json.JSONDecodeError:
                pass
        if isinstance(right, str) and isinstance(left, Mapping):
            try:
                right = json.loads(right)
            except json.JSONDecodeError:
                pass
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            try:
                return json.dumps(left, sort_keys=True, default=str) == json.dumps(
                    right, sort_keys=True, default=str
                )
            except (TypeError, ValueError):
                return str(left) == str(right)
        try:
            left_float = float(left)
            right_float = float(right)
        except (TypeError, ValueError):
            return str(left) == str(right)
        return math.isfinite(left_float) and math.isfinite(right_float) and math.isclose(
            left_float, right_float, rel_tol=0.0, abs_tol=1e-12
        )

    common_columns = [
        column for column in summaries.columns
        if column in raw.columns and column not in key_columns
    ]
    raw_indexed = raw.set_index(key_columns)
    summary_indexed = summaries.set_index(key_columns)
    for key in sorted(raw_keys):
        for column in common_columns:
            _require(
                equal(raw_indexed.loc[key, column], summary_indexed.loc[key, column]),
                f"{label} aggregate/cell summary mismatch at {key}, {column}",
            )

    specs = _load_worker_specs(root)
    _require(len(specs) == expected_rows, f"{label} worker spec count mismatch")
    spec_keys = set(specs)
    _require(
        spec_keys == raw_keys,
        f"{label} worker spec key set mismatch",
    )
    current_production = production_code_sha256()
    current_ablation = (
        required_ablation_code_sha256 or ablation_code_sha256()
    )
    for key, spec in specs.items():
        _require(
            str(spec.get("production_code_sha256", "")) == current_production,
            f"{label} worker spec production digest mismatch: {key}",
        )
        _require(
            str(spec.get("ablation_code_sha256", "")) == current_ablation,
            f"{label} worker spec ablation digest mismatch: {key}",
        )
        tta = spec.get("tta_config")
        _require(isinstance(tta, Mapping), f"{label} worker spec runtime config missing: {key}")
        _require(
            str(tta.get("dusafe_logging_mode", "")) == "evidence",
            f"{label} worker spec is not evidence logging: {key}",
        )
        requested = str(tta.get("candidate_cuda_graph_requested_mode", "disabled")).lower()
        _require(
            requested in {"off", "auto", "force", "disabled", "false"},
            f"{label} worker spec graph request is invalid: {key}",
        )
        _require(
            not _panel_bool(tta.get("candidate_cuda_graph_enabled", False)),
            f"{label} worker spec graph is enabled: {key}",
        )
        status = tta.get("candidate_cuda_graph_status")
        if status is not None:
            _require(
                str(status) == "disabled_evidence_logging",
                f"{label} worker spec graph status mismatch: {key}",
            )


def _mechanism_source_reference(
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    key: tuple[str, str, int],
) -> tuple[dict[str, Any] | None, Path]:
    """Resolve the explicitly registered source identity for one panel cell."""

    dataset, _scenario, seed = key
    if str(spec.get("source_identity_reference", "")) == "paper_table_golden_cell":
        checkpoints = spec.get("paper_source_checkpoints")
        _require(isinstance(checkpoints, Mapping), "paper-table source checkpoints are missing")
        raw = checkpoints.get(str(int(seed)))
        _require(isinstance(raw, Mapping), f"paper-table source checkpoint is missing: {key}")
        reference = dict(raw)
        reference["source_checkpoint_path"] = str(
            _norm_path(ROOT / str(reference.get("source_checkpoint_path", "")))
        )
        cache_root = _norm_path(ROOT / str(spec.get("source_cache_root", "")))
        return reference, cache_root
    reference = protocol.get("_source_reference", {}).get(key)
    return reference, _norm_path(protocol["source_cache_roots"][dataset])


def _validate_mechanism_panel(
    root: Path,
    *,
    label: str,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    manifest = _json(root / "manifest.json")
    # Prefer the signed raw export but enrich each row from its worker spec;
    # compact evidence summaries intentionally omit some runtime fields.
    frame = _read_panel(root)
    expected_rows = int(spec["raw_rows"])
    allow_legacy_top_contract = label.startswith("augmentation_controls[")
    _validate_evidence_panel_manifest(
        root, manifest, label=label, spec=spec,
        raw_rows=expected_rows, current_hash=production_code_sha256(),
        allow_legacy_top_contract=allow_legacy_top_contract,
        protocol=protocol,
    )
    causal_required = bool(spec.get("causal_evidence_code_required", False))
    # Prospective causal summaries may intentionally omit path/context columns
    # from the compact raw export.  Reconstruct them only from the registered
    # seed-specific source reference and the current row's deployment fields;
    # supplied non-empty values are still checked and cannot be overwritten.
    if causal_required:
        frame = frame.copy()
        identity_columns = {
            "dataset", "scenario", "source_seed", "source_model_sha256"
        }
        _require(identity_columns.issubset(frame.columns), f"{label} lacks causal source identity columns")
        references = protocol.get("_source_reference", {})
        path_missing = "source_checkpoint_path" not in frame.columns
        if not path_missing:
            path_values = frame["source_checkpoint_path"].astype(str).str.strip()
            if path_values.eq("").all() or path_values.str.lower().isin({"nan", "none"}).all():
                path_missing = True
            else:
                _require(
                    not path_values.eq("").any()
                    and not path_values.str.lower().isin({"nan", "none"}).any(),
                    f"{label} has partially missing source checkpoint paths",
                )
        if path_missing:
            frame["source_checkpoint_path"] = ""
            for index, row in frame.iterrows():
                key = (
                    str(row["dataset"]).upper(),
                    str(row["scenario"]).replace("→", "->"),
                    int(row["source_seed"]),
                )
                reference, _cache_root = _mechanism_source_reference(
                    spec, protocol, key
                )
                _require(reference is not None, f"{label} missing source reference: {key}")
                expected_path = str(reference.get("source_checkpoint_path", "")).strip()
                _require(expected_path, f"{label} source reference path is missing: {key}")
                frame.at[index, "source_checkpoint_path"] = expected_path
        context_missing = "source_metadata_context_sha256" not in frame.columns
        if not context_missing:
            context_values = frame["source_metadata_context_sha256"].astype(str).str.strip()
            if context_values.eq("").all() or context_values.str.lower().isin({"nan", "none"}).all():
                context_missing = True
            else:
                _require(
                    not context_values.eq("").any()
                    and not context_values.str.lower().isin({"nan", "none"}).any(),
                    f"{label} has partially missing source metadata contexts",
                )
        if context_missing:
            frame["source_metadata_context_sha256"] = ""
            for index, row in frame.iterrows():
                derived = _metadata_context_sha256(row)
                _require(SHA256_RE.fullmatch(derived), f"{label} cannot reconstruct source context at row {index}")
                frame.at[index, "source_metadata_context_sha256"] = derived
    required = {
        "dataset", "scenario", "source_seed", "stream_seed", "source_model_sha256",
        "source_checkpoint_path", "source_metadata_context_sha256", "status",
        "production_code_sha256", "confirmatory", "target_labels_used_for_parameter_selection",
    }
    _require(required.issubset(frame.columns), f"{label} missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.upper()
    frame["scenario"] = frame["scenario"].astype(str).str.replace("→", "->", regex=False)
    frame["source_seed"] = frame["source_seed"].astype(int)
    frame["stream_seed"] = frame["stream_seed"].astype(int)
    _require(len(frame) == expected_rows, f"{label} raw row count mismatch: {len(frame)} != {expected_rows}")
    _require(frame["status"].astype(str).eq("ok").all(), f"{label} contains failed rows")
    current_hash = production_code_sha256()
    _require(frame["production_code_sha256"].astype(str).eq(current_hash).all(), f"{label} production hash mismatch")
    required_ablation = None
    if spec.get("ablation_code_required", False):
        required_ablation = (
            _registered_ablation_artifact_digest(protocol, "augmentation_controls")
            if label.startswith("augmentation_controls[")
            else ablation_code_sha256()
        )
    if required_ablation is not None:
        _require("ablation_code_sha256" in frame.columns, f"{label} missing ablation code digest")
        _require(frame["ablation_code_sha256"].astype(str).eq(required_ablation).all(), f"{label} ablation code digest mismatch")
    required_causal = causal_evidence_code_sha256() if spec.get("causal_evidence_code_required", False) else None
    if required_causal is not None:
        _require("causal_evidence_code_sha256" in frame.columns, f"{label} missing causal evidence code digest")
        _require(frame["causal_evidence_code_sha256"].astype(str).eq(required_causal).all(), f"{label} causal evidence code digest mismatch")
    variants_col = "runner" if "runner" in frame.columns else "variant"
    _require(variants_col in frame.columns, f"{label} lacks runner/variant column")
    frame[variants_col] = frame[variants_col].astype(str)
    expected_datasets = tuple(str(value).upper() for value in spec.get("datasets", [spec.get("dataset")]))
    expected_flows = {}
    if "dataset" in spec:
        expected_flows[str(spec["dataset"]).upper()] = [str(flow) for flow in spec["flows"]]
    else:
        for dataset in expected_datasets:
            expected_flows[dataset] = list(protocol["formal_flows"][dataset])
    observed_datasets = set(frame["dataset"])
    _require(observed_datasets == set(expected_datasets), f"{label} dataset set mismatch")
    for dataset in expected_datasets:
        _require(set(frame.loc[frame["dataset"] == dataset, "scenario"]) == set(expected_flows[dataset]), f"{label} formal flow set mismatch for {dataset}")
    _require(set(frame["source_seed"]) == set(SOURCE_SEEDS), f"{label} source seed set mismatch")
    _require(set(frame["stream_seed"]) == {STREAM_SEED}, f"{label} stream seed set mismatch")
    expected_runners = set(spec.get("variants", spec.get("runners", [])))
    _require(set(frame[variants_col]) == expected_runners, f"{label} runner set mismatch")
    for index, row in frame.iterrows():
        _validate_evidence_runtime_row(
            row,
            label=label,
            index=index,
            graph=str(spec["graph"]),
            causal=causal_required,
        )
        dataset = str(row["dataset"])
        scenario = str(row["scenario"])
        seed = int(row["source_seed"])
        key = (dataset, scenario, seed)
        reference, cache_root = _mechanism_source_reference(
            spec, protocol, key
        )
        _require(reference is not None, f"{label} missing source reference: {key}")
        _require(str(row["source_model_sha256"]) == str(reference["source_model_sha256"]), f"{label} source hash mismatch: {key}")
        _require(str(row["source_checkpoint_path"]) == str(reference["source_checkpoint_path"]), f"{label} source checkpoint path mismatch: {key}")
        _require(SHA256_RE.fullmatch(str(row["source_metadata_context_sha256"])), f"{label} invalid source context: {key}")
        _require(_metadata_context_sha256(row) == str(row["source_metadata_context_sha256"]), f"{label} reconstructed source context mismatch: {key}")
        reference_context = str(
            reference.get(
                "source_metadata_context_sha256",
                row["source_metadata_context_sha256"],
            )
        )
        _require(
            SHA256_RE.fullmatch(reference_context),
            f"{label} source reference context is invalid: {key}",
        )
        frame.at[index, "source_reference_metadata_context_sha256"] = reference_context
        _check_cache_root(str(row["source_checkpoint_path"]), cache_root, key)
    identity_keys = ["dataset", "scenario", "source_seed", "stream_seed", variants_col]
    if "condition" in frame.columns:
        identity_keys.append("condition")
    if "batch_index" in frame.columns:
        identity_keys.append("batch_index")
    if "horizon" in frame.columns:
        identity_keys.append("horizon")
    _require(not frame.duplicated(identity_keys).any(), f"{label} duplicate row keys")
    for key, group in frame.groupby(["dataset", "scenario", "source_seed"]):
        _require(group["source_model_sha256"].astype(str).nunique() == 1, f"{label} source hash differs within {key}")
        _require(group["source_checkpoint_path"].astype(str).nunique() == 1, f"{label} source path differs within {key}")
        _require(group["source_metadata_context_sha256"].astype(str).nunique() == 1, f"{label} source context differs within {key}")
    if label == "heldout_hhar":
        _require(set(frame["condition"]) == {"clean"}, f"{label} condition set mismatch")
        _require(set(frame["horizon"]) == {1}, f"{label} horizon mismatch")
        _require((frame.groupby("source_seed").size() == 50).all(), f"{label} must have 50 rows per source seed")
    elif label == "confidence_eeg":
        conditions = set(spec["conditions"])
        _require(set(frame["condition"]) == conditions, f"{label} condition set mismatch")
        counts = frame.groupby(["source_seed", "condition"]).size()
        _require(len(counts) == 6 and (counts == 10).all(), f"{label} must have 10 rows per seed/condition")
    else:
        required_keys = {
            (dataset, scenario, seed, runner)
            for dataset in expected_datasets
            for scenario in expected_flows[dataset]
            for seed in SOURCE_SEEDS
            for runner in expected_runners
        }
        observed_keys = set(zip(frame["dataset"], frame["scenario"], frame["source_seed"], frame[variants_col]))
        _require(observed_keys == required_keys, f"{label} augmentation key set mismatch")
    metric = str(spec["metric"])
    _require(metric in frame.columns, f"{label} metric column missing: {metric}")
    if allow_legacy_top_contract:
        _validate_augmentation_cell_summaries(
            root,
            frame,
            expected_rows=expected_rows,
            label=label,
            required_ablation_code_sha256=required_ablation,
        )
        # Return a normalized manifest without mutating the legacy input.
        manifest = dict(manifest)
        manifest["status"] = "complete"
        manifest["logging_mode"] = "evidence"
        manifest["candidate_cuda_graph_mode"] = "disabled"
        manifest["candidate_cuda_graph_enabled"] = False
        manifest["raw_rows"] = expected_rows
        manifest["raw_sha256"] = _artifact_digest(root / "raw.csv")
    return frame, manifest


def _validate_multi_augmentation_panel(
    dirs: Iterable[Path],
    *,
    label: str,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate HAR/HHAR augmentation caches independently, then merge them."""

    paths = [_norm_path(path) for path in dirs]
    _require(len(paths) >= 2, f"{label} requires separate dataset cache directories")
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    datasets = tuple(str(value).upper() for value in spec["datasets"])
    rows_per_dataset = int(spec["raw_rows"]) // len(datasets)
    _require(rows_per_dataset * len(datasets) == int(spec["raw_rows"]), f"{label} row count is not divisible by dataset count")
    for path in paths:
        manifest = _json(path / "manifest.json")
        declared = manifest.get("datasets")
        if isinstance(declared, list) and len(declared) == 1:
            dataset = str(declared[0]).upper()
        else:
            name = path.name.lower()
            dataset = "HHAR" if "hhar" in name else "HAR"
        _require(dataset in datasets, f"{label} unexpected dataset directory: {dataset}")
        local_spec = dict(spec)
        local_spec["datasets"] = [dataset]
        local_spec["raw_rows"] = rows_per_dataset
        frame, checked_manifest = _validate_mechanism_panel(
            path,
            label=f"{label}[{dataset}]",
            spec=local_spec,
            protocol=protocol,
        )
        _require(set(frame["dataset"]) == {dataset}, f"{label}[{dataset}] contains another dataset")
        frames.append(frame)
        manifests.append(checked_manifest)
    merged = pd.concat(frames, ignore_index=True)
    runner_column = "runner" if "runner" in merged.columns else "variant"
    keys = ["dataset", "scenario", "source_seed", runner_column]
    _require(not merged.duplicated(keys).any(), f"{label} duplicate keys across augmentation directories")
    expected = {
        (dataset, scenario, seed, runner)
        for dataset in datasets
        for scenario in protocol["formal_flows"][dataset]
        for seed in SOURCE_SEEDS
        for runner in spec["runners"]
    }
    observed = set(zip(merged["dataset"], merged["scenario"], merged["source_seed"], merged[runner_column]))
    _require(len(merged) == int(spec["raw_rows"]) and observed == expected, f"{label} combined key set/count mismatch")
    combined_manifest = dict(manifests[0])
    combined_manifest["input_manifests"] = manifests
    combined_manifest["input_dirs"] = [str(path) for path in paths]
    return merged, combined_manifest


def _aggregate_mechanism_panel(frame: pd.DataFrame, *, spec: Mapping[str, Any]) -> pd.DataFrame:
    runner = "runner" if "runner" in frame.columns else "variant"
    metric = str(spec["metric"])
    grouping = ["dataset", "scenario", runner]
    if "condition" in frame.columns:
        grouping.append("condition")
    return frame.groupby(grouping, as_index=False).agg(
        rows=(metric, "size"), source_seeds=("source_seed", "nunique"),
        metric_mean=(metric, "mean"), metric_std=(metric, "std"),
    )


def _old_v4_mechanism_comparison(
    new_frame: pd.DataFrame,
    *,
    old_root: Path,
    label: str,
    spec: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    old_path = Path(old_root) / "raw.csv"
    if not old_path.is_file():
        return {"panel": label, "status": "missing_old_v4", "old_path": str(old_path), "cell_rows": 0, "equivalent_cells": 0}
    old = pd.read_csv(old_path)
    for frame in (new_frame, old):
        frame["dataset"] = frame["dataset"].astype(str).str.upper()
        frame["scenario"] = frame["scenario"].astype(str).str.replace("→", "->", regex=False)
        frame["source_seed"] = frame["source_seed"].astype(int)
    new_runner = "runner" if "runner" in new_frame.columns else "variant"
    old_runner = "runner" if "runner" in old.columns else "variant"
    metric = str(spec["metric"])
    old_metric = metric if metric in old.columns else ("f1" if "f1" in old.columns else "future_macro_f1")
    _require(metric in new_frame.columns and old_metric in old.columns, f"{label} old-v4 comparison metric missing")
    keys = ["dataset", "scenario", "source_seed", new_runner]
    if "condition" in new_frame.columns and "condition" in old.columns:
        keys.append("condition")
    if "batch_index" in new_frame.columns and "batch_index" in old.columns:
        keys.append("batch_index")
    if "horizon" in new_frame.columns and "horizon" in old.columns:
        keys.append("horizon")
    left = new_frame[[*keys, metric]].rename(columns={new_runner: "runner", metric: "metric_v5"})
    right = old[["dataset", "scenario", "source_seed", old_runner, *[key for key in keys if key in {"condition", "batch_index", "horizon"}], old_metric]].rename(columns={old_runner: "runner", old_metric: "metric_v4"})
    merged = left.merge(right, on=[key if key != new_runner else "runner" for key in keys], how="outer", indicator=True)
    merged["metric_delta"] = merged["metric_v5"] - merged["metric_v4"]
    merged["equivalent"] = (merged["_merge"] == "both") & np.isclose(merged["metric_v5"], merged["metric_v4"], rtol=0.0, atol=0.0, equal_nan=False)
    _write_csv(output / f"old_v4_{label}_cell_comparison.csv", merged)
    summary_keys = [key for key in ("dataset", "runner", "condition") if key in merged.columns]
    summary = merged.groupby(summary_keys, dropna=False, as_index=False).agg(
        cell_rows=("_merge", "size"), metric_v5_mean=("metric_v5", "mean"), metric_v4_mean=("metric_v4", "mean"), metric_delta_mean=("metric_delta", "mean"), equivalent_cells=("equivalent", "sum"),
    )
    _write_csv(output / f"old_v4_{label}_summary_comparison.csv", summary)
    return {"panel": label, "status": "compared", "old_path": str(old_path), "cell_rows": int(len(merged)), "equivalent_cells": int(merged["equivalent"].sum()), "summary_rows": int(len(summary))}


def _parse_profile_mapping(value: Any) -> dict[str, Any]:
    """Parse the compact historical profile or the current JSON profile.

    The historical golden artifacts stored a Python ``repr`` while current
    evidence rows store either that representation or canonical JSON.  Only
    the explicitly registered algorithm/profile fields are compared; opaque
    state hashes that include logging-only buffers are deliberately excluded
    by the protocol and replaced by model-buffer and optimizer hashes.
    """

    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("paper-table profile is missing")
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise EvidenceError("paper-table profile is not a mapping")


def _series_exact_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return row-wise exact equality while treating paired missing values alike."""

    return left.eq(right) | (left.isna() & right.isna())


def _validate_paper_table_golden_panel(
    current: pd.DataFrame,
    current_manifest: Mapping[str, Any],
    *,
    label: str,
    spec: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Fail closed unless a current panel preserves the signed paper table.

    This contract is intentionally independent from ``old_v4_root``.  The
    latter remains a historical execution comparison and is not allowed to
    stand in for the positive values printed in the paper.  The signed golden
    CSV is pinned by digest, then every paper-table metric and every registered
    semantic provenance field is compared at the batch/seed cell level.
    """

    golden_root = _norm_path(ROOT / str(spec["golden_root"]))
    golden_manifest_path = golden_root / "manifest.json"
    golden_raw_path = golden_root / "raw.csv"
    _require(
        _artifact_digest(golden_manifest_path) == str(spec["golden_manifest_sha256"]),
        f"{label} paper-table golden manifest digest mismatch",
    )
    _require(
        _artifact_digest(golden_raw_path) == str(spec["golden_raw_sha256"]),
        f"{label} paper-table golden raw digest mismatch",
    )
    golden_manifest = _json(golden_manifest_path)
    golden = _csv(golden_raw_path)

    manifest_fields = tuple(str(value) for value in spec["manifest_provenance_fields"])
    for field in manifest_fields:
        _require(field in golden_manifest, f"{label} golden manifest lacks provenance field: {field}")
        _require(field in current_manifest, f"{label} current manifest lacks provenance field: {field}")
        _require(
            _canonical_hash(golden_manifest[field]) == _canonical_hash(current_manifest[field]),
            f"{label} manifest provenance mismatch: {field}",
        )

    key_columns = tuple(str(value) for value in spec["key_columns"])
    table_fields = tuple(str(value) for value in spec["paper_table_fields"])
    provenance_fields = tuple(str(value) for value in spec["semantic_provenance_fields"])
    required = set(key_columns) | set(table_fields) | set(provenance_fields) | {"profile"}
    _require(required.issubset(current.columns), f"{label} current paper-table columns missing: {sorted(required - set(current.columns))}")
    _require(required.issubset(golden.columns), f"{label} golden paper-table columns missing: {sorted(required - set(golden.columns))}")

    variants = {str(value) for value in spec["paper_variants"]}
    current = current[current["variant"].astype(str).isin(variants)].copy()
    golden = golden[golden["variant"].astype(str).isin(variants)].copy()
    for frame in (current, golden):
        frame["dataset"] = frame["dataset"].astype(str).str.upper()
        frame["scenario"] = frame["scenario"].astype(str).str.replace("→", "->", regex=False)
        frame["source_seed"] = frame["source_seed"].astype(int)
        frame["stream_seed"] = frame["stream_seed"].astype(int)
        _require(not frame.duplicated(list(key_columns)).any(), f"{label} paper-table duplicate cell keys")

    expected_profile = dict(spec["profile_fields"])
    for frame_label, frame in (("golden", golden), ("current", current)):
        for row_index, raw_profile in frame["profile"].items():
            profile = _parse_profile_mapping(raw_profile)
            for field, expected in expected_profile.items():
                _require(field in profile, f"{label} {frame_label} profile lacks {field} at row {row_index}")
                _require(
                    profile[field] == expected,
                    f"{label} {frame_label} profile mismatch for {field} at row {row_index}",
                )

    columns = [*key_columns, *table_fields, *provenance_fields]
    merged = current[columns].merge(
        golden[columns],
        on=list(key_columns),
        how="outer",
        suffixes=("_current", "_golden"),
        indicator=True,
        validate="one_to_one",
    )
    _require(
        len(merged) == len(current) == len(golden) and merged["_merge"].eq("both").all(),
        f"{label} paper-table cell key set mismatch",
    )
    for field in (*table_fields, *provenance_fields):
        equivalent = _series_exact_equal(
            merged[f"{field}_current"], merged[f"{field}_golden"]
        )
        merged[f"{field}_equivalent"] = equivalent
        _require(equivalent.all(), f"{label} paper-table cell mismatch: {field}")
    merged["all_table_fields_equivalent"] = merged[
        [f"{field}_equivalent" for field in table_fields]
    ].all(axis=1)
    merged["all_semantic_provenance_equivalent"] = merged[
        [f"{field}_equivalent" for field in provenance_fields]
    ].all(axis=1)
    _write_csv(output / f"paper_table_golden_{label}_cell_comparison.csv", merged)

    group_columns = ["dataset", "scenario", "variant"]
    if "condition" in key_columns:
        group_columns.append("condition")
    table_rows: list[dict[str, Any]] = []
    for group_key, current_group in current.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        golden_mask = pd.Series(True, index=golden.index)
        record = dict(zip(group_columns, group_key))
        for field, value in record.items():
            golden_mask &= golden[field].eq(value)
        golden_group = golden.loc[golden_mask]
        _require(len(current_group) == len(golden_group), f"{label} aggregate group size mismatch: {record}")
        for metric in table_fields:
            current_mean = float(current_group[metric].astype(float).mean())
            golden_mean = float(golden_group[metric].astype(float).mean())
            _require(current_mean == golden_mean, f"{label} paper-table aggregate mismatch: {record}, {metric}")
            table_rows.append(
                {
                    **record,
                    "metric": metric,
                    "current_mean": current_mean,
                    "golden_mean": golden_mean,
                    "equivalent": True,
                    "rows": int(len(current_group)),
                }
            )
    table = pd.DataFrame(table_rows)
    _write_csv(output / f"paper_table_golden_{label}_table_comparison.csv", table)
    return {
        "panel": label,
        "status": "passed",
        "role": "paper_table_claim_golden_reference",
        "golden_root": str(golden_root),
        "golden_manifest_sha256": str(spec["golden_manifest_sha256"]),
        "golden_raw_sha256": str(spec["golden_raw_sha256"]),
        "cell_rows": int(len(merged)),
        "paper_table_fields": list(table_fields),
        "semantic_provenance_fields": list(provenance_fields),
        "manifest_provenance_fields": list(manifest_fields),
        "excluded_opaque_state_hashes": dict(spec.get("excluded_opaque_state_hashes", {})),
        "table_cells_equivalent": int(merged["all_table_fields_equivalent"].sum()),
        "provenance_cells_equivalent": int(merged["all_semantic_provenance_equivalent"].sum()),
        "aggregate_rows": int(len(table)),
    }


def _validate_paper_table_golden_references(
    current_frames: Mapping[str, pd.DataFrame],
    current_manifests: Mapping[str, Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    _require(
        str(contract.get("contract", "")) == "paper_table_claim_preservation_v1",
        "paper-table golden reference contract is not registered",
    )
    _require(contract.get("required") is True, "paper-table golden reference must be required")
    _require(
        contract.get("historical_execution_comparisons_are_not_paper_table_references") is True,
        "historical execution comparison is incorrectly allowed as a paper-table reference",
    )
    panels = contract.get("panels")
    _require(isinstance(panels, Mapping) and panels, "paper-table golden reference has no panels")
    results: dict[str, Any] = {}
    for label, spec in panels.items():
        _require(label in current_frames, f"missing current paper-table panel: {label}")
        _require(label in current_manifests, f"missing current paper-table manifest: {label}")
        results[str(label)] = _validate_paper_table_golden_panel(
            current_frames[str(label)],
            current_manifests[str(label)],
            label=str(label),
            spec=spec,
            output=output,
        )
    return {
        "contract": str(contract["contract"]),
        "status": "passed",
        "comparison": str(contract.get("comparison", "")),
        "historical_execution_comparisons_are_not_paper_table_references": True,
        "panels": results,
    }


def _aggregate_main(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = {"confidence_only": "No_SSAW", "hard_ssaw": "Full"}
    per_seed = frame.groupby(["dataset", "source_seed", "runner"], as_index=False).agg(formal_flows=("scenario", "nunique"), f1=("f1", "mean"))
    _require((per_seed["formal_flows"] == 5).all(), "main is not five-flow per seed")
    pivot = per_seed.pivot(index=["dataset", "source_seed"], columns="runner", values="f1").reset_index()
    rows = []
    for dataset, group in pivot.groupby("dataset"):
        row: dict[str, Any] = {"dataset": dataset, "formal_flows": 5, "source_seeds": 3}
        for runner, label in labels.items():
            values = group[runner].astype(float)
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_std"] = float(values.std(ddof=1))
        row["Full_minus_No_SSAW_pp"] = 100 * (row["Full_mean"] - row["No_SSAW_mean"])
        rows.append(row)
    dataset = pd.DataFrame(rows).sort_values("dataset")
    flow = frame.groupby(["dataset", "scenario", "runner"], as_index=False).agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    means = flow.pivot(index=["dataset", "scenario", "source_seeds"], columns="runner", values="f1_mean").reset_index()
    stds = flow.pivot(index=["dataset", "scenario", "source_seeds"], columns="runner", values="f1_std").reset_index()
    flow_table = means[["dataset", "scenario", "source_seeds"]].copy()
    for runner, label in labels.items():
        flow_table[f"{label}_mean"] = means[runner]
        flow_table[f"{label}_std"] = stds[runner]
    flow_table["Full_minus_No_SSAW_pp"] = 100 * (flow_table["Full_mean"] - flow_table["No_SSAW_mean"])
    return dataset, flow_table.sort_values(["dataset", "scenario"])


def _aggregate_core(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = {"accept_all_raw": "Raw TTA", "confidence_only": "Confidence-only", "random_eligible_spline": "Confidence + Random", "hard_ssaw": "Full"}
    seed = frame.groupby(["dataset", "source_seed", "runner"], as_index=False).agg(formal_flows=("scenario", "nunique"), f1=("f1", "mean"))
    _require((seed["formal_flows"] == 5).all(), "core is not five-flow per seed")
    dataset = seed.groupby(["dataset", "runner"], as_index=False).agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    dataset["variant"] = dataset["runner"].map(labels)
    dataset = dataset[["dataset", "variant", "runner", "source_seeds", "f1_mean", "f1_std"]].sort_values(["dataset", "runner"])
    flow = frame.groupby(["dataset", "scenario", "runner"], as_index=False).agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    flow["variant"] = flow["runner"].map(labels)
    return dataset, flow[["dataset", "scenario", "variant", "runner", "source_seeds", "f1_mean", "f1_std"]].sort_values(["dataset", "scenario", "runner"])


def _validate_efficiency(root: Path, protocol: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    manifest = _json(root / "manifest.json")
    # The v4 overhead runner writes the completion state in
    # ``finalization.json`` while its runtime manifest remains a provenance
    # record.  Prefer the signed finalization state when present.
    finalization_path = root / "finalization.json"
    finalization = _json(finalization_path) if finalization_path.is_file() else {}
    expected_protocol = protocol["efficiency"]["protocol"]
    _require((manifest.get("protocol") or finalization.get("protocol")) == expected_protocol, "efficiency protocol mismatch")
    _require((finalization.get("status") or manifest.get("status")) == "complete", "efficiency panel is incomplete")
    _require(int(finalization.get("expected_cells", manifest.get("expected_cells", -1))) == 12, "efficiency expected row count mismatch")
    frame = _csv(root / "method_overhead.csv")
    _require(len(frame) == 12, "efficiency overhead row count mismatch")
    _require(frame["status"].astype(str).eq("ok").all(), "efficiency contains failed rows")
    _require(set(frame["method"].astype(str)) == set(EFFICIENCY_METHODS), "efficiency methods mismatch")
    _require(set(frame.loc[frame["method"].astype(str) == "DuSafe", "variant"].astype(str)) == {"full", "no_ssaw"}, "efficiency DuSafe variants mismatch")
    _require(frame["prediction_timing_scope"].astype(str).isin({"source_inference", "online_update_plus_post_update_prediction"}).all(), "efficiency prediction timing scope mismatch")
    for field in ("source_checkpoint_sha256", "registered_source_checkpoint_sha256", "registered_source_config_sha256"):
        if field in frame.columns:
            values = frame[field].astype(str)
            _require(values.str.fullmatch(SHA256_RE.pattern).all(), f"efficiency invalid {field}")
    if {"source_checkpoint_sha256", "registered_source_checkpoint_sha256"}.issubset(frame.columns):
        _require((frame["source_checkpoint_sha256"].astype(str) == frame["registered_source_checkpoint_sha256"].astype(str)).all(), "efficiency source checkpoint registration mismatch")
    if "flow_source_profile_applied" in frame.columns:
        _require(frame["flow_source_profile_applied"].astype(str).str.lower().isin({"true", "1"}).all(), "efficiency flow source profile was not applied")
    for field, expected in (("target_selected_descriptive", True), ("confirmatory", False)):
        if field in frame.columns:
            values = frame[field].astype(str).str.lower()
            _require(values.isin({str(expected).lower(), "1" if expected else "0"}).all(), f"efficiency {field} mismatch")
    return frame, {**manifest, **finalization}


def _old_v4_comparison(new_frame: pd.DataFrame, old_root: Path, panel: str, output: Path) -> dict[str, Any]:
    old_path = Path(old_root) / ("main_full_no_ssaw" if panel == "main" else "core_ablation_har_hhar") / "raw.csv"
    if not old_path.is_file():
        return {"panel": panel, "status": "missing_old_v4", "old_path": str(old_path), "cell_rows": 0, "equivalent_cells": 0}
    old = pd.read_csv(old_path)
    old["dataset"] = old["dataset"].astype(str).str.upper()
    old["scenario"] = old["scenario"].astype(str).str.replace("→", "->", regex=False)
    keys = MAIN_KEYS
    columns = [*keys, "f1"]
    left = new_frame[columns].rename(columns={"f1": "f1_v5"})
    right = old[columns].rename(columns={"f1": "f1_v4"})
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    merged["f1_delta"] = merged["f1_v5"] - merged["f1_v4"]
    merged["equivalent"] = (merged["_merge"] == "both") & np.isclose(merged["f1_v5"], merged["f1_v4"], rtol=0.0, atol=0.0, equal_nan=False)
    _write_csv(output / f"old_v4_{panel}_cell_comparison.csv", merged)
    summary = merged.groupby(["dataset", "runner"], dropna=False, as_index=False).agg(cell_rows=("_merge", "size"), f1_v5_mean=("f1_v5", "mean"), f1_v4_mean=("f1_v4", "mean"), f1_delta_mean=("f1_delta", "mean"), equivalent_cells=("equivalent", "sum"))
    _write_csv(output / f"old_v4_{panel}_summary_comparison.csv", summary)
    return {"panel": panel, "status": "compared", "old_path": str(old_path), "cell_rows": int(len(merged)), "equivalent_cells": int(merged["equivalent"].sum()), "summary_rows": int(len(summary))}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--main-dir", type=Path)
    parser.add_argument("--main-dirs", nargs="+", type=Path)
    parser.add_argument("--core-dir", type=Path)
    parser.add_argument("--core-dirs", nargs="+", type=Path)
    parser.add_argument("--safety-dir", type=Path)
    parser.add_argument("--heldout-dir", type=Path)
    parser.add_argument("--confidence-dir", type=Path)
    parser.add_argument("--augmentation-dir", type=Path)
    parser.add_argument("--augmentation-dirs", nargs="+", type=Path)
    parser.add_argument(
        "--allow-missing-mechanism-panels",
        action="store_true",
        help="write an incomplete manifest when one of the three required evidence panels is absent",
    )
    parser.add_argument("--old-v4-root", type=Path, default=DEFAULT_OLD_V4_ROOT)
    parser.add_argument("--efficiency-dir", type=Path, default=DEFAULT_EFFICIENCY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _json(args.protocol)
    _require(config.get("protocol", "").startswith("paper_evidence_v5_"), "not a paper_evidence_v5 protocol")
    current_hash = production_code_sha256()
    profiles_path = _norm_path(ROOT / config["source_profile_json"])
    profiles = _load_profiles(profiles_path)
    reference_path = _norm_path(ROOT / config["source_reference_csv"])
    source_reference = _load_source_reference(reference_path)
    protocol = dict(config)
    protocol["_profiles"] = profiles
    protocol["_source_reference"] = source_reference
    protocol["source_cache_roots"] = {dataset: _norm_path(ROOT / value) for dataset, value in config["source_cache_roots"].items()}
    root = _norm_path(args.root)
    if args.main_dirs:
        main_dirs = _expand_path_values(args.main_dirs)
    elif args.main_dir:
        main_dirs = [_norm_path(args.main_dir)]
    else:
        legacy_main = root / "main_full_no_ssaw"
        split_main = [root / "main_nonhhar", root / "main_hhar"]
        main_dirs = [legacy_main] if legacy_main.is_dir() else [path for path in split_main if path.is_dir()]
    if args.core_dirs:
        core_dirs = _expand_path_values(args.core_dirs)
    elif args.core_dir:
        core_dirs = [_norm_path(args.core_dir)]
    else:
        legacy_core = root / "core_ablation_har_hhar"
        split_core = [root / "core_har", root / "core_hhar"]
        core_dirs = [legacy_core] if legacy_core.is_dir() else [path for path in split_core if path.is_dir()]
    safety_dir = _norm_path(args.safety_dir or root / "safety_har_12_to_16_physical_s3_s6")
    output = _norm_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main, main_manifest = _validate_multi_panel(main_dirs, label="main", protocol=protocol, all_datasets=tuple(config["datasets"]), variants=MAIN_VARIANTS, expected_total=int(config["counts"]["main_cells"]))
    core, core_manifest = _validate_multi_panel(core_dirs, label="core", protocol=protocol, all_datasets=tuple(config["core_datasets"]), variants=CORE_VARIANTS, expected_total=int(config["counts"]["core_cells"]))
    _validate_core_logging(core)
    safety, safety_manifest = _validate_safety(safety_dir, protocol)
    efficiency, efficiency_manifest = _validate_efficiency(_norm_path(args.efficiency_dir), protocol)
    panel_specs = config.get("mechanism_panels", {})
    panel_args = {
        "heldout_hhar": args.heldout_dir,
        "confidence_eeg": args.confidence_dir,
        "augmentation_controls": (
            _expand_path_values(args.augmentation_dirs)
            if args.augmentation_dirs
            else args.augmentation_dir
        ),
    }
    mechanism_frames: dict[str, pd.DataFrame] = {}
    mechanism_manifests: dict[str, dict[str, Any]] = {}
    mechanism_panel_paths: dict[str, list[Path]] = {}
    mechanism_comparisons: list[dict[str, Any]] = []
    missing_mechanism: list[str] = []
    for panel_name, panel_spec in panel_specs.items():
        configured_panel = panel_args.get(panel_name)
        if panel_name == "augmentation_controls" and configured_panel is None:
            split_defaults = [root / "augmentation_har", root / "augmentation_hhar"]
            panel_dirs = split_defaults if all(path.is_dir() for path in split_defaults) else [root / Path(str(panel_spec["root"])).name]
        elif isinstance(configured_panel, (list, tuple)):
            panel_dirs = list(configured_panel)
        else:
            default_panel = root / Path(str(panel_spec["root"]).split("/")[-1])
            panel_dirs = [configured_panel or default_panel]
        panel_dirs = [_norm_path(path) for path in panel_dirs]
        mechanism_panel_paths[panel_name] = panel_dirs
        if not panel_dirs or any(not panel_dir.is_dir() for panel_dir in panel_dirs):
            missing_mechanism.append(panel_name)
            continue
        if panel_name == "augmentation_controls" and len(panel_dirs) > 1:
            panel_frame, panel_manifest = _validate_multi_augmentation_panel(
                panel_dirs,
                label=panel_name,
                spec=panel_spec,
                protocol=protocol,
            )
        else:
            panel_frame, panel_manifest = _validate_mechanism_panel(
                panel_dirs[0],
                label=panel_name,
                spec=panel_spec,
                protocol=protocol,
            )
        mechanism_frames[panel_name] = panel_frame
        mechanism_manifests[panel_name] = panel_manifest
        _write_csv(output / f"{panel_name}_raw_normalized.csv", panel_frame)
        _write_csv(output / f"{panel_name}_summary.csv", _aggregate_mechanism_panel(panel_frame, spec=panel_spec))
        old_root = _norm_path(ROOT / panel_spec["old_v4_root"])
        mechanism_comparisons.append(_old_v4_mechanism_comparison(panel_frame, old_root=old_root, label=panel_name, spec=panel_spec, output=output))
    paper_table_contract = config.get("paper_table_golden_reference", {})
    paper_table_labels = set(paper_table_contract.get("panels", {}))
    missing_paper_table_panels = sorted(paper_table_labels - set(mechanism_frames))
    if missing_paper_table_panels:
        paper_table_golden = {
            "contract": str(paper_table_contract.get("contract", "")),
            "status": "incomplete_missing_current_panels",
            "missing_panels": missing_paper_table_panels,
            "historical_execution_comparisons_are_not_paper_table_references": True,
        }
    else:
        paper_table_golden = _validate_paper_table_golden_references(
            mechanism_frames,
            mechanism_manifests,
            contract=paper_table_contract,
            output=output,
        )
    if missing_mechanism and not args.allow_missing_mechanism_panels:
        raise EvidenceError(f"missing required mechanism evidence panels: {missing_mechanism}")
    # Main/core share two exact variants on the ten common flows.  This is an
    # invariant of the paired evidence design, not a statistical comparison.
    shared_columns = [*MAIN_KEYS, "f1"]
    shared = core[core["runner"].isin(MAIN_VARIANTS)][shared_columns].merge(
        main[main["dataset"].isin(config["core_datasets"])][shared_columns],
        on=list(MAIN_KEYS), suffixes=("_core", "_main"), how="outer", indicator=True,
    )
    _require(len(shared) == 60 and (shared["_merge"] == "both").all(), "main/core shared key set mismatch")
    _require(np.allclose(shared["f1_core"], shared["f1_main"], rtol=0.0, atol=0.0), "main/core shared F1 is not bitwise identical")
    main_dataset, main_flow = _aggregate_main(main)
    core_dataset, core_flow = _aggregate_core(core)
    _write_csv(output / "main_raw_normalized.csv", main)
    _write_csv(output / "core_raw_normalized.csv", core)
    _write_csv(output / "safety_raw_normalized.csv", safety)
    _write_csv(output / "main_dataset_summary.csv", main_dataset)
    _write_csv(output / "main_flow_summary.csv", main_flow)
    _write_csv(output / "core_ablation_dataset_summary.csv", core_dataset)
    _write_csv(output / "core_ablation_flow_summary.csv", core_flow)
    _write_csv(output / "efficiency_overhead.csv", efficiency)
    if (safety_dir / "summary_aggregate.csv").is_file():
        _write_csv(output / "safety_summary.csv", _csv(safety_dir / "summary_aggregate.csv"))
    else:
        _write_csv(output / "safety_summary.csv", safety)
    comparisons = [
        _old_v4_comparison(main, args.old_v4_root, "main", output),
        _old_v4_comparison(core, args.old_v4_root, "core", output),
    ]
    mechanism_status = {
        panel_name: {
            "status": "validated" if panel_name in mechanism_frames else "missing_optional",
            "rows": int(len(mechanism_frames[panel_name])) if panel_name in mechanism_frames else 0,
            "root": [str(path) for path in mechanism_panel_paths.get(panel_name, [])],
            "logging_mode": panel_specs[panel_name].get("logging_mode"),
            "graph": panel_specs[panel_name].get("graph"),
        }
        for panel_name in panel_specs
    }
    final = {
        "protocol": config["protocol"],
        "status": "incomplete_missing_mechanism_panels" if missing_mechanism else "complete",
        "evidence_status": "target_selected_descriptive_not_confirmatory",
        "target_selected_descriptive": True,
        "confirmatory": False,
        "production_code_sha256": current_hash,
        "ablation_code_sha256": ablation_code_sha256(),
        "ablation_code_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in ABLATION_CODE_FILES],
        "causal_evidence_code_sha256": causal_evidence_code_sha256(),
        "causal_evidence_code_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CAUSAL_EVIDENCE_CODE_FILES],
        "counts": config["counts"],
        "input_dirs": {"main": [str(path) for path in main_dirs], "core": [str(path) for path in core_dirs], "safety": str(safety_dir)},
        "source_cache_roots": {key: str(value) for key, value in protocol["source_cache_roots"].items()},
        "source_metadata_context_definition": config["source_identity"]["metadata_context_definition"],
        "logging_modes": {"main": "production", "core": "production", "safety": "evidence_graph_disabled", "mechanism": "evidence_graph_disabled"},
        "mechanism_panels": mechanism_status,
        "contract_checks": {
            "source_checkpoint_identity": "passed",
            "source_metadata_context_identity": "passed",
            "source_cache_root_partition": "passed",
            "unique_cell_keys": "passed",
            "main_core_shared_cells_bitwise_equal": "passed",
            "safety_evidence_logging_graph_off": "passed",
            "production_code_hash": "passed",
            "ablation_code_hash": "passed",
            "causal_evidence_code_hash": "passed",
            "efficiency_v4_reference": "passed",
            "mechanism_panels": "passed" if not missing_mechanism else "incomplete",
            "paper_table_golden_claim_preservation": paper_table_golden["status"],
        },
        "old_v4_equivalence": comparisons,
        "old_v4_mechanism_equivalence": {
            "role": "historical_execution_comparison_only_not_paper_table_claim_reference",
            "comparisons": mechanism_comparisons,
        },
        "paper_table_golden_reference": paper_table_golden,
        "efficiency_reference": {
            "root": str(_norm_path(args.efficiency_dir)),
            "protocol": efficiency_manifest.get("protocol"),
            "rows": int(len(efficiency)),
            "rerun": False,
        },
        "outputs": {name: str(output / name) for name in (
            "main_dataset_summary.csv", "main_flow_summary.csv", "core_ablation_dataset_summary.csv",
            "core_ablation_flow_summary.csv", "safety_summary.csv", "efficiency_overhead.csv",
            "main_raw_normalized.csv", "core_raw_normalized.csv", "safety_raw_normalized.csv",
        )},
    }
    for panel_name in mechanism_frames:
        final["outputs"][f"{panel_name}_raw_normalized.csv"] = str(output / f"{panel_name}_raw_normalized.csv")
        final["outputs"][f"{panel_name}_summary.csv"] = str(output / f"{panel_name}_summary.csv")
    _write_json(output / "manifest.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
