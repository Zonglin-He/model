"""Formal, resume-safe HHAR experiment queue.

The queue is a protocol planner and serial executor for the audited AdaTime
HHAR transfer.  It keeps the formal contract in one place:

* the ten registered AdaTime flows are never inferred from a user-supplied
  subset;
* source seeds ``1,2,3`` are independent replication units and stream seed
  ``42`` is paired across methods;
* every method receives the source checkpoint from the same cache directory;
* target labels are never a selection input;
* the HHAR orientation script performs both orientation and source-only
  stage-2 coordinate calibration before any formal transfer cell is released;
* the overhead representative is the registered HHAR ``0->6`` flow with the
  same eleven-method benchmark registry and common batch size 48.

Dry-run is the default.  The executor never starts more than one subprocess at
a time.  A requested CUDA execution additionally holds an exclusive lock for
the whole queue.  Status and manifest updates use atomic replacement so a
reader can resume from the last completed stage after interruption or OOM.

This module does not import or modify ``algorithms.dusafe``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data_model_configs import HHAR, validate_scenario  # noqa: E402


PROTOCOL_VERSION = "formal_hhar_adatime_queue_v1"
DEFAULT_PREREQUISITE = ROOT / "results" / "background" / "reviewer_remaining_queue.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "hhar_formal_queue"
DEFAULT_PRETRAIN_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "hhar_formal"
DEFAULT_FISHER_CACHE_DIR = ROOT / "results" / "eata_fisher_cache" / "hhar_formal"
GPU_LOCK_PATH = ROOT / "results" / ".hhar_formal_gpu.lock"

FORMAL_SOURCE_SEEDS = (1, 2, 3)
FORMAL_STREAM_SEED = 42
FORMAL_FLOWS = tuple((str(source), str(target)) for source, target in HHAR.scenarios)
FORMAL_METHODS = (
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
FORMAL_CORRUPTIONS = (
    "signal_freeze",
    "blackout",
    "attenuation",
    "amplitude_drift",
    "packet_loss",
    "saturation",
)
FORMAL_SEVERITIES = ("moderate", "severe")
# Frozen dimensions emitted by calibrate_hhar_orientation_source.py: nine
# source users, three source seeds, six registered orientation strengths, then
# 5 auxiliary-weight + 5 learning-rate + 4 steps coordinate candidates across
# two source-only conditions.
FORMAL_ORIENTATION_CALIBRATION_CELLS = 9 * len(FORMAL_SOURCE_SEEDS) * 6
FORMAL_STAGE2_COORDINATE_CANDIDATES = 5 + 5 + 4
FORMAL_STAGE2_CALIBRATION_CELLS = (
    FORMAL_STAGE2_COORDINATE_CANDIDATES
    * 9
    * len(FORMAL_SOURCE_SEEDS)
    * 2
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _is_gpu_device(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _explicit_true(value: Any) -> bool:
    """Recognize a literal true claim, not prose such as ``not verified``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _target_selection_in_mapping(payload: Mapping[str, Any]) -> bool:
    """Find target-label selection claims without rejecting target-free flags."""

    for raw_key, value in payload.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if isinstance(value, Mapping) and _target_selection_in_mapping(value):
            return True
        if not _truthy(value):
            continue
        if key in {
            "target_labels_used_for_selection",
            "target_labels_used_for_tuning",
            "target_selected",
            "target_selection",
            "selection_uses_target_labels",
            "uses_target_labels_for_selection",
        }:
            return True
        if "target" in key and "label" in key and (
            "select" in key or "tuning" in key or "tune" in key
        ):
            return True
        if key in {"selection_provenance", "selection_source", "selection_split"}:
            normalized = str(value).strip().lower().replace("_", "-")
            if normalized in {
                "target",
                "target-selected",
                "target-labels",
                "target-test",
                "target-validation",
            }:
                return True
    return False


def reject_target_selected_config(
    *,
    target_labels_used_for_selection: bool = False,
    selection_provenance: str = "source-only",
    selection_config: Mapping[str, Any] | None = None,
) -> None:
    """Reject any formal queue configuration selected with target labels."""

    provenance = str(selection_provenance).strip().lower().replace("_", "-")
    if target_labels_used_for_selection or provenance in {
        "target",
        "target-selected",
        "target-labels",
        "target-test",
        "target-validation",
    }:
        raise ValueError(
            "Formal HHAR queue rejects target-selected configuration: "
            "target labels may not select or tune any stage."
        )
    if selection_config is not None and _target_selection_in_mapping(selection_config):
        raise ValueError(
            "Formal HHAR queue rejects a selection manifest that uses target labels."
        )


def _load_manifest_mapping(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None, f"missing calibration manifest: {manifest_path}"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid calibration manifest {manifest_path}: {exc}"
    if not isinstance(payload, Mapping):
        return None, f"calibration manifest must be a JSON object: {manifest_path}"
    return dict(payload), None


def validate_source_only_calibration_manifests(
    orientation_manifest: str | Path,
    tta_manifest: str | Path,
) -> tuple[bool, list[str]]:
    """Validate the complete HHAR orientation+stage-2 calibration outputs.

    ``calibrate_hhar_orientation_source.py`` is intentionally one calibration
    stage.  Its ``manifest.json`` is the completion gate and its
    ``selected_profile.json`` is the frozen runtime selection.  A path to an
    orientation-only artifact, a missing stage-2 selection, or any missing
    target-exclusion flag fails closed.
    """

    errors: list[str] = []
    orientation, orientation_error = _load_manifest_mapping(orientation_manifest)
    profile, profile_error = _load_manifest_mapping(tta_manifest)
    if orientation_error:
        errors.append(orientation_error)
    if profile_error:
        errors.append(profile_error)

    def _require_false_flags(
        payload: Mapping[str, Any], label: str, keys: tuple[str, ...]
    ) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            errors.append(f"{label} lacks explicit target-label exclusion: {missing}")
        if any(_explicit_true(payload.get(key)) for key in keys if key in payload):
            errors.append(f"{label} selected with target labels or target data")
        if _target_selection_in_mapping(payload):
            errors.append(f"{label} uses target labels for selection")

    def _number(
        payload: Mapping[str, Any], key: str, label: str, *, positive: bool = False
    ) -> float | None:
        try:
            value = float(payload[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} lacks numeric {key}")
            return None
        if not math.isfinite(value) or value < 0 or (positive and value <= 0):
            errors.append(f"{label} {key} must be finite and {'positive' if positive else 'non-negative'}")
            return None
        return value

    if orientation is not None:
        strength_path = Path(orientation_manifest).expanduser().parent / "selected_strength.json"
        strength_selection, strength_error = _load_manifest_mapping(strength_path)
        if strength_error:
            errors.append(strength_error)
        elif strength_selection is not None:
            for key in ("target_labels_used", "target_metrics_used"):
                if key not in strength_selection:
                    errors.append(f"selected_strength.json lacks explicit {key}=false")
                elif _explicit_true(strength_selection[key]):
                    errors.append("selected_strength.json selected with target labels")
            if _target_selection_in_mapping(strength_selection):
                errors.append("selected_strength.json uses target labels for selection")
        if str(orientation.get("dataset", "")).upper() != "HHAR":
            errors.append("orientation calibration dataset must be HHAR")
        if str(orientation.get("status", "")).lower() != "complete":
            errors.append("orientation calibration manifest status must be complete (stage 2 included)")
        _require_false_flags(
            orientation,
            "orientation calibration manifest",
            (
                "target_labels_used",
                "target_labels_used_for_selection",
                "target_metrics_used",
                "target_data_used",
            ),
        )
        if orientation.get("target_transfer_flows_excluded") is not True:
            errors.append("orientation calibration must explicitly exclude target transfer flows")
        orientation_selection = orientation.get("orientation_selection")
        if not isinstance(orientation_selection, Mapping):
            errors.append("orientation calibration lacks completed orientation_selection")
            orientation_selection = {}
        _number(orientation_selection, "selected_strength_deg", "orientation selection")
        second_stage = orientation.get("second_stage")
        if not isinstance(second_stage, Mapping):
            errors.append("orientation calibration lacks second_stage manifest")
            second_stage = {}
        if str(second_stage.get("status", "")).lower() != "complete":
            errors.append("orientation calibration second_stage status must be complete")
        if not isinstance(second_stage.get("selected_profile"), Mapping):
            errors.append("orientation calibration lacks second_stage.selected_profile")
        serialized = json.dumps(orientation, sort_keys=True, default=str).lower()
        if "source" not in serialized or not (
            "holdout" in serialized or "corruption" in serialized
        ):
            errors.append(
                "orientation calibration must document source-domain holdout/controlled-corruption selection"
            )

    if profile is not None:
        # The final selected_profile.json deliberately contains only the
        # frozen selection; dataset/status are verified against manifest.json.
        _require_false_flags(
            profile,
            "selected calibration profile",
            ("target_labels_used", "target_data_used"),
        )
        selected = profile.get("selected_profile")
        if isinstance(selected, Mapping):
            selected = selected
        else:
            selected = profile
        orientation_selection = selected.get("orientation")
        adaptation = selected.get("adaptation")
        if not isinstance(orientation_selection, Mapping):
            errors.append("selected calibration profile lacks orientation selection")
            orientation_selection = {}
        if not isinstance(adaptation, Mapping):
            errors.append("selected calibration profile lacks adaptation selection")
            adaptation = {}
        _number(orientation_selection, "selected_strength_deg", "selected orientation")
        aux = adaptation.get("auxiliary_weight", adaptation.get("ssaw_auxiliary_weight"))
        if aux is None:
            errors.append("selected calibration profile lacks auxiliary_weight")
        else:
            try:
                aux_value = float(aux)
                if not math.isfinite(aux_value) or aux_value < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("selected calibration auxiliary_weight must be finite and non-negative")
        _number(adaptation, "learning_rate", "selected adaptation", positive=True)
        try:
            steps = int(adaptation["steps"])
            if steps < 1 or float(adaptation["steps"]) != steps:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append("selected adaptation steps must be a positive integer")
        serialized = json.dumps(profile, sort_keys=True, default=str).lower()
        if "source" not in serialized or not (
            "holdout" in serialized or "corruption" in serialized
        ):
            errors.append(
                "selected calibration profile must document source-domain holdout/controlled-corruption selection"
            )

    return not errors, errors


def read_status(path: str | Path) -> Mapping[str, Any]:
    """Read a reviewer prerequisite status object."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Queue prerequisite must be a JSON object: {path}")
    return payload


def prerequisite_complete(path: str | Path = DEFAULT_PREREQUISITE) -> bool:
    """Return true only after the reviewer queue reports ``phase=complete``."""

    try:
        payload = read_status(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if str(payload.get("phase", "")).strip().lower() != "complete":
        return False
    # A literal true claim for HHAR in the prerequisite must not certify a
    # source-only formal queue.  ``not verified`` remains an explicit review
    # gap but does not claim target selection.
    formal_claim = payload.get("formal_queue_target_labels_used_for_selection")
    if isinstance(formal_claim, Mapping) and _explicit_true(formal_claim.get("HHAR")):
        return False
    if _explicit_true(payload.get("hhar_target_labels_used_for_selection")):
        return False
    return True


def _validate_formal_seeds(
    source_seeds: Iterable[int], stream_seed: int
) -> tuple[list[int], int]:
    seeds = [int(seed) for seed in source_seeds]
    if seeds != list(FORMAL_SOURCE_SEEDS):
        raise ValueError(
            "Formal HHAR queue requires source seeds exactly [1, 2, 3] "
            f"in that order; received {seeds!r}."
        )
    stream_seed = int(stream_seed)
    if stream_seed != FORMAL_STREAM_SEED:
        raise ValueError(
            f"Formal HHAR queue requires paired stream seed 42; received {stream_seed}."
        )
    return seeds, stream_seed


def _validate_formal_flows(
    flows: Iterable[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    observed = FORMAL_FLOWS if flows is None else tuple(
        (str(source), str(target)) for source, target in flows
    )
    if observed != FORMAL_FLOWS:
        raise ValueError(
            "Formal HHAR queue requires the exact AdaTime ten-flow order: "
            + ", ".join(f"{source}->{target}" for source, target in FORMAL_FLOWS)
        )
    for source, target in observed:
        validate_scenario("HHAR", source, target)
    return list(observed)


def _hhar_data_dir(data_path: str | Path) -> Path:
    path = Path(data_path).expanduser()
    if path.name.strip().upper() == "HHAR":
        return path
    return path / "HHAR"


def _command(*parts: object) -> list[str]:
    return [str(part) for part in parts]


def _override_args(overrides: Mapping[str, Any] | None) -> list[str]:
    """Encode frozen calibration values for runners with repeatable flags."""

    result: list[str] = []
    for key, value in (overrides or {}).items():
        result.extend(("--override", f"{key}={value}"))
    return result


def calibration_runtime_overrides(profile_manifest: str | Path) -> dict[str, Any]:
    """Read and validate the final HHAR source-only runtime selection."""

    payload, error = _load_manifest_mapping(profile_manifest)
    if error or payload is None:
        raise ValueError(error or f"missing selected calibration profile: {profile_manifest}")
    selected = payload.get("selected_profile")
    if not isinstance(selected, Mapping):
        selected = payload
    orientation = selected.get("orientation")
    adaptation = selected.get("adaptation")
    if not isinstance(orientation, Mapping) or not isinstance(adaptation, Mapping):
        raise ValueError(
            "selected calibration profile must contain orientation and adaptation mappings"
        )
    try:
        strength = float(orientation["selected_strength_deg"])
        auxiliary_raw = adaptation.get("auxiliary_weight")
        if auxiliary_raw is None:
            auxiliary_raw = adaptation.get("ssaw_auxiliary_weight")
        auxiliary_weight = float(auxiliary_raw)
        learning_rate = float(adaptation["learning_rate"])
        steps_float = float(adaptation["steps"])
        steps = int(adaptation["steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete selected calibration runtime profile: {exc}") from exc
    if (
        not math.isfinite(strength)
        or strength < 0
        or not math.isfinite(auxiliary_weight)
        or auxiliary_weight < 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or steps < 1
        or steps_float != steps
    ):
        raise ValueError("selected calibration runtime values are outside the formal domain")
    return {
        "ssaw_strength": strength,
        "ssaw_auxiliary_weight": auxiliary_weight,
        "learning_rate": learning_rate,
        "steps": steps,
        "ssaw_sigma": 0.0,
        "normalization_reference": "source",
    }


def _append_runtime_overrides(command: list[str], overrides: Mapping[str, Any]) -> None:
    command.extend(_override_args(overrides))


def apply_runtime_overrides(
    plan: dict[str, Any], overrides: Mapping[str, Any]
) -> None:
    """Apply one frozen source-only profile to every formal adaptation stage."""

    adaptation_stages = {
        "smoke",
        "main_table",
        "controlled_safety",
        "full_no_ssaw",
        "factorial",
        "overhead",
    }
    for stage in plan.get("stages", []):
        if str(stage.get("id")) not in adaptation_stages:
            continue
        commands = stage.get("commands", [])
        for command in commands:
            # Avoid duplicate flags when a plan was built from an existing
            # selected_profile and then resumed after orientation completion.
            if any(str(item) == "--override" for item in command):
                existing = {
                    str(command[index + 1]).split("=", 1)[0]
                    for index, item in enumerate(command[:-1])
                    if str(item) == "--override" and "=" in str(command[index + 1])
                }
                for key, value in overrides.items():
                    if key not in existing:
                        command.extend(("--override", f"{key}={value}"))
            else:
                _append_runtime_overrides(command, overrides)
        if len(commands) == 1:
            stage["command"] = commands[0]
    plan["runtime_overrides"] = dict(overrides)
    plan["runtime_overrides_applied"] = True
    plan.setdefault("calibration_artifacts", {})["runtime_overrides_status"] = "applied"


def validate_smoke_output(output_dir: str | Path) -> tuple[bool, list[str]]:
    """Require all eleven representative-flow smoke cells to succeed."""

    raw_path = Path(output_dir) / "per_source_seed_results.csv"
    if not raw_path.is_file():
        return False, [f"smoke output is missing: {raw_path}"]
    try:
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return False, [f"cannot read smoke output {raw_path}: {exc}"]
    expected = {
        ("HHAR", "0->6", method, "1", "42") for method in FORMAL_METHODS
    }
    observed: set[tuple[str, str, str, str, str]] = set()
    errors: list[str] = []
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("scenario", "")),
            str(row.get("method", "")),
            str(row.get("source_seed", "")),
            str(row.get("stream_seed", "")),
        )
        if key in observed:
            errors.append(f"duplicate smoke cell: {key}")
        observed.add(key)
        if key in expected and str(row.get("status", "")).lower() != "ok":
            errors.append(
                f"smoke cell failed/OOM: {key} status={row.get('status')} "
                f"is_oom={row.get('is_oom')}"
            )
    missing = expected - observed
    if missing:
        errors.append(f"smoke output is incomplete; missing {sorted(missing)}")
    unexpected = observed - expected
    if unexpected:
        errors.append(f"smoke output contains unexpected cells: {sorted(unexpected)}")
    return not errors, errors


def _stage(
    *,
    stage_id: str,
    phase: str,
    script: str | None,
    commands: list[list[str]],
    uses_gpu: bool,
    depends_on: Iterable[str] = (),
    status: str = "planned",
    blocked_reason: str | None = None,
    flow_count: int | None = None,
    expected_cells: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": stage_id,
        "phase": phase,
        "script": script,
        "commands": commands,
        # Keep a singular command for callers that inspect one-command stages.
        "command": commands[0] if len(commands) == 1 else None,
        "command_count": len(commands),
        "uses_gpu": bool(uses_gpu),
        "depends_on": list(depends_on),
        "status": status,
        "target_labels_used_for_selection": False,
        "attempts": 0,
    }
    if flow_count is not None:
        record["flow_count"] = int(flow_count)
    if expected_cells is not None:
        record["expected_cells"] = int(expected_cells)
    if blocked_reason:
        record["blocked_reason"] = blocked_reason
    return record


def build_formal_plan(
    *,
    data_path: str | Path,
    device: str = "cpu",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pretrain_cache_dir: str | Path = DEFAULT_PRETRAIN_CACHE_DIR,
    fisher_cache_dir: str | Path = DEFAULT_FISHER_CACHE_DIR,
    backbone: str = "CNN",
    source_seeds: Iterable[int] = FORMAL_SOURCE_SEEDS,
    stream_seed: int = FORMAL_STREAM_SEED,
    target_labels_used_for_selection: bool = False,
    selection_provenance: str = "source-only",
    selection_config: Mapping[str, Any] | None = None,
    orientation_manifest: str | Path | None = None,
    tta_calibration_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Build the formal HHAR queue without touching data or starting a job."""

    source_seeds, stream_seed = _validate_formal_seeds(source_seeds, stream_seed)
    flows = _validate_formal_flows()
    reject_target_selected_config(
        target_labels_used_for_selection=target_labels_used_for_selection,
        selection_provenance=selection_provenance,
        selection_config=selection_config,
    )

    data_root = Path(data_path).expanduser()
    output_root = Path(output_dir).expanduser()
    cache_root = Path(pretrain_cache_dir).expanduser()
    fisher_root = Path(fisher_cache_dir).expanduser()
    hhar_dir = _hhar_data_dir(data_root)
    python = sys.executable
    scripts = ROOT / "scripts"
    schema_dir = output_root / "schema_audit"
    orientation_dir = output_root / "orientation_calibration"
    smoke_dir = output_root / "smoke"
    main_dir = output_root / "main_table"
    safety_dir = output_root / "controlled_safety"
    bundle_dir = output_root / "full_no_ssaw"
    factorial_dir = output_root / "factorial"
    orientation_manifest_path = Path(
        orientation_manifest or (orientation_dir / "manifest.json")
    ).expanduser()
    profile_manifest_path = Path(
        tta_calibration_manifest or (orientation_dir / "selected_profile.json")
    ).expanduser()

    calibration_validation_status = "pending_calibration"
    calibration_validation_errors: list[str] = []
    runtime_overrides: dict[str, Any] = {}
    if orientation_manifest_path.is_file() and profile_manifest_path.is_file():
        calibration_valid, calibration_validation_errors = (
            validate_source_only_calibration_manifests(
                orientation_manifest_path, profile_manifest_path
            )
        )
        if calibration_valid:
            try:
                runtime_overrides = calibration_runtime_overrides(profile_manifest_path)
            except ValueError as exc:
                calibration_validation_errors.append(str(exc))
                calibration_valid = False
        calibration_validation_status = "valid" if calibration_valid else "invalid"

    schema_command = _command(
        python,
        scripts / "audit_hhar_schema.py",
        "--data-dir",
        _absolute(hhar_dir),
        "--json-out",
        _absolute(schema_dir / "schema_audit.json"),
    )
    smoke_command = _command(
        python,
        scripts / "run_full_main_table.py",
        "--data-path",
        _absolute(data_root),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--datasets",
        "HHAR",
        "--methods",
        ",".join(FORMAL_METHODS),
        "--scenarios",
        "0->6",
        "--source-seeds",
        "1",
        "--stream-seed",
        str(stream_seed),
        "--pretrain-cache-dir",
        _absolute(cache_root),
        "--eata-fisher-cache-dir",
        _absolute(fisher_root),
        "--output-dir",
        _absolute(smoke_dir),
        "--batch-policy",
        "common",
        "--common-batch-sizes",
        "HHAR=48",
        "--limit-jobs",
        str(len(FORMAL_METHODS)),
        "--retry-failures",
        "--run-signature",
        "formal-hhar-smoke-v1",
    )
    if runtime_overrides:
        _append_runtime_overrides(smoke_command, runtime_overrides)
    main_command = _command(
        python,
        scripts / "run_full_main_table.py",
        "--data-path",
        _absolute(data_root),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--datasets",
        "HHAR",
        "--methods",
        ",".join(FORMAL_METHODS),
        "--source-seeds",
        ",".join(str(seed) for seed in source_seeds),
        "--stream-seed",
        str(stream_seed),
        "--pretrain-cache-dir",
        _absolute(cache_root),
        "--eata-fisher-cache-dir",
        _absolute(fisher_root),
        "--output-dir",
        _absolute(main_dir),
        "--batch-policy",
        "common",
        "--common-batch-sizes",
        "HHAR=48",
        "--run-signature",
        "formal-hhar-main-v1",
    )
    if runtime_overrides:
        _append_runtime_overrides(main_command, runtime_overrides)

    safety_commands = []
    for source, target in flows:
        safety_commands.append(
            _command(
                python,
                scripts / "run_controlled_safety_benchmark.py",
                "--data_path",
                _absolute(data_root),
                "--device",
                str(device),
                "--backbone",
                str(backbone),
                "--registry",
                "benchmark",
                "--datasets",
                "HHAR",
                "--methods",
                ",".join(FORMAL_METHODS),
                "--variants",
                "full",
                "--scenarios",
                f"HHAR:{source}->{target}",
                "--corruptions",
                ",".join(FORMAL_CORRUPTIONS),
                "--severities",
                ",".join(FORMAL_SEVERITIES),
                "--source_seeds",
                ",".join(str(seed) for seed in source_seeds),
                "--stream_seeds",
                str(stream_seed),
                "--corruption_seed",
                "1",
                "--pretrain_cache_dir",
                _absolute(cache_root),
                "--fisher_cache_dir",
                _absolute(fisher_root),
                "--output_dir",
                _absolute(safety_dir),
            )
        )
        if runtime_overrides:
            _append_runtime_overrides(safety_commands[-1], runtime_overrides)

    # ``dual_gate_only`` is the exact no-SSAW counterpart: confidence and
    # source-semantic admission remain enabled while the complete SSAW branch
    # is removed.  The generic factorial runner already enumerates all HHAR
    # scenarios, unlike the older five-flow cumulative runner.
    bundle_command = _command(
        python,
        scripts / "run_dusafe_factorial_ablation.py",
        "--output-root",
        _absolute(bundle_dir),
        "--data-path",
        _absolute(data_root),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--datasets",
        "HHAR",
        "--source-seeds",
        ",".join(str(seed) for seed in source_seeds),
        "--stream-seed",
        str(stream_seed),
        "--runners",
        "dual_gate_only,full",
        "--pretrain-cache-dir",
        _absolute(cache_root),
    )
    factorial_command = _command(
        python,
        scripts / "run_dusafe_factorial_ablation.py",
        "--output-root",
        _absolute(factorial_dir),
        "--data-path",
        _absolute(data_root),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--datasets",
        "HHAR",
        "--source-seeds",
        ",".join(str(seed) for seed in source_seeds),
        "--stream-seed",
        str(stream_seed),
        "--runners",
        ",".join(
            (
                "raw_only",
                "confidence_only",
                "semantic_only",
                "dual_gate_only",
                "ssaw_only",
                "ssaw_confidence",
                "ssaw_semantic",
                "full",
            )
        ),
        "--pretrain-cache-dir",
        _absolute(cache_root),
    )
    if runtime_overrides:
        _append_runtime_overrides(bundle_command, runtime_overrides)
        _append_runtime_overrides(factorial_command, runtime_overrides)

    overhead_dir = output_root / "overhead"
    overhead_command = _command(
        python,
        scripts / "run_compute_overhead_v2.py",
        "--data-path",
        _absolute(data_root),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--registry",
        "benchmark",
        "--datasets",
        "HHAR",
        "--methods",
        ",".join(FORMAL_METHODS),
        "--profiles",
        "default,common",
        "--common-batch-sizes",
        "HHAR=48",
        "--source-seed",
        "1",
        "--stream-seed",
        str(stream_seed),
        "--pretrain-cache-dir",
        _absolute(cache_root),
        "--eata-fisher-cache-dir",
        _absolute(fisher_root),
        "--output-dir",
        _absolute(overhead_dir),
    )
    if runtime_overrides:
        _append_runtime_overrides(overhead_command, runtime_overrides)

    stages = [
        _stage(
            stage_id="schema_audit",
            phase="schema_audit",
            script="scripts/audit_hhar_schema.py",
            commands=[schema_command],
            uses_gpu=False,
        ),
        _stage(
            stage_id="orientation_calibration",
            phase="source_only_orientation_calibration",
            script="scripts/calibrate_hhar_orientation_source.py",
            commands=[
                _command(
                    python,
                    scripts / "calibrate_hhar_orientation_source.py",
                    "--data-path",
                    _absolute(data_root),
                    "--device",
                    str(device),
                    "--backbone",
                    str(backbone),
                    "--source-seeds",
                    ",".join(str(seed) for seed in source_seeds),
                    "--stream-seed",
                    str(stream_seed),
                    "--pretrain-cache-dir",
                    _absolute(cache_root),
                    "--output-dir",
                    _absolute(orientation_dir),
                    "--sigma",
                    "0",
                    "--eval-batch-size",
                    "48",
                )
            ],
            uses_gpu=_is_gpu_device(device),
            depends_on=("schema_audit",),
            expected_cells=(
                FORMAL_ORIENTATION_CALIBRATION_CELLS
                + FORMAL_STAGE2_CALIBRATION_CELLS
            ),
        ),
        _stage(
            stage_id="smoke",
            phase="smoke",
            script="scripts/run_full_main_table.py",
            commands=[smoke_command],
            uses_gpu=_is_gpu_device(device),
            depends_on=("orientation_calibration",),
            expected_cells=len(FORMAL_METHODS),
        ),
        _stage(
            stage_id="main_table",
            phase="fixed_source_main_table",
            script="scripts/run_full_main_table.py",
            commands=[main_command],
            uses_gpu=_is_gpu_device(device),
            depends_on=("smoke",),
            flow_count=len(flows),
            expected_cells=len(flows) * len(FORMAL_METHODS) * len(source_seeds),
        ),
        _stage(
            stage_id="controlled_safety",
            phase="dusafe_safety",
            script="scripts/run_controlled_safety_benchmark.py",
            commands=safety_commands,
            uses_gpu=_is_gpu_device(device),
            depends_on=("main_table",),
            flow_count=len(flows),
            expected_cells=(
                len(flows)
                * len(FORMAL_METHODS)
                * len(FORMAL_CORRUPTIONS)
                * len(FORMAL_SEVERITIES)
                * len(source_seeds)
            ),
        ),
        _stage(
            stage_id="full_no_ssaw",
            phase="full_vs_no_ssaw",
            script="scripts/run_dusafe_factorial_ablation.py",
            commands=[bundle_command],
            uses_gpu=_is_gpu_device(device),
            depends_on=("controlled_safety",),
            flow_count=len(flows),
            expected_cells=len(flows) * len(source_seeds) * 2,
        ),
        _stage(
            stage_id="factorial",
            phase="factorial",
            script="scripts/run_dusafe_factorial_ablation.py",
            commands=[factorial_command],
            uses_gpu=_is_gpu_device(device),
            depends_on=("full_no_ssaw",),
            flow_count=len(flows),
            expected_cells=len(flows) * len(source_seeds) * 8,
        ),
        _stage(
            stage_id="overhead",
            phase="overhead",
            script="scripts/run_compute_overhead_v2.py",
            depends_on=("factorial",),
            commands=[overhead_command],
            uses_gpu=_is_gpu_device(device),
            flow_count=1,
            expected_cells=2 * len(FORMAL_METHODS),
        ),
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": "Formal HHAR AdaTime queue",
        "formal": True,
        "formal_ready": calibration_validation_status == "valid",
        "completion_blocked": calibration_validation_status != "valid",
        "dataset": "HHAR",
        "data_path": _absolute(data_root),
        "processed_data_path": _absolute(hhar_dir),
        "flows": [f"{source}->{target}" for source, target in flows],
        "flow_count": len(flows),
        "adatime_flow_order": True,
        "source_seeds": source_seeds,
        "stream_seed": stream_seed,
        "source_seed_is_independent_unit": True,
        "stream_seed_is_paired_control": True,
        "resume_key_fields": [
            "dataset",
            "flow",
            "method",
            "source_seed",
            "stream_seed",
            "profile",
        ],
        "methods": list(FORMAL_METHODS),
        "safety_corruptions": list(FORMAL_CORRUPTIONS),
        "safety_severities": list(FORMAL_SEVERITIES),
        "safety_methods": list(FORMAL_METHODS),
        "safety_flow_count": len(safety_commands),
        "calibration_artifacts": {
            "orientation_manifest": _absolute(orientation_manifest_path),
            "profile_manifest": _absolute(profile_manifest_path),
            # Compatibility alias retained for consumers that called the
            # profile artifact a TTA manifest; it is the same HHAR file, not
            # a separate calibration stage.
            "tta_manifest": _absolute(profile_manifest_path),
            "selected_strength": _absolute(orientation_dir / "selected_strength.json"),
            "required_before_smoke": True,
            "validation_status": calibration_validation_status,
            "validation_errors": calibration_validation_errors,
            "runtime_overrides_status": (
                "applied" if runtime_overrides else "pending_calibration"
            ),
            "validator": "validate_source_only_calibration_manifests",
        },
        "source_only_ssaw_calibration": {
            "order": ["orientation_calibration"],
            "script": "scripts/calibrate_hhar_orientation_source.py",
            "includes_stage2_coordinate_calibration": True,
            "selection_split": "source-domain holdout plus controlled corruption",
            "target_labels_used_for_selection": False,
            "parameters_to_freeze": [
                "ssaw_auxiliary_weight",
                "learning_rate",
                "steps",
            ],
            "orientation_definition": (
                "source-only physical orientation calibration must precede TTA "
                "calibration for complete three-axis HHAR windows"
            ),
            "status": (
                "complete_manifest_validated"
                if calibration_validation_status == "valid"
                else "pending_complete_manifest"
            ),
        },
        "runtime_overrides": dict(runtime_overrides),
        "runtime_overrides_applied": bool(runtime_overrides),
        "overhead": {
            "representative_flow": "0->6",
            "profiles": ["default", "common"],
            "common_batch_size": 48,
            "method_count": len(FORMAL_METHODS),
            "expected_cells": 2 * len(FORMAL_METHODS),
        },
        "shared_source_checkpoint": {
            "cache_dir": _absolute(cache_root),
            "key": "dataset/source-domain/source-seed",
            "shared_by_methods": True,
            "source_training_registry": "benchmark source recipe",
            "verification": "checkpoint_consistency.csv and source_model_sha256",
        },
        "paired_panels": {
            "full_vs_no_ssaw": {
                "no_ssaw_profile": "dual_gate_only",
                "full_profile": "full",
                "paired_by": ["flow", "source_seed", "stream_seed"],
            },
            "factorial": {
                "runner_count": 8,
                "factors": ["SSAW", "confidence_gate", "source_semantic_gate"],
                "interaction_required": True,
            },
        },
        "target_labels_used_for_selection": False,
        "target_labels_used_for_tuning": False,
        "target_labels_used_online": False,
        "selection_provenance": "source-only",
        "reviewer_prerequisite_status": _absolute(DEFAULT_PREREQUISITE),
        "gpu_execution": {
            "requested_device": str(device),
            "serial": True,
            "lock_path": _absolute(GPU_LOCK_PATH),
        },
        "resume": True,
        "atomic_status": True,
        "oom_policy": (
            "record return-code/OOM as a terminal stage status, preserve all "
            "completed artifacts, release the queue lock, and resume only the "
            "failed stage on a later invocation"
        ),
        "stages": stages,
        "capability_gaps": [],
        "created_at": _utc_now(),
    }


def build_queue(
    *,
    data_path: str | Path,
    device: str = "cpu",
    da_method: str = "DuSafe",
    num_runs: int = 1,
    output_dir: str | Path = ROOT / "results" / "hhar_queue",
    source_seed: int = 1,
    stream_seed: int = FORMAL_STREAM_SEED,
) -> list[dict[str, Any]]:
    """Build the legacy ten-flow smoke commands.

    The formal multi-method queue is returned by :func:`build_formal_plan`.
    This compatibility helper remains one command per flow for callers that
    used the earlier scaffold and pins source/stream seeds explicitly.
    """

    if da_method not in {"DuSafe", "NoAdap"}:
        raise ValueError("HHAR smoke queue supports only DuSafe or NoAdap")
    if int(num_runs) < 1:
        raise ValueError("num_runs must be positive")
    output_dir = Path(output_dir)
    jobs = []
    for source, target in _validate_formal_flows():
        command = [
            sys.executable,
            str(ROOT / "trainers" / "tta_trainer.py"),
            "--da_method",
            da_method,
            "--dataset",
            "HHAR",
            "--data-path",
            str(Path(data_path).resolve()),
            "--device",
            str(device),
            "--num_runs",
            str(int(num_runs)),
            "--seed",
            str(int(stream_seed)),
            "--source_seed",
            str(int(source_seed)),
            "--scenario",
            f"{source}->{target}",
            "--save_dir",
            str(output_dir.resolve()),
        ]
        jobs.append(
            {
                "id": f"HHAR_{source}_to_{target}",
                "dataset": "HHAR",
                "scenario": f"{source}->{target}",
                "device": str(device),
                "source_seed": int(source_seed),
                "stream_seed": int(stream_seed),
                "uses_gpu": _is_gpu_device(device),
                "command": command,
            }
        )
    return jobs


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    retries: int = 20,
) -> None:
    """Publish a JSON status file atomically, tolerating transient readers."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(max(1, int(retries))):
            try:
                temporary.replace(destination)
                break
            except PermissionError:
                if attempt + 1 >= max(1, int(retries)):
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _status_path(output_dir: Path) -> Path:
    return output_dir / "status.json"


def _manifest_payload(plan: Mapping[str, Any], *, status: str, dry_run: bool) -> dict[str, Any]:
    payload = dict(plan)
    payload["status"] = status
    payload["dry_run"] = bool(dry_run)
    payload["updated_at"] = _utc_now()
    payload["stage_statuses"] = {
        str(stage["id"]): str(stage.get("status", "planned"))
        for stage in payload.get("stages", [])
    }
    return payload


def _publish_status(
    plan: dict[str, Any],
    *,
    status: str,
    dry_run: bool,
    output_dir: Path,
    json_out: str | Path | None = None,
) -> dict[str, Any]:
    payload = _manifest_payload(plan, status=status, dry_run=dry_run)
    atomic_write_json(_status_path(output_dir), payload)
    atomic_write_json(output_dir / "manifest.json", payload)
    if json_out:
        atomic_write_json(json_out, payload)
    return payload


def _merge_resume_status(plan: dict[str, Any], output_dir: Path) -> None:
    """Restore only compatible terminal records from the previous status."""

    previous = _load_json_object(_status_path(output_dir))
    if not previous or previous.get("protocol_version") != PROTOCOL_VERSION:
        return
    old_stages = {
        str(stage.get("id")): stage
        for stage in previous.get("stages", [])
        if isinstance(stage, Mapping)
    }
    for stage in plan.get("stages", []):
        old = old_stages.get(str(stage["id"]))
        if not old:
            continue
        old_status = str(old.get("status", ""))
        if old_status in {"completed"}:
            # Keep the new command declaration but carry timestamps, attempts,
            # return code and subcommand progress for auditability.
            for key in (
                "status",
                "attempts",
                "started_at",
                "finished_at",
                "returncode",
                "oom",
                "error",
                "subcommands",
            ):
                if key in old:
                    stage[key] = old[key]
        elif old_status in {"failed", "oom", "running"}:
            stage["previous_status"] = old_status
            stage["previous_error"] = old.get("error", "")


class _GPUQueueLock:
    """Small cross-platform O_EXCL lock for the whole formal queue."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "created_at": _utc_now()})
        try:
            descriptor = os.open(
                str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as exc:
            raise RuntimeError(f"HHAR GPU queue lock already exists: {self.path}") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self.owned = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.owned:
            self.path.unlink(missing_ok=True)
            self.owned = False
        return False


def is_oom_text(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda out of memory",
            "cudnn_status_alloc_failed",
            "cuda error: out of memory",
        )
    )


def is_retryable_failure(returncode: int, output: str, oom: bool) -> bool:
    """Classify transient native/resource failures for bounded retries."""

    if oom:
        return True
    # Windows access-violation and POSIX signal-style native crashes are
    # commonly emitted by CUDA extensions without useful text.
    if int(returncode) in {
        -1073741819,  # 0xC0000005
        3221225477,
        -11,  # SIGSEGV
        -6,  # SIGABRT
    }:
        return True
    lowered = str(output).lower()
    deterministic_markers = (
        "usage:",
        "invalid override",
        "unknown dataset",
        "unknown method",
        "target labels",
        "manifest",
        "schema audit",
        "requires --",
        "must be ",
        "valueerror",
        "keyerror",
    )
    return not any(marker in lowered for marker in deterministic_markers)


def _run_command(command: list[str]) -> tuple[int, str, bool]:
    """Run one child command and return ``(returncode, output, oom)``."""

    try:
        completed = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except TypeError:
        # Keep monkeypatched/test subprocess runners and older Python wrappers
        # compatible with the original subprocess.run call shape.
        completed = subprocess.run(command, cwd=ROOT, check=False)
        output = str(getattr(completed, "stdout", "") or "")[-4000:]
        returncode = int(getattr(completed, "returncode", 1))
        return returncode, output, bool(returncode and is_oom_text(output))

    # Do not retain multi-hour calibration/safety logs in memory.  The child
    # output is echoed for monitoring while only a bounded diagnostic tail is
    # kept for status.json and OOM classification.
    tail_lines: deque[str] = deque(maxlen=200)
    oom_seen = False
    stream = getattr(completed, "stdout", None)
    if stream is not None:
        for line in stream:
            text_line = str(line)
            tail_lines.append(text_line)
            oom_seen = oom_seen or is_oom_text(text_line)
            print(text_line, end="" if text_line.endswith("\n") else "\n", flush=True)
    returncode = int(completed.wait())
    output = "".join(tail_lines)[-4000:]
    return returncode, output, bool(returncode and (oom_seen or is_oom_text(output)))


def execute_plan(
    plan: dict[str, Any],
    *,
    output_dir: str | Path,
    prerequisite_status: str | Path = DEFAULT_PREREQUISITE,
    dry_run: bool = True,
    resume: bool = True,
    retry_failures: bool = True,
    max_stage_retries: int = 3,
    json_out: str | Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute or publish a plan and return ``(returncode, manifest)``."""

    if int(max_stage_retries) < 1:
        raise ValueError("max_stage_retries must be positive")
    max_stage_retries = int(max_stage_retries)
    plan["retry_policy"] = {
        "max_attempts_per_subcommand": max_stage_retries,
        "retryable": "OOM/native crash and non-validation nonzero exits",
        "resume_completed_subcommands": True,
    }
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    if resume:
        _merge_resume_status(plan, output_root)
    reviewer_ok = prerequisite_complete(prerequisite_status)
    plan["reviewer_prerequisite_status"] = _absolute(prerequisite_status)
    plan["reviewer_prerequisite_complete"] = bool(reviewer_ok)

    if not reviewer_ok:
        for stage in plan.get("stages", []):
            if str(stage.get("status")) not in {"completed"}:
                stage["status"] = "blocked_waiting_for_reviewer_queue"
                stage["blocked_reason"] = (
                    "reviewer prerequisite phase is not complete; no HHAR command was launched"
                )
        payload = _publish_status(
            plan,
            status="blocked_waiting_for_reviewer_queue",
            dry_run=True,
            output_dir=output_root,
            json_out=json_out,
        )
        return 2, payload

    if dry_run:
        payload = _publish_status(
            plan,
            status="dry_run",
            dry_run=True,
            output_dir=output_root,
            json_out=json_out,
        )
        return 0, payload

    requested_device = str(plan.get("gpu_execution", {}).get("requested_device", "cpu"))
    lock_context = _GPUQueueLock(GPU_LOCK_PATH) if _is_gpu_device(requested_device) else nullcontext()
    overall_status = "planned"
    returncode = 0
    with lock_context:
        _publish_status(
            plan,
            status="running",
            dry_run=False,
            output_dir=output_root,
            json_out=json_out,
        )
        stages = list(plan.get("stages", []))
        stage_by_id = {str(stage["id"]): stage for stage in stages}
        for position, stage in enumerate(stages):
            stage_id = str(stage["id"])
            current = str(stage.get("status", "planned"))
            if current in {"completed"}:
                continue
            if current in {"failed", "oom"} and not retry_failures:
                stage["status"] = "blocked_previous_failure"
                stage["blocked_reason"] = "resume found a prior failure and retry_failures=False"
                overall_status = "blocked"
                returncode = 1
                break
            dependency_statuses = [
                str(stage_by_id[dependency].get("status", "planned"))
                for dependency in stage.get("depends_on", [])
                if dependency in stage_by_id
            ]
            if any(status != "completed" for status in dependency_statuses):
                stage["status"] = "blocked_dependency"
                stage["blocked_reason"] = "dependency did not complete"
                overall_status = "blocked"
                returncode = 3
                for later in stages[position + 1 :]:
                    if str(later.get("status")) == "planned":
                        later["status"] = "blocked_dependency"
                        later["blocked_reason"] = f"blocked after stage {stage_id}"
                break

            if stage_id == "smoke":
                artifacts = plan.get("calibration_artifacts", {})
                valid_calibration, calibration_errors = (
                    validate_source_only_calibration_manifests(
                        artifacts.get("orientation_manifest", ""),
                        artifacts.get(
                            "profile_manifest", artifacts.get("tta_manifest", "")
                        ),
                    )
                )
                plan["calibration_validation"] = {
                    "status": "valid" if valid_calibration else "blocked",
                    "errors": calibration_errors,
                    "checked_at": _utc_now(),
                }
                if not valid_calibration:
                    stage["status"] = "blocked_missing_calibration_manifest"
                    stage["blocked_reason"] = (
                        "source-only orientation/TTA selections are required before smoke: "
                        + "; ".join(calibration_errors)
                    )
                    overall_status = "blocked"
                    returncode = 4
                    for later in stages[position + 1 :]:
                        if str(later.get("status")) == "planned":
                            later["status"] = "blocked_dependency"
                            later["blocked_reason"] = (
                                "blocked because source-only calibration manifests are incomplete"
                            )
                    break

            stage["status"] = "running"
            stage["attempts"] = int(stage.get("attempts", 0)) + 1
            stage["started_at"] = _utc_now()
            prior_subcommands = (
                list(stage.get("subcommands", []))
                if current in {"failed", "oom"}
                else []
            )
            stage["subcommands"] = prior_subcommands
            completed_subcommands = {
                int(entry.get("index"))
                for entry in prior_subcommands
                if str(entry.get("returncode")) == "0"
            }
            _publish_status(
                plan,
                status="running",
                dry_run=False,
                output_dir=output_root,
                json_out=json_out,
            )
            stage_failed = False
            for index, command in enumerate(stage.get("commands", [])):
                if index in completed_subcommands:
                    continue
                command_succeeded = False
                returncode_one = 1
                output = ""
                oom = False
                for attempt in range(1, max_stage_retries + 1):
                    returncode_one, output, oom = _run_command(list(command))
                    subcommand = {
                        "index": index,
                        "attempt": attempt,
                        "max_attempts": max_stage_retries,
                        "command": list(command),
                        "returncode": returncode_one,
                        "oom": oom,
                        "retryable": bool(
                            returncode_one != 0
                            and is_retryable_failure(returncode_one, output, oom)
                        ),
                        "finished_at": _utc_now(),
                    }
                    if output:
                        subcommand["output_tail"] = output[-4000:]
                    stage["subcommands"].append(subcommand)
                    _publish_status(
                        plan,
                        status="running",
                        dry_run=False,
                        output_dir=output_root,
                        json_out=json_out,
                    )
                    if returncode_one == 0:
                        command_succeeded = True
                        completed_subcommands.add(index)
                        break
                    if (
                        attempt >= max_stage_retries
                        or not is_retryable_failure(returncode_one, output, oom)
                    ):
                        break
                if not command_succeeded:
                    stage_failed = True
                    stage["returncode"] = int(returncode_one)
                    stage["oom"] = bool(oom)
                    stage["error"] = (
                        "subprocess reported out-of-memory; artifacts are preserved"
                        if oom
                        else (
                            f"subprocess exited with return code {returncode_one} "
                            f"after at most {max_stage_retries} attempts"
                        )
                    )
                    stage["status"] = "oom" if oom else "failed"
                    stage["finished_at"] = _utc_now()
                    overall_status = "oom" if oom else "failed"
                    returncode = 75 if oom else int(returncode_one or 1)
                    break
            if stage_failed:
                for later in stages[position + 1 :]:
                    if str(later.get("status")) == "planned":
                        later["status"] = "blocked_dependency"
                        later["blocked_reason"] = f"blocked after stage {stage_id} failure"
                break
            stage["status"] = "completed"
            stage["returncode"] = 0
            stage["finished_at"] = _utc_now()
            if stage_id == "orientation_calibration":
                artifacts = plan.get("calibration_artifacts", {})
                orientation_path = artifacts.get("orientation_manifest", "")
                profile_path = artifacts.get(
                    "profile_manifest", artifacts.get("tta_manifest", "")
                )
                valid_calibration, calibration_errors = (
                    validate_source_only_calibration_manifests(
                        orientation_path, profile_path
                    )
                )
                plan["calibration_validation"] = {
                    "status": "valid" if valid_calibration else "blocked",
                    "errors": calibration_errors,
                    "checked_at": _utc_now(),
                }
                if valid_calibration:
                    try:
                        apply_runtime_overrides(
                            plan, calibration_runtime_overrides(profile_path)
                        )
                    except ValueError as exc:
                        plan["calibration_validation"] = {
                            "status": "blocked",
                            "errors": [str(exc)],
                            "checked_at": _utc_now(),
                        }
            _publish_status(
                plan,
                status="running",
                dry_run=False,
                output_dir=output_root,
                json_out=json_out,
            )
            if stage_id == "smoke":
                smoke_valid, smoke_errors = validate_smoke_output(
                    Path(output_root) / "smoke"
                )
                plan["smoke_validation"] = {
                    "status": "complete" if smoke_valid else "blocked",
                    "errors": smoke_errors,
                    "checked_at": _utc_now(),
                    "expected_cells": len(FORMAL_METHODS),
                }
                if not smoke_valid:
                    stage["status"] = "blocked_smoke_validation"
                    stage["blocked_reason"] = "; ".join(smoke_errors)
                    overall_status = "blocked"
                    returncode = 5
                    for later in stages[position + 1 :]:
                        if str(later.get("status")) == "planned":
                            later["status"] = "blocked_dependency"
                            later["blocked_reason"] = (
                                "blocked because representative smoke did not complete all eleven methods"
                            )
                    _publish_status(
                        plan,
                        status="blocked",
                        dry_run=False,
                        output_dir=output_root,
                        json_out=json_out,
                    )
                    break

        if overall_status == "planned":
            overall_status = "complete"
            returncode = 0
        payload = _publish_status(
            plan,
            status=overall_status,
            dry_run=False,
            output_dir=output_root,
            json_out=json_out,
        )
    return returncode, payload


def _parse_csv(raw: str, cast=str) -> list[Any]:
    values = []
    for item in str(raw).split(","):
        text = item.strip()
        if text:
            values.append(cast(text))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pretrain-cache-dir", default=str(DEFAULT_PRETRAIN_CACHE_DIR))
    parser.add_argument("--fisher-cache-dir", default=str(DEFAULT_FISHER_CACHE_DIR))
    parser.add_argument(
        "--orientation-manifest",
        default=None,
        help="Complete HHAR calibration manifest.json (orientation + stage 2).",
    )
    parser.add_argument(
        "--tta-calibration-manifest",
        default=None,
        help="Frozen HHAR selected_profile.json emitted by the same calibrator.",
    )
    parser.add_argument("--prerequisite-status", default=str(DEFAULT_PREREQUISITE))
    parser.add_argument(
        "--source-seeds",
        default=",".join(str(seed) for seed in FORMAL_SOURCE_SEEDS),
    )
    parser.add_argument("--stream-seed", type=int, default=FORMAL_STREAM_SEED)
    parser.add_argument(
        "--target-labels-used-for-selection",
        "--target-selected",
        dest="target_labels_used_for_selection",
        action="store_true",
        help="Rejected: formal HHAR selection must be source-only.",
    )
    parser.add_argument(
        "--target-labels-used-for-tuning",
        dest="target_labels_used_for_tuning",
        action="store_true",
        help="Rejected: formal HHAR tuning must be source-only.",
    )
    parser.add_argument(
        "--selection-provenance",
        choices=("source-only", "target-selected"),
        default="source-only",
    )
    parser.add_argument(
        "--selection-config",
        default=None,
        help="Optional JSON provenance; target-selected claims are rejected.",
    )
    # Compatibility options from the original ten-flow scaffold.
    parser.add_argument("--da-method", choices=("DuSafe", "NoAdap"), default="DuSafe")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Execute serially; dry-run is the default.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--no-retry-failures",
        action="store_true",
        help="Leave a prior failed stage blocked instead of retrying it.",
    )
    parser.add_argument(
        "--max-stage-retries",
        type=int,
        default=3,
        help="Maximum attempts per failed queue subcommand (OOM/native-safe).",
    )
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_runs < 1:
        parser.error("--num-runs must be positive")
    if args.max_stage_retries < 1:
        parser.error("--max-stage-retries must be positive")
    if args.target_labels_used_for_tuning:
        parser.error(
            "Formal HHAR queue rejects --target-labels-used-for-tuning; "
            "target labels cannot select or tune the protocol."
        )
    selection_config = None
    if args.selection_config:
        config_path = Path(args.selection_config)
        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --selection-config: {exc}")
        if not isinstance(raw_config, Mapping):
            parser.error("--selection-config must contain a JSON object")
        selection_config = raw_config
    try:
        source_seeds, stream_seed = _validate_formal_seeds(
            _parse_csv(args.source_seeds, int), args.stream_seed
        )
        reject_target_selected_config(
            target_labels_used_for_selection=args.target_labels_used_for_selection,
            selection_provenance=args.selection_provenance,
            selection_config=selection_config,
        )
        plan = build_formal_plan(
            data_path=args.data_path,
            device=args.device,
            output_dir=args.output_dir,
            pretrain_cache_dir=args.pretrain_cache_dir,
            fisher_cache_dir=args.fisher_cache_dir,
            backbone=args.backbone,
            source_seeds=source_seeds,
            stream_seed=stream_seed,
            target_labels_used_for_selection=False,
            selection_provenance="source-only",
            selection_config=selection_config,
            orientation_manifest=args.orientation_manifest,
            tta_calibration_manifest=args.tta_calibration_manifest,
        )
    except ValueError as exc:
        parser.error(str(exc))
    returncode, payload = execute_plan(
        plan,
        output_dir=args.output_dir,
        prerequisite_status=args.prerequisite_status,
        dry_run=not args.no_dry_run,
        resume=not args.no_resume,
        retry_failures=not args.no_retry_failures,
        max_stage_retries=args.max_stage_retries,
        json_out=args.json_out,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_CORRUPTIONS",
    "FORMAL_FLOWS",
    "FORMAL_METHODS",
    "FORMAL_SEVERITIES",
    "FORMAL_SOURCE_SEEDS",
    "FORMAL_STREAM_SEED",
    "PROTOCOL_VERSION",
    "atomic_write_json",
    "build_formal_plan",
    "build_parser",
    "build_queue",
    "calibration_runtime_overrides",
    "execute_plan",
    "is_oom_text",
    "is_retryable_failure",
    "main",
    "prerequisite_complete",
    "read_status",
    "reject_target_selected_config",
    "validate_smoke_output",
    "validate_source_only_calibration_manifests",
]
