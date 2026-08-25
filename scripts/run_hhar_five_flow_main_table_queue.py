"""Resume-safe HHAR five-flow clean main-table queue.

The benchmark runner accepts one set of runtime overrides for the complete
invocation.  That is unsafe for this panel because the frozen HHAR DuSafe
profile must never be applied to the ten benchmark baselines.  This module
therefore plans 165 isolated child invocations of
:mod:`run_full_main_table`, one for each method/flow/source-seed cell.  The
ten benchmark methods receive no runtime override; DuSafe receives only the
frozen ``tta_config`` loaded from the HHAR tuning state and manifest.

The child process boundary also contains native crashes and allocator state.
The queue is CPU/dry-run by default; GPU execution is opt-in and delegates the
single-GPU lock to ``run_full_main_table.py``.  No trainer or CUDA module is
imported while constructing a plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import (  # noqa: E402
    HHAR_CONFIRMATORY,
    HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
    HHAR_REPORTED_FLOWS,
    HHAR_REPORTED_PARTITION,
    evaluation_partition_metadata,
    formal_flow_metadata,
    formal_scenario_pairs,
)


QUEUE_PROTOCOL_VERSION = "hhar_five_flow_main_table_queue_v2_one_cell"
# Public aliases make the protocol signature easy for supervisors and tests to
# consume without depending on one historical constant name.
PROTOCOL_VERSION = QUEUE_PROTOCOL_VERSION
MAIN_TABLE_PROTOCOL_VERSION = QUEUE_PROTOCOL_VERSION

DATASET = "HHAR"
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
BASELINE_METHODS = METHODS[:-1]
DUSAFE_METHODS = ("DuSafe",)
FORMAL_METHODS = METHODS
FORMAL_SOURCE_SEEDS = SOURCE_SEEDS
FORMAL_STREAM_SEED = STREAM_SEED
FORMAL_FLOWS = tuple(str(flow) for flow in HHAR_REPORTED_FLOWS)
EXPECTED_CELL_COUNT = len(FORMAL_FLOWS) * len(METHODS) * len(SOURCE_SEEDS)
EXPECTED_CELLS = EXPECTED_CELL_COUNT

DEFAULT_OUTPUT_DIR = ROOT / "results" / "hhar_five_flow_main_table"
DEFAULT_PRETRAIN_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "hhar_formal"
DEFAULT_EATA_FISHER_CACHE_DIR = (
    ROOT / "results" / "eata_fisher_cache" / "full_main_table"
)
DEFAULT_HHAR_TUNING_DIR = ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1"
DEFAULT_HHAR_STATE = DEFAULT_HHAR_TUNING_DIR / "state.json"
DEFAULT_HHAR_MANIFEST = DEFAULT_HHAR_TUNING_DIR / "manifest.json"
GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"

KEY_COLUMNS = ("dataset", "scenario", "method", "source_seed", "stream_seed")
REQUIRED_HASH_COLUMNS = ("source_model_sha256", "source_checkpoint_file_sha256")
METADATA_COLUMNS = (
    "formal_protocol_version",
    "evaluation_partition",
    "parameter_selection_data_overlap",
    "selection_overlap",
    "confirmatory",
    "target_labels_used_for_parameter_selection",
    "target_labels_used_for_updates",
    "target_labels_used_for_metrics",
    "main_table_group",
)


@dataclass(frozen=True)
class MainTableGroup:
    """One isolated ``run_full_main_table.py`` invocation."""

    name: str
    methods: tuple[str, ...]
    scenario: str
    source_seed: int
    output_dir: Path
    run_signature: str
    overrides: Mapping[str, Any]
    command: tuple[str, ...]

    @property
    def expected_cells(self) -> int:
        return 1

    @property
    def expected_keys(self) -> frozenset[tuple[str, str, str, int, int]]:
        return frozenset(
            _key_tuple(
                {
                    "dataset": DATASET,
                    "scenario": self.scenario,
                    "method": method,
                    "source_seed": self.source_seed,
                    "stream_seed": STREAM_SEED,
                }
            )
            for method in self.methods
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically publish a JSON status/manifest file."""

    target = _absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(_json_safe(dict(payload)), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically publish a CSV so a killed queue cannot expose a partial file."""

    target = _absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _absolute(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Mapping[str, Any]:
    target = _absolute(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {target}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON document must be an object: {target}")
    return payload


def _nested_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _first_mapping(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Mapping[str, Any] | None:
    for path in paths:
        found = _nested_mapping(payload, *path)
        if found is not None:
            return found
    return None


def _first_value(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def _declared_flows(state: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = (
        manifest.get("evaluation_flows"),
        manifest.get("reported_flows"),
        state.get("evaluation_flows"),
        state.get("single_flow_protocol", {}).get("evaluation_flows")
        if isinstance(state.get("single_flow_protocol"), Mapping)
        else None,
        state.get("signature", {}).get("evaluation_flows")
        if isinstance(state.get("signature"), Mapping)
        else None,
        state.get("hhar_five_flow_protocol", {}).get("development_flows")
        if isinstance(state.get("hhar_five_flow_protocol"), Mapping)
        else None,
    )
    for candidate in candidates:
        if candidate is not None:
            return tuple(str(value) for value in candidate)
    return ()


def _declared_bool(
    state: Mapping[str, Any], manifest: Mapping[str, Any], key: str, default: Any = None
) -> Any:
    for payload in (manifest, state):
        if key in payload:
            return payload[key]
        for container in (
            payload.get("single_flow_protocol"),
            payload.get("signature"),
            payload.get("validation_gate"),
            payload.get("hhar_five_flow_protocol"),
        ):
            if isinstance(container, Mapping) and key in container:
                return container[key]
    return default


def load_frozen_tta_config(
    state_path: str | Path = DEFAULT_HHAR_STATE,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the frozen DuSafe profile from state and manifest.

    The returned mapping is the exact runtime profile.  Both documents must
    carry the profile when both are supplied, and their values must agree.
    Protocol metadata is validated independently so a profile from an older
    ten-flow/holdout queue cannot be resumed silently.
    """

    state_file = _absolute(state_path)
    state = _read_json(state_file)
    manifest_file = _absolute(manifest_path) if manifest_path is not None else state_file.with_name("manifest.json")
    manifest = _read_json(manifest_file)
    state_config = state.get("tta_config")
    manifest_config = manifest.get("current_tta_config", manifest.get("tta_config"))
    if not isinstance(state_config, Mapping):
        raise ValueError(f"HHAR frozen state lacks tta_config: {state_file}")
    if not isinstance(manifest_config, Mapping):
        raise ValueError(f"HHAR tuning manifest lacks current_tta_config: {manifest_file}")
    state_config = dict(state_config)
    manifest_config = dict(manifest_config)
    if canonical_json(state_config) != canonical_json(manifest_config):
        raise ValueError("HHAR frozen state and manifest tta_config differ")

    expected_flows = tuple(FORMAL_FLOWS)
    declared_flows = _declared_flows(state, manifest)
    if declared_flows != expected_flows:
        raise ValueError(
            "HHAR frozen profile has a different five-flow protocol: "
            f"expected={expected_flows}, observed={declared_flows}"
        )
    overlap = _declared_bool(state, manifest, "parameter_selection_data_overlap")
    if overlap is None:
        overlap = _declared_bool(state, manifest, "selection_overlap")
    confirmatory = _declared_bool(state, manifest, "confirmatory")
    target_selected = _declared_bool(state, manifest, "target_labels_used_for_selection")
    if target_selected is None:
        target_selected = _declared_bool(state, manifest, "target_selected")
    partition = _first_value(
        manifest,
        ("evaluation_partition",),
        ("validation_gate", "evaluation_partition"),
    )
    if partition is None:
        partition = _first_value(
            state,
            ("evaluation_partition",),
            ("single_flow_protocol", "evaluation_partition"),
        )
    if str(partition) != HHAR_REPORTED_PARTITION:
        raise ValueError("HHAR frozen profile must use target_selected_evaluation")
    if overlap is not True:
        raise ValueError("HHAR frozen profile must declare parameter-selection overlap")
    if confirmatory is not False:
        raise ValueError("HHAR target-selected profile must be non-confirmatory")
    if target_selected is not True:
        raise ValueError("HHAR frozen profile must declare target-label selection")
    # Fail closed on all completion markers.  In particular, an old
    # early-freeze/tuning document must not be treated as a runnable profile.
    required_completion = {
        "manifest.status": manifest.get("status") == "complete",
        "manifest.phase": manifest.get("phase") == "complete",
        "manifest.tuning_complete": manifest.get("tuning_complete") is True,
        "state.phase": state.get("phase") == "complete",
        "state.completed": state.get("completed") is True,
    }
    incomplete = [name for name, passed in required_completion.items() if not passed]
    if incomplete:
        raise ValueError(
            "HHAR frozen profile is not complete; missing completion markers: "
            + ", ".join(incomplete)
        )
    return dict(state_config)


# Compatibility aliases used by queue/supervisor callers.
load_frozen_profile = load_frozen_tta_config
load_frozen_config = load_frozen_tta_config


def _key_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    try:
        return (
            str(row["dataset"]).upper(),
            str(row["scenario"]),
            str(row["method"]),
            int(row["source_seed"]),
            int(row["stream_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid main-table key row: {row}") from exc


def expected_key_set(methods: Iterable[str] = METHODS) -> frozenset[tuple[str, str, str, int, int]]:
    selected = tuple(str(method) for method in methods)
    return frozenset(
        (DATASET, scenario, method, source_seed, STREAM_SEED)
        for scenario in FORMAL_FLOWS
        for method in selected
        for source_seed in SOURCE_SEEDS
    )


def expected_keys(methods: Iterable[str] = METHODS) -> frozenset[tuple[str, str, str, int, int]]:
    return expected_key_set(methods)


def expected_cell_count() -> int:
    """Return the frozen clean-panel cell count (5 flows × 11 × 3)."""

    return EXPECTED_CELL_COUNT


def expected_stream_cell_count() -> int:
    """Compatibility alias: one fresh stream child is one clean cell."""

    return EXPECTED_CELL_COUNT


def expected_baseline_cell_count() -> int:
    return len(FORMAL_FLOWS) * len(BASELINE_METHODS) * len(SOURCE_SEEDS)


def expected_dusafe_cell_count() -> int:
    return len(FORMAL_FLOWS) * len(DUSAFE_METHODS) * len(SOURCE_SEEDS)


def _format_override_value(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return value
    return repr(value)


def override_arguments(overrides: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for key in sorted(overrides):
        result.extend(("--override", f"{key}={_format_override_value(overrides[key])}"))
    return result


def _profile_signature(config: Mapping[str, Any], state_path: Path, manifest_path: Path) -> str:
    material = {
        "queue_protocol": QUEUE_PROTOCOL_VERSION,
        "state_sha256": file_sha256(state_path),
        "manifest_sha256": file_sha256(manifest_path),
        "tta_config_sha256": config_sha256(config),
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()[:20]


def build_group_command(
    *,
    group_name: str,
    methods: Sequence[str],
    output_dir: str | Path,
    data_path: str | Path,
    device: str = "cpu",
    backbone: str = "CNN",
    pretrain_cache_dir: str | Path = DEFAULT_PRETRAIN_CACHE_DIR,
    eata_fisher_cache_dir: str | Path = DEFAULT_EATA_FISHER_CACHE_DIR,
    frozen_config: Mapping[str, Any] | None = None,
    run_signature: str | None = None,
    scenario: str | None = None,
    source_seed: int | None = None,
) -> tuple[str, ...]:
    """Build one single-cell child command.

    ``run_full_main_table.py`` is intentionally invoked with exactly one
    method, one formal flow, and one source seed.  This bounds a native crash
    or allocator failure to one cell while preserving the shared source cache.
    """

    selected = tuple(str(method) for method in methods)
    if not selected or any(method not in METHODS for method in selected):
        raise ValueError(f"unsupported HHAR main-table methods: {selected}")
    if len(selected) != 1:
        raise ValueError("each HHAR main-table child must contain exactly one method")
    is_dusafe = selected == DUSAFE_METHODS or group_name.lower() == "dusafe"
    if is_dusafe and selected != DUSAFE_METHODS:
        raise ValueError("DuSafe group must contain only DuSafe")
    if not is_dusafe and frozen_config:
        raise ValueError("baseline group cannot receive DuSafe overrides")
    if scenario is None or str(scenario) not in FORMAL_FLOWS:
        raise ValueError(f"single-cell child requires one registered HHAR flow: {scenario}")
    if source_seed is None or int(source_seed) not in SOURCE_SEEDS:
        raise ValueError(f"single-cell child requires one formal source seed: {source_seed}")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_full_main_table.py"),
        "--data-path",
        str(_absolute(data_path)),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--datasets",
        DATASET,
        "--methods",
        ",".join(selected),
        "--scenarios",
        str(scenario),
        "--source-seeds",
        str(int(source_seed)),
        "--stream-seed",
        str(STREAM_SEED),
        "--pretrain-cache-dir",
        str(_absolute(pretrain_cache_dir)),
        "--eata-fisher-cache-dir",
        str(_absolute(eata_fisher_cache_dir)),
        "--output-dir",
        str(_absolute(output_dir)),
        # Preserve each method's benchmark defaults.  In particular, do not
        # force the tuned DuSafe batch size on EATA or other baselines.
        "--batch-policy",
        "method_default",
        "--run-signature",
        str(run_signature or ""),
        "--retry-failures",
    ]
    if is_dusafe:
        if not isinstance(frozen_config, Mapping) or not frozen_config:
            raise ValueError("DuSafe group requires a non-empty frozen tta_config")
        command.extend(override_arguments(frozen_config))
    return tuple(command)


def _validate_formal_registry() -> None:
    configured = tuple(f"{source}->{target}" for source, target in formal_scenario_pairs(DATASET))
    if configured != FORMAL_FLOWS:
        raise ValueError(
            f"formal HHAR flow registry drifted: expected={FORMAL_FLOWS}, observed={configured}"
        )
    if len(FORMAL_FLOWS) != 5:
        raise ValueError("formal HHAR main-table queue requires exactly five flows")
    for scenario in FORMAL_FLOWS:
        metadata = evaluation_partition_metadata(DATASET, scenario)
        if metadata.get("evaluation_partition") != HHAR_REPORTED_PARTITION:
            raise ValueError("HHAR formal main-table partition drifted")
        if metadata.get("selection_overlap") is not True:
            raise ValueError("HHAR formal main-table must declare selection overlap")
        if metadata.get("confirmatory") is not False:
            raise ValueError("HHAR formal main-table must be non-confirmatory")


def build_plan(
    *,
    data_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pretrain_cache_dir: str | Path = DEFAULT_PRETRAIN_CACHE_DIR,
    eata_fisher_cache_dir: str | Path = DEFAULT_EATA_FISHER_CACHE_DIR,
    device: str = "cpu",
    backbone: str = "CNN",
    frozen_state: str | Path = DEFAULT_HHAR_STATE,
    frozen_manifest: str | Path | None = None,
    hhar_frozen_state: str | Path | None = None,
    hhar_frozen_config: str | Path | None = None,
) -> dict[str, Any]:
    """Build the complete five-flow plan without launching a trainer."""

    _validate_formal_registry()
    if hhar_frozen_state is not None:
        frozen_state = hhar_frozen_state
    if hhar_frozen_config is not None:
        frozen_state = hhar_frozen_config
    if frozen_state is None:
        raise ValueError("HHAR main-table queue requires a frozen state.json")
    state_path = _absolute(frozen_state)
    manifest_path = (
        state_path.with_name("manifest.json")
        if frozen_manifest is None
        else _absolute(frozen_manifest)
    )
    frozen = load_frozen_tta_config(state_path, manifest_path)
    output_root = _absolute(output_dir)
    profile_signature = _profile_signature(frozen, state_path, manifest_path)
    groups: list[MainTableGroup] = []
    ordinal = 0
    for scenario in FORMAL_FLOWS:
        for source_seed in SOURCE_SEEDS:
            for method in METHODS:
                ordinal += 1
                safe_scenario = scenario.replace("->", "-to-")
                name = f"cell-{ordinal:03d}-{safe_scenario}-{method}-s{source_seed}"
                group_dir = output_root / "cells" / name
                dusafe = method == "DuSafe"
                signature = (
                    f"{QUEUE_PROTOCOL_VERSION}:{method}:{scenario}:source={source_seed}"
                )
                if dusafe:
                    signature += f":profile={profile_signature}"
                groups.append(
                    MainTableGroup(
                        name=name,
                        methods=(method,),
                        scenario=scenario,
                        source_seed=int(source_seed),
                        output_dir=group_dir,
                        run_signature=signature,
                        overrides=dict(frozen) if dusafe else {},
                        command=build_group_command(
                            group_name=method,
                            methods=(method,),
                            scenario=scenario,
                            source_seed=source_seed,
                            output_dir=group_dir,
                            data_path=data_path,
                            device=device,
                            backbone=backbone,
                            pretrain_cache_dir=pretrain_cache_dir,
                            eata_fisher_cache_dir=eata_fisher_cache_dir,
                            frozen_config=frozen if dusafe else None,
                            run_signature=signature,
                        ),
                    )
                )
    serialized_groups = [_group_to_dict(group) for group in groups]
    return {
        "protocol": QUEUE_PROTOCOL_VERSION,
        "protocol_version": QUEUE_PROTOCOL_VERSION,
        "dataset": DATASET,
        "data_path": str(_absolute(data_path)),
        "output_dir": str(output_root),
        "pretrain_cache_dir": str(_absolute(pretrain_cache_dir)),
        "eata_fisher_cache_dir": str(_absolute(eata_fisher_cache_dir)),
        "device": str(device),
        "backbone": str(backbone),
        "flows": list(FORMAL_FLOWS),
        "formal_flow_metadata": _json_safe(dict(formal_flow_metadata(DATASET))),
        "evaluation_partition_metadata": {
            scenario: _json_safe(dict(evaluation_partition_metadata(DATASET, scenario)))
            for scenario in FORMAL_FLOWS
        },
        "flow_count": len(FORMAL_FLOWS),
        "methods": list(METHODS),
        "baseline_methods": list(BASELINE_METHODS),
        "dusafe_methods": list(DUSAFE_METHODS),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "expected_groups": len(groups),
        "expected_group_count": len(groups),
        "group_count": len(groups),
        "expected_cells": EXPECTED_CELL_COUNT,
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "selection_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_metrics": True,
        "hhar_frozen_state": str(state_path),
        "hhar_frozen_manifest": str(manifest_path),
        "frozen_tta_config": dict(frozen),
        "frozen_tta_config_sha256": config_sha256(frozen),
        "frozen_state_sha256": file_sha256(state_path),
        "frozen_manifest_sha256": file_sha256(manifest_path),
        "gpu_lock": str(GPU_LOCK_PATH),
        "gpu_lock_required": str(device).lower().startswith("cuda"),
        "process_isolation": "one fresh child process per method-flow-source cell; runner cleans each trainer",
        "groups": serialized_groups,
        # ``cells`` is a compatibility alias used by the other formal queue
        # planners; each entry is both a group and exactly one cell here.
        "cells": serialized_groups,
    }


# Queue modules in this repository historically exposed ``build_queue``;
# retain that spelling for supervisors while keeping the plan implementation
# explicit.
build_queue = build_plan


def _group_to_dict(group: MainTableGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "id": group.name,
        "key": "|".join(
            (
                DATASET,
                group.scenario,
                f"method={group.methods[0]}",
                f"source_seed={group.source_seed}",
                f"stream_seed={STREAM_SEED}",
            )
        ),
        "methods": list(group.methods),
        "method": group.methods[0] if len(group.methods) == 1 else None,
        "scenario": group.scenario,
        "source_seed": int(group.source_seed),
        "stream_seed": STREAM_SEED,
        "output_dir": str(group.output_dir),
        "run_signature": group.run_signature,
        "overrides": dict(group.overrides),
        "command": list(group.command),
        "expected_cells": group.expected_cells,
        "expected_keys": [list(key) for key in sorted(group.expected_keys)],
        "status": "planned",
        "attempts": 0,
    }


def groups_from_plan(plan: Mapping[str, Any]) -> tuple[MainTableGroup, ...]:
    result = []
    for raw in plan.get("groups", plan.get("cells", ())):
        if not isinstance(raw, Mapping):
            raise ValueError("malformed main-table group plan")
        result.append(
            MainTableGroup(
                name=str(raw["name"]),
                methods=tuple(str(value) for value in raw["methods"]),
                scenario=str(raw["scenario"]),
                source_seed=int(raw["source_seed"]),
                output_dir=Path(str(raw["output_dir"])),
                run_signature=str(raw["run_signature"]),
                overrides=dict(raw.get("overrides", {})),
                command=tuple(str(value) for value in raw["command"]),
            )
        )
    return tuple(result)


def _raw_path(group_dir: str | Path) -> Path:
    return _absolute(group_dir) / "per_source_seed_results.csv"


def _load_raw(group_dir: str | Path) -> pd.DataFrame:
    path = _raw_path(group_dir)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _coerce_key_frame(frame: pd.DataFrame) -> list[tuple[str, str, str, int, int]]:
    keys = []
    for record in frame.to_dict("records"):
        keys.append(_key_tuple(record))
    return keys


def _hash_value(row: Mapping[str, Any], column: str) -> str:
    value = row.get(column, "")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _source_unit(row: Mapping[str, Any]) -> tuple[str, int]:
    source = row.get("src_id")
    if source is None or str(source).strip() in {"", "nan", "None"}:
        scenario = str(row.get("scenario", ""))
        source = scenario.split("->", 1)[0] if "->" in scenario else ""
    return str(source), int(row["source_seed"])


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _validate_checkpoint_hash_columns(
    frame: pd.DataFrame,
    *,
    errors: list[str],
    label: str,
) -> None:
    """Require one immutable source identity per independent source seed."""

    if "source_model_sha256" not in frame.columns:
        errors.append(f"{label} output lacks source_model_sha256")
        return
    for column in ("source_model_sha256", "source_checkpoint_file_sha256"):
        if column not in frame.columns:
            # Older unit fixtures contain only the tensor identity.  Real
            # runner outputs include both; the tensor identity remains the
            # minimum required resume key.
            if column == "source_checkpoint_file_sha256":
                continue
            errors.append(f"{label} output lacks {column}")
            continue
        grouped: dict[tuple[str, int], set[str]] = {}
        for row in frame.to_dict("records"):
            try:
                source, seed = _source_unit(row)
            except (KeyError, TypeError, ValueError):
                continue
            value = _hash_value(row, column)
            if not value:
                errors.append(f"{label} source={source} source_seed={seed} has missing {column}")
            grouped.setdefault((source, seed), set()).add(value)
        for (source, seed), hashes in sorted(grouped.items()):
            if len(hashes) != 1:
                errors.append(
                    f"{label} source={source} source_seed={seed} maps to multiple {column} values"
                )


def validate_group_output(
    group: MainTableGroup,
    frame: pd.DataFrame | None = None,
    *,
    require_metadata: bool = False,
) -> tuple[bool, list[str]]:
    """Validate exact group keys and protocol signatures for safe resume."""

    if frame is None:
        frame = _load_raw(group.output_dir)
    errors: list[str] = []
    if frame.empty:
        return False, [f"{group.name} output is missing or empty"]
    missing_columns = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"{group.name} output lacks key columns {missing_columns}")
        return False, errors
    try:
        keys = _coerce_key_frame(frame)
    except ValueError as exc:
        return False, [str(exc)]
    observed = set(keys)
    duplicates = sorted(key for key in observed if keys.count(key) > 1)
    if duplicates:
        errors.append(f"{group.name} output has duplicate keys: {duplicates[:3]}")
    if observed != set(group.expected_keys):
        errors.append(
            f"{group.name} key set mismatch: expected={len(group.expected_keys)}, observed={len(observed)}"
        )
    if len(frame) != group.expected_cells:
        errors.append(
            f"{group.name} row count {len(frame)} != expected {group.expected_cells}"
        )
    if "status" in frame.columns and frame["status"].astype(str).ne("ok").any():
        errors.append(f"{group.name} output contains failed cells")
    if "run_signature" not in frame.columns and group.run_signature:
        errors.append(f"{group.name} output lacks run_signature")
    elif group.run_signature and frame["run_signature"].astype(str).ne(group.run_signature).any():
        errors.append(f"{group.name} output run_signature is stale")
    if group.methods != DUSAFE_METHODS and "runtime_hparams" in frame.columns:
        # A baseline child must be invoked without DuSafe overrides.  The
        # runner manifest is checked separately; this catches contaminated
        # rows even when a hand-edited manifest claims no override.
        forbidden = set(_frozen_override_names_from_frame(frame)) & set(
            _known_dusafe_override_names()
        )
        if forbidden:
            errors.append(f"baseline rows contain DuSafe override keys: {sorted(forbidden)}")
    child_manifest = _absolute(group.output_dir) / "manifest.json"
    if child_manifest.is_file():
        try:
            child_payload = _read_json(child_manifest)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            child_overrides = child_payload.get("runtime_overrides", {})
            if not isinstance(child_overrides, Mapping):
                errors.append(f"{group.name} child manifest runtime_overrides is malformed")
            elif group.methods != DUSAFE_METHODS and child_overrides:
                errors.append("baseline child manifest contains runtime overrides")
            elif group.methods == DUSAFE_METHODS and canonical_json(child_overrides) != canonical_json(group.overrides):
                errors.append("DuSafe child manifest overrides differ from frozen tta_config")
    if require_metadata:
        for column in METADATA_COLUMNS:
            if column not in frame.columns:
                errors.append(f"merged output lacks {column}")
    # Every source cell needs an auditable checkpoint identity.  A source
    # seed may map to exactly one tensor/file hash across all flows/methods.
    _validate_checkpoint_hash_columns(frame, errors=errors, label=group.name)
    if "fisher_enabled" in frame.columns and "EATA" in group.methods:
        eata = frame[frame["method"].astype(str).eq("EATA")]
        if not eata.empty and eata["fisher_enabled"].map(lambda value: not _bool_value(value)).any():
            errors.append("EATA rows do not declare fisher_enabled=true")
        for column in ("fisher_cache_path", "fisher_cache_hash", "fisher_source_checkpoint_sha256"):
            if column not in eata.columns or eata[column].map(lambda value: not bool(str(value).strip()) if not pd.isna(value) else True).any():
                errors.append(f"EATA rows lack validated {column}")
        if "source_model_sha256" in eata.columns and "fisher_source_checkpoint_sha256" in eata.columns:
            mismatched_fisher_hash = (
                eata["source_model_sha256"].astype(str).str.strip()
                != eata["fisher_source_checkpoint_sha256"].astype(str).str.strip()
            )
            if mismatched_fisher_hash.any():
                errors.append("EATA Fisher source hash does not match source checkpoint hash")
    return not errors, errors


def _known_dusafe_override_names() -> tuple[str, ...]:
    # Names used by current source profiles.  This is intentionally a denylist
    # only for baseline contamination detection; DuSafe config validation is
    # performed against the frozen mapping itself.
    return (
        "confidence_keep_fraction",
        "confidence_reference_samples",
        "enable_confidence_gate",
        "enable_source_semantic_gate",
        "enable_ssaw",
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
    )


def _frozen_override_names_from_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    names: set[str] = set()
    for value in frame.get("runtime_hparams", pd.Series(dtype=object)).dropna():
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            names.update(str(key) for key in parsed)
    return tuple(sorted(names))


def validate_merged_output(
    frame: pd.DataFrame,
    *,
    expected_hashes: Mapping[Any, str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate the merged 165-cell clean panel."""

    errors: list[str] = []
    expected = expected_key_set()
    if frame.empty:
        return False, ["merged main-table output is empty"]
    try:
        keys = _coerce_key_frame(frame)
    except ValueError as exc:
        return False, [str(exc)]
    observed = set(keys)
    if len(keys) != len(observed):
        errors.append("merged main-table output contains duplicate keys")
    if observed != set(expected):
        errors.append(
            f"merged main-table key set mismatch: expected={len(expected)}, observed={len(observed)}"
        )
    if len(frame) != EXPECTED_CELL_COUNT:
        errors.append(f"merged main-table row count {len(frame)} != {EXPECTED_CELL_COUNT}")
    if "status" in frame.columns and frame["status"].astype(str).ne("ok").any():
        errors.append("merged main-table contains failed cells")
    _validate_checkpoint_hash_columns(frame, errors=errors, label="merged main-table")
    if expected_hashes is not None and "source_model_sha256" in frame.columns:
        grouped: dict[tuple[str, int], set[str]] = {}
        for row in frame.to_dict("records"):
            source, seed = _source_unit(row)
            grouped.setdefault((source, seed), set()).add(
                _hash_value(row, "source_model_sha256")
            )
        for (source, seed), values in grouped.items():
            expected = expected_hashes.get((source, seed), expected_hashes.get(seed, ""))
            if values != {str(expected)}:
                errors.append(
                    f"source={source} source_seed={seed} checkpoint hash differs from prior merged output"
                )
    required_metadata = {
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": True,
        "selection_overlap": True,
        "confirmatory": False,
    }
    for column, expected_value in required_metadata.items():
        if column not in frame.columns:
            errors.append(f"merged main-table lacks {column}")
        else:
            observed_values = {_normalise_scalar(value) for value in frame[column].tolist()}
            if observed_values != {_normalise_scalar(expected_value)}:
                errors.append(f"merged main-table {column} metadata drifted")
    if "fisher_enabled" not in frame.columns:
        errors.append("merged main-table lacks fisher_enabled")
    else:
        eata = frame[frame["method"].astype(str).eq("EATA")]
        if len(eata) != len(FORMAL_FLOWS) * len(SOURCE_SEEDS):
            errors.append("merged main-table EATA key count drifted")
        elif eata["fisher_enabled"].map(lambda value: not _bool_value(value)).any():
            errors.append("merged main-table EATA Fisher is disabled")
    return not errors, errors


def validate_cells(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    """Validate serialized 165-cell rows and return an audit summary."""

    raw_rows = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    if raw_rows and all("expected_keys" in row and "method" in row for row in raw_rows):
        observed: list[tuple[str, str, str, int, int]] = []
        errors: list[str] = []
        for row in raw_rows:
            expected = row.get("expected_keys")
            if not isinstance(expected, Sequence) or len(expected) != 1:
                errors.append(f"group {row.get('name')} must contain exactly one expected key")
                continue
            try:
                observed.append(tuple(expected[0]))
            except (TypeError, IndexError):
                errors.append(f"group {row.get('name')} has malformed expected key")
        expected_set = set(expected_key_set())
        if len(observed) != len(set(observed)):
            errors.append("duplicate planned cell keys")
        if set(observed) != expected_set:
            errors.append(
                f"planned cell key set mismatch: expected={len(expected_set)}, observed={len(set(observed))}"
            )
        return {
            "expected_cell_count": EXPECTED_CELL_COUNT,
            "observed_cell_count": len(observed),
            "passed": not errors,
            "errors": errors,
        }
    frame = pd.DataFrame(raw_rows)
    try:
        valid, errors = validate_merged_output(frame)
    except (KeyError, TypeError, ValueError) as exc:
        valid, errors = False, [str(exc)]
    return {
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "observed_cell_count": int(len(frame)),
        "passed": bool(valid),
        "errors": list(errors),
    }


def _normalise_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"nan", "none", "null"}:
            return None
        return text
    return value


def _annotate_group(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    metadata = {
        "formal_protocol_version": QUEUE_PROTOCOL_VERSION,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": True,
        "selection_overlap": True,
        "confirmatory": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_metrics": True,
        "main_table_group": name,
    }
    for column, value in metadata.items():
        result[column] = value
    return result


def merge_group_outputs(
    groups: Sequence[pd.DataFrame] | pd.DataFrame,
    dusafe: pd.DataFrame | None = None,
    *,
    output_path: str | Path | None = None,
    group_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Strictly merge all single-cell group outputs.

    The optional second positional argument preserves the earlier two-frame
    helper API; the queue itself passes all 165 frames in one sequence.
    """

    if isinstance(groups, pd.DataFrame):
        frames = [groups] if dusafe is None else [groups, dusafe]
    else:
        frames = list(groups)
        if dusafe is not None:
            frames.append(dusafe)
    if not frames:
        raise ValueError("cannot merge an empty main-table group list")
    observed_keys: set[tuple[str, str, str, int, int]] = set()
    annotated: list[pd.DataFrame] = []
    validation_errors: list[str] = []
    for index, frame in enumerate(frames):
        if frame.empty:
            validation_errors.append(f"group {index} output is empty")
            continue
        try:
            keys = set(_coerce_key_frame(frame))
        except ValueError as exc:
            validation_errors.append(str(exc))
            continue
        if observed_keys & keys:
            validation_errors.append(f"group {index} overlaps a prior main-table key")
        observed_keys.update(keys)
        methods = tuple(sorted({str(value) for value in frame["method"].tolist()}))
        scenarios = tuple(sorted({str(value) for value in frame["scenario"].tolist()}))
        seeds = tuple(sorted({int(value) for value in frame["source_seed"].tolist()}))
        # Current queue outputs are one cell.  Keep validating legacy
        # multi-cell fixtures by exact key/hash checks, but never relax the
        # final merged 165-key gate below.
        if len(methods) == len(scenarios) == len(seeds) == 1 and len(frame) == 1:
            group_name = str(group_names[index]) if group_names and index < len(group_names) else methods[0]
            synthetic = MainTableGroup(
                name=group_name,
                methods=methods,
                scenario=scenarios[0],
                source_seed=seeds[0],
                output_dir=Path("."),
                run_signature="",
                overrides={},
                command=(),
            )
            valid, errors = validate_group_output(synthetic, frame)
            if not valid:
                validation_errors.extend(errors)
            annotated.append(_annotate_group(frame, group_name))
        else:
            if "status" in frame.columns and frame["status"].astype(str).ne("ok").any():
                validation_errors.append(f"group {index} contains failed cells")
            hash_errors: list[str] = []
            _validate_checkpoint_hash_columns(frame, errors=hash_errors, label=f"group {index}")
            validation_errors.extend(hash_errors)
            group_name = str(group_names[index]) if group_names and index < len(group_names) else (methods[0] if methods else f"group-{index}")
            annotated.append(_annotate_group(frame, group_name))
    if validation_errors:
        raise ValueError("cannot merge invalid groups: " + "; ".join(validation_errors))
    merged = pd.concat(annotated, ignore_index=True)
    merged = merged.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    valid, errors = validate_merged_output(merged)
    if not valid:
        raise ValueError("merged main-table validation failed: " + "; ".join(errors))
    if output_path is not None:
        atomic_write_csv(merged, output_path)
    return merged


def _status_payload(
    plan: Mapping[str, Any],
    *,
    status: str,
    current_group: str | None,
    groups: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed_groups = sum(str(group.get("status")) == "complete" for group in groups)
    completed_cells = sum(
        int(group.get("expected_cells", 0))
        for group in groups
        if str(group.get("status")) == "complete"
    )
    return {
        "protocol_version": QUEUE_PROTOCOL_VERSION,
        "status": status,
        "dry_run": status == "dry_run",
        "phase": status,
        "dataset": DATASET,
        "flows": list(FORMAL_FLOWS),
        "methods": list(METHODS),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "expected_cells": EXPECTED_CELL_COUNT,
        "completed_groups": completed_groups,
        "expected_groups": int(plan.get("expected_groups", EXPECTED_CELL_COUNT)),
        "completed_cells": completed_cells,
        "current_group": current_group,
        "failures": list(failures),
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "selection_overlap": True,
        "confirmatory": False,
        "updated_at": _utc_now(),
    }


def _write_queue_manifest(plan: Mapping[str, Any], output_root: Path, *, status: str, groups: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]], merged: pd.DataFrame | None = None) -> None:
    payload = dict(plan)
    payload.update(
        {
            "status": status,
            "dry_run": status == "dry_run",
            "created_or_updated_at": _utc_now(),
            "groups": [dict(group) for group in groups],
            "cells": [dict(group) for group in groups],
            "failures": list(failures),
            "raw_rows": 0 if merged is None else int(len(merged)),
            "successful_rows": 0 if merged is None else int(len(merged)),
            "outputs": {
                "cells": "cells/<cell-id>/per_source_seed_results.csv",
                "merged_raw": "per_source_seed_results.csv",
                "main_table": "main_table.csv",
                "status": "status.json",
            },
        }
    )
    atomic_write_json(payload, output_root / "manifest.json")


def _run_child(group: MainTableGroup, *, device: str, output_root: Path, max_attempts: int) -> tuple[bool, list[str], int | None]:
    group.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{group.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_return: int | None = None
    errors: list[str] = []
    for attempt in range(1, int(max_attempts) + 1):
        env = os.environ.copy()
        if str(device).lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nATTEMPT {attempt} COMMAND {json.dumps(list(group.command))}\n")
            handle.flush()
            result = subprocess.run(
                list(group.command),
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
            handle.write(f"RETURN_CODE {result.returncode}\n")
        last_return = int(result.returncode)
        frame = _load_raw(group.output_dir)
        valid, errors = validate_group_output(group, frame)
        if result.returncode == 0 and valid:
            return True, [], last_return
        # A fresh run is deliberately used for every attempt; the child owns
        # all trainer/model memory and exits before the next attempt starts.
        time.sleep(min(0.25 * attempt, 1.0))
    return False, errors or [f"child returned {last_return}"], last_return


def is_retryable_failure(returncode: int, output: str = "", is_oom: bool = False) -> bool:
    """Classify allocator failures/native crashes for a fresh child retry."""

    if int(returncode) == 0:
        return False
    lowered = str(output).lower()
    if is_oom or any(
        marker in lowered
        for marker in (
            "out of memory",
            "memoryerror",
            "cannot allocate memory",
            "cudnn_status_alloc_failed",
            "std::bad_alloc",
        )
    ):
        return True
    return int(returncode) < 0 or int(returncode) in {
        -1073741819,
        -1073740791,
        0xC0000005,
        0xC0000409,
        0xC0000017,
        0xC000009A,
    } or "segmentation fault" in lowered


def is_oom_text(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "memoryerror",
            "cannot allocate memory",
            "cudnn_status_alloc_failed",
            "std::bad_alloc",
        )
    )


def wait_for_hhar_profile(
    state_path: str | Path,
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    poll_seconds: int = 60,
) -> dict[str, Any]:
    """Wait for the strict tuner completion gate without acquiring a GPU lock."""

    if int(poll_seconds) < 1:
        raise ValueError("poll_seconds must be positive")
    output_root = _absolute(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            config = load_frozen_tta_config(state_path, manifest_path)
        except (OSError, ValueError) as exc:
            atomic_write_json(
                {
                    "protocol_version": QUEUE_PROTOCOL_VERSION,
                    "status": "waiting_for_hhar",
                    "phase": "waiting_for_hhar",
                    "expected_groups": EXPECTED_CELL_COUNT,
                    "expected_cells": EXPECTED_CELL_COUNT,
                    "evaluation_partition": HHAR_REPORTED_PARTITION,
                    "selection_overlap": True,
                    "confirmatory": False,
                    "reason": str(exc),
                    "state_path": str(_absolute(state_path)),
                    "manifest_path": str(_absolute(manifest_path)),
                    "updated_at": _utc_now(),
                    "gpu_lock_acquired": False,
                },
                output_root / "status.json",
            )
            time.sleep(int(poll_seconds))
            continue
        return config


def run_queue(
    plan: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = True,
    max_attempts: int = 3,
) -> tuple[int, dict[str, Any]]:
    """Execute/resume a plan, returning ``(returncode, status_payload)``."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    output_root = _absolute(output_dir or plan["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    groups = [
        dict(group)
        for group in plan.get("groups", plan.get("cells", ()))
    ]
    failures: list[dict[str, Any]] = []
    if dry_run:
        for group in groups:
            group["status"] = "planned"
        status = _status_payload(plan, status="dry_run", current_group=None, groups=groups, failures=failures)
        atomic_write_json(status, output_root / "status.json")
        _write_queue_manifest(plan, output_root, status="dry_run", groups=groups, failures=failures)
        return 0, status

    group_objects = groups_from_plan(plan)
    for index, group in enumerate(group_objects):
        group_record = groups[index]
        frame = _load_raw(group.output_dir)
        valid, errors = validate_group_output(group, frame)
        if valid:
            group_record["status"] = "complete"
            group_record["attempts"] = 0
            continue
        group_record["status"] = "running"
        group_record["attempts"] = int(group_record.get("attempts", 0)) + 1
        atomic_write_json(
            _status_payload(plan, status="running", current_group=group.name, groups=groups, failures=failures),
            output_root / "status.json",
        )
        success, child_errors, returncode = _run_child(
            group,
            device=str(plan.get("device", "cpu")),
            output_root=output_root,
            max_attempts=max_attempts,
        )
        if not success:
            group_record["status"] = "failed"
            failures.append(
                {
                    "group": group.name,
                    "returncode": returncode,
                    "errors": child_errors or errors,
                }
            )
            status = _status_payload(plan, status="failed", current_group=group.name, groups=groups, failures=failures)
            atomic_write_json(status, output_root / "status.json")
            _write_queue_manifest(plan, output_root, status="failed", groups=groups, failures=failures)
            return 2, status
        group_record["status"] = "complete"
        group_record["attempts"] = int(group_record.get("attempts", 0)) + 1
        atomic_write_json(
            _status_payload(plan, status="running", current_group=None, groups=groups, failures=failures),
            output_root / "status.json",
        )

    try:
        group_frames = [_load_raw(group.output_dir) for group in group_objects]
        merged = merge_group_outputs(
            group_frames,
            output_path=output_root / "per_source_seed_results.csv",
            group_names=[group.name for group in group_objects],
        )
    except (ValueError, OSError) as exc:
        failures.append({"group": "merge", "errors": [str(exc)]})
        status = _status_payload(plan, status="failed", current_group="merge", groups=groups, failures=failures)
        atomic_write_json(status, output_root / "status.json")
        _write_queue_manifest(plan, output_root, status="failed", groups=groups, failures=failures)
        return 2, status

    # Reuse the runner's analyzer only through its safe read-only API; this
    # does not start a trainer and preserves the exact merged raw rows.
    _write_main_table_summary(merged, output_root / "main_table.csv")
    status = _status_payload(plan, status="complete", current_group=None, groups=groups, failures=failures)
    status["completed_cells"] = int(len(merged))
    atomic_write_json(status, output_root / "status.json")
    _write_queue_manifest(plan, output_root, status="complete", groups=groups, failures=failures, merged=merged)
    return 0, status


# Compatibility spelling used by other process-isolated queue modules.
execute_queue = run_queue
group_command = build_group_command
validate_groups = validate_cells


def _write_main_table_summary(frame: pd.DataFrame, path: Path) -> None:
    rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_successful_cells": int(len(group)),
                "n_source_seeds": int(group["source_seed"].nunique()),
                "n_scenarios": int(group["scenario"].nunique()),
                "evaluation_partition": HHAR_REPORTED_PARTITION,
                "selection_overlap": True,
                "confirmatory": False,
                "f1_mean": float(pd.to_numeric(group["f1"], errors="coerce").mean())
                if "f1" in group
                else float("nan"),
                "accuracy_mean": float(pd.to_numeric(group["accuracy"], errors="coerce").mean())
                if "accuracy" in group
                else float("nan"),
                "auroc_mean": float(pd.to_numeric(group["auroc"], errors="coerce").mean())
                if "auroc" in group
                else float("nan"),
            }
        )
    atomic_write_csv(pd.DataFrame(rows), path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-path", "--data_path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pretrain-cache-dir", "--pretrain_cache_dir", default=str(DEFAULT_PRETRAIN_CACHE_DIR))
    parser.add_argument("--eata-fisher-cache-dir", "--eata_fisher_cache_dir", default=str(DEFAULT_EATA_FISHER_CACHE_DIR))
    parser.add_argument("--hhar-tuning-dir", default=str(DEFAULT_HHAR_TUNING_DIR))
    parser.add_argument("--frozen-state", "--hhar-frozen-state", default=None)
    parser.add_argument("--frozen-manifest", "--hhar-frozen-manifest", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--wait-for-hhar", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tuning_dir = _absolute(args.hhar_tuning_dir)
    state_path = _absolute(args.frozen_state) if args.frozen_state else tuning_dir / "state.json"
    manifest_path = _absolute(args.frozen_manifest) if args.frozen_manifest else tuning_dir / "manifest.json"
    try:
        if args.wait_for_hhar:
            wait_for_hhar_profile(
                state_path,
                manifest_path,
                output_dir=args.output_dir,
                poll_seconds=args.poll_seconds,
            )
        plan = build_plan(
            data_path=args.data_path,
            output_dir=args.output_dir,
            pretrain_cache_dir=args.pretrain_cache_dir,
            eata_fisher_cache_dir=args.eata_fisher_cache_dir,
            device=args.device,
            backbone=args.backbone,
            frozen_state=state_path,
            frozen_manifest=manifest_path,
        )
        if args.analyze_only:
            output_root = _absolute(args.output_dir)
            merged_path = output_root / "per_source_seed_results.csv"
            frame = pd.read_csv(merged_path)
            valid, errors = validate_merged_output(frame)
            if not valid:
                raise ValueError("existing merged output is invalid: " + "; ".join(errors))
            _write_main_table_summary(frame, output_root / "main_table.csv")
            atomic_write_json({"status": "complete", "protocol_version": QUEUE_PROTOCOL_VERSION, "completed_cells": len(frame)}, output_root / "status.json")
            return 0
        code, _ = run_queue(plan, dry_run=bool(args.dry_run), max_attempts=args.max_attempts)
        return code
    except (OSError, ValueError, KeyError) as exc:
        print(f"[HHAR main-table queue] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "QUEUE_PROTOCOL_VERSION",
    "PROTOCOL_VERSION",
    "DATASET",
    "SOURCE_SEEDS",
    "STREAM_SEED",
    "METHODS",
    "BASELINE_METHODS",
    "DUSAFE_METHODS",
    "FORMAL_METHODS",
    "FORMAL_FLOWS",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_CELLS",
    "MainTableGroup",
    "atomic_write_json",
    "atomic_write_csv",
    "load_frozen_tta_config",
    "load_frozen_profile",
    "expected_key_set",
    "expected_keys",
    "expected_cell_count",
    "expected_stream_cell_count",
    "expected_baseline_cell_count",
    "expected_dusafe_cell_count",
    "override_arguments",
    "build_group_command",
    "build_plan",
    "build_queue",
    "validate_group_output",
    "validate_merged_output",
    "validate_cells",
    "validate_groups",
    "merge_group_outputs",
    "groups_from_plan",
    "run_queue",
    "execute_queue",
    "group_command",
    "is_retryable_failure",
    "is_oom_text",
    "wait_for_hhar_profile",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
