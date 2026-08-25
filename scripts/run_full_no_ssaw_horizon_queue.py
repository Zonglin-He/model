"""Process-isolated queue for the formal Full/no-SSAW horizon audit.

The queue is intentionally a planner plus a small serial supervisor.  A stream
cell is one registered flow, source seed, and corruption condition; horizons
1/3/5 are evaluated together on the exact same online trajectory.  Every
stream cell is launched in a fresh subprocess, so a Python OOM, native crash, or
broken CUDA context is contained to that cell and can be retried without
losing the rest of the manifest.  The default mode is a CPU dry-run; no
experiment subprocess is started unless ``--no-dry-run`` is supplied.

The protocol grid is frozen here rather than inferred from arbitrary CLI
subsets:

* EEG/HAR/FD/HHAR registered flows;
* source seeds 1, 2, 3 and paired stream seed 42;
* horizons 1, 3, 5;
* clean plus six registered categorical corruptions at moderate/severe.

HHAR cells require the completed F1 profile selected on the registered first
five flows.  The formal queue reports those same five target-selected flows;
the remaining five data-model flows are not part of this A--F panel.  HHAR
formal results are descriptive and non-confirmatory because parameter
selection and evaluation overlap.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data_model_configs import DATASET_NAMES, validate_scenario  # noqa: E402
from configs.formal_evaluation_protocol import (  # noqa: E402
    HHAR_DEVELOPMENT_FLOWS,
    HHAR_REPORTED_FLOWS,
    evaluation_partition_metadata,
    formal_scenario_pairs,
)
from dataloader.corruption_transforms import CORRUPTION_REGISTRY  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    DEFAULT_PAPER_FLOW_PROFILE_JSON,
)


PROTOCOL_VERSION = "full_no_ssaw_horizon_queue_v3_five_formal_flows"
DATASETS = ("EEG", "HAR", "FD", "HHAR")
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
HORIZONS = (1, 3, 5)
# Explicit aliases follow the naming used by the other formal queue modules.
FORMAL_DATASETS = DATASETS
FORMAL_SOURCE_SEEDS = SOURCE_SEEDS
FORMAL_STREAM_SEED = STREAM_SEED
FORMAL_HORIZONS = HORIZONS
# The six conditions are the categorical corruption subset used by the
# controlled-safety protocol.  Other registry entries remain available to
# exploratory scripts but are not silently added to this formal queue.
FORMAL_CORRUPTIONS = (
    "signal_freeze",
    "blackout",
    "attenuation",
    "amplitude_drift",
    "packet_loss",
    "saturation",
)
FORMAL_SEVERITIES = ("moderate", "severe")
FORMAL_CONDITIONS = (
    "clean",
    *(f"{corruption}:{severity}"
      for corruption in FORMAL_CORRUPTIONS
      for severity in FORMAL_SEVERITIES),
)
CLEAN_CONDITION = "clean"
CORRUPTION_FRACTION = 0.5
CORRUPTION_SEED = 1
MAX_RETRIES = 3
DEFAULT_OUTPUT_DIR = ROOT / "results" / "full_no_ssaw_horizon_queue"
DEFAULT_PRETRAIN_CACHE_DIR = ROOT / "results" / "pretrain_cache"
DEFAULT_HHAR_FROZEN_STATE = (
    ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1" / "state.json"
)
GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"
# Compatibility export only.  The current dataset-level tuning protocol uses
# one five-flow set for both selection and reporting and has no holdout split.
HHAR_HOLDOUT_FLOWS: tuple[str, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _is_gpu_device(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Publish a complete JSON manifest through a same-directory replace."""

    target = Path(path)
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
        for attempt in range(20):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _explicit_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _target_selection_claim(payload: Any) -> bool:
    """Find explicit target-label/data selection claims recursively."""

    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if _target_selection_claim(value):
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
            if not _explicit_true(value):
                continue
            if key in {
                "target_labels_used",
                "target_data_used",
                "target_metrics_used",
                "target_labels_used_for_selection",
                "target_labels_used_for_tuning",
                "target_selected",
                "selection_uses_target_labels",
                "target_data_used_for_selection",
                "target_data_used_for_tuning",
            }:
                return True
            if "target" in key and ("label" in key or "data" in key) and (
                "select" in key or "tune" in key or "metric" in key
            ):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(_target_selection_claim(item) for item in payload)
    return False


def frozen_hhar_provenance(path: str | Path) -> dict[str, Any]:
    """Validate/fingerprint HHAR's single five-flow dataset-level tuning."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        return {
            "status": "missing",
            "wait_status": "waiting_for_hhar_frozen_state",
            "path": _absolute(config_path),
            "target_labels_used_for_selection": True,
            "target_data_used_for_selection": True,
            "selection_mode": "target_development_f1_pending",
        }
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HHAR frozen state {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("HHAR frozen state must contain a JSON object")
    signature = payload.get("signature")
    if isinstance(payload.get("tta_config"), Mapping) and isinstance(
        signature, Mapping
    ):
        if payload.get("completed") is not True:
            return {
                "status": "incomplete",
                "wait_status": "waiting_for_hhar_frozen_state",
                "path": _absolute(config_path),
                "target_labels_used_for_selection": True,
                "target_data_used_for_selection": True,
                "selection_mode": "target_development_f1_pending",
            }
        if signature.get("target_labels_used_for_selection") is not True:
            raise ValueError("HHAR final state must declare target-label selection")
        if tuple(signature.get("evaluation_flows", ())) != tuple(HHAR_REPORTED_FLOWS):
            raise ValueError("HHAR formal five-flow protocol drifted")
        if signature.get("parameter_selection_data_overlap") is not True:
            raise ValueError("HHAR signature must declare parameter-selection overlap")
        if signature.get("confirmatory") is not False:
            raise ValueError("HHAR target-selected evaluation cannot be confirmatory")
        manifest_path = config_path.with_name("manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid HHAR tuning manifest {manifest_path}: {exc}") from exc
        if manifest.get("status") != "complete":
            raise ValueError("HHAR tuning manifest is not complete")
        if tuple(manifest.get("evaluation_flows", ())) != tuple(HHAR_REPORTED_FLOWS):
            raise ValueError("HHAR tuning manifest has a different five-flow protocol")
        if manifest.get("parameter_selection_data_overlap") is not True:
            raise ValueError("HHAR tuning manifest must declare selection overlap")
        if manifest.get("confirmatory") is not False:
            raise ValueError("HHAR tuning manifest must be non-confirmatory")
        return {
            "status": "ready",
            "path": _absolute(config_path),
            "sha256": file_sha256(config_path),
            "manifest_path": _absolute(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "selection_mode": "target_selected_five_flow_f1",
            "target_labels_used_for_selection": True,
            "target_data_used_for_selection": True,
            "evaluation_flows": list(HHAR_REPORTED_FLOWS),
            "evaluation_partition": "target_selected_evaluation",
            "parameter_selection_data_overlap": True,
            "confirmatory": False,
            "source_only": False,
        }

    # Backward-compatible source-only fixture/profile path.
    if _target_selection_claim(payload):
        raise ValueError(
            "HHAR frozen state selected with target labels/data but uses an unsupported schema"
        )
    for key in ("target_labels_used", "target_data_used"):
        if key not in payload or _explicit_true(payload.get(key)):
            raise ValueError(f"source-only HHAR state must declare {key}=false")
    return {
        "status": "ready",
        "path": _absolute(config_path),
        "sha256": file_sha256(config_path),
        "profile_id": payload.get("profile_id", payload.get("selected_profile_id")),
        # These are protocol declarations, not inferred from absent fields.
        "target_labels_used_for_selection": False,
        "target_data_used_for_selection": False,
        "source_only": True,
        "selection_mode": "source_only",
    }


def _normalize_datasets(datasets: Iterable[str] | None) -> tuple[str, ...]:
    raw = DATASETS if datasets is None else datasets
    if isinstance(raw, str):
        raw = [value.strip() for value in raw.split(",") if value.strip()]
    requested = tuple(str(value).strip().upper() for value in raw if str(value).strip())
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("datasets must be non-empty and unique")
    unknown = sorted(set(requested) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return requested


def _normalize_source_seeds(source_seeds: Iterable[int] | None) -> tuple[int, ...]:
    raw = SOURCE_SEEDS if source_seeds is None else source_seeds
    if isinstance(raw, str):
        raw = [value.strip() for value in raw.split(",") if value.strip()]
    requested = tuple(int(value) for value in raw)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("source_seeds must be non-empty and unique")
    unknown = sorted(set(requested) - set(SOURCE_SEEDS))
    if unknown:
        raise ValueError(f"source_seeds must be registered values 1,2,3; unknown={unknown}")
    return tuple(seed for seed in SOURCE_SEEDS if seed in set(requested))


def _normalize_horizons(horizons: Iterable[int] | None) -> tuple[int, ...]:
    raw = HORIZONS if horizons is None else horizons
    if isinstance(raw, str):
        raw = [value.strip() for value in raw.split(",") if value.strip()]
    requested = tuple(int(value) for value in raw)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("horizons must be non-empty and unique")
    unknown = sorted(set(requested) - set(HORIZONS))
    if unknown:
        raise ValueError(f"horizons must be registered values 1,3,5; unknown={unknown}")
    return tuple(horizon for horizon in HORIZONS if horizon in set(requested))


def _normalize_conditions(conditions: Iterable[str] | None) -> tuple[str, ...]:
    if conditions is None:
        requested = tuple(row["condition"] for row in _condition_rows())
    else:
        raw = (
            [value.strip() for value in conditions.split(",") if value.strip()]
            if isinstance(conditions, str)
            else conditions
        )
        requested = tuple(str(value).strip() for value in raw if str(value).strip())
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("conditions must be non-empty and unique")
    registered = tuple(row["condition"] for row in _condition_rows())
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        raise ValueError(
            "conditions must be clean or registered corruption:severity values; "
            f"unknown={unknown}"
        )
    return tuple(condition for condition in registered if condition in set(requested))


def _normalize_scenarios(
    datasets: Iterable[str],
    scenarios: Mapping[str, Iterable[str]] | Iterable[str] | None,
) -> dict[str, tuple[str, ...]]:
    requested_datasets = _normalize_datasets(datasets)
    registered = {
        dataset: tuple(
            f"{source}->{target}" for source, target in formal_scenario_pairs(dataset)
        )
        for dataset in requested_datasets
    }
    if scenarios is None:
        return registered
    if isinstance(scenarios, Mapping):
        normalized_mapping = {str(key).strip().upper(): value for key, value in scenarios.items()}
        foreign = sorted(set(normalized_mapping) - set(requested_datasets))
        if foreign:
            raise ValueError(f"scenario filter contains foreign datasets: {foreign}")
        result = {}
        for dataset in requested_datasets:
            if dataset not in normalized_mapping:
                raise ValueError(f"scenario filter is missing dataset {dataset}")
            values = normalized_mapping[dataset]
            if isinstance(values, str):
                values = [item.strip() for item in values.split(",") if item.strip()]
            requested = tuple(str(value).strip() for value in values if str(value).strip())
            if not requested or len(set(requested)) != len(requested):
                raise ValueError(f"{dataset}: scenarios must be non-empty and unique")
            unknown = sorted(set(requested) - set(registered[dataset]))
            if unknown:
                raise ValueError(f"{dataset}: scenarios must be registered flows; unknown={unknown}")
            result[dataset] = tuple(item for item in registered[dataset] if item in set(requested))
        return result
    if len(requested_datasets) != 1:
        raise ValueError("a bare scenario iterable requires exactly one dataset")
    values = (
        [item.strip() for item in scenarios.split(",") if item.strip()]
        if isinstance(scenarios, str)
        else scenarios
    )
    return _normalize_scenarios(requested_datasets, {requested_datasets[0]: values})


def _flow_rows(
    datasets: Iterable[str] | None = None,
    scenarios: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    requested_datasets = _normalize_datasets(datasets)
    selected = _normalize_scenarios(requested_datasets, scenarios)
    for dataset in requested_datasets:
        for scenario in selected[dataset]:
            source, target = scenario.split("->", 1)
            validate_scenario(dataset, source, target)
            rows.append(
                {
                    "dataset": dataset,
                    "source_domain": str(source),
                    "target_domain": str(target),
                    "scenario": f"{source}->{target}",
                }
            )
    return rows


FORMAL_FLOWS = tuple((row["dataset"], row["scenario"]) for row in _flow_rows())


def _condition_rows(conditions: Iterable[str] | None = None) -> list[dict[str, Any]]:
    rows = [{"condition": CLEAN_CONDITION, "corruption": "none", "severity": None}]
    rows.extend(
        {
            "condition": f"{corruption}:{severity}",
            "corruption": corruption,
            "severity": severity,
        }
        for corruption in FORMAL_CORRUPTIONS
        for severity in FORMAL_SEVERITIES
    )
    if conditions is None:
        return rows
    selected = _normalize_conditions(conditions)
    return [row for row in rows if row["condition"] in set(selected)]


EXPECTED_STREAM_CELL_COUNT = len(FORMAL_FLOWS) * len(SOURCE_SEEDS) * len(
    _condition_rows()
)
EXPECTED_CELL_COUNT = EXPECTED_STREAM_CELL_COUNT * len(HORIZONS)


def expected_cell_count() -> int:
    """Return the frozen count including the explicit horizon dimension."""

    return EXPECTED_CELL_COUNT


def expected_stream_cell_count() -> int:
    """Return the count if all three horizons are run in one process."""

    return EXPECTED_STREAM_CELL_COUNT


def _expected_key_set_for(
    *,
    datasets: Iterable[str] | None = None,
    scenarios: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    source_seeds: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
) -> frozenset[str]:
    """Return endpoint keys for a validated explicit subset."""

    keys: set[str] = set()
    selected_horizons = _normalize_horizons(horizons)
    selected_seeds = _normalize_source_seeds(source_seeds)
    selected_conditions = _condition_rows(_normalize_conditions(conditions))
    for flow in _flow_rows(datasets, scenarios):
        for source_seed in selected_seeds:
            for horizon in selected_horizons:
                for condition in selected_conditions:
                    keys.add(
                        make_cell_key(
                            dataset=flow["dataset"],
                            scenario=flow["scenario"],
                            source_seed=source_seed,
                            stream_seed=STREAM_SEED,
                            horizon=horizon,
                            corruption=condition["corruption"],
                            severity=condition["severity"],
                            horizons=selected_horizons,
                        )
                    )
    return frozenset(keys)


def _expected_stream_key_set_for(
    *,
    datasets: Iterable[str] | None = None,
    scenarios: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    source_seeds: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
) -> frozenset[str]:
    """Return subprocess keys for a validated explicit subset."""

    keys: set[str] = set()
    selected_horizons = _normalize_horizons(horizons)
    selected_seeds = _normalize_source_seeds(source_seeds)
    selected_conditions = _condition_rows(_normalize_conditions(conditions))
    for flow in _flow_rows(datasets, scenarios):
        for source_seed in selected_seeds:
            for condition in selected_conditions:
                keys.add(
                    make_stream_cell_key(
                        dataset=flow["dataset"],
                        scenario=flow["scenario"],
                        source_seed=source_seed,
                        stream_seed=STREAM_SEED,
                        corruption=condition["corruption"],
                        severity=condition["severity"],
                        horizons=selected_horizons,
                    )
                )
    return frozenset(keys)


def _expected_key_set() -> frozenset[str]:
    """Default endpoint keys, retained for compatibility (2340 total)."""

    return _expected_key_set_for()


def _expected_stream_key_set() -> frozenset[str]:
    """Default subprocess keys, retained for compatibility (780 total)."""

    return _expected_stream_key_set_for()


def make_cell_key(
    *,
    dataset: str,
    scenario: str,
    source_seed: int,
    stream_seed: int,
    horizon: int,
    corruption: str,
    severity: str | None,
    horizons: Iterable[int] = HORIZONS,
) -> str:
    """Build a canonical, human-readable unique cell key."""

    return "|".join(
        (
            str(dataset).upper(),
            str(scenario),
            f"source_seed={int(source_seed)}",
            f"stream_seed={int(stream_seed)}",
            f"horizon={int(horizon)}",
            f"corruption={str(corruption)}",
            f"severity={str(severity) if severity is not None else CLEAN_CONDITION}",
        )
    )


def make_stream_cell_key(
    *,
    dataset: str,
    scenario: str,
    source_seed: int,
    stream_seed: int,
    corruption: str,
    severity: str | None,
    horizons: Iterable[int] = HORIZONS,
) -> str:
    """Build a canonical subprocess key shared by horizons 1/3/5."""

    return "|".join(
        (
            str(dataset).upper(),
            str(scenario),
            f"source_seed={int(source_seed)}",
            f"stream_seed={int(stream_seed)}",
            "horizons=" + ",".join(str(int(value)) for value in horizons),
            f"corruption={str(corruption)}",
            f"severity={str(severity) if severity is not None else CLEAN_CONDITION}",
        )
    )


def _cell_id(key: str, ordinal: int) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"cell-{ordinal:04d}-{digest}"


def _single_cell_command(
    *,
    cell: Mapping[str, Any],
    data_path: str | Path,
    device: str,
    backbone: str,
    pretrain_cache_dir: str | Path,
    output_dir: str | Path,
    hhar_frozen_state: str | Path | None,
    flow_profile_json: str | Path = DEFAULT_PAPER_FLOW_PROFILE_JSON,
) -> list[str]:
    dataset = str(cell["dataset"]).upper()
    cache_dir = (
        Path(pretrain_cache_dir) / "hhar_formal"
        if dataset == "HHAR"
        else Path(pretrain_cache_dir) / "optuna_stepwise"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_full_no_ssaw_horizon_audit.py"),
        "--data-path",
        str(data_path),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--dataset",
        str(cell["dataset"]),
        "--scenario",
        str(cell["scenario"]),
        "--source-seed",
        str(cell["source_seed"]),
        "--stream-seed",
        str(cell["stream_seed"]),
        "--horizons",
        ",".join(str(value) for value in cell["horizons"]),
        "--corruption",
        str(cell["corruption"]),
        "--corruption-fraction",
        str(CORRUPTION_FRACTION),
        "--corruption-seed",
        str(CORRUPTION_SEED),
        "--pretrain-cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--queue-cell-key",
        str(cell["key"]),
        "--low-memory",
        "--flow-profile-json",
        str(flow_profile_json),
    ]
    # Clean cells ignore severity in the single-cell runner.  Supplying a
    # valid categorical value keeps the command parser strict and leaves the
    # canonical condition in the queue key/manifest as ``clean``.
    command.extend(("--severity", str(cell["severity"] or "moderate")))
    if dataset == "HHAR":
        if hhar_frozen_state is not None:
            command.extend(("--hhar-frozen-config", str(hhar_frozen_state)))
    return command


def build_queue(
    *,
    data_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pretrain_cache_dir: str | Path = DEFAULT_PRETRAIN_CACHE_DIR,
    device: str = "cpu",
    backbone: str = "CNN",
    hhar_frozen_state: str | Path | None = DEFAULT_HHAR_FROZEN_STATE,
    hhar_frozen_config: str | Path | None = None,
    datasets: Iterable[str] | None = None,
    scenarios: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    source_seeds: Iterable[int] | None = None,
    horizons: Iterable[int] | None = None,
    flow_profile_json: str | Path = DEFAULT_PAPER_FLOW_PROFILE_JSON,
) -> dict[str, Any]:
    """Build and validate the complete formal cell manifest in memory."""

    if hhar_frozen_config is not None:
        hhar_frozen_state = hhar_frozen_config
    selected_datasets = _normalize_datasets(datasets)
    # Validate the reviewed per-flow TTA profile during planning; the child
    # command receives the same path and applies the exact selected flow.
    from scripts.paper_flow_profiles import load_paper_flow_profiles

    load_paper_flow_profiles(flow_profile_json, selected_datasets)
    selected_scenarios = _normalize_scenarios(selected_datasets, scenarios)
    selected_seeds = _normalize_source_seeds(source_seeds)
    selected_conditions = _normalize_conditions(conditions)
    selected_horizons = _normalize_horizons(horizons)
    selected_flows = _flow_rows(selected_datasets, selected_scenarios)
    selected_condition_rows = _condition_rows(selected_conditions)
    if tuple(DATASET_NAMES) != DATASETS:
        raise ValueError(
            f"dataset registry drifted: expected {DATASETS}, observed {tuple(DATASET_NAMES)}"
        )
    if tuple(SOURCE_SEEDS) != (1, 2, 3) or tuple(HORIZONS) != (1, 3, 5):
        raise ValueError("formal source seeds/horizons drifted")
    unknown = sorted(set(FORMAL_CORRUPTIONS) - set(CORRUPTION_REGISTRY))
    if unknown:
        raise ValueError(f"formal corruption registry is missing {unknown}")
    output_root = Path(output_dir).expanduser().resolve()
    data_root = _absolute(data_path)
    pretrain_root = _absolute(pretrain_cache_dir)
    frozen_path = None if hhar_frozen_state is None else _absolute(hhar_frozen_state)
    frozen = (
        {"status": "not_required"}
        if frozen_path is None
        else frozen_hhar_provenance(frozen_path)
    )
    cells: list[dict[str, Any]] = []
    ordinal = 0
    for flow in selected_flows:
        for source_seed in selected_seeds:
            for condition in selected_condition_rows:
                ordinal += 1
                key = make_stream_cell_key(
                    dataset=flow["dataset"],
                    scenario=flow["scenario"],
                    source_seed=source_seed,
                    stream_seed=STREAM_SEED,
                    corruption=condition["corruption"],
                    severity=condition["severity"],
                    horizons=selected_horizons,
                )
                cell_id = _cell_id(key, ordinal)
                cell_output = output_root / "cells" / cell_id
                log_path = output_root / "logs" / f"{cell_id}.log"
                scenario = flow["scenario"]
                partition_metadata = evaluation_partition_metadata(
                    flow["dataset"], scenario
                )
                evaluation_partition = str(
                    partition_metadata["evaluation_partition"]
                )
                selection_overlap = bool(
                    partition_metadata["selection_overlap"]
                )
                target_selected = bool(
                    frozen.get("target_labels_used_for_selection", True)
                    if flow["dataset"] == "HHAR"
                    else True
                )
                cell = {
                    "id": cell_id,
                    "key": key,
                    "unique_key": key,
                    "dataset": flow["dataset"],
                    "scenario": flow["scenario"],
                    "source_domain": flow["source_domain"],
                    "target_domain": flow["target_domain"],
                    "source_seed": int(source_seed),
                    "stream_seed": int(STREAM_SEED),
                    "horizons": list(selected_horizons),
                    "expected_endpoint_keys": [
                        make_cell_key(
                            dataset=flow["dataset"],
                            scenario=flow["scenario"],
                            source_seed=source_seed,
                            stream_seed=STREAM_SEED,
                            horizon=horizon,
                            corruption=condition["corruption"],
                            severity=condition["severity"],
                        )
                        for horizon in selected_horizons
                    ],
                    "corruption": condition["corruption"],
                    "severity": condition["severity"],
                    "condition": condition["condition"],
                    "target_labels_used_for_updates": False,
                    "target_labels_used_for_parameter_selection": target_selected,
                    "parameter_selection_data_overlap": selection_overlap,
                    "evaluation_partition": evaluation_partition,
                    "target_labels_used_for_metrics": True,
                    "hhar_frozen_config": frozen_path
                    if flow["dataset"] == "HHAR"
                    else None,
                    "output_dir": str(cell_output),
                    "log_path": str(log_path),
                    "status": "planned",
                    "attempts": 0,
                }
                cell["command"] = _single_cell_command(
                    cell=cell,
                    data_path=data_root,
                    device=device,
                    backbone=backbone,
                    pretrain_cache_dir=pretrain_root,
                    output_dir=cell_output,
                    hhar_frozen_state=frozen_path,
                    flow_profile_json=flow_profile_json,
                )
                cells.append(cell)
    validate_cells(
        cells,
        datasets=selected_datasets,
        scenarios=selected_scenarios,
        source_seeds=selected_seeds,
        conditions=selected_conditions,
        horizons=selected_horizons,
    )
    full_scope = (
        tuple(selected_datasets) == tuple(DATASETS)
        and all(
            tuple(selected_scenarios[dataset])
            == tuple(f"{source}->{target}" for source, target in formal_scenario_pairs(dataset))
            for dataset in selected_datasets
        )
        and tuple(selected_seeds) == tuple(SOURCE_SEEDS)
        and tuple(selected_conditions) == tuple(row["condition"] for row in _condition_rows())
        and tuple(selected_horizons) == tuple(HORIZONS)
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": _utc_now(),
        "status": "planned",
        "dry_run": True,
        "data_path": _absolute(data_path),
        "output_dir": str(output_root),
        "pretrain_cache_dir": pretrain_root,
        "flow_profile_json": _absolute(flow_profile_json),
        "device": str(device),
        "gpu_lock_required": _is_gpu_device(device),
        "backbone": str(backbone),
        "datasets": list(selected_datasets),
        "flow_count": len(selected_flows),
        "flows": [row["scenario"] for row in selected_flows],
        "flows_by_dataset": {
            dataset: [
                row["scenario"] for row in selected_flows if row["dataset"] == dataset
            ]
            for dataset in selected_datasets
        },
        "scenario_scope": (
            "registered_formal_full"
            if full_scope
            else "registered_representative_subset"
        ),
        "source_seeds": list(selected_seeds),
        "stream_seed": STREAM_SEED,
        "horizons": list(selected_horizons),
        "conditions": list(selected_conditions),
        "corruptions": [
            CLEAN_CONDITION,
            *tuple(
                corruption
                for corruption in FORMAL_CORRUPTIONS
                if any(
                    row["corruption"] == corruption for row in selected_condition_rows
                )
            ),
        ],
        "severities": sorted(
            {
                row["severity"]
                for row in selected_condition_rows
                if row["severity"] is not None
            }
        ),
        "corruption_fraction": CORRUPTION_FRACTION,
        "expected_stream_cell_count": len(cells),
        "expected_cell_count": len(cells) * len(selected_horizons),
        "expected_cells": len(cells) * len(selected_horizons),
        "target_labels_used_for_updates": False,
        "target_labels_used_for_parameter_selection": {
            **{dataset: True for dataset in selected_datasets if dataset != "HHAR"},
            **(
                {"HHAR": bool(frozen.get("target_labels_used_for_selection", True))}
                if "HHAR" in selected_datasets
                else {}
            ),
        },
        "target_labels_used_for_metrics": True,
        "evidence_role": "A_confidence_accept_all_vs_admitted_future_horizon",
        "evidence_role_policy": {
            "horizon": 5,
            "future_metrics": ["clean_f1", "corrupted_f1"],
            "safety_metrics_source": "controlled_safety_benchmark_same_batch_start",
            "two_by_two_grid": "audit_only",
        },
        "hhar_frozen_state": frozen,
        "max_retries": MAX_RETRIES,
        "process_isolation": "one fresh subprocess per stream cell; horizons 1/3/5 share one trajectory",
        "cells": cells,
    }


def validate_cells(
    cells: Iterable[Mapping[str, Any]],
    *,
    datasets: Iterable[str] | None = None,
    scenarios: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    source_seeds: Iterable[int] | None = None,
    conditions: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Fail closed on duplicate, missing, or malformed protocol keys."""

    rows = list(cells)
    selected_datasets = _normalize_datasets(datasets)
    selected_scenarios = _normalize_scenarios(selected_datasets, scenarios)
    selected_seeds = _normalize_source_seeds(source_seeds)
    selected_conditions = _normalize_conditions(conditions)
    selected_horizons = _normalize_horizons(horizons)
    expected_keys = set(
        _expected_stream_key_set_for(
            datasets=selected_datasets,
            scenarios=selected_scenarios,
            source_seeds=selected_seeds,
            conditions=selected_conditions,
            horizons=selected_horizons,
        )
    )
    expected_endpoint_keys = set(
        _expected_key_set_for(
            datasets=selected_datasets,
            scenarios=selected_scenarios,
            source_seeds=selected_seeds,
            conditions=selected_conditions,
            horizons=selected_horizons,
        )
    )
    observed = [str(row.get("key", "")) for row in rows]
    duplicate_keys = sorted(
        key for key, count in Counter(observed).items() if count > 1
    )
    if duplicate_keys:
        raise ValueError(f"duplicate formal stream cell keys: {duplicate_keys[:5]}")
    observed_set = set(observed)
    missing = sorted(expected_keys - observed_set)
    unexpected = sorted(observed_set - expected_keys)
    if missing or unexpected or len(rows) != len(expected_keys):
        raise ValueError(
            "formal stream queue key set mismatch: "
            f"expected={len(expected_keys)}, observed={len(rows)}, "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for row in rows:
        if tuple(row.get("horizons", ())) != selected_horizons:
            raise ValueError(
                "every stream cell must evaluate the explicitly selected horizons"
            )
        endpoint_keys = tuple(row.get("expected_endpoint_keys", ()))
        if len(endpoint_keys) != len(selected_horizons) or any(
            key not in expected_endpoint_keys for key in endpoint_keys
        ):
            raise ValueError("stream cell endpoint keys do not match the formal horizon grid")
        if row.get("target_labels_used_for_updates") is not False:
            raise ValueError("target labels cannot be used by online updates")
        target_selected = bool(row.get("target_labels_used_for_parameter_selection"))
        partition = str(row.get("evaluation_partition", ""))
        if target_selected and partition != "target_selected_evaluation":
            raise ValueError("target-selected cells require an evaluation partition")
        if partition == "target_selected_evaluation" and row.get(
            "parameter_selection_data_overlap"
        ) is not True:
            raise ValueError("target-selected formal cells must declare selection overlap")
        if row.get("target_labels_used_for_metrics") is not True:
            raise ValueError("true target labels must be explicitly limited to offline metrics")
    return {
        "expected_stream_cell_count": len(expected_keys),
        "expected_endpoint_cell_count": len(expected_endpoint_keys),
        "observed_stream_cell_count": len(rows),
        "key_count": len(observed_set),
        "duplicate_keys": duplicate_keys,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "passed": not duplicate_keys and not missing and not unexpected,
    }


def is_oom_text(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        token in lowered
        for token in (
            "out of memory",
            "cuda error: out of memory",
            "cudnn_status_alloc_failed",
            "memoryerror",
            "std::bad_alloc",
            "cannot allocate memory",
        )
    )


def is_retryable_failure(returncode: int, output: str, is_oom: bool) -> bool:
    """Classify failures that are safe to retry in a fresh process."""

    if int(returncode) == 0:
        return False
    if bool(is_oom):
        return True
    # Negative POSIX values and common Windows access-violation/stack-overflow
    # statuses indicate a native crash; the parent remains alive and can retry.
    if int(returncode) < 0 or int(returncode) in {
        -1073741819,  # 0xC0000005 access violation
        -1073740791,  # 0xC0000409 stack buffer overrun
        0xC0000005,
        0xC0000409,
        0xC0000017,  # STATUS_NO_MEMORY
        0xC000009A,  # STATUS_INSUFFICIENT_RESOURCES
    }:
        return True
    return "segmentation fault" in str(output).lower()


@contextlib.contextmanager
def _gpu_lock(path: str | Path):
    """Hold the repository-wide recoverable GPU lock.

    The main-table runner, physical/baseline queues, tuner, and protocol
    supervisor all use :class:`GPUExperimentLock`'s ``O_EXCL`` lock file.  The
    former horizon-specific byte-range lock did not contend with that file on
    Windows, so two GPU stages could overlap despite sharing the same path.
    """

    with wait_for_gpu_experiment_lock(path):
        yield


def _read_tail(path: str | Path, limit: int = 12000) -> str:
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit), os.SEEK_SET)
            return handle.read()
    except OSError:
        return ""


def _publish(plan: Mapping[str, Any], output_dir: str | Path) -> None:
    atomic_write_json(plan, Path(output_dir) / "manifest.json")


def _frozen_ready_for_cell(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> tuple[bool, str]:
    if str(cell.get("dataset", "")).upper() != "HHAR":
        return True, "not_required"
    state = plan.get("hhar_frozen_state", {})
    path = state.get("path") if isinstance(state, Mapping) else None
    if not path:
        return False, "no HHAR frozen state path configured"
    if not Path(path).is_file():
        return False, f"HHAR frozen state is not available: {path}"
    observed = frozen_hhar_provenance(path)
    if observed.get("status") != "ready":
        return False, f"HHAR frozen state is {observed.get('status', 'not ready')}"
    expected_hash = state.get("sha256") if isinstance(state, Mapping) else None
    if expected_hash and observed.get("sha256") != expected_hash:
        return False, "HHAR frozen state changed after queue planning"
    return True, "ready"


def _wait_for_frozen(
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    enabled: bool,
    timeout: float,
    poll_seconds: float,
) -> tuple[bool, str]:
    ready, reason = _frozen_ready_for_cell(plan, cell)
    if ready or not enabled:
        return ready, reason
    started = time.monotonic()
    while True:
        time.sleep(max(0.01, float(poll_seconds)))
        ready, reason = _frozen_ready_for_cell(plan, cell)
        if ready:
            return True, reason
        if timeout > 0 and time.monotonic() - started >= timeout:
            return False, f"HHAR frozen-state wait timed out after {timeout:.1f}s: {reason}"


def _run_cell_once(
    cell: dict[str, Any],
    *,
    gpu_lock_path: str | Path | None = None,
) -> tuple[int, bool, str]:
    command = [str(item) for item in cell["command"]]
    log_path = Path(cell["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(cell["output_dir"]).mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{started}] command={json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        try:
            lock_context = (
                _gpu_lock(gpu_lock_path)
                if gpu_lock_path is not None
                else contextlib.nullcontext()
            )
            with lock_context:
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
                returncode = int(process.wait())
        except (OSError, ValueError) as exc:
            log.write(f"parent failed to launch cell: {exc!r}\n")
            return 1, False, repr(exc)
        log.write(f"[{_utc_now()}] returncode={returncode}\n")
    tail = _read_tail(log_path)
    oom = is_oom_text(tail)
    return returncode, oom, tail


def execute_queue(
    plan: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = True,
    resume: bool = True,
    max_retries: int = MAX_RETRIES,
    wait_for_hhar_frozen: bool = False,
    hhar_wait_timeout: float = 0.0,
    hhar_poll_seconds: float = 10.0,
    gpu_lock_path: str | Path = GPU_LOCK_PATH,
) -> tuple[int, dict[str, Any]]:
    """Execute/resume a serial process-isolated queue and atomically publish it."""

    if int(max_retries) < 1:
        raise ValueError("max_retries must be positive")
    output_root = Path(output_dir or plan.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if resume and manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            prior = None
        if isinstance(prior, Mapping) and prior.get("protocol") == plan.get("protocol"):
            prior_cells = {
                str(row.get("key")): row
                for row in prior.get("cells", [])
                if isinstance(row, Mapping)
            }
            for cell in plan["cells"]:
                old = prior_cells.get(str(cell["key"]))
                if old is not None:
                    for field in (
                        "status",
                        "attempts",
                        "returncode",
                        "is_oom",
                        "error",
                        "started_at",
                        "finished_at",
                        "output_tail",
                    ):
                        if field in old:
                            cell[field] = old[field]
    plan["max_retries"] = int(max_retries)
    plan["dry_run"] = bool(dry_run)
    plan["resume"] = bool(resume)
    plan["started_at"] = _utc_now()
    plan["status"] = "dry_run" if dry_run else "running"
    _publish(plan, output_root)
    if dry_run:
        plan["finished_at"] = _utc_now()
        _publish(plan, output_root)
        return 0, plan

    # Wait for the HHAR tuner before taking the shared GPU lock.  Otherwise a
    # waiting queue can deadlock the tuner that must acquire the same lock.
    hhar_cell = next(
        (cell for cell in plan["cells"] if str(cell.get("dataset", "")).upper() == "HHAR"),
        None,
    )
    if hhar_cell is not None:
        ready, reason = _wait_for_frozen(
            plan,
            hhar_cell,
            enabled=wait_for_hhar_frozen,
            timeout=float(hhar_wait_timeout),
            poll_seconds=float(hhar_poll_seconds),
        )
        if not ready:
            plan["status"] = "waiting_for_hhar_frozen_state"
            plan["blocked_reason"] = reason
            _publish(plan, output_root)
            return 2, plan
        frozen_path = plan.get("hhar_frozen_state", {}).get("path")
        if frozen_path:
            plan["hhar_frozen_state"] = frozen_hhar_provenance(frozen_path)

    lock_required = _is_gpu_device(str(plan.get("device", "cpu")))
    # Lock per stream cell, not for the complete multi-day parent queue.  This
    # preserves single-GPU mutual exclusion while allowing other formal queues
    # to make progress between independently isolated cells.
    lock_context = contextlib.nullcontext()
    plan["gpu_lock_scope"] = "per_stream_cell" if lock_required else "not_required"
    plan["gpu_lock_busy_consumes_attempt"] = False
    returncode = 0
    had_failure = False
    with lock_context:
        for index, cell in enumerate(plan["cells"]):
            status = str(cell.get("status", "planned"))
            if status == "completed":
                continue
            if status in {"failed", "oom"} and int(cell.get("attempts", 0)) >= int(max_retries):
                had_failure = True
                returncode = returncode or int(cell.get("returncode") or 1)
                continue
            ready, reason = _wait_for_frozen(
                plan,
                cell,
                enabled=wait_for_hhar_frozen,
                timeout=float(hhar_wait_timeout),
                poll_seconds=float(hhar_poll_seconds),
            )
            if not ready:
                cell["status"] = "waiting_for_hhar_frozen_state"
                cell["blocked_reason"] = reason
                plan["status"] = "waiting_for_hhar_frozen_state"
                _publish(plan, output_root)
                return 2, plan
            if str(cell.get("dataset", "")).upper() == "HHAR":
                # If the profile appeared after planning, freeze its
                # fingerprint before the first HHAR subprocess starts.
                frozen_path = plan.get("hhar_frozen_state", {}).get("path")
                if frozen_path:
                    plan["hhar_frozen_state"] = frozen_hhar_provenance(frozen_path)
            attempts = int(cell.get("attempts", 0))
            cell["status"] = "running"
            plan["status"] = "running"
            _publish(plan, output_root)
            succeeded = False
            for attempt in range(attempts + 1, int(max_retries) + 1):
                cell["attempts"] = attempt
                cell["started_at"] = _utc_now()
                return_one, oom, output_tail = _run_cell_once(
                    cell,
                    gpu_lock_path=(gpu_lock_path if lock_required else None),
                )
                cell["returncode"] = return_one
                cell["is_oom"] = bool(oom)
                cell["output_tail"] = output_tail[-12000:]
                cell["finished_at"] = _utc_now()
                if return_one == 0:
                    # A zero exit must leave the child manifest; this catches
                    # accidental early exits that would otherwise masquerade
                    # as a completed protocol cell.
                    child_manifest = Path(cell["output_dir"]) / "manifest.json"
                    if not child_manifest.is_file():
                        cell["status"] = "failed"
                        cell["error"] = "cell exited zero but emitted no manifest.json"
                        return_one = 1
                    else:
                        try:
                            child_payload = json.loads(
                                child_manifest.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError, json.JSONDecodeError) as exc:
                            child_payload = None
                            cell["error"] = f"invalid child manifest: {exc}"
                        child_horizons = (
                            tuple(sorted(int(value) for value in child_payload.get("horizons", ())))
                            if isinstance(child_payload, Mapping)
                            else ()
                        )
                        expected_child_horizons = tuple(
                            sorted(int(value) for value in cell.get("horizons", ()))
                        )
                        if not isinstance(child_payload, Mapping) or (
                            child_payload.get("queue_cell_key") != cell["key"]
                        ) or child_horizons != expected_child_horizons or child_payload.get(
                            "protocol_passed"
                        ) is not True:
                            cell["status"] = "failed"
                            cell["error"] = (
                                cell.get("error")
                                or "child manifest key/horizons/protocol status does not match parent"
                            )
                            return_one = 1
                        else:
                            cell["status"] = "completed"
                            cell["error"] = None
                            succeeded = True
                if succeeded:
                    break
                retryable = is_retryable_failure(return_one, output_tail, oom)
                cell["status"] = "oom" if oom else "failed"
                cell["error"] = (
                    "subprocess reported out-of-memory"
                    if oom
                    else f"subprocess exited {return_one}"
                )
                if attempt >= int(max_retries) or not retryable:
                    break
                cell["status"] = "retry_pending"
                _publish(plan, output_root)
            if not succeeded:
                had_failure = True
                failure_code = 75 if cell.get("is_oom") else int(
                    cell.get("returncode") or 1
                )
                if returncode == 0:
                    returncode = failure_code
                _publish(plan, output_root)
                # Cells are independent.  Continue after exhausting this
                # cell's retries so one OOM/native crash cannot block the
                # remaining protocol grid.
                continue
            _publish(plan, output_root)
    plan["status"] = "complete_with_failures" if had_failure else "complete"
    plan["finished_at"] = _utc_now()
    plan["completed_cell_count"] = sum(
        str(cell.get("status")) == "completed" for cell in plan["cells"]
    )
    plan["failed_cell_count"] = sum(
        str(cell.get("status")) in {"failed", "oom"} for cell in plan["cells"]
    )
    _publish(plan, output_root)
    return returncode, plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--pretrain-cache-dir", default=str(DEFAULT_PRETRAIN_CACHE_DIR)
    )
    parser.add_argument(
        "--hhar-frozen-state",
        "--hhar-frozen-config",
        default=str(DEFAULT_HHAR_FROZEN_STATE),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(DEFAULT_PAPER_FLOW_PROFILE_JSON),
        help=(
            "Per-flow TTA override JSON; defaults to "
            "configs/paper_flow_profiles_v1.json."
        ),
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--wait-for-hhar-frozen", action="store_true")
    parser.add_argument("--hhar-wait-timeout", type=float, default=0.0)
    parser.add_argument("--hhar-poll-seconds", type=float, default=10.0)
    parser.add_argument("--gpu-lock-path", default=str(GPU_LOCK_PATH))
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated registered datasets (default: all four)",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Registered flow filter, e.g. HAR:12->16 or 12->16 for one dataset",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated conditions, e.g. clean,signal_freeze:severe",
    )
    parser.add_argument(
        "--source-seeds",
        default=",".join(str(seed) for seed in SOURCE_SEEDS),
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(horizon) for horizon in HORIZONS),
    )
    parser.add_argument("--json-out", default=None)
    return parser


def run_queue(*args, **kwargs):
    """Compatibility alias for callers that use the queue verb."""

    return execute_queue(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_retries < 1:
        parser.error("--max-retries must be positive")
    if args.hhar_wait_timeout < 0 or args.hhar_poll_seconds <= 0:
        parser.error("HHAR wait timeout must be non-negative and poll seconds positive")
    datasets = tuple(
        value.strip().upper() for value in str(args.datasets).split(",") if value.strip()
    )
    scenarios = None
    if args.scenarios is not None:
        parsed: dict[str, list[str]] = {}
        for item in (value.strip() for value in str(args.scenarios).split(",")):
            if not item:
                continue
            if ":" in item:
                dataset, scenario = item.split(":", 1)
                dataset = dataset.strip().upper()
            else:
                if len(datasets) != 1:
                    parser.error("bare --scenarios requires exactly one selected dataset")
                dataset, scenario = datasets[0], item
            parsed.setdefault(dataset, []).append(scenario.strip())
        scenarios = parsed
    conditions = tuple(
        value.strip() for value in str(args.conditions).split(",") if value.strip()
    ) if args.conditions is not None else None
    source_seeds = tuple(
        int(value.strip()) for value in str(args.source_seeds).split(",") if value.strip()
    )
    horizons = tuple(
        int(value.strip()) for value in str(args.horizons).split(",") if value.strip()
    )
    try:
        plan = build_queue(
            data_path=args.data_path,
            output_dir=args.output_dir,
            pretrain_cache_dir=args.pretrain_cache_dir,
            device=args.device,
            backbone=args.backbone,
            hhar_frozen_state=args.hhar_frozen_state,
            flow_profile_json=args.flow_profile_json,
            datasets=datasets,
            scenarios=scenarios,
            conditions=conditions,
            source_seeds=source_seeds,
            horizons=horizons,
        )
        returncode, payload = execute_queue(
            plan,
            output_dir=args.output_dir,
            dry_run=bool(args.dry_run),
            resume=not args.no_resume,
            max_retries=args.max_retries,
            wait_for_hhar_frozen=args.wait_for_hhar_frozen,
            hhar_wait_timeout=args.hhar_wait_timeout,
            hhar_poll_seconds=args.hhar_poll_seconds,
            gpu_lock_path=args.gpu_lock_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.json_out:
        atomic_write_json(payload, args.json_out)
    print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), flush=True)
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLEAN_CONDITION",
    "DATASETS",
    "FORMAL_CONDITIONS",
    "FORMAL_CORRUPTIONS",
    "FORMAL_DATASETS",
    "FORMAL_FLOWS",
    "FORMAL_HORIZONS",
    "FORMAL_SEVERITIES",
    "FORMAL_SOURCE_SEEDS",
    "FORMAL_STREAM_SEED",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_STREAM_CELL_COUNT",
    "GPU_LOCK_PATH",
    "HORIZONS",
    "PROTOCOL_VERSION",
    "SOURCE_SEEDS",
    "STREAM_SEED",
    "atomic_write_json",
    "build_parser",
    "build_queue",
    "execute_queue",
    "expected_cell_count",
    "expected_stream_cell_count",
    "file_sha256",
    "frozen_hhar_provenance",
    "is_oom_text",
    "is_retryable_failure",
    "main",
    "make_cell_key",
    "make_stream_cell_key",
    "run_queue",
    "validate_cells",
]
