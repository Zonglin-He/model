"""Measure current deployment compute overhead under an explicit registry.

This runner is intentionally separate from the historical compute-overhead
scripts.  It measures only methods registered in the current tree, keeps the
source checkpoint fixed across method/profile pairs, reports both the current
dataset default batch and an explicit common-batch profile, and records OOM
fallbacks instead of silently changing the requested protocol.

The stream timer includes host-to-device transfer and CUDA synchronization.
The profiler reports one deployment invocation: the online update (including
all DuSafe inner steps) plus the post-update prediction scored by the paper
evaluator. It excludes data-loader and host-to-device work.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tta_hparams_new import get_hparams_class
from configs.formal_evaluation_protocol import (
    HHAR_CONFIRMATORY,
    HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
    HHAR_REPORTED_FLOWS,
    HHAR_REPORTED_PARTITION,
    evaluation_partition_metadata,
    formal_flow_metadata,
    formal_scenario_pairs,
)
from benchmark_baselines.fisher import ensure_source_fisher
from scripts.supplementary_utils import (
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    move_data_to_device,
)
from scripts.run_full_main_table import (
    GPUExperimentLock,
    wait_for_gpu_experiment_lock,
)
from scripts.paper_flow_profiles import (
    DEFAULT_PAPER_FLOW_PROFILE_JSON,
    load_paper_flow_profiles,
    profile_for_flow,
)
from trainers.tta_abstract_trainer import _predict_after_adaptation


# Formal overhead uses one registered source-training/checkpoint profile per
# flow, shared by every method and DuSafe variant in that flow. HHAR is the
# only dataset for which the data-model registry is intentionally narrowed;
# the five flows are target-selected development/evaluation rows and are
# descriptive, not confirmatory. Keep this mapping immutable in the manifest
# and queue keys.
FORMAL_DATASETS = ("EEG", "HAR", "FD", "HHAR")
FORMAL_SCENARIOS = {
    dataset: tuple(
        f"{source}->{target}" for source, target in formal_scenario_pairs(dataset)
    )
    for dataset in FORMAL_DATASETS
}
FORMAL_HHAR_FLOWS = tuple(str(flow) for flow in HHAR_REPORTED_FLOWS)

# ``SCENARIOS`` used to contain one representative pair per dataset.  It is
# now the formal flow registry.  ``REPRESENTATIVE_SCENARIOS`` is retained for
# callers that only need a cheap smoke cell and is deliberately not used by
# the formal queue/finalizer.
SCENARIOS = FORMAL_SCENARIOS
REPRESENTATIVE_SCENARIOS = {
    dataset: tuple(flow.split("->", 1))
    for dataset, flow in {
        "EEG": "16->1",
        "HAR": "12->16",
        "FD": "2->3",
        "HHAR": "0->6",
    }.items()
}

REQUESTED_METHODS = (
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
BASELINE_METHODS = tuple(method for method in REQUESTED_METHODS if method != "DuSafe")
DUSAFE_METHOD = "DuSafe"
DUSAFE_VARIANTS = ("full", "no_ssaw")
FORMAL_VARIANTS = DUSAFE_VARIANTS
METHOD_DISPLAY_NAMES = {
    "NoAdap": "Source",
    "Tent": "TENT",
    "ACCUPOfficial": "ACCUP",
}
FORMAL_SOURCE_SEEDS = (1,)
FORMAL_STREAM_SEED = 42
DEFAULT_HHAR_TUNER_STATE = (
    ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1" / "state.json"
)
DEFAULT_HHAR_TUNER_MANIFEST = DEFAULT_HHAR_TUNER_STATE.with_name("manifest.json")
DEFAULT_FLOWWISE_SOURCE_PROFILE_JSON = (
    ROOT
    / "results"
    / "optuna"
    / "flowwise_ssaw_deadline_v3"
    / "selected_profiles.json"
)
OVERHEAD_PROTOCOL_VERSION = "compute_overhead_formal_v4"
GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"

# These values make the fairness profile distinct from the current production
# defaults while remaining practical on the 8 GiB reviewer GPU.  They are
# still one fixed value per dataset and are applied identically to every
# method that is present in the production registry.
DEFAULT_COMMON_BATCHES = {"EEG": 96, "HAR": 16, "FD": 128, "HHAR": 48}

# The current unified SSAW implementation has one fixed physical view per
# target batch (plus its inverse when antithetic views are enabled).  It does
# expose ``last_metadata['view_count']`` as a diagnostic, but it does not
# expose a registered candidate grid or a candidate-selection axis.  A
# candidate/view *curve* would therefore be an invented experimental factor;
# finalization records this explicitly instead of fabricating rows.
CANDIDATE_VIEW_CURVE = {
    "status": "not_applicable",
    "candidate_count_available": False,
    "reason": (
        "current unified SSAW exposes a per-batch view_count diagnostic, not a "
        "meaningful registered candidate-count/view-count hyperparameter curve"
    ),
    "diagnostic_field": "ssaw_view_count",
}

PARAMETER_DEFINITION = {
    "legacy_total_parameters": (
        "Historical total_parameters is the deployed tta_model.model "
        "backbone/model count only; it excludes frozen auxiliary modules."
    ),
    "backbone_parameters": "Unique parameters registered by tta_model.model.",
    "frozen_auxiliary_parameters": (
        "Unique non-deployed parameters with requires_grad=False, "
        "including DuSafe's frozen source semantic extractor."
    ),
    "trainable_parameters": (
        "Deployed parameters present in the optimizer; zero for source-only inference."
    ),
    "resident_parameter_count": (
        "Unique parameters registered by the complete tta_model wrapper, "
        "including deployed and auxiliary modules."
    ),
    "optimizer_state_tensor_count_bytes": (
        "Unique tensor leaves in optimizer.state and their element_size-based bytes; "
        "the primary columns are sampled after the complete online stream. Initial "
        "pre-update values are retained in *_initial columns."
    ),
    "resident_buffers_bytes": (
        "Unique registered tta_model buffers, including deployed and auxiliary buffers, "
        "counted as numel * element_size."
    ),
}


def canonical_json(value: Any) -> str:
    """Serialize protocol metadata deterministically for signatures/hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_flowwise_source_profiles(
    path: str | Path,
    datasets: Iterable[str] = FORMAL_DATASETS,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the registered flow-specific source-training/checkpoint contract.

    The Optuna artifact also contains per-flow TTA settings.  Those settings
    are intentionally ignored here: paper TTA profiles have their own
    priority chain, while this loader owns source preparation only.
    """

    profile_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load flow-specific source profiles from {profile_path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ValueError("flow-specific source profile root must be a JSON object")

    selected_datasets = tuple(str(dataset).strip().upper() for dataset in datasets)
    unknown = [dataset for dataset in selected_datasets if dataset not in FORMAL_SCENARIOS]
    if unknown:
        raise ValueError(f"unknown source-profile dataset(s): {unknown}")
    expected = {
        (dataset, scenario)
        for dataset in selected_datasets
        for scenario in FORMAL_SCENARIOS[dataset]
    }
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for key, payload in raw.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"source profile {key!r} must be a JSON object")
        key_text = str(key)
        if ":" not in key_text:
            raise ValueError(f"malformed source profile key {key_text!r}")
        key_dataset, key_scenario = key_text.split(":", 1)
        dataset = str(payload.get("dataset", key_dataset)).strip().upper()
        flow = payload.get("flow")
        if isinstance(flow, (list, tuple)) and len(flow) == 2:
            scenario = f"{flow[0]}->{flow[1]}"
        else:
            scenario = key_scenario
        if dataset != key_dataset.strip().upper() or scenario != key_scenario:
            raise ValueError(
                f"source profile key/payload mismatch: {key_text!r} vs "
                f"{dataset}:{scenario}"
            )
        pair = (dataset, scenario)
        if pair not in expected:
            # Profiles for datasets outside this plan are ignored, but an
            # unregistered flow inside a selected dataset is always an error.
            if dataset in selected_datasets:
                raise ValueError(f"unregistered source profile {dataset}:{scenario}")
            continue
        source_config = payload.get("source_config")
        if not isinstance(source_config, Mapping) or not source_config:
            raise ValueError(f"{dataset}:{scenario} has no source_config")
        source_config = dict(source_config)
        required_source_keys = {
            "pre_learning_rate",
            "num_epochs",
            "batch_size",
            "weight_decay",
        }
        missing = sorted(required_source_keys - set(source_config))
        if missing:
            raise ValueError(
                f"{dataset}:{scenario} source_config lacks {missing}"
            )
        config_sha256 = _sha256_json(source_config)
        registered_config_sha256 = str(payload.get("source_config_sha256", "")).strip()
        if registered_config_sha256 and registered_config_sha256 != config_sha256:
            raise ValueError(
                f"{dataset}:{scenario} source_config_sha256 mismatch"
            )
        checkpoint_sha256 = str(payload.get("source_checkpoint_sha256", "")).strip()
        checkpoint_path_text = str(payload.get("source_checkpoint_path", "")).strip()
        if not checkpoint_sha256 or not checkpoint_path_text:
            raise ValueError(
                f"{dataset}:{scenario} lacks registered source checkpoint identity"
            )
        if pair in parsed:
            raise ValueError(f"duplicate source profile {dataset}:{scenario}")
        parsed[pair] = {
            "dataset": dataset,
            "scenario": scenario,
            "source_config": source_config,
            "source_config_sha256": config_sha256,
            "source_checkpoint_sha256": checkpoint_sha256,
            # Do not use Path.resolve() here.  Besides following links (which
            # is not part of the checkpoint identity), Windows realpath can
            # enter native filesystem code and has proved unstable under the
            # full torch/pytest process.  An absolute, normalized text path is
            # sufficient for the registered cache identity.
            "source_checkpoint_path": os.path.abspath(
                os.path.expanduser(checkpoint_path_text)
            ),
        }
    missing_profiles = sorted(expected - set(parsed))
    if missing_profiles:
        raise ValueError(f"missing flow-specific source profiles: {missing_profiles}")
    return parsed


def validate_registered_source_identity(
    profile: Mapping[str, Any],
    *,
    source_seed: int,
    actual_sha256: str,
    actual_path: str | Path,
) -> None:
    """Fail closed when seed-1 overhead does not reuse the registered source."""

    if int(source_seed) != 1:
        return
    expected_sha256 = str(profile.get("source_checkpoint_sha256", "")).strip()
    expected_path = os.path.abspath(
        os.path.expanduser(str(profile.get("source_checkpoint_path", "")))
    )
    observed_path = os.path.abspath(os.path.expanduser(os.fspath(actual_path)))
    if str(actual_sha256).strip() != expected_sha256:
        raise RuntimeError(
            "registered source checkpoint tensor hash mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    if os.path.normcase(observed_path) != os.path.normcase(expected_path):
        raise RuntimeError(
            "registered source checkpoint path mismatch: "
            f"expected={expected_path}, actual={observed_path}"
        )


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 for a checkpoint/cache file."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_gpu_device(device: str | torch.device) -> bool:
    return str(device).strip().lower().startswith("cuda")


def _json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {target}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON document must be an object: {target}")
    return dict(value)


def _nested_value(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
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


def _declared_hhar_flows(
    state: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    candidates = (
        manifest.get("evaluation_flows"),
        manifest.get("reported_flows"),
        state.get("evaluation_flows"),
        state.get("reported_flows"),
        _nested_value(
            state,
            ("hhar_five_flow_protocol", "evaluation_flows"),
            ("single_flow_protocol", "evaluation_flows"),
            ("signature", "evaluation_flows"),
        ),
    )
    for candidate in candidates:
        if candidate is not None:
            return tuple(str(value) for value in candidate)
    return ()


def _declared_hhar_flag(
    state: Mapping[str, Any], manifest: Mapping[str, Any], key: str
) -> Any:
    for payload in (manifest, state):
        if key in payload:
            return payload[key]
        for container_name in (
            "hhar_five_flow_protocol",
            "single_flow_protocol",
            "signature",
            "validation_gate",
        ):
            container = payload.get(container_name)
            if isinstance(container, Mapping) and key in container:
                return container[key]
    return None


def load_hhar_tuner_state(
    state_path: str | Path = DEFAULT_HHAR_TUNER_STATE,
    manifest_path: str | Path | None = None,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Load the frozen HHAR tuning state and fail closed on stale profiles.

    The overhead runner must never benchmark HHAR with the current source
    defaults after a tuner has started.  Completion is checked in both state
    and manifest, and the state/manifest runtime profiles must be identical.
    ``require_complete=False`` is only for dry-run planning; such a plan is
    marked non-runnable and cannot be executed by :func:`run_overhead_queue`.
    """

    state_file = Path(state_path).expanduser().resolve()
    manifest_file = (
        state_file.with_name("manifest.json")
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    errors: list[str] = []
    try:
        state = _json_object(state_file)
    except ValueError as error:
        if require_complete:
            raise
        state = {}
        errors.append(str(error))
    try:
        manifest = _json_object(manifest_file)
    except ValueError as error:
        if require_complete:
            raise
        manifest = {}
        errors.append(str(error))

    state_config = state.get("tta_config")
    manifest_config = manifest.get("current_tta_config", manifest.get("tta_config"))
    if not isinstance(state_config, Mapping):
        errors.append(f"HHAR tuner state lacks tta_config: {state_file}")
    if not isinstance(manifest_config, Mapping):
        errors.append(f"HHAR tuner manifest lacks current_tta_config: {manifest_file}")
    if isinstance(state_config, Mapping) and isinstance(manifest_config, Mapping):
        if canonical_json(state_config) != canonical_json(manifest_config):
            errors.append("HHAR tuner state and manifest tta_config differ")

    observed_flows = _declared_hhar_flows(state, manifest)
    if observed_flows != FORMAL_HHAR_FLOWS:
        errors.append(
            "HHAR tuner flow protocol mismatch: "
            f"expected={FORMAL_HHAR_FLOWS}, observed={observed_flows}"
        )
    partition = _nested_value(
        manifest,
        ("evaluation_partition",),
        ("validation_gate", "evaluation_partition"),
    )
    if partition is None:
        partition = _nested_value(
            state,
            ("evaluation_partition",),
            ("single_flow_protocol", "evaluation_partition"),
        )
    if str(partition) != HHAR_REPORTED_PARTITION:
        errors.append(
            "HHAR tuner must declare evaluation_partition="
            f"{HHAR_REPORTED_PARTITION}"
        )
    if _declared_hhar_flag(state, manifest, "parameter_selection_data_overlap") is not True:
        errors.append("HHAR tuner must declare parameter-selection overlap")
    if _declared_hhar_flag(state, manifest, "confirmatory") is not False:
        errors.append("HHAR target-selected overhead must be non-confirmatory")
    if _declared_hhar_flag(state, manifest, "target_labels_used_for_selection") is not True:
        errors.append("HHAR tuner must declare target-label selection")

    completion = {
        "manifest.status": manifest.get("status") == "complete",
        "manifest.phase": manifest.get("phase") == "complete",
        "manifest.tuning_complete": manifest.get("tuning_complete") is True,
        "state.phase": state.get("phase") == "complete",
        "state.completed": state.get("completed") is True,
    }
    if require_complete:
        errors.extend(
            f"HHAR tuner is incomplete: {name}"
            for name, passed in completion.items()
            if not passed
        )
    if errors and require_complete:
        raise ValueError("; ".join(errors))

    result = {
        "state_path": str(state_file),
        "manifest_path": str(manifest_file),
        "state_sha256": file_sha256(state_file) if state_file.is_file() else "",
        "manifest_sha256": file_sha256(manifest_file) if manifest_file.is_file() else "",
        "tta_config": dict(state_config) if isinstance(state_config, Mapping) else {},
        "state": state,
        "manifest": manifest,
        "complete": not errors and all(completion.values()),
        "completion": completion,
        "errors": errors,
    }
    return result


def validate_hhar_tuner_complete(
    state_path: str | Path = DEFAULT_HHAR_TUNER_STATE,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Strict convenience wrapper used immediately before GPU execution."""

    return load_hhar_tuner_state(state_path, manifest_path, require_complete=True)


def formal_flow_registry() -> dict[str, tuple[str, ...]]:
    """Return a copy of the formal flow map for plans/manifests/tests."""

    return {dataset: tuple(flows) for dataset, flows in FORMAL_SCENARIOS.items()}


def _canonical_method(method: str) -> str:
    aliases = {
        "source": "NoAdap",
        "noadapt": "NoAdap",
        "tent": "Tent",
        "accup": "ACCUPOfficial",
        "accupofficial": "ACCUPOfficial",
        "cotta": "CoTTA",
        "sotta": "SoTTA",
        "rotta": "RoTTA",
        "dusafe": "DuSafe",
    }
    value = str(method).strip()
    return aliases.get(value.lower(), value)


def _canonical_variant(variant: str, method: str) -> str:
    value = str(variant).strip().lower().replace("-", "_")
    if _canonical_method(method) != DUSAFE_METHOD:
        if value not in {"", "none", "default", "baseline"}:
            raise ValueError(f"baseline method {method} cannot use variant {variant}")
        return "baseline"
    aliases = {"full": "full", "no_ssaw": "no_ssaw", "nossaw": "no_ssaw"}
    if value not in aliases:
        raise ValueError(f"DuSafe variant must be full or no_ssaw, got {variant!r}")
    return aliases[value]


def effective_overhead_registry(method: str, requested_registry: str) -> str:
    """Use production DuSafe while retaining benchmark baseline adapters.

    ``benchmark_baselines.registry`` intentionally exposes a legacy base
    DuSafe alias for provenance tests.  A mixed overhead panel must not time
    that alias as the production sampled-spline method, so DuSafe cells are
    routed to the production registry and all other methods retain the
    requested registry.
    """

    registry = str(requested_registry).strip().lower()
    if registry not in {"production", "benchmark"}:
        raise ValueError(f"unknown overhead registry: {requested_registry!r}")
    return "production" if method == DUSAFE_METHOD else registry


def dusafe_variant_runtime_hparams(variant: str) -> dict[str, Any]:
    """Return explicit production variant switches for one overhead cell."""

    value = _canonical_variant(variant, DUSAFE_METHOD)
    if value == "full":
        return {
            "dusafe_variant": "spline_residual",
            "enable_ssaw": True,
            "enable_source_semantic_router": False,
        }
    return {
        "dusafe_variant": "confidence_raw",
        "enable_ssaw": False,
        "enable_source_semantic_router": False,
    }


def expand_method_variants(
    methods: Iterable[str], variants: Iterable[str] = FORMAL_VARIANTS
) -> tuple[tuple[str, str], ...]:
    """Expand methods to deterministic baseline/Full/no-SSAW cells."""

    selected_methods = tuple(_canonical_method(method) for method in methods)
    if not selected_methods:
        raise ValueError("at least one overhead method is required")
    unknown = [method for method in selected_methods if method not in REQUESTED_METHODS]
    if unknown:
        raise ValueError(f"unsupported overhead methods: {unknown}")
    selected_variants = tuple(str(value) for value in variants)
    result: list[tuple[str, str]] = []
    for method in selected_methods:
        if method == DUSAFE_METHOD:
            for variant in selected_variants:
                result.append((method, _canonical_variant(variant, method)))
        else:
            result.append((method, _canonical_variant("baseline", method)))
    return tuple(result)


def expected_overhead_cell_count(
    datasets: Iterable[str] = FORMAL_DATASETS,
    methods: Iterable[str] = REQUESTED_METHODS,
    variants: Iterable[str] = FORMAL_VARIANTS,
    profiles: Iterable[str] = ("default",),
    source_seeds: Iterable[int] = FORMAL_SOURCE_SEEDS,
) -> int:
    """Return the exact number of independent process-isolated cells."""

    dataset_values = tuple(str(dataset).upper() for dataset in datasets)
    for dataset in dataset_values:
        if dataset not in FORMAL_SCENARIOS:
            raise ValueError(f"unknown formal overhead dataset: {dataset}")
    profile_values = tuple(str(profile) for profile in profiles)
    seed_values = tuple(int(seed) for seed in source_seeds)
    return sum(len(FORMAL_SCENARIOS[dataset]) for dataset in dataset_values) * len(
        expand_method_variants(methods, variants)
    ) * len(profile_values) * len(seed_values)


# Public protocol counts for the default formal queue: four datasets × five
# flows × ten benchmark baselines plus DuSafe Full/no-SSAW × one source seed ×
# one dataset-level/default batch profile.
FORMAL_FLOW_COUNT = sum(len(flows) for flows in FORMAL_SCENARIOS.values())
FORMAL_METHOD_VARIANT_COUNT = len(expand_method_variants(REQUESTED_METHODS, FORMAL_VARIANTS))
EXPECTED_FORMAL_OVERHEAD_CELLS = (
    FORMAL_FLOW_COUNT * FORMAL_METHOD_VARIANT_COUNT * len(FORMAL_SOURCE_SEEDS)
)


@dataclass(frozen=True)
class OverheadCell:
    """One process-isolated overhead measurement cell."""

    name: str
    dataset: str
    scenario: str
    method: str
    variant: str
    profile: str
    source_seed: int
    stream_seed: int
    output_dir: Path
    command: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, str, str, str, int, int]:
        return (
            self.dataset,
            self.scenario,
            self.method,
            self.variant,
            self.profile,
            self.source_seed,
            self.stream_seed,
        )


def _cell_to_dict(cell: OverheadCell) -> dict[str, Any]:
    return {
        "name": cell.name,
        "dataset": cell.dataset,
        "scenario": cell.scenario,
        "method": cell.method,
        "variant": cell.variant,
        "profile": cell.profile,
        "source_seed": cell.source_seed,
        "stream_seed": cell.stream_seed,
        "output_dir": str(cell.output_dir),
        "command": list(cell.command),
        "status": "planned",
        "attempts": 0,
    }


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_override_entries(entries: Iterable[str] | None) -> dict[str, Any]:
    """Parse repeatable trainer runtime overrides."""

    result: dict[str, Any] = {}
    for entry in entries or ():
        if "=" not in str(entry):
            raise ValueError(f"Invalid override '{entry}'; expected key=value")
        key, raw_value = str(entry).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{entry}'; empty key")
        text = raw_value.strip()
        lowered = text.lower()
        if lowered == "none":
            value: Any = None
        elif lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        else:
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                value = text
        result[key] = value
    return result


def parse_batch_map(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            dataset, value = item.split("=", 1)
        elif ":" in item:
            dataset, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid batch map entry '{item}'; use DATASET=INTEGER."
            )
        dataset = dataset.strip()
        if dataset not in SCENARIOS:
            raise ValueError(f"Unknown dataset in batch map: {dataset}")
        batch_size = int(value)
        if batch_size < 1:
            raise ValueError("Batch sizes must be positive.")
        result[dataset] = batch_size
    return result


def synchronize(device: torch.device | str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def is_oom_text(text: str) -> bool:
    """Classify Python/CUDA/native stderr without importing CUDA state."""

    normalized = str(text).lower()
    return any(
        marker in normalized
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "cudnn_status_alloc_failed",
            "cuda_malloc_async",
            "resource exhausted",
        )
    )


def is_native_crash(return_code: int | None, text: str = "") -> bool:
    """Recognize native/allocator child failures for queue status logging."""

    if return_code is None or int(return_code) == 0:
        return False
    if is_oom_text(text):
        return False
    # POSIX signals are negative.  Windows access-violation/stack-overflow
    # codes are returned as unsigned HRESULT-like values by subprocess.
    native_codes = {
        -6,  # SIGABRT
        -11,  # SIGSEGV
        -1073741819,  # 0xC0000005 access violation
        -1073741571,  # 0xC00000FD stack overflow
    }
    return int(return_code) in native_codes or int(return_code) < 0


def clear_cuda_after_oom() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def invoke(model: torch.nn.Module, data: Any, source_only: bool) -> torch.Tensor:
    if source_only:
        with torch.inference_mode():
            return model({"data": data})
    model_inputs = {"data": data}
    model(model_inputs)
    # The paper evaluator scores the model after all requested online update
    # steps. Computational overhead must include that deployment prediction;
    # timing only the adaptation wrapper both undercounts work and reports the
    # wrapper's pre-update logits for DuSafe.
    return _predict_after_adaptation(model, model_inputs)


def primary_batch_size(data: Any) -> int:
    primary = data[0] if isinstance(data, (tuple, list)) else data
    return int(primary.size(0))


def tensor_state_sha256(model: torch.nn.Module) -> str:
    """Hash model tensors in stable name order without serializing metadata."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _unique_tensors(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Return tensors once by object identity, preserving traversal order."""

    unique: list[torch.Tensor] = []
    seen: set[int] = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(tensor)
    return unique


def _iter_nested_tensors(value: Any) -> Iterable[torch.Tensor]:
    """Yield tensor leaves from optimizer state without assuming an optimizer."""

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_nested_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_nested_tensors(child)


def _optimizer_state_metrics(optimizer: Any) -> tuple[int, int]:
    """Count unique optimizer-state tensors and their resident byte size."""

    if optimizer is None:
        return 0, 0
    tensors = _unique_tensors(_iter_nested_tensors(getattr(optimizer, "state", {})))
    return int(len(tensors)), int(
        sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)
    )


def parameter_counts(tta_model: torch.nn.Module, source_only: bool) -> dict[str, int]:
    """Return compatible and explicit resident-parameter accounting.

    ``total_parameters`` is retained as the historical deployed-backbone
    count.  The explicit resident count additionally includes registered
    non-deployed auxiliary modules (the current DuSafe semantic reference
    extractor is frozen) and deduplicates shared parameter objects by id.
    """

    deployed = getattr(tta_model, "model", tta_model)
    deployed_parameters = _unique_tensors(deployed.parameters())
    wrapper_parameters = _unique_tensors(tta_model.parameters())
    deployed_ids = {id(parameter) for parameter in deployed_parameters}
    auxiliary_parameters = [
        parameter for parameter in wrapper_parameters if id(parameter) not in deployed_ids
    ]
    # Current adapters register only frozen non-deployed copies here (for
    # example DuSafe's source semantic extractor, CoTTA's teacher/anchor, and
    # RoTTA's teacher).  Keep a separate diagnostic for an unusual trainable
    # auxiliary module while retaining the requested frozen-auxiliary field.
    frozen_auxiliary_parameters = [
        parameter for parameter in auxiliary_parameters if not parameter.requires_grad
    ]
    trainable_auxiliary_parameters = [
        parameter for parameter in auxiliary_parameters if parameter.requires_grad
    ]
    requires_grad = sum(
        parameter.numel() for parameter in deployed_parameters if parameter.requires_grad
    )
    optimizer = getattr(tta_model, "optimizer", None)
    if optimizer is None or source_only:
        optimizer_ids: set[int] = set()
        optimizer_count = 0
    else:
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        optimizer_count = sum(
            parameter.numel()
            for parameter in deployed_parameters
            if id(parameter) in optimizer_ids
        )
    optimizer_state_tensor_count, optimizer_state_bytes = _optimizer_state_metrics(
        optimizer
    )
    resident_buffers = _unique_tensors(tta_model.buffers())
    resident_buffers_bytes = sum(
        int(buffer.numel()) * int(buffer.element_size()) for buffer in resident_buffers
    )
    backbone_parameters = sum(parameter.numel() for parameter in deployed_parameters)
    frozen_auxiliary_count = sum(
        parameter.numel() for parameter in frozen_auxiliary_parameters
    )
    trainable_auxiliary_count = sum(
        parameter.numel() for parameter in trainable_auxiliary_parameters
    )
    resident_parameter_count = sum(parameter.numel() for parameter in wrapper_parameters)
    # ``trainable_parameters`` means parameters receiving deployment updates;
    # ``requires_grad_parameters`` keeps source-only inference unambiguous.
    return {
        # Legacy fields retained for CSV consumers.  total_parameters is
        # explicitly the deployed backbone/model count, not all resident data.
        "total_parameters": int(backbone_parameters),
        "trainable_parameters": int(0 if source_only else optimizer_count),
        "optimizer_parameters": int(optimizer_count),
        "requires_grad_parameters": int(requires_grad),
        "wrapper_total_parameters": int(resident_parameter_count),
        # Explicit resident accounting.
        "backbone_parameters": int(backbone_parameters),
        "frozen_auxiliary_parameters": int(frozen_auxiliary_count),
        "trainable_auxiliary_parameters": int(trainable_auxiliary_count),
        "resident_parameter_count": int(resident_parameter_count),
        "optimizer_state_tensor_count": int(optimizer_state_tensor_count),
        "optimizer_state_bytes": int(optimizer_state_bytes),
        "resident_buffer_tensor_count": int(len(resident_buffers)),
        "resident_buffers_bytes": int(resident_buffers_bytes),
    }


def with_post_stream_optimizer_state(
    initial_counts: Mapping[str, int], post_stream_counts: Mapping[str, int]
) -> dict[str, int]:
    """Publish deployment optimizer state after adaptation, retaining time zero.

    Adam-style state is lazily allocated on the first committed update.  Counting
    it before the stream makes every adapting method look like it has zero
    optimizer-state memory, even though peak VRAM already includes that state.
    Parameter and buffer counts remain the initial structural inventory; only the
    two optimizer-state fields are replaced by their post-stream measurements.
    """

    merged = dict(initial_counts)
    merged["optimizer_state_tensor_count_initial"] = int(
        initial_counts.get("optimizer_state_tensor_count", 0)
    )
    merged["optimizer_state_bytes_initial"] = int(
        initial_counts.get("optimizer_state_bytes", 0)
    )
    merged["optimizer_state_tensor_count"] = int(
        post_stream_counts.get("optimizer_state_tensor_count", 0)
    )
    merged["optimizer_state_bytes"] = int(
        post_stream_counts.get("optimizer_state_bytes", 0)
    )
    return merged


def profile_flops(
    model: torch.nn.Module,
    data: Any,
    source_only: bool,
    device: torch.device,
) -> tuple[float, str, str, dict[str, dict[str, float]]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    stage_owner = model if hasattr(model, "record_runtime_stage_markers") else None
    graph_owner = (
        model
        if hasattr(model, "candidate_cuda_graph_runtime_enabled")
        else None
    )
    previous_stage_markers = (
        bool(stage_owner.record_runtime_stage_markers)
        if stage_owner is not None
        else None
    )
    if stage_owner is not None:
        stage_owner.record_runtime_stage_markers = True
    previous_graph_runtime = (
        bool(graph_owner.candidate_cuda_graph_runtime_enabled)
        if graph_owner is not None
        else None
    )
    if graph_owner is not None:
        # A replayed graph appears as one opaque profiler event. FLOPs and
        # stage attribution must therefore use the exact eager execution.
        graph_owner.candidate_cuda_graph_runtime_enabled = False
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as profiler:
            invoke(model, data, source_only)
            synchronize(device)
        events = profiler.key_averages()
        flops = sum(float(event.flops or 0.0) for event in events)
        stage_times = {}
        for event in events:
            if not str(event.key).startswith("dusafe."):
                continue
            device_time_us = float(
                getattr(
                    event,
                    "device_time_total",
                    getattr(event, "cuda_time_total", 0.0),
                )
                or 0.0
            )
            stage_times[str(event.key)] = {
                "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
                "device_total_ms": device_time_us / 1000.0,
                "calls": int(event.count),
            }
        if not math.isfinite(flops):
            return (
                float("nan"),
                "non_finite",
                "Profiler returned non-finite FLOPs.",
                stage_times,
            )
        return flops, "ok", "", stage_times
    except Exception as error:  # profiler support varies by torch/operator.
        clear_cuda_after_oom()
        return (
            float("nan"),
            "oom" if is_cuda_oom(error) else "error",
            repr(error),
            {},
        )
    finally:
        if stage_owner is not None:
            stage_owner.record_runtime_stage_markers = previous_stage_markers
        if graph_owner is not None:
            graph_owner.candidate_cuda_graph_runtime_enabled = (
                previous_graph_runtime
            )


def next_loader_batch(loader: Iterable[Any], iterator: Any) -> tuple[Any, Any, Any, Any]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return iterator, batch[0], batch[1], batch[2]


def reset_to_source(
    tta_model: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
) -> None:
    deployed = getattr(tta_model, "model", tta_model)
    deployed.load_state_dict(source_state, strict=True)
    optimizer = getattr(tta_model, "optimizer", None)
    if optimizer is not None:
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
    if hasattr(tta_model, "_last_gate_log"):
        tta_model._last_gate_log = {}
    if hasattr(tta_model, "_last_batch_log"):
        tta_model._last_batch_log = {}


def stream_and_measure(
    *,
    trainer: Any,
    tta_model: torch.nn.Module,
    target_loader: Any,
    source_only: bool,
    warmup_batches: int,
    measure_batches: int,
    device: torch.device,
    profile: str,
    dataset: str,
    method: str,
    scenario: str,
    requested_batch_size: int,
    effective_batch_size: int,
    oom_fallback: bool,
    oom_history: list[str],
    source_checkpoint_sha256: str,
    variant: str = "baseline",
) -> dict[str, Any]:
    configure_graph_workload = getattr(
        tta_model, "configure_candidate_graph_workload", None
    )
    loader_batch_size = getattr(target_loader, "batch_size", None)
    loader_dataset = getattr(target_loader, "dataset", None)
    if (
        callable(configure_graph_workload)
        and loader_batch_size
        and loader_dataset is not None
    ):
        configure_graph_workload(
            expected_full_batch_searches=(
                (len(loader_dataset) // int(loader_batch_size))
                * max(1, int(getattr(tta_model, "steps", 1)))
            )
        )
    source_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in getattr(tta_model, "model", tta_model).state_dict().items()
    }
    source_hash = tensor_state_sha256(getattr(tta_model, "model", tta_model))
    initial_counts = parameter_counts(tta_model, source_only)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    stream_start = time.perf_counter()
    stream_samples = 0
    stream_predictions: list[torch.Tensor] = []
    stream_labels: list[torch.Tensor] = []
    stream_iterator = iter(target_loader)
    for data, labels, _ in stream_iterator:
        stream_samples += primary_batch_size(data)
        data = move_data_to_device(data, device)
        outputs = invoke(tta_model, data, source_only)
        stream_predictions.append(outputs.detach().argmax(dim=1).cpu())
        stream_labels.append(labels.view(-1).long().cpu())
        del outputs, data
    synchronize(device)
    stream_seconds = time.perf_counter() - stream_start
    predictions = torch.cat(stream_predictions) if stream_predictions else torch.empty(0, dtype=torch.long)
    labels = torch.cat(stream_labels) if stream_labels else torch.empty(0, dtype=torch.long)
    stream_accuracy = (
        float((predictions == labels).float().mean().item()) if labels.numel() else float("nan")
    )
    stream_f1 = (
        float(f1_score(labels.numpy(), predictions.numpy(), average="macro", zero_division=0))
        if labels.numel()
        else float("nan")
    )
    # ``stream_seconds`` is the complete online target pass.  For adapting
    # methods this is the wall-clock adaptation cost (transfer + update + CUDA
    # synchronization); Source has no adaptation update and reports zero in
    # the explicit adaptation field while retaining the full inference pass.
    total_adaptation_seconds = 0.0 if source_only else float(stream_seconds)
    ssaw_view_count: int | None = None
    ssaw_candidate_forward_count: int | None = None
    ssaw_candidate_search_execution: str | None = None
    ssaw_candidate_materialization: str | None = None
    candidate_cuda_graph_diagnostics: dict[str, Any] = {}
    stream_candidate_cuda_graph_diagnostics: dict[str, Any] = {}
    ssaw = getattr(tta_model, "ssaw", None)
    metadata = getattr(ssaw, "last_metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("view_count") is not None:
        try:
            ssaw_view_count = int(metadata["view_count"])
        except (TypeError, ValueError):
            ssaw_view_count = None
    if isinstance(metadata, Mapping):
        if metadata.get("candidate_forward_count") is not None:
            try:
                ssaw_candidate_forward_count = int(
                    metadata["candidate_forward_count"]
                )
            except (TypeError, ValueError):
                ssaw_candidate_forward_count = None
        if metadata.get("candidate_search_execution") is not None:
            ssaw_candidate_search_execution = str(
                metadata["candidate_search_execution"]
            )
        if metadata.get("candidate_materialization") is not None:
            ssaw_candidate_materialization = str(
                metadata["candidate_materialization"]
            )
    graph_cache = getattr(tta_model, "_candidate_cuda_graph", None)
    if graph_cache is not None and hasattr(graph_cache, "diagnostics"):
        stream_candidate_cuda_graph_diagnostics = {
            f"stream_{key}": value
            for key, value in graph_cache.diagnostics().items()
        }
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if torch.cuda.is_available() and str(device).startswith("cuda")
        else float("nan")
    )
    counts = with_post_stream_optimizer_state(
        initial_counts,
        parameter_counts(tta_model, source_only),
    )

    # Per-batch latency starts from the untouched source checkpoint.  This
    # avoids measuring a DuSafe model whose weights have already drifted over
    # the complete stream, while preserving the same source state/hash.
    reset_to_source(tta_model, source_state)
    timing_iterator = iter(target_loader)
    timings: list[float] = []
    timed_samples: list[int] = []
    for index in range(warmup_batches + measure_batches):
        timing_iterator, raw_data, _, _ = next_loader_batch(target_loader, timing_iterator)
        batch_size = primary_batch_size(raw_data)
        synchronize(device)
        start = time.perf_counter()
        data = move_data_to_device(raw_data, device)
        outputs = invoke(tta_model, data, source_only)
        synchronize(device)
        elapsed = time.perf_counter() - start
        del outputs, data, raw_data
        if index >= warmup_batches:
            timings.append(elapsed)
            timed_samples.append(batch_size)

    total_timed_seconds = sum(timings)
    timed_sample_count = sum(timed_samples)
    latency_mean = statistics.mean(timings) * 1000.0 if timings else float("nan")
    latency_std = statistics.pstdev(timings) * 1000.0 if len(timings) > 1 else 0.0
    throughput = timed_sample_count / total_timed_seconds if total_timed_seconds > 0 else float("nan")
    # The unprefixed graph diagnostics describe the exact execution state used
    # by the latency loop. Stream-prefixed fields retain the complete online
    # pass state separately; freezing diagnostics before timing can otherwise
    # misreport an eager fallback as a graph timing result (or vice versa).
    if graph_cache is not None and hasattr(graph_cache, "diagnostics"):
        candidate_cuda_graph_diagnostics = dict(graph_cache.diagnostics())

    # Profile a fresh-source batch after timing.  A profiler OOM is recorded
    # separately and does not trigger a second expensive stream retry.
    reset_to_source(tta_model, source_state)
    profile_iterator = iter(target_loader)
    profile_iterator, profile_raw_data, _, _ = next_loader_batch(target_loader, profile_iterator)
    profile_data = move_data_to_device(profile_raw_data, device)
    (
        profiler_flops,
        profiler_status,
        profiler_error,
        profiler_stage_times,
    ) = profile_flops(
        tta_model, profile_data, source_only, device
    )
    del profile_data, profile_raw_data

    row: dict[str, Any] = {
        "status": "ok",
        "dataset": dataset,
        "scenario": scenario,
        "method": method,
        "variant": variant,
        "profile": profile,
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "batch_size": int(round(statistics.mean(timed_samples))) if timed_samples else int(effective_batch_size),
        "oom_fallback": bool(oom_fallback),
        "oom_history": " | ".join(oom_history),
        "warmup_batches": int(warmup_batches),
        "measure_batches": int(measure_batches),
        "stream_batches": int(len(stream_labels)),
        "stream_samples": int(stream_samples),
        "stream_seconds": float(stream_seconds),
        "total_stream_time_seconds": float(stream_seconds),
        "total_adaptation_seconds": float(total_adaptation_seconds),
        "total_adaptation_time_seconds": float(total_adaptation_seconds),
        "stream_samples_per_second": float(stream_samples / stream_seconds) if stream_seconds > 0 else float("nan"),
        "samples_per_second": float(stream_samples / stream_seconds) if stream_seconds > 0 else float("nan"),
        "stream_accuracy": stream_accuracy,
        "stream_macro_f1": stream_f1,
        "prediction_timing_scope": (
            "source_inference"
            if source_only
            else "online_update_plus_post_update_prediction"
        ),
        "latency_mean_ms": float(latency_mean),
        "latency_std_ms": float(latency_std),
        "throughput_samples_per_second": float(throughput),
        "peak_cuda_memory_mb": float(peak_memory),
        "peak_vram_mb": float(peak_memory),
        "peak_vram_bytes": float(peak_memory * (1024**2)) if math.isfinite(peak_memory) else float("nan"),
        "profiler_flops_per_batch": float(profiler_flops),
        "profiler_macs_per_batch_approx": float(profiler_flops / 2.0),
        "flops_per_batch": float(profiler_flops),
        "macs_per_batch": float(profiler_flops / 2.0),
        "profiler_status": profiler_status,
        "profiler_error": profiler_error,
        "profiler_stage_times": json.dumps(
            profiler_stage_times, sort_keys=True
        ),
        # This is the state after the adapter has configured the deployment
        # model (for example, methods may remove BN running-stat buffers).
        # Keep it separate from the canonical pre-adapter checkpoint hash.
        "source_state_sha256": source_hash,
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "ssaw_view_count": ssaw_view_count,
        "ssaw_candidate_forward_count": ssaw_candidate_forward_count,
        "ssaw_candidate_search_execution": ssaw_candidate_search_execution,
        "ssaw_candidate_materialization": ssaw_candidate_materialization,
        **stream_candidate_cuda_graph_diagnostics,
        **candidate_cuda_graph_diagnostics,
        "candidate_view_curve_status": CANDIDATE_VIEW_CURVE["status"],
        **counts,
    }
    return row


def measure_once(
    args: argparse.Namespace,
    *,
    dataset: str,
    method: str,
    profile: str,
    requested_batch_size: int,
    scenario: str | None = None,
    variant: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    dataset = str(dataset).upper()
    method = _canonical_method(method)
    if dataset not in FORMAL_SCENARIOS:
        raise ValueError(f"unknown formal overhead dataset: {dataset}")
    if scenario is None:
        # A direct legacy call still gets a deterministic representative flow;
        # formal queue callers always pass an explicit flow.
        src_id, trg_id = REPRESENTATIVE_SCENARIOS[dataset]
        scenario = f"{src_id}->{trg_id}"
    scenario = str(scenario)
    if scenario not in FORMAL_SCENARIOS[dataset]:
        raise ValueError(
            f"unregistered {dataset} overhead flow {scenario}; "
            f"expected one of {FORMAL_SCENARIOS[dataset]}"
        )
    variant = _canonical_variant(variant or ("baseline" if method != DUSAFE_METHOD else "full"), method)
    source_profiles = getattr(args, "flow_source_profiles", {}) or {}
    source_profile = source_profiles.get((dataset, scenario))
    if not isinstance(source_profile, Mapping):
        raise ValueError(
            f"missing flow-specific source profile for {dataset}:{scenario}"
        )
    source_profile = dict(source_profile)
    method_registry = effective_overhead_registry(method, args.registry)
    src_id, trg_id = scenario.split("->", 1)
    current_batch_size = int(requested_batch_size)
    fallback_history: list[str] = []
    while current_batch_size >= 1:
        trainer = None
        tta_model = None
        pre_trained_model = None
        fisher_info: dict[str, Any] = {}
        source_checkpoint_sha256 = ""
        try:
            trainer = build_trainer(
                data_path=args.data_path,
                device=args.device,
                dataset=dataset,
                da_method=method,
                backbone=args.backbone,
                exp_name=f"compute_overhead_v2_{profile}_{method}",
                seed=args.stream_seed,
                source_seed=args.source_seed,
                pretrain_cache_dir=args.pretrain_cache_dir,
                algorithm_registry=method_registry,
            )
            # Source preparation is a separate, immutable protocol layer.
            # Apply it before scenario data loading/pretraining for every
            # baseline and DuSafe variant. Runtime/TTA overrides below never
            # mutate ``source_hparams``.
            trainer.source_hparams.update(
                dict(source_profile["source_config"])
            )
            if int(args.source_seed) == 1:
                expected_cache_parent = Path(
                    source_profile["source_checkpoint_path"]
                ).expanduser().resolve().parent
                actual_cache_parent = Path(
                    args.pretrain_cache_dir
                ).expanduser().resolve()
                if os.path.normcase(str(actual_cache_parent)) != os.path.normcase(
                    str(expected_cache_parent)
                ):
                    raise RuntimeError(
                        "registered source checkpoint cache directory mismatch: "
                        f"expected={expected_cache_parent}, actual={actual_cache_parent}"
                    )
            # Benchmark baselines always use their frozen registry defaults.
            # Runtime overrides are reserved for DuSafe's frozen dataset-level
            # profile (and the explicit no-SSAW variant); allowing a generic
            # CLI override to reach Tent/EATA/etc. would silently invalidate
            # the comparison.
            runtime_overrides = (
                dict(getattr(args, "overrides", {}) or {})
                if method == DUSAFE_METHOD
                else {}
            )
            flow_profiles = getattr(args, "flow_profile_overrides", {}) or {}
            flow_profile = dict(flow_profiles.get((dataset, scenario), {}))
            if method == DUSAFE_METHOD and flow_profile:
                trainer.set_runtime_hparams(flow_profile)
            if runtime_overrides:
                # Explicit CLI/tuner overrides remain the final caller
                # choice; the paper flow profile fills only missing keys.
                trainer.set_runtime_hparams(runtime_overrides)
            if method == DUSAFE_METHOD:
                trainer.set_runtime_hparams(
                    dusafe_variant_runtime_hparams(variant)
                )
            # The measurement profile owns batch size even when a dataset
            # overlay records a different tuned deployment batch.
            trainer.set_runtime_hparams({"batch_size": int(current_batch_size)})

            def pre_tta_hook(hook_trainer, hook_model):
                nonlocal source_checkpoint_sha256
                source_checkpoint_sha256 = tensor_state_sha256(hook_model)
                source_cache_path = hook_trainer._pretrain_cache_path()
                if not source_cache_path:
                    raise RuntimeError(
                        "source checkpoint path is unavailable after pretraining"
                    )
                validate_registered_source_identity(
                    source_profile,
                    source_seed=args.source_seed,
                    actual_sha256=source_checkpoint_sha256,
                    actual_path=source_cache_path,
                )
                if method != "EATA":
                    return
                fisher_info.update(
                    ensure_source_fisher(
                        model=hook_model,
                        source_loader=hook_trainer.src_train_dl,
                        cache_dir=args.eata_fisher_cache_dir,
                        dataset=dataset,
                        source_seed=args.source_seed,
                        source_checkpoint_sha256=source_checkpoint_sha256,
                        samples=int(
                            hook_trainer.hparams.get(
                                "fisher_samples", args.eata_fisher_samples
                            )
                        ),
                        adapt_keywords=hook_trainer.hparams.get(
                            "adapt_keywords", ("classifier", "adapter")
                        ),
                    )
                )
                hook_trainer.hparams["fisher_enabled"] = True
                hook_trainer.hparams["fisher_path"] = fisher_info[
                    "fisher_cache_path"
                ]

            tta_model, pre_trained_model = create_tta_model(
                trainer,
                src_id,
                trg_id,
                run_seed=args.stream_seed,
                pre_tta_hook=pre_tta_hook,
            )
            row = stream_and_measure(
                trainer=trainer,
                tta_model=tta_model,
                target_loader=trainer.trg_whole_dl,
                source_only=method == "NoAdap",
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
                device=trainer.device,
                profile=profile,
                dataset=dataset,
                method=method,
                scenario=scenario,
                requested_batch_size=requested_batch_size,
                effective_batch_size=current_batch_size,
                oom_fallback=bool(fallback_history),
                oom_history=fallback_history,
                source_checkpoint_sha256=source_checkpoint_sha256,
                variant=variant,
            )
            row["effective_method_registry"] = method_registry
            row["source_cache_path"] = str(trainer._pretrain_cache_path() or "")
            source_cache = Path(row["source_cache_path"]) if row["source_cache_path"] else None
            row["source_checkpoint_path"] = str(source_cache.resolve()) if source_cache else ""
            row["source_checkpoint_file_sha256"] = (
                file_sha256(source_cache) if source_cache is not None and source_cache.is_file() else ""
            )
            row["source_hparams"] = json.dumps(trainer.source_hparams, sort_keys=True, default=str)
            row["runtime_hparams"] = json.dumps(trainer.hparams, sort_keys=True, default=str)
            row.update(
                {
                    "flow_source_profile_applied": True,
                    "flow_source_profile_json": str(
                        Path(args.source_profile_json).expanduser().resolve()
                    ),
                    "registered_source_config_sha256": source_profile[
                        "source_config_sha256"
                    ],
                    "registered_source_checkpoint_sha256": source_profile[
                        "source_checkpoint_sha256"
                    ],
                    "registered_source_checkpoint_path": source_profile[
                        "source_checkpoint_path"
                    ],
                }
            )
            if method == "EATA":
                row.update(fisher_info)
            else:
                row.update(
                    {
                        "fisher_enabled": False,
                        "fisher_cache_path": "",
                        "fisher_cache_hash": "",
                        "fisher_cache_bytes": 0,
                        "fisher_cache_hit": False,
                        "fisher_compute_seconds": 0.0,
                        "fisher_load_seconds": 0.0,
                        "fisher_samples": 0,
                        "fisher_batches": 0,
                        "fisher_source_checkpoint_sha256": "",
                        "fisher_parameter_count": 0,
                    }
                )
            return row, None
        except Exception as error:
            if not is_cuda_oom(error) or current_batch_size <= 1:
                error_text = "".join(traceback.format_exception_only(type(error), error)).strip()
                failure_row = {
                    "status": "failed",
                    "dataset": dataset,
                    "scenario": scenario,
                    "method": method,
                    "variant": variant,
                    "profile": profile,
                    "requested_batch_size": int(requested_batch_size),
                    "effective_batch_size": int(current_batch_size),
                    "oom_fallback": bool(fallback_history),
                    "oom_history": " | ".join(fallback_history),
                    "error": error_text,
                    "flow_source_profile_applied": bool(source_profile),
                    "registered_source_config_sha256": source_profile.get(
                        "source_config_sha256", ""
                    ),
                    "registered_source_checkpoint_sha256": source_profile.get(
                        "source_checkpoint_sha256", ""
                    ),
                    "registered_source_checkpoint_path": source_profile.get(
                        "source_checkpoint_path", ""
                    ),
                }
                if fisher_info:
                    failure_row.update(fisher_info)
                return failure_row, error_text
            fallback_history.append(f"{current_batch_size}->{max(1, current_batch_size // 2)}: {type(error).__name__}")
            clear_cuda_after_oom()
            current_batch_size = max(1, current_batch_size // 2)
        finally:
            if trainer is not None:
                cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)

    raise RuntimeError("unreachable batch fallback state")


def registry_status(methods: list[str], registry: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "NoAdap":
            rows.append({"method": method, "available": True, "reason": "trainer source-only path"})
            continue
        try:
            if registry == "benchmark":
                from benchmark_baselines.registry import get_algorithm_class
            else:
                from algorithms.get_tta_class import get_algorithm_class

            get_algorithm_class(method)
        except Exception as error:
            rows.append({"method": method, "available": False, "reason": str(error)})
        else:
            rows.append({"method": method, "available": True, "reason": "current algorithms registry"})
    return rows


def config_snapshot(datasets: list[str], registry: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in datasets:
        if registry == "benchmark":
            from configs.benchmark_baselines import get_benchmark_hparams_class

            hparams = get_benchmark_hparams_class(dataset)()
        else:
            hparams = get_hparams_class(dataset)()
        result[dataset] = {
            "source_train_params": hparams.source_train_params,
            "target_runtime_params": hparams.train_params,
            "DuSafe": hparams.alg_hparams.get("DuSafe"),
            "algorithm_hparams": {
                method: hparams.alg_hparams.get(method)
                for method in hparams.alg_hparams
                if method != "NoAdap"
            },
        }
    return result


def _atomic_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Publish a small queue/finalizer JSON atomically."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def build_overhead_command(
    cell: OverheadCell,
    *,
    data_path: str | Path,
    device: str,
    backbone: str,
    pretrain_cache_dir: str | Path,
    eata_fisher_cache_dir: str | Path,
    hhar_tuner_state: str | Path,
    hhar_tuner_manifest: str | Path,
    gpu_lock_path: str | Path = GPU_LOCK_PATH,
    registry: str = "benchmark",
    overrides: Mapping[str, Any] | None = None,
    flow_profile_json: str | Path | None = DEFAULT_PAPER_FLOW_PROFILE_JSON,
    source_profile_json: str | Path = DEFAULT_FLOWWISE_SOURCE_PROFILE_JSON,
    warmup_batches: int = 5,
    measure_batches: int = 20,
) -> tuple[str, ...]:
    """Build one child command containing exactly one measurement cell."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell-mode",
        "--data-path",
        str(Path(data_path).expanduser().resolve()),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--registry",
        str(registry),
        "--datasets",
        cell.dataset,
        "--scenarios",
        cell.scenario,
        "--methods",
        cell.method,
        "--variants",
        cell.variant,
        "--profiles",
        cell.profile,
        "--source-seed",
        str(cell.source_seed),
        "--stream-seed",
        str(cell.stream_seed),
        "--pretrain-cache-dir",
        str(Path(pretrain_cache_dir).expanduser().resolve()),
        "--eata-fisher-cache-dir",
        str(Path(eata_fisher_cache_dir).expanduser().resolve()),
        "--hhar-tuner-state",
        str(Path(hhar_tuner_state).expanduser().resolve()),
        "--hhar-tuner-manifest",
        str(Path(hhar_tuner_manifest).expanduser().resolve()),
        "--gpu-lock-path",
        str(Path(gpu_lock_path).expanduser().resolve()),
        "--output-dir",
        str(cell.output_dir.expanduser().resolve()),
        "--warmup-batches",
        str(int(warmup_batches)),
        "--measure-batches",
        str(int(measure_batches)),
        "--source-profile-json",
        str(Path(source_profile_json).expanduser().resolve()),
    ]
    if flow_profile_json is not None:
        command.extend(
            (
                "--flow-profile-json",
                str(Path(flow_profile_json).expanduser().resolve()),
            )
        )
    for key in sorted((overrides or {})):
        value = overrides[key]
        if value is None:
            rendered = "None"
        elif value is True:
            rendered = "True"
        elif value is False:
            rendered = "False"
        elif isinstance(value, str):
            rendered = value
        else:
            rendered = repr(value)
        command.extend(("--override", f"{key}={rendered}"))
    return tuple(command)


def build_overhead_plan(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    datasets: Iterable[str] = FORMAL_DATASETS,
    scenarios: Iterable[str] | None = None,
    methods: Iterable[str] = REQUESTED_METHODS,
    variants: Iterable[str] = FORMAL_VARIANTS,
    profiles: Iterable[str] = ("default",),
    source_seeds: Iterable[int] = FORMAL_SOURCE_SEEDS,
    stream_seed: int = FORMAL_STREAM_SEED,
    device: str = "cpu",
    backbone: str = "CNN",
    registry: str = "benchmark",
    pretrain_cache_dir: str | Path = ROOT / "results" / "pretrain_cache",
    eata_fisher_cache_dir: str | Path = ROOT / "results" / "eata_fisher_cache",
    hhar_tuner_state: str | Path = DEFAULT_HHAR_TUNER_STATE,
    hhar_tuner_manifest: str | Path | None = None,
    gpu_lock_path: str | Path = GPU_LOCK_PATH,
    overrides: Mapping[str, Any] | None = None,
    flow_profile_json: str | Path | None = DEFAULT_PAPER_FLOW_PROFILE_JSON,
    source_profile_json: str | Path = DEFAULT_FLOWWISE_SOURCE_PROFILE_JSON,
    require_hhar_complete: bool | None = None,
    warmup_batches: int = 5,
    measure_batches: int = 20,
) -> dict[str, Any]:
    """Construct a deterministic, CPU-only plan without launching trainers.

    One plan cell is one ``dataset × formal flow × method/variant × profile ×
    source seed``.  The child command is intentionally one-cell-only so a
    Python exception, CUDA OOM, or native process crash cannot contaminate the
    next source checkpoint or adapter.
    """

    selected_datasets = tuple(str(dataset).strip().upper() for dataset in datasets)
    if not selected_datasets:
        raise ValueError("at least one formal overhead dataset is required")
    unknown = [dataset for dataset in selected_datasets if dataset not in FORMAL_SCENARIOS]
    if unknown:
        raise ValueError(f"unknown formal overhead dataset(s): {unknown}")
    flow_profiles = load_paper_flow_profiles(flow_profile_json, selected_datasets)
    flow_source_profiles = load_flowwise_source_profiles(
        source_profile_json, selected_datasets
    )
    requested_scenarios = (
        None
        if scenarios is None
        else tuple(str(value).strip() for value in scenarios)
    )
    if requested_scenarios is not None:
        if len(selected_datasets) != 1:
            raise ValueError(
                "scenario-filtered overhead queues require exactly one dataset"
            )
        if not requested_scenarios:
            raise ValueError("at least one overhead scenario is required")
        dataset = selected_datasets[0]
        invalid_scenarios = [
            value
            for value in requested_scenarios
            if value not in FORMAL_SCENARIOS[dataset]
        ]
        if invalid_scenarios:
            raise ValueError(
                f"unknown {dataset} overhead scenarios: {invalid_scenarios}"
            )
    selected_scenarios = {
        dataset: (
            tuple(FORMAL_SCENARIOS[dataset])
            if requested_scenarios is None
            else requested_scenarios
        )
        for dataset in selected_datasets
    }
    selected_profiles = tuple(str(profile).strip() for profile in profiles)
    if not selected_profiles or any(profile not in {"default", "common"} for profile in selected_profiles):
        raise ValueError(f"overhead profiles must be default/common, got {selected_profiles}")
    selected_seeds = tuple(int(seed) for seed in source_seeds)
    if not selected_seeds:
        raise ValueError("at least one source seed is required")
    selected_methods = tuple(_canonical_method(method) for method in methods)
    registry_name = str(registry).strip().lower()
    if registry_name not in {"production", "benchmark"}:
        raise ValueError(f"unknown overhead registry: {registry!r}")
    method_variants = expand_method_variants(selected_methods, variants)
    tuner_manifest = (
        Path(hhar_tuner_state).expanduser().resolve().with_name("manifest.json")
        if hhar_tuner_manifest is None
        else Path(hhar_tuner_manifest).expanduser().resolve()
    )
    tuner: dict[str, Any] = {
        "complete": True,
        "state_path": str(Path(hhar_tuner_state).expanduser().resolve()),
        "manifest_path": str(tuner_manifest),
        "tta_config": {},
        "errors": [],
    }
    if "HHAR" in selected_datasets:
        try:
            tuner = load_hhar_tuner_state(
                hhar_tuner_state,
                tuner_manifest,
                require_complete=bool(
                    str(device).lower().startswith("cuda")
                    if require_hhar_complete is None
                    else require_hhar_complete
                ),
            )
        except ValueError:
            # A dry-run plan remains inspectable, but it is explicitly
            # non-runnable.  GPU plans always fail closed above.
            if str(device).lower().startswith("cuda") or require_hhar_complete:
                raise
            tuner = load_hhar_tuner_state(
                hhar_tuner_state, tuner_manifest, require_complete=False
            )
    frozen_overrides = dict(tuner.get("tta_config", {})) if tuner.get("complete") else {}
    explicit_overrides = dict(overrides or {})
    output_root = Path(output_dir).expanduser().resolve()
    cells: list[OverheadCell] = []
    ordinal = 0
    for dataset in selected_datasets:
        for scenario in selected_scenarios[dataset]:
            # Validate the formal metadata at plan creation, not after GPU
            # execution.  HHAR in particular cannot silently pick a holdout.
            metadata = evaluation_partition_metadata(dataset, scenario)
            if metadata.get("confirmatory") is not False:
                raise ValueError(f"formal overhead flow is confirmatory: {dataset} {scenario}")
            for source_seed in selected_seeds:
                for method, variant in method_variants:
                    for profile in selected_profiles:
                        ordinal += 1
                        safe = scenario.replace("->", "-to-")
                        name = (
                            f"cell-{ordinal:04d}-{dataset}-{safe}-{method}-"
                            f"{variant}-p{profile}-s{source_seed}"
                        )
                        cell_dir = output_root / "cells" / name
                        cell = OverheadCell(
                            name=name,
                            dataset=dataset,
                            scenario=scenario,
                            method=method,
                            variant=variant,
                            profile=profile,
                            source_seed=source_seed,
                            stream_seed=int(stream_seed),
                            output_dir=cell_dir,
                            command=(),
                        )
                        flow_overrides = (
                            profile_for_flow(flow_profiles, dataset, scenario)
                            if method == DUSAFE_METHOD
                            else {}
                        )
                        cell_overrides = (
                            # The retired dataset-level HHAR tuner is fallback
                            # metadata only. Current paper experiments use one
                            # registered profile per flow, while an explicit
                            # CLI override remains the final caller choice.
                            {
                                **frozen_overrides,
                                **flow_overrides,
                                **explicit_overrides,
                            }
                            if method == DUSAFE_METHOD
                            else {}
                        )
                        command = build_overhead_command(
                            cell,
                            data_path=data_path,
                            device=device,
                            backbone=backbone,
                            pretrain_cache_dir=pretrain_cache_dir,
                            eata_fisher_cache_dir=eata_fisher_cache_dir,
                            hhar_tuner_state=hhar_tuner_state,
                            hhar_tuner_manifest=tuner_manifest,
                            gpu_lock_path=gpu_lock_path,
                            registry=effective_overhead_registry(
                                method, registry_name
                            ),
                            # Baseline children must not inherit DuSafe or
                            # caller overrides; their benchmark registry
                            # defaults are the frozen comparison point.
                            overrides=cell_overrides,
                            flow_profile_json=flow_profile_json,
                            source_profile_json=source_profile_json,
                            warmup_batches=warmup_batches,
                            measure_batches=measure_batches,
                        )
                        cells.append(
                            OverheadCell(
                                **{**cell.__dict__, "command": command}
                            )
                        )
    snapshot = config_snapshot(list(selected_datasets), registry_name)
    production_snapshot = config_snapshot(
        list(selected_datasets), "production"
    )
    return {
        "protocol": OVERHEAD_PROTOCOL_VERSION,
        "protocol_version": OVERHEAD_PROTOCOL_VERSION,
        "data_path": str(Path(data_path).expanduser().resolve()),
        "output_dir": str(output_root),
        "device": str(device),
        "gpu_lock_path": str(Path(gpu_lock_path).expanduser().resolve()),
        "gpu_lock_required": is_gpu_device(device),
        "gpu_lock_acquired": False,
        "backbone": str(backbone),
        "algorithm_registry": registry_name,
        "effective_method_registry": {
            method: effective_overhead_registry(method, registry_name)
            for method in selected_methods
        },
        "paper_flow_profile_json": (
            str(Path(flow_profile_json).expanduser().resolve())
            if flow_profile_json is not None
            else ""
        ),
        "paper_flow_profile_overrides": {
            f"{dataset}:{scenario}": dict(values)
            for (dataset, scenario), values in flow_profiles.items()
        },
        "flow_source_profile_json": str(
            Path(source_profile_json).expanduser().resolve()
        ),
        "flow_source_profiles": {
            f"{dataset}:{scenario}": dict(values)
            for (dataset, scenario), values in flow_source_profiles.items()
        },
        "source_profile_priority": (
            "flow-specific source_config is applied directly to source_hparams "
            "before pretraining for every method; TTA profiles and CLI runtime "
            "overrides cannot mutate it"
        ),
        "datasets": list(selected_datasets),
        "formal_scenarios": {
            dataset: list(selected_scenarios[dataset])
            for dataset in selected_datasets
        },
        "scenario_scope": (
            "all_formal_flows"
            if requested_scenarios is None
            else "registered_representative_subset"
        ),
        "formal_flow_metadata": {
            dataset: dict(formal_flow_metadata(dataset)) for dataset in selected_datasets
        },
        "methods": list(selected_methods),
        "method_display_names": {
            method: METHOD_DISPLAY_NAMES.get(method, method) for method in selected_methods
        },
        "method_variants": [list(item) for item in method_variants],
        "profiles": list(selected_profiles),
        "source_seeds": list(selected_seeds),
        "stream_seed": int(stream_seed),
        "warmup_batches": int(warmup_batches),
        "measure_batches": int(measure_batches),
        "expected_cells": len(cells),
        "expected_cell_count": len(cells),
        "target_selected_descriptive": True,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "selection_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_metrics": True,
        "measurement_purpose": "computational overhead only; no performance inference",
        "evidence_role": "compute_efficiency_full_vs_confidence_only",
        "efficiency_metrics": [
            "latency_mean_ms",
            "throughput_samples_per_second",
            "peak_vram_mb",
        ],
        "peak_vram_policy": (
            "CPU plans intentionally report NaN; valid VRAM comparison requires "
            "the same explicit CUDA device and is not inferred from CPU RSS."
        ),
        "source_seed_role": "fixed source checkpoint identity, not a performance replication",
        "timing_repeats_role": "warmup/measure batches are within-cell timing repeats, not statistical units",
        "process_isolation": "one fresh child process per dataset-flow-method-variant-profile-source cell",
        "same_hardware_required": True,
        "hhar_tuner": tuner,
        "hhar_tuner_state": str(Path(hhar_tuner_state).expanduser().resolve()),
        "hhar_tuner_manifest": str(tuner_manifest),
        "frozen_dataset_config": snapshot,
        "production_dusafe_config": production_snapshot,
        "candidate_view_curve": dict(CANDIDATE_VIEW_CURVE),
        "cells": [_cell_to_dict(cell) for cell in cells],
    }


# Public aliases used by supervisors and CPU tests.
build_plan = build_overhead_plan
build_queue = build_overhead_plan


def _cells_from_plan(plan: Mapping[str, Any]) -> tuple[OverheadCell, ...]:
    cells: list[OverheadCell] = []
    for raw in plan.get("cells", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("malformed overhead cell plan")
        cells.append(
            OverheadCell(
                name=str(raw["name"]),
                dataset=str(raw["dataset"]).upper(),
                scenario=str(raw["scenario"]),
                method=str(raw["method"]),
                variant=str(raw.get("variant", "baseline")),
                profile=str(raw["profile"]),
                source_seed=int(raw["source_seed"]),
                stream_seed=int(raw["stream_seed"]),
                output_dir=Path(str(raw["output_dir"])),
                command=tuple(str(value) for value in raw["command"]),
            )
        )
    return tuple(cells)


def expected_overhead_keys(plan: Mapping[str, Any]) -> frozenset[tuple[str, str, str, str, str, int, int]]:
    return frozenset(cell.key for cell in _cells_from_plan(plan))


def _read_cell_rows(cell: OverheadCell) -> list[dict[str, Any]]:
    path = cell.output_dir / "method_overhead.csv"
    manifest_path = cell.output_dir / "manifest.json"
    if (
        not path.is_file()
        or path.stat().st_size == 0
        or not manifest_path.is_file()
    ):
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != OVERHEAD_PROTOCOL_VERSION:
            return []
        return pd.read_csv(path).to_dict("records")
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        return []


REQUIRED_OVERHEAD_METRICS = (
    "latency_mean_ms",
    "throughput_samples_per_second",
    "total_adaptation_time_seconds",
    "peak_vram_mb",
    "profiler_flops_per_batch",
    "profiler_macs_per_batch_approx",
    "trainable_parameters",
)


def validate_overhead_rows(
    plan: Mapping[str, Any], frame: pd.DataFrame
) -> tuple[bool, list[str]]:
    """Validate exact cells, metric coverage, protocol flags and shared hashes."""

    errors: list[str] = []
    if str(plan.get("protocol", "")) != OVERHEAD_PROTOCOL_VERSION:
        errors.append(
            "overhead plan protocol does not match the current measurement "
            f"contract: {plan.get('protocol')!r} != {OVERHEAD_PROTOCOL_VERSION!r}"
        )
    expected = expected_overhead_keys(plan)
    key_columns = (
        "dataset",
        "scenario",
        "method",
        "variant",
        "profile",
        "source_seed",
        "stream_seed",
    )
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        return False, [f"merged overhead output lacks key columns {missing}"]
    observed: list[tuple[str, str, str, str, str, int, int]] = []
    for row in frame.to_dict("records"):
        try:
            observed.append(
                (
                    str(row["dataset"]).upper(),
                    str(row["scenario"]),
                    str(row["method"]),
                    str(row["variant"]),
                    str(row["profile"]),
                    int(row["source_seed"]),
                    int(row["stream_seed"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid overhead key row: {row}: {error}")
    observed_set = set(observed)
    if len(observed) != len(observed_set):
        errors.append("merged overhead output contains duplicate cell keys")
    if observed_set != set(expected):
        errors.append(
            f"overhead key set mismatch: expected={len(expected)}, observed={len(observed_set)}"
        )
    if len(frame) != len(expected):
        errors.append(f"overhead row count {len(frame)} != expected {len(expected)}")
    for column in REQUIRED_OVERHEAD_METRICS:
        if column not in frame.columns:
            errors.append(f"overhead output lacks required metric {column}")
    if "prediction_timing_scope" not in frame.columns:
        errors.append("overhead output lacks prediction_timing_scope")
    else:
        allowed_scopes = {
            "source_inference",
            "online_update_plus_post_update_prediction",
        }
        observed_scopes = set(frame["prediction_timing_scope"].astype(str))
        if not observed_scopes.issubset(allowed_scopes):
            errors.append(
                "overhead output has invalid prediction_timing_scope values: "
                f"{sorted(observed_scopes - allowed_scopes)}"
            )
        for row in frame.to_dict("records"):
            expected_scope = (
                "source_inference"
                if str(row.get("method")) == "NoAdap"
                else "online_update_plus_post_update_prediction"
            )
            if str(row.get("prediction_timing_scope")) != expected_scope:
                errors.append(
                    f"{row.get('method')} uses the wrong prediction timing scope"
                )
    if "status" in frame.columns and frame["status"].astype(str).ne("ok").any():
        errors.append("overhead output contains failed measurement rows")
    if "confirmatory" in frame.columns and frame["confirmatory"].map(lambda value: str(value).lower() in {"true", "1"}).any():
        errors.append("target-selected overhead rows cannot be confirmatory")
    if "target_selected_descriptive" in frame.columns and frame["target_selected_descriptive"].map(lambda value: str(value).lower() not in {"true", "1"}).any():
        errors.append("overhead rows must declare target_selected_descriptive=true")
    for column in ("warmup_batches", "measure_batches"):
        if column not in frame.columns:
            errors.append(f"overhead output lacks timing contract field {column}")
            continue
        expected_value = int(plan.get(column, -1))
        parsed = pd.to_numeric(frame[column], errors="coerce")
        if parsed.isna().any() or parsed.astype(int).ne(expected_value).any():
            errors.append(
                f"overhead rows do not match plan {column}={expected_value}"
            )
    if "hardware" not in frame.columns:
        errors.append("overhead output lacks hardware identity")
    else:
        hardware_values = {
            str(value).strip()
            for value in frame["hardware"].tolist()
            if value is not None and not (isinstance(value, float) and pd.isna(value))
            and str(value).strip()
        }
        if not hardware_values:
            errors.append("overhead rows have no hardware identity")
        elif len(hardware_values) != 1:
            errors.append(f"overhead rows use multiple hardware identities: {sorted(hardware_values)}")

    # Every method and Full/no-SSAW variant in one flow must share the exact
    # source tensor/file identity.  Profiles are measured separately but still
    # start from the same source checkpoint.
    for column in ("source_checkpoint_sha256", "source_checkpoint_file_sha256"):
        if column not in frame.columns:
            if column == "source_checkpoint_sha256":
                errors.append(f"overhead output lacks {column}")
            continue
        grouped: dict[tuple[str, str, int], set[str]] = {}
        for row in frame.to_dict("records"):
            value = row.get(column, "")
            if value is None or (isinstance(value, float) and pd.isna(value)):
                value = ""
            value = str(value).strip()
            key = (str(row.get("dataset", "")), str(row.get("scenario", "")), int(row.get("source_seed", -1)))
            if not value:
                errors.append(f"{key} has missing {column}")
            grouped.setdefault(key, set()).add(value)
        for key, values in grouped.items():
            if len(values) > 1:
                errors.append(f"{key} maps to multiple {column} values")

    registered_profiles = plan.get("flow_source_profiles", {})
    if not isinstance(registered_profiles, Mapping) or not registered_profiles:
        errors.append("overhead plan lacks flow-specific source profiles")
    else:
        required_identity_columns = (
            "source_checkpoint_path",
            "registered_source_checkpoint_sha256",
            "registered_source_checkpoint_path",
            "registered_source_config_sha256",
            "flow_source_profile_applied",
        )
        for column in required_identity_columns:
            if column not in frame.columns:
                errors.append(f"overhead output lacks {column}")
        if all(column in frame.columns for column in required_identity_columns):
            for row in frame.to_dict("records"):
                dataset = str(row.get("dataset", "")).upper()
                scenario = str(row.get("scenario", ""))
                source_seed = int(row.get("source_seed", -1))
                profile_key = f"{dataset}:{scenario}"
                registered = registered_profiles.get(profile_key)
                if not isinstance(registered, Mapping):
                    errors.append(f"missing registered source profile {profile_key}")
                    continue
                applied = str(row.get("flow_source_profile_applied", "")).lower()
                if applied not in {"true", "1"}:
                    errors.append(f"{profile_key} source profile was not applied")
                for row_column, profile_field in (
                    (
                        "registered_source_checkpoint_sha256",
                        "source_checkpoint_sha256",
                    ),
                    (
                        "registered_source_config_sha256",
                        "source_config_sha256",
                    ),
                ):
                    if str(row.get(row_column, "")).strip() != str(
                        registered.get(profile_field, "")
                    ).strip():
                        errors.append(
                            f"{profile_key} {row_column} does not match plan"
                        )
                expected_path = Path(
                    str(registered.get("source_checkpoint_path", ""))
                ).expanduser().resolve()
                path_columns = ["registered_source_checkpoint_path"]
                if source_seed == 1:
                    path_columns.append("source_checkpoint_path")
                for row_column in path_columns:
                    actual_path = Path(
                        str(row.get(row_column, ""))
                    ).expanduser().resolve()
                    if os.path.normcase(str(actual_path)) != os.path.normcase(
                        str(expected_path)
                    ):
                        errors.append(
                            f"{profile_key} {row_column} does not match plan"
                        )
                if source_seed == 1 and str(
                    row.get("source_checkpoint_sha256", "")
                ).strip() != str(
                    registered.get("source_checkpoint_sha256", "")
                ).strip():
                    errors.append(
                        f"{profile_key} seed-1 source checkpoint hash does not "
                        "match registered profile"
                    )
    return not errors, errors


def finalize_overhead_queue(
    plan: Mapping[str, Any],
    *,
    allow_failures: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Merge child rows only after exact-key/hash/metric validation."""

    output_root = Path(str(plan["output_dir"])).expanduser().resolve()
    cells = _cells_from_plan(plan)
    rows: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    for cell in cells:
        cell_rows = _read_cell_rows(cell)
        if len(cell_rows) != 1:
            missing_cells.append(cell.name)
            continue
        row = dict(cell_rows[0])
        row.setdefault("variant", cell.variant)
        row.setdefault("profile", cell.profile)
        row.setdefault("source_seed", cell.source_seed)
        row.setdefault("stream_seed", cell.stream_seed)
        row.setdefault("dataset", cell.dataset)
        row.setdefault("scenario", cell.scenario)
        row.setdefault("method", cell.method)
        rows.append(row)
    frame = pd.DataFrame(rows)
    valid, errors = validate_overhead_rows(plan, frame)
    if missing_cells:
        errors = [f"missing/empty cell outputs: {missing_cells}", *errors]
    if errors and not allow_failures:
        status = {
            "status": "blocked",
            "protocol": plan.get("protocol", OVERHEAD_PROTOCOL_VERSION),
            "expected_cells": len(cells),
            "observed_rows": len(frame),
            "missing_cells": missing_cells,
            "errors": errors,
            "candidate_view_curve": dict(CANDIDATE_VIEW_CURVE),
        }
        _atomic_json(status, output_root / "finalization.json")
        return 2, status
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, output_root / "method_overhead.csv")
    status = {
        "status": "complete" if valid and not errors else "complete_with_failures",
        "protocol": plan.get("protocol", OVERHEAD_PROTOCOL_VERSION),
        "expected_cells": len(cells),
        "observed_rows": len(frame),
        "missing_cells": missing_cells,
        "errors": errors,
        "candidate_view_curve": dict(CANDIDATE_VIEW_CURVE),
    }
    _atomic_json(status, output_root / "finalization.json")
    return (0 if valid and not errors else 2), status


def _run_overhead_children(
    plan: Mapping[str, Any],
    cells: tuple[OverheadCell, ...],
    output_root: Path,
    *,
    max_attempts: int,
    timeout_seconds: float | None,
    gpu_lock_path: str | Path | None = None,
    gpu_lock_required: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Run children with one shared-lock acquisition per measurement cell."""

    status_rows: list[dict[str, Any]] = []
    for cell in cells:
        cell.output_dir.mkdir(parents=True, exist_ok=True)
        cell_status = {
            "cell": cell.name,
            "dataset": cell.dataset,
            "scenario": cell.scenario,
            "method": cell.method,
            "variant": cell.variant,
            "profile": cell.profile,
            "source_seed": cell.source_seed,
            "stream_seed": cell.stream_seed,
            "status": "planned",
            "attempts": 0,
            "return_code": None,
            "error": "",
        }
        for attempt in range(max(1, int(max_attempts))):
            cell_status["attempts"] = attempt + 1
            try:
                lock_context = (
                    wait_for_gpu_experiment_lock(gpu_lock_path)
                    if gpu_lock_required and gpu_lock_path is not None
                    else nullcontext()
                )
                with lock_context:
                    completed = subprocess.run(
                        list(cell.command),
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                (cell.output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
                (cell.output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
                cell_status["return_code"] = int(completed.returncode)
                combined = stdout + "\n" + stderr
                if completed.returncode == 0 and _read_cell_rows(cell):
                    cell_status["status"] = "ok"
                    break
                if is_oom_text(combined):
                    cell_status["status"] = "oom"
                    cell_status["error"] = "child reported OOM"
                elif is_native_crash(completed.returncode, combined):
                    cell_status["status"] = "native_crash"
                    cell_status["error"] = "child exited with a native crash code"
                else:
                    cell_status["status"] = "failed"
                    cell_status["error"] = combined[-4000:]
            except subprocess.TimeoutExpired as error:
                cell_status["status"] = "timeout"
                cell_status["error"] = str(error)
            except OSError as error:
                cell_status["status"] = "launcher_error"
                cell_status["error"] = repr(error)
            if cell_status["status"] == "ok":
                break
        status_rows.append(cell_status)
        atomic_write_csv(pd.DataFrame(status_rows), output_root / "cell_status.csv")
    failed = [row for row in status_rows if row["status"] != "ok"]
    code, final_status = finalize_overhead_queue(plan, allow_failures=bool(failed))
    final_status["cell_failures"] = failed
    return (2 if failed else code), final_status


def run_overhead_queue(
    plan: Mapping[str, Any],
    *,
    dry_run: bool = True,
    max_attempts: int = 1,
    timeout_seconds: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute cells serially in fresh processes and persist crash/OOM status."""

    output_root = Path(str(plan["output_dir"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cells = _cells_from_plan(plan)
    if int(plan.get("expected_cells", len(cells))) != len(cells):
        raise ValueError("overhead plan expected_cells does not match cell list")
    gate = plan.get("hhar_tuner", {})
    if not dry_run and "HHAR" in plan.get("datasets", ()) and not gate.get("complete", False):
        status = {
            "status": "blocked_waiting_for_hhar_tuner",
            "expected_cells": len(cells),
            "errors": gate.get("errors", ["HHAR tuner is not complete"]),
        }
        _atomic_json(status, output_root / "queue_status.json")
        return 2, status
    if dry_run:
        status = {
            "status": "dry_run",
            "protocol": plan.get("protocol", OVERHEAD_PROTOCOL_VERSION),
            "expected_cells": len(cells),
            "process_isolation": plan.get("process_isolation"),
            "candidate_view_curve": dict(CANDIDATE_VIEW_CURVE),
        }
        _atomic_json({**dict(plan), "queue_status": status}, output_root / "manifest.json")
        _atomic_json(status, output_root / "queue_status.json")
        return 0, status

    lock_path = str(plan.get("gpu_lock_path", GPU_LOCK_PATH))
    lock_required = is_gpu_device(plan.get("device", "cpu"))
    # Acquire and release around each isolated measurement cell.  Lock
    # contention is handled by the waiting wrapper before the attempt starts;
    # it is never recorded as an OOM/worker failure or charged as a retry.
    runtime_plan = dict(plan)
    runtime_plan["gpu_lock_path"] = lock_path
    runtime_plan["gpu_lock_required"] = lock_required
    runtime_plan["gpu_lock_scope"] = "per_measurement_cell" if lock_required else "not_required"
    runtime_plan["gpu_lock_busy_consumes_attempt"] = False
    runtime_plan["gpu_lock_acquired"] = bool(lock_required)
    code, final_status = _run_overhead_children(
        runtime_plan,
        cells,
        output_root,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
        gpu_lock_path=lock_path,
        gpu_lock_required=lock_required,
    )
    final_status["gpu_lock_path"] = lock_path
    final_status["gpu_lock_required"] = lock_required
    final_status["gpu_lock_scope"] = runtime_plan["gpu_lock_scope"]
    final_status["gpu_lock_busy_consumes_attempt"] = False
    final_status["gpu_lock_acquired"] = bool(lock_required)
    _atomic_json(final_status, output_root / "queue_status.json")
    _atomic_json({**dict(runtime_plan), "queue_status": final_status}, output_root / "manifest.json")
    return code, final_status


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", "--data_path", dest="data_path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--registry",
        choices=("production", "benchmark"),
        default="benchmark",
        help="Explicit algorithm registry; benchmark is isolated from production DuSafe.",
    )
    parser.add_argument("--methods", default=",".join(REQUESTED_METHODS))
    parser.add_argument("--datasets", default=",".join(FORMAL_DATASETS))
    parser.add_argument("--scenarios", default=None, help="Optional comma-separated one-cell flow(s).")
    parser.add_argument("--variants", default=",".join(FORMAL_VARIANTS))
    parser.add_argument("--profiles", default="default")
    parser.add_argument(
        "--common-batch-sizes", default="EEG=96,HAR=16,FD=128,HHAR=48"
    )
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Runtime key=value override applied to every requested method.",
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument("--pretrain-cache-dir", default=str(ROOT / "results" / "pretrain_cache"))
    parser.add_argument(
        "--eata-fisher-cache-dir",
        default=str(ROOT / "results" / "eata_fisher_cache"),
        help="Cache directory for source-only diagonal Fisher tensors.",
    )
    parser.add_argument(
        "--eata-fisher-samples",
        type=int,
        default=2000,
        help="Maximum source-train samples for the official-style EATA Fisher pass.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "compute_overhead_current_v2"))
    parser.add_argument("--hhar-tuner-state", default=str(DEFAULT_HHAR_TUNER_STATE))
    parser.add_argument("--hhar-tuner-manifest", default=None)
    parser.add_argument(
        "--flow-profile-json",
        default=str(DEFAULT_PAPER_FLOW_PROFILE_JSON),
        help=(
            "Per-flow TTA override JSON; defaults to "
            "configs/paper_flow_profiles_v1.json."
        ),
    )
    parser.add_argument(
        "--source-profile-json",
        default=str(DEFAULT_FLOWWISE_SOURCE_PROFILE_JSON),
        help=(
            "Flow-specific source_config and registered seed-1 checkpoint "
            "identity from the completed flowwise tuner. This source-only "
            "contract is applied to every baseline and DuSafe."
        ),
    )
    parser.add_argument("--gpu-lock-path", default=str(GPU_LOCK_PATH))
    parser.add_argument("--cell-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Build and execute one fresh child process per formal cell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the isolated queue without launching children.",
    )
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args()
    args.overrides = parse_override_entries(args.override)
    if args.warmup_batches < 0 or args.measure_batches < 1:
        raise ValueError("warmup-batches must be >= 0 and measure-batches must be >= 1")
    if args.eata_fisher_samples < 1:
        raise ValueError("eata-fisher-samples must be positive")

    datasets = [dataset.upper() for dataset in parse_list(args.datasets)]
    methods = [_canonical_method(method) for method in parse_list(args.methods)]
    variants = parse_list(args.variants)
    profiles = parse_list(args.profiles)
    scenarios = parse_list(args.scenarios) if args.scenarios else None
    args.flow_profile_overrides = load_paper_flow_profiles(
        args.flow_profile_json, datasets
    )
    args.flow_source_profiles = load_flowwise_source_profiles(
        args.source_profile_json, datasets
    )
    unknown_datasets = [dataset for dataset in datasets if dataset not in FORMAL_SCENARIOS]
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s): {unknown_datasets}")
    unknown_profiles = [profile for profile in profiles if profile not in {"default", "common"}]
    if unknown_profiles:
        raise ValueError(f"Unknown profile(s): {unknown_profiles}")
    common_batches = parse_batch_map(args.common_batch_sizes)
    method_variants = expand_method_variants(methods, variants)

    if args.queue or args.dry_run or (is_gpu_device(args.device) and not args.cell_mode):
        plan = build_overhead_plan(
            data_path=args.data_path,
            output_dir=args.output_dir,
            datasets=datasets,
            scenarios=scenarios,
            methods=methods,
            variants=variants,
            profiles=profiles,
            source_seeds=(args.source_seed,),
            stream_seed=args.stream_seed,
            device=args.device,
            backbone=args.backbone,
            registry=args.registry,
            pretrain_cache_dir=args.pretrain_cache_dir,
            eata_fisher_cache_dir=args.eata_fisher_cache_dir,
            hhar_tuner_state=args.hhar_tuner_state,
            hhar_tuner_manifest=args.hhar_tuner_manifest,
            gpu_lock_path=args.gpu_lock_path,
            overrides=args.overrides,
            flow_profile_json=args.flow_profile_json,
            source_profile_json=args.source_profile_json,
            require_hhar_complete=(str(args.device).lower().startswith("cuda")),
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
        )
        code, status = run_overhead_queue(
            plan,
            dry_run=bool(args.dry_run),
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(status, indent=2, default=str), flush=True)
        if code:
            raise SystemExit(code)
        return

    if "HHAR" in datasets:
        # Direct execution is still fail-closed.  The queue path is preferred
        # for native-crash containment, but a legacy direct invocation must not
        # benchmark HHAR against an unfinished/stale tuner profile.
        tuner = validate_hhar_tuner_complete(
            args.hhar_tuner_state, args.hhar_tuner_manifest
        )
        args.overrides = {**dict(tuner["tta_config"]), **args.overrides}
    config_cache = config_snapshot(datasets, args.registry)
    output_dir = ensure_dir(args.output_dir)
    method_path = output_dir / "method_overhead.csv"
    status_path = output_dir / "method_status.csv"
    source_path = output_dir / "source_checkpoints.csv"
    rows = pd.read_csv(method_path).to_dict("records") if method_path.exists() else []
    statuses = registry_status(methods, args.registry)
    atomic_write_csv(pd.DataFrame(statuses), status_path)
    available = [row["method"] for row in statuses if row["available"]]
    completed = {
        (
            str(row.get("dataset")),
            str(row.get("scenario")),
            str(row.get("method")),
            str(row.get("variant", "baseline")),
            str(row.get("profile")),
        )
        for row in rows
        if str(row.get("status")) == "ok"
    }
    source_rows: list[dict[str, Any]] = []

    print(f"Current registry methods: {available}; skipped: {[row['method'] for row in statuses if not row['available']]}", flush=True)
    for dataset in datasets:
        flow_values = tuple(scenarios) if scenarios is not None else FORMAL_SCENARIOS[dataset]
        invalid_flows = [flow for flow in flow_values if flow not in FORMAL_SCENARIOS[dataset]]
        if invalid_flows:
            raise ValueError(
                f"Unknown {dataset} formal flow(s): {invalid_flows}; "
                f"expected {FORMAL_SCENARIOS[dataset]}"
            )
        for scenario in flow_values:
            for method in available:
                variants_for_method = (
                    [variant for candidate_method, variant in method_variants if candidate_method == method]
                    if method == DUSAFE_METHOD
                    else ["baseline"]
                )
                for variant in variants_for_method:
                    for profile in profiles:
                        key = (dataset, scenario, method, variant, profile)
                        if key in completed:
                            print(
                                f"[skip] {dataset} {scenario} {method} {variant} {profile} already recorded",
                                flush=True,
                            )
                            continue
                        if profile == "default":
                            requested_batch_size = int(
                                (
                                    args.overrides.get("batch_size")
                                    if method == DUSAFE_METHOD and "batch_size" in args.overrides
                                    else None
                                )
                                or config_cache[dataset]["target_runtime_params"]["batch_size"]
                            )
                        else:
                            if dataset not in common_batches:
                                raise ValueError(f"No common batch size supplied for {dataset}")
                            requested_batch_size = int(common_batches[dataset])
                        print(
                            f"[Compute v2] {dataset} {scenario} {method} {variant} {profile} "
                            f"requested_batch={requested_batch_size}",
                            flush=True,
                        )
                        row, error = measure_once(
                            args,
                            dataset=dataset,
                            scenario=scenario,
                            method=method,
                            variant=variant,
                            profile=profile,
                            requested_batch_size=requested_batch_size,
                        )
                        row["target_selected_descriptive"] = True
                        row["evaluation_partition"] = HHAR_REPORTED_PARTITION
                        row["parameter_selection_data_overlap"] = True
                        row["selection_overlap"] = True
                        row["confirmatory"] = False
                        row["hardware"] = (
                            torch.cuda.get_device_name(0)
                            if torch.cuda.is_available()
                            else "CPU"
                        )
                        rows.append(row)
                        if row.get("source_state_sha256"):
                            source_rows.append(
                                {
                                    "dataset": dataset,
                                    "scenario": row.get("scenario"),
                                    "method": method,
                                    "variant": variant,
                                    "profile": profile,
                                    "source_seed": args.source_seed,
                                    "source_cache_path": row.get("source_cache_path", ""),
                                    "source_state_sha256": row.get(
                                        "source_state_sha256", ""
                                    ),
                                    "source_checkpoint_sha256": row.get(
                                        "source_checkpoint_sha256", ""
                                    ),
                                    "source_checkpoint_file_sha256": row.get(
                                        "source_checkpoint_file_sha256", ""
                                    ),
                                }
                            )
                        atomic_write_csv(pd.DataFrame(rows), method_path)
                        if error:
                            print(
                                f"[failed] {dataset} {scenario} {method} {variant} {profile}: {error}",
                                flush=True,
                            )

    fisher_columns = [
        "dataset",
        "method",
        "profile",
        "fisher_enabled",
        "fisher_cache_path",
        "fisher_cache_hash",
        "fisher_cache_bytes",
        "fisher_cache_hit",
        "fisher_compute_seconds",
        "fisher_load_seconds",
        "fisher_samples",
        "fisher_batches",
        "fisher_source_checkpoint_sha256",
        "fisher_parameter_count",
    ]
    fisher_rows = []
    fisher_keys = set()
    for row in rows:
        if row.get("method") != "EATA" or row.get("status") != "ok":
            continue
        key = (
            row.get("dataset"),
            row.get("fisher_cache_hash"),
            row.get("fisher_cache_path"),
        )
        if key in fisher_keys:
            continue
        fisher_keys.add(key)
        fisher_rows.append({column: row.get(column, "") for column in fisher_columns})

    manifest = {
        "measurement_version": OVERHEAD_PROTOCOL_VERSION,
        "protocol": OVERHEAD_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(args.device),
        "gpu_lock_path": str(Path(args.gpu_lock_path).expanduser().resolve()),
        "gpu_lock_required": is_gpu_device(args.device),
        "gpu_lock_acquired": False,
        "data_path": str(Path(args.data_path).resolve()),
        "source_seed": int(args.source_seed),
        "stream_seed": int(args.stream_seed),
        "datasets": datasets,
        "scenarios": {
            dataset: list(FORMAL_SCENARIOS[dataset]) for dataset in datasets
        },
        "methods_requested": methods,
        "method_display_names": {
            method: METHOD_DISPLAY_NAMES.get(method, method) for method in methods
        },
        "method_variants": [list(item) for item in method_variants],
        "algorithm_registry": args.registry,
        "methods_available": available,
        "methods_skipped": [row for row in statuses if not row["available"]],
        "effective_method_registry": {
            method: effective_overhead_registry(method, args.registry)
            for method in methods
        },
        "method_provenance": (
            __import__("configs.benchmark_baselines", fromlist=["PROVENANCE"]).PROVENANCE
            if args.registry == "benchmark"
            else {}
        ),
        "profiles": profiles,
        "common_batch_sizes": common_batches,
        "warmup_batches": int(args.warmup_batches),
        "measure_batches": int(args.measure_batches),
        "runtime_overrides": dict(args.overrides),
        "flow_source_profile_json": str(
            Path(args.source_profile_json).expanduser().resolve()
        ),
        "flow_source_profiles": {
            f"{dataset}:{scenario}": dict(values)
            for (dataset, scenario), values in args.flow_source_profiles.items()
        },
        "source_profile_priority": (
            "flow-specific source_config is fixed before pretraining for all "
            "methods and is not affected by TTA runtime overrides"
        ),
        "target_selected_descriptive": True,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": True,
        "selection_overlap": True,
        "confirmatory": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_metrics": True,
        "measurement_purpose": "computational overhead only; no performance inference",
        "source_seed_role": "fixed source checkpoint identity, not a performance replication",
        "timing_repeats_role": "warmup/measure batches are within-cell timing repeats, not statistical units",
        "process_isolation": (
            "direct legacy mode; use --queue for one fresh child per "
            "dataset-flow-method-variant-profile-source cell"
        ),
        "same_hardware_required": True,
        "stream_timer": (
            "host-to-device transfer + online update + the post-update "
            "prediction scored by the paper evaluator + CUDA synchronization; "
            "starts from source checkpoint"
        ),
        "batch_timer": (
            "host-to-device transfer + online update + post-update prediction "
            "+ CUDA synchronization; starts from source checkpoint after reset"
        ),
        "profiler": "torch.profiler with_flops=True; MACs are FLOPs/2 approximation and exclude transfer",
        "peak_memory": "CUDA max_memory_allocated reset immediately before stream; profiler allocation excluded",
        "parameter_definition": PARAMETER_DEFINITION,
        "candidate_view_curve": dict(CANDIDATE_VIEW_CURVE),
        "hhar_tuner_state": str(Path(args.hhar_tuner_state).expanduser().resolve()),
        "hhar_tuner_manifest": str(
            Path(args.hhar_tuner_manifest).expanduser().resolve()
            if args.hhar_tuner_manifest
            else Path(args.hhar_tuner_state).expanduser().resolve().with_name("manifest.json")
        ),
        "eata_fisher_cache_dir": str(Path(args.eata_fisher_cache_dir).resolve()),
        "eata_fisher_requested_samples": int(args.eata_fisher_samples),
        "eata_fisher_calibration": fisher_rows,
        "eata_fisher_cost_boundary": (
            "source Fisher wall time/cache bytes are reported in "
            "eata_fisher_calibration.csv and fisher_* columns; they are "
            "excluded from online stream_seconds and batch latency"
        ),
        "config_snapshot": config_cache,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if source_rows:
        existing_source = pd.read_csv(source_path).to_dict("records") if source_path.exists() else []
        atomic_write_csv(pd.DataFrame(existing_source + source_rows), source_path)
    atomic_write_csv(
        pd.DataFrame(fisher_rows, columns=fisher_columns),
        output_dir / "eata_fisher_calibration.csv",
    )
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
