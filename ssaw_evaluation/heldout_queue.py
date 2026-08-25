"""Formal Full/no-SSAW held-out evidence queue.

This module is the data-generation and queue layer for the offline mechanism
panel.  It is intentionally separate from the online algorithm and from the
historical corruption benchmark.  A worker receives one registered transfer
flow, source seed, test seed, and variant; it extracts clean and held-out
logits/features from the existing trainer target stream, saves true labels in
the cell artifact for offline F1/source-label-accuracy only, and records a source-checkpoint
hash.  The parent process pairs Full and no_ssaw cells before aggregation.

The queue is safe to import and exercise on CPU.  The actual trainer imports
are lazy and only occur inside ``run_worker_cell``; no GPU is initialized by
plan construction, dry-run, metric aggregation, or synthetic tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from .heldout_mechanism import (
    COMMON_PREDICTIVE_METRICS,
    HeldOutCase,
    compute_mechanism_metrics,
    heldout_direction_diagnostics,
    summarize_heldout_direction_diagnostics,
    validate_case,
)
from scripts.run_full_main_table import (
    GPUExperimentLock,
    wait_for_gpu_experiment_lock,
)


QUEUE_PROTOCOL_VERSION = "ssaw_full_no_ssaw_heldout_queue_v3_spline_direction_bank"
DATASETS = ("EEG", "HAR", "FD", "HHAR")
SOURCE_SEEDS = (1, 2, 3)
VARIANTS = ("Full", "no_ssaw")
DEFAULT_TEST_SEED = 42
DEFAULT_TRAINING_VIEW_SEED = 1729

# The checked-in physical protocol still contains SO(3)/sensor nuisance
# operators.  Those are useful for a separate plausibility audit, but they
# are not the mechanism family used by production DuSafe anymore.  Keep the
# old names below for HeldOutCase/backward-compatible physical artifacts and
# publish the production mechanism names explicitly in every new queue
# manifest/row.
CURRENT_SSAW_DIRECTION_FAMILY = "spline_residual_sobol_direction"
HELDOUT_SSAW_DIRECTION_FAMILY = "unseen_spline_residual_sobol_direction"
HELDOUT_SSAW_DIRECTION_SIGNS = (1.0, -1.0)
HELDOUT_SSAW_DIRECTION_RADII = (1.0, 0.5, 0.25)
HELDOUT_SSAW_DIRECTION_COUNT = 4
HELDOUT_SSAW_CANDIDATE_COUNT = (
    HELDOUT_SSAW_DIRECTION_COUNT
    * len(HELDOUT_SSAW_DIRECTION_SIGNS)
    * len(HELDOUT_SSAW_DIRECTION_RADII)
)

# One deterministic operator per dataset is selected from the pre-registered
# held-out families.  The names are cross-checked against the protected
# protocol at runtime; this module never changes that protocol.
OPERATOR_FAMILIES = {
    "EEG": "smooth_channel_gain_drift",
    "HAR": "smooth_so3_orientation_trajectory",
    "FD": "smooth_sensor_response_drift",
    "HHAR": "smooth_so3_orientation_trajectory",
}
TRAINING_VIEW_FAMILIES = {
    "EEG": "window_constant_channel_gain",
    "HAR": "window_constant_bounded_so3",
    "FD": "window_constant_sensor_gain",
    "HHAR": "window_constant_bounded_so3",
}
TRAINING_VIEW_PROVENANCE_KEYS = (
    "enable_ssaw",
    "ssaw_sobol_seed",
    "ssaw_temporal_mode",
    "ssaw_sigma",
    "ssaw_control_points",
    "ssaw_strength",
    "ssaw_antithetic",
    "ssaw_antithetic_pairs",
    # Current production DuSafe uses sampled spline directions rather than
    # the archived window-constant SSAW controls above.  Keep the legacy
    # fields for old manifests, but include every spline setting that can
    # change the held-out direction family in new provenance hashes.
    "dusafe_variant",
    "enable_source_semantic_router",
    "spline_control_points",
    "spline_num_directions",
    "spline_log_strength",
    "spline_radius_levels",
)
REQUIRED_PHYSICAL_METADATA = {
    "EEG": ("sampling_rate_hz", "sampling_rate_provenance"),
    "HAR": ("sampling_rate_hz", "sampling_rate_provenance"),
    # FD rate/order references are not silently invented.  When absent, the
    # mechanism layer emits normalized sample-axis metrics and nulls raw/order
    # quantities in paired aggregation.
    "FD": (),
    # HHAR rows are not resampled to a common clock; sample-axis invariants
    # remain valid without copying HAR's 50 Hz.
    "HHAR": (),
}
LABEL_LEAKAGE_FLAGS = {
    "label_leakage_online_updates": False,
    "target_labels_used_online": False,
    "target_labels_used_for_parameter_selection": True,
    "target_labels_used_for_tuning": True,
    "target_selected_datasets": ["EEG", "HAR", "FD", "HHAR"],
    "confirmatory_partition": "none: HHAR formal flows are target-selected descriptive",
    "true_labels_saved_for_offline_f1_and_source_label_accuracy_only": True,
    "ground_truth_lpr_observed": False,
    "independent_reannotation_available": False,
}


def _canonical_dataset(dataset: str) -> str:
    value = str(dataset).strip().upper()
    if value == "MFD":
        value = "FD"
    if value not in DATASETS:
        raise ValueError(f"Unsupported held-out queue dataset {dataset!r}")
    return value


# Formal A--F uses the same five HHAR flows as the current tuning protocol.
# Keep the complete ten-flow model registry untouched; this queue must use the
# lightweight formal registry so it cannot accidentally plan excluded flows.
from configs.formal_evaluation_protocol import (
    HHAR_DEVELOPMENT_FLOWS,
    HHAR_REPORTED_FLOWS,
    evaluation_partition_metadata,
    formal_scenario_pairs,
)

HHAR_HOLDOUT_FLOWS: frozenset[str] = frozenset()


def evaluation_partition(dataset: str, scenario: str) -> tuple[str, bool, bool]:
    """Return partition, selection-overlap, and confirmatory status."""

    metadata = evaluation_partition_metadata(_canonical_dataset(dataset), str(scenario))
    return (
        str(metadata["evaluation_partition"]),
        bool(metadata["selection_overlap"]),
        bool(metadata["confirmatory"]),
    )


def _canonical_variant(variant: str) -> str:
    value = str(variant).strip()
    lowered = value.lower().replace("-", "_")
    if lowered in {"full", "production"}:
        return "Full"
    if lowered in {"no_ssaw", "nossaw"}:
        return "no_ssaw"
    raise ValueError("variant must be Full or no_ssaw")


def cell_file_stem(cell: QueueCell | str) -> str:
    """Return a Windows/POSIX-safe deterministic artifact stem."""

    key = cell.key_string if isinstance(cell, QueueCell) else str(cell)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def _explicit_bool(value: Any, *, field: str) -> bool:
    """Parse serialized booleans without treating ``\"False\"`` as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} must be an explicit boolean")


def _scenario_label(source: str, target: str) -> str:
    return f"{str(source)}->{str(target)}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Publish a JSON document through a same-directory atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model: torch.nn.Module | Mapping[str, Any]) -> str:
    """Hash source tensors in stable name/dtype/shape order."""

    state = model.state_dict() if hasattr(model, "state_dict") else model
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OperatorSpec:
    dataset: str
    training_view_family: str
    held_out_view_family: str
    operator_id: str
    required_metadata: tuple[str, ...]


def operator_spec(dataset: str) -> OperatorSpec:
    """Return one held-out operator cross-checked against the registered config."""

    dataset = _canonical_dataset(dataset)
    family = OPERATOR_FAMILIES[dataset]
    training_family = TRAINING_VIEW_FAMILIES[dataset]
    if family == training_family:
        raise ValueError(f"{dataset}: held-out operator reuses training view family")
    # This import is intentionally read-only.  It prevents the queue from
    # silently introducing a family absent from the pre-registered protocol.
    from configs.ssaw_evaluation_protocol import get_dataset_physical_protocol

    protocol = get_dataset_physical_protocol(dataset)
    if family not in protocol.held_out_view_families:
        raise ValueError(
            f"{dataset}: operator {family!r} is not pre-registered in the protocol"
        )
    if training_family != protocol.training_view_family:
        raise ValueError(
            f"{dataset}: queue training family disagrees with the registered protocol"
        )
    return OperatorSpec(
        dataset=dataset,
        training_view_family=training_family,
        held_out_view_family=family,
        operator_id=f"{dataset.lower()}_{family}_v1",
        required_metadata=tuple(REQUIRED_PHYSICAL_METADATA[dataset]),
    )


def _operator_unit_interval(
    dataset: str, family: str, trajectory_id: str, seed: int, suffix: str
) -> float:
    material = f"{dataset}|{family}|{trajectory_id}|{int(seed)}|{suffix}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _validate_operator_signal(inputs: torch.Tensor) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("held-out operator input must be a torch.Tensor")
    if inputs.device.type != "cpu":
        raise ValueError("held-out operators are CPU-only; apply before device transfer")
    if inputs.ndim != 3:
        raise ValueError("held-out operator input must have shape [batch, channels, time]")
    if any(int(size) < 1 for size in inputs.shape) or inputs.size(-1) < 4:
        raise ValueError("held-out operator input must have positive dimensions and time >= 4")
    if inputs.is_complex():
        raise TypeError("held-out operator input must be real-valued")
    value = inputs.to(dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("held-out operator input must be finite")
    return value


def _eeg_gain_drift(inputs: torch.Tensor, *, seed: int, trajectory_id: str) -> torch.Tensor:
    batch, channels, time = inputs.shape
    phase = 2.0 * math.pi * _operator_unit_interval(
        "EEG", OPERATOR_FAMILIES["EEG"], trajectory_id, seed, "phase"
    )
    amplitude = 0.04 + 0.05 * _operator_unit_interval(
        "EEG", OPERATOR_FAMILIES["EEG"], trajectory_id, seed, "amplitude"
    )
    axis = torch.linspace(0.0, 1.0, time, dtype=inputs.dtype)
    channel_phase = torch.arange(channels, dtype=inputs.dtype)[:, None] * 0.17
    gain = 1.0 + amplitude * axis[None, :] * torch.sin(
        2.0 * math.pi * axis[None, :] + phase + channel_phase
    )
    return inputs * gain.view(1, channels, time)


def _so3_orientation_drift(
    inputs: torch.Tensor, *, seed: int, trajectory_id: str, dataset: str
) -> torch.Tensor:
    channels = inputs.size(1)
    if channels < 3 or channels % 3:
        raise ValueError(f"{dataset} held-out SO(3) operator requires 3-axis channel groups")
    batch, groups, _three, time = inputs.reshape(
        inputs.size(0), channels // 3, 3, inputs.size(-1)
    ).shape
    del batch
    phase = 2.0 * math.pi * _operator_unit_interval(
        dataset, OPERATOR_FAMILIES[dataset], trajectory_id, seed, "phase"
    )
    amplitude = 0.025 + 0.045 * _operator_unit_interval(
        dataset, OPERATOR_FAMILIES[dataset], trajectory_id, seed, "amplitude"
    )
    axis = torch.linspace(0.0, 1.0, time, dtype=inputs.dtype)
    yaw = amplitude * axis * torch.sin(2.0 * math.pi * axis + phase)
    pitch = 0.7 * amplitude * axis * torch.sin(2.0 * math.pi * axis + phase + 0.8)
    roll = 0.5 * amplitude * axis * torch.sin(2.0 * math.pi * axis + phase + 1.4)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    # Rz(yaw) @ Ry(pitch) @ Rx(roll), represented elementwise over time.
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    values = inputs.reshape(inputs.size(0), groups, 3, inputs.size(-1))
    x, y, z = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    rotated = torch.stack(
        (
            r00 * x + r01 * y + r02 * z,
            r10 * x + r11 * y + r12 * z,
            r20 * x + r21 * y + r22 * z,
        ),
        dim=2,
    )
    return rotated.reshape_as(inputs)


def _fd_response_drift(inputs: torch.Tensor, *, seed: int, trajectory_id: str) -> torch.Tensor:
    time = inputs.size(-1)
    phase = 2.0 * math.pi * _operator_unit_interval(
        "FD", OPERATOR_FAMILIES["FD"], trajectory_id, seed, "phase"
    )
    amplitude = 0.04 + 0.05 * _operator_unit_interval(
        "FD", OPERATOR_FAMILIES["FD"], trajectory_id, seed, "amplitude"
    )
    axis = torch.linspace(0.0, 1.0, time, dtype=inputs.dtype)
    gain = 1.0 + amplitude * axis * torch.cos(2.0 * math.pi * axis + phase)
    return inputs * gain.view(1, 1, time)


def apply_held_out_operator(
    dataset: str,
    inputs: torch.Tensor,
    *,
    seed: int,
    trajectory_id: str,
    held_out_view_family: Optional[str] = None,
) -> torch.Tensor:
    """Apply one deterministic, pre-registered held-out operator on CPU."""

    dataset = _canonical_dataset(dataset)
    spec = operator_spec(dataset)
    family = spec.held_out_view_family if held_out_view_family is None else str(held_out_view_family)
    if family != spec.held_out_view_family:
        raise ValueError(
            f"{dataset}: unsupported operator family {family!r}; expected {spec.held_out_view_family!r}"
        )
    trajectory_id = str(trajectory_id).strip()
    if not trajectory_id:
        raise ValueError("trajectory_id must be non-empty")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("held-out operator seed must be an integer") from exc
    value = _validate_operator_signal(inputs)
    if dataset == "EEG":
        output = _eeg_gain_drift(value, seed=seed, trajectory_id=trajectory_id)
    elif dataset in {"HAR", "HHAR"}:
        output = _so3_orientation_drift(
            value, seed=seed, trajectory_id=trajectory_id, dataset=dataset
        )
    else:
        output = _fd_response_drift(value, seed=seed, trajectory_id=trajectory_id)
    if tuple(output.shape) != tuple(value.shape) or not torch.isfinite(output).all():
        raise RuntimeError("held-out operator violated shape/finite invariants")
    return output.to(dtype=inputs.dtype)


class DeterministicHeldOutOperator:
    """Callable adapter for ``HeldOutMechanismRunner`` and queue workers."""

    def __init__(self, dataset: str, trajectory_id: str):
        self.dataset = _canonical_dataset(dataset)
        self.trajectory_id = str(trajectory_id)
        self.spec = operator_spec(self.dataset)

    def __call__(self, inputs: torch.Tensor, seed: int = DEFAULT_TEST_SEED) -> torch.Tensor:
        return apply_held_out_operator(
            self.dataset,
            inputs,
            seed=seed,
            trajectory_id=self.trajectory_id,
            held_out_view_family=self.spec.held_out_view_family,
        )


def make_held_out_operator(dataset: str, trajectory_id: str) -> DeterministicHeldOutOperator:
    return DeterministicHeldOutOperator(dataset, trajectory_id)


@dataclass(frozen=True)
class QueueCell:
    dataset: str
    source: str
    target: str
    source_seed: int
    variant: str
    training_view_seed: int = DEFAULT_TRAINING_VIEW_SEED
    test_seed: int = DEFAULT_TEST_SEED
    trajectory_id: str = ""

    def __post_init__(self) -> None:
        dataset = _canonical_dataset(self.dataset)
        variant = _canonical_variant(self.variant)
        source = str(self.source).strip()
        target = str(self.target).strip()
        if not source or not target:
            raise ValueError("queue cell source/target must be non-empty")
        source_seed = int(self.source_seed)
        training_view_seed = int(self.training_view_seed)
        test_seed = int(self.test_seed)
        if training_view_seed == test_seed:
            raise ValueError(
                "training_view_seed (ssaw_sobol_seed) and heldout_test_seed must be separated"
            )
        if training_view_seed == source_seed or test_seed == source_seed:
            raise ValueError(
                "source_seed, training_view_seed, and heldout_test_seed must be distinct"
            )
        trajectory = str(self.trajectory_id).strip() or (
            f"{dataset}:{source}->{target}:trajectory"
        )
        if trajectory == TRAINING_VIEW_FAMILIES[dataset]:
            raise ValueError("held-out trajectory must differ from training view family")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source_seed", source_seed)
        object.__setattr__(self, "training_view_seed", training_view_seed)
        object.__setattr__(self, "test_seed", test_seed)
        object.__setattr__(self, "trajectory_id", trajectory)

    @property
    def scenario(self) -> str:
        return _scenario_label(self.source, self.target)

    @property
    def operator_spec(self) -> OperatorSpec:
        return operator_spec(self.dataset)

    @property
    def operator_id(self) -> str:
        return self.operator_spec.operator_id

    @property
    def held_out_view_family(self) -> str:
        return self.operator_spec.held_out_view_family

    @property
    def training_view_family(self) -> str:
        return self.operator_spec.training_view_family

    @property
    def key(self) -> tuple[str, str, int, int, int, str]:
        """Stable resume key including all three seed roles and variant."""

        return (
            self.dataset,
            self.scenario,
            int(self.source_seed),
            int(self.training_view_seed),
            int(self.test_seed),
            self.variant,
        )

    @property
    def key_string(self) -> str:
        return "|".join(
            (
                self.dataset,
                self.scenario,
                f"src{self.source_seed}",
                f"view{self.training_view_seed}",
                f"seed{self.test_seed}",
                self.variant,
            )
        )

    @property
    def heldout_test_seed(self) -> int:
        return self.test_seed


def cell_key(cell: QueueCell | Mapping[str, Any]) -> tuple[str, str, int, int, int, str]:
    if not isinstance(cell, QueueCell):
        cell = QueueCell(**dict(cell))
    return cell.key


def registered_flows(dataset: str) -> tuple[tuple[str, str], ...]:
    return formal_scenario_pairs(_canonical_dataset(dataset))


def _scenario_label_set(values: Iterable[str], *, dataset: str) -> tuple[str, ...]:
    """Validate a selected subset of the dataset's registered flow labels."""

    registered = tuple(
        _scenario_label(source, target) for source, target in registered_flows(dataset)
    )
    requested = tuple(str(value).strip() for value in values if str(value).strip())
    if not requested:
        raise ValueError(f"{dataset}: scenarios must be non-empty")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{dataset}: scenarios must be unique")
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        raise ValueError(
            f"{dataset}: scenarios must be registered flows; unknown={unknown}"
        )
    # Keep the frozen registry order so manifests and resume keys are stable
    # regardless of CLI ordering.
    return tuple(scenario for scenario in registered if scenario in set(requested))


def selected_scenarios(
    datasets: Iterable[str],
    scenarios: Optional[Mapping[str, Iterable[str]] | Iterable[str]] = None,
) -> dict[str, tuple[str, ...]]:
    """Normalize optional flow filters without changing the default five-flow plan.

    A bare iterable is only accepted for a single dataset.  Dataset-keyed
    mappings are used for multi-dataset plans and reject foreign keys.
    """

    requested_datasets = tuple(_canonical_dataset(dataset) for dataset in datasets)
    if not requested_datasets or len(set(requested_datasets)) != len(requested_datasets):
        raise ValueError("datasets must be non-empty and unique")
    if scenarios is None:
        return {
            dataset: tuple(
                _scenario_label(source, target) for source, target in registered_flows(dataset)
            )
            for dataset in requested_datasets
        }
    if isinstance(scenarios, Mapping):
        normalized_mapping = {
            str(key).strip().upper(): value for key, value in scenarios.items()
        }
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
            result[dataset] = _scenario_label_set(values or (), dataset=dataset)
        return result
    if len(requested_datasets) != 1:
        raise ValueError("a bare scenario iterable requires exactly one dataset")
    values = (
        [item.strip() for item in scenarios.split(",") if item.strip()]
        if isinstance(scenarios, str)
        else scenarios
    )
    return {requested_datasets[0]: _scenario_label_set(values, dataset=requested_datasets[0])}


def default_training_view_seed(dataset: str) -> int:
    """Read the repository's frozen default SSAW Sobol seed without a trainer."""

    dataset = _canonical_dataset(dataset)
    from configs.tta_hparams_new import get_hparams_class

    hparams = get_hparams_class(dataset)()
    try:
        seed = int(hparams.alg_hparams["DuSafe"]["ssaw_sobol_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{dataset}: configs default lacks ssaw_sobol_seed") from exc
    return seed


def training_view_provenance(
    dataset: str, tta_config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Hash the frozen training-view family and its physical-view parameters."""

    dataset = _canonical_dataset(dataset)
    if "ssaw_sobol_seed" not in tta_config:
        raise ValueError(f"{dataset}: TTA config lacks ssaw_sobol_seed")
    parameters = {
        key: tta_config[key]
        for key in TRAINING_VIEW_PROVENANCE_KEYS
        if key in tta_config
    }
    parameters["ssaw_sobol_seed"] = int(tta_config["ssaw_sobol_seed"])
    payload = {
        "dataset": dataset,
        "training_view_family": TRAINING_VIEW_FAMILIES[dataset],
        "current_training_view_family": CURRENT_SSAW_DIRECTION_FAMILY,
        "heldout_direction_family": HELDOUT_SSAW_DIRECTION_FAMILY,
        "direction_bank": {
            "num_directions": int(
                tta_config.get("spline_num_directions", HELDOUT_SSAW_DIRECTION_COUNT)
            ),
            "signs": list(HELDOUT_SSAW_DIRECTION_SIGNS),
            "radius_levels": list(
                tta_config.get("spline_radius_levels", HELDOUT_SSAW_DIRECTION_RADII)
            ),
        },
        "parameters": parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return {
        # ``training_view_family`` is retained as the physical-protocol
        # compatibility field consumed by HeldOutCase validation.  New
        # mechanism consumers must use the explicit current/held-out fields.
        "training_view_family": TRAINING_VIEW_FAMILIES[dataset],
        "physical_training_view_family": TRAINING_VIEW_FAMILIES[dataset],
        "current_training_view_family": CURRENT_SSAW_DIRECTION_FAMILY,
        "heldout_direction_family": HELDOUT_SSAW_DIRECTION_FAMILY,
        "direction_bank": payload["direction_bank"],
        "parameters": parameters,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_queue_cells(
    *,
    datasets: Iterable[str] = DATASETS,
    source_seeds: Iterable[int] = SOURCE_SEEDS,
    variants: Iterable[str] = VARIANTS,
    test_seed: int = DEFAULT_TEST_SEED,
    training_view_seeds: Optional[Mapping[str, int]] = None,
    scenarios: Optional[Mapping[str, Iterable[str]] | Iterable[str]] = None,
) -> tuple[QueueCell, ...]:
    datasets = tuple(_canonical_dataset(dataset) for dataset in datasets)
    source_seeds = tuple(int(seed) for seed in source_seeds)
    variants = tuple(_canonical_variant(variant) for variant in variants)
    if set(variants) != set(VARIANTS) or len(variants) != len(VARIANTS):
        raise ValueError("queue must contain exactly Full and no_ssaw variants")
    if len(source_seeds) != len(set(source_seeds)) or not source_seeds:
        raise ValueError("source_seeds must be non-empty and unique")
    scenarios_by_dataset = selected_scenarios(datasets, scenarios)
    cells = []
    for dataset in datasets:
        training_view_seed = (
            int(training_view_seeds[dataset])
            if training_view_seeds is not None and dataset in training_view_seeds
            else default_training_view_seed(dataset)
        )
        for scenario in scenarios_by_dataset[dataset]:
            source, target = scenario.split("->", 1)
            trajectory = f"{dataset}:{source}->{target}:heldout_trajectory_v1"
            for source_seed in source_seeds:
                for variant in variants:
                    cells.append(
                        QueueCell(
                            dataset=dataset,
                            source=source,
                            target=target,
                            source_seed=source_seed,
                            variant=variant,
                            training_view_seed=training_view_seed,
                            test_seed=int(test_seed),
                            trajectory_id=trajectory,
                        )
                    )
    validate_cell_plan(
        cells,
        datasets=datasets,
        source_seeds=source_seeds,
        scenarios=scenarios_by_dataset,
    )
    return tuple(cells)


def validate_cell_plan(
    cells: Sequence[QueueCell | Mapping[str, Any]],
    *,
    datasets: Optional[Iterable[str]] = None,
    source_seeds: Optional[Iterable[int]] = None,
    scenarios: Optional[Mapping[str, Iterable[str]] | Iterable[str]] = None,
) -> tuple[QueueCell, ...]:
    normalized = tuple(
        cell if isinstance(cell, QueueCell) else QueueCell(**dict(cell))
        for cell in cells
    )
    if len({cell.key for cell in normalized}) != len(normalized):
        raise ValueError("queue contains duplicate restoration keys")
    expected_datasets = tuple(_canonical_dataset(value) for value in (datasets or DATASETS))
    expected_seeds = tuple(int(value) for value in (source_seeds or SOURCE_SEEDS))
    expected_scenarios = selected_scenarios(expected_datasets, scenarios)
    if set(cell.dataset for cell in normalized) != set(expected_datasets):
        raise ValueError("queue dataset registry is incomplete or contains foreign cells")
    grouped: Dict[tuple[str, str, int, int, int], set[str]] = {}
    for cell in normalized:
        if cell.dataset not in expected_datasets or cell.source_seed not in expected_seeds:
            raise ValueError(f"foreign queue cell: {cell.key}")
        base = (
            cell.dataset,
            cell.scenario,
            cell.source_seed,
            cell.training_view_seed,
            cell.test_seed,
        )
        grouped.setdefault(base, set()).add(cell.variant)
    expected_variants = set(VARIANTS)
    for base, observed in grouped.items():
        if observed != expected_variants:
            raise ValueError(f"incomplete Full/no_ssaw pairing for {base}: {observed}")
    expected_flows = {
        (dataset, scenario)
        for dataset in expected_datasets
        for scenario in expected_scenarios[dataset]
    }
    actual_flows = {(cell.dataset, cell.scenario) for cell in normalized}
    if actual_flows != expected_flows:
        raise ValueError("queue flows do not exactly match the registered scenario registry")
    expected_count = sum(
        len(expected_scenarios[dataset]) * len(expected_seeds) * len(expected_variants)
        for dataset in expected_datasets
    )
    if len(normalized) != expected_count:
        raise ValueError(
            f"queue cell count mismatch: expected {expected_count}, got {len(normalized)}"
        )
    return normalized


def validate_cell_metadata(cell: QueueCell, metadata: Mapping[str, Any]) -> HeldOutCase:
    """Build the strict split case for a cell and reject missing physical metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("dataset physical metadata must be a mapping")
    spec = cell.operator_spec
    for flag in (
        "label_leakage",
        "target_labels_used_online",
        "target_labels_used_for_selection",
        "target_labels_used_for_tuning",
        "lpr_estimated",
        "ground_truth_lpr_observed",
        "independent_reannotation_available",
    ):
        if flag in metadata and _explicit_bool(metadata[flag], field=flag):
            raise ValueError(f"{cell.dataset}: metadata cannot assert {flag}=true")
    if "operator_family" in metadata and str(metadata["operator_family"]) != spec.held_out_view_family:
        raise ValueError(f"{cell.dataset}: metadata operator_family disagrees with registered operator")
    if (
        "training_view_family" in metadata
        and str(metadata["training_view_family"]) != spec.training_view_family
    ):
        raise ValueError(
            f"{cell.dataset}: metadata training_view_family disagrees with registered view"
        )
    if "training_view_config_sha256" in metadata:
        training_hash = str(metadata["training_view_config_sha256"]).strip().lower()
        if len(training_hash) != 64 or any(
            character not in "0123456789abcdef" for character in training_hash
        ):
            raise ValueError("training_view_config_sha256 must be a 64-character hex digest")
    if "training_view_seed" in metadata and int(metadata["training_view_seed"]) != cell.training_view_seed:
        raise ValueError(f"{cell.dataset}: metadata training_view_seed disagrees with cell")
    if "ssaw_sobol_seed" in metadata and int(metadata["ssaw_sobol_seed"]) != cell.training_view_seed:
        raise ValueError(f"{cell.dataset}: metadata ssaw_sobol_seed disagrees with cell")
    if "heldout_test_seed" in metadata and int(metadata["heldout_test_seed"]) != cell.test_seed:
        raise ValueError(f"{cell.dataset}: metadata heldout_test_seed disagrees with cell")
    if cell.training_view_seed == cell.test_seed:
        raise ValueError(
            f"{cell.dataset}: training_view_seed and heldout_test_seed must be distinct"
        )
    missing = [key for key in spec.required_metadata if key not in metadata]
    if missing:
        raise ValueError(f"{cell.dataset}: missing required physical metadata {missing}")
    merged = dict(metadata)
    merged.update(
        {
            "training_seed": cell.source_seed,
            "training_view_seed": cell.training_view_seed,
            "ssaw_sobol_seed": cell.training_view_seed,
            "test_seed": cell.test_seed,
            "heldout_test_seed": cell.test_seed,
            "held_out_trajectory": cell.trajectory_id,
            "held_out_operator": cell.operator_id,
            "training_view_family": spec.training_view_family,
            "operator_family": spec.held_out_view_family,
            "label_leakage": False,
            "target_labels_used_online": False,
            "lpr_estimated": False,
            "rotation_rate_unverified": bool(
                cell.dataset == "FD"
                and not any(
                    metadata.get(name) is not None
                    for name in (
                        "rotation_frequency_hz",
                        "operating_frequency_hz",
                        "shaft_frequency_hz",
                        "order_reference_hz",
                    )
                )
            ),
        }
    )
    return validate_case(
        HeldOutCase(
            dataset=cell.dataset,
            training_view_family=spec.training_view_family,
            held_out_view_family=spec.held_out_view_family,
            held_out_trajectory=cell.trajectory_id,
            held_out_operator=cell.operator_id,
            training_seed=cell.source_seed,
            test_seed=cell.test_seed,
            metadata=merged,
            algorithm=cell.variant,
        )
    )


def queue_manifest(
    cells: Sequence[QueueCell],
    *,
    metadata_by_dataset: Optional[Mapping[str, Mapping[str, Any]]] = None,
    output_dir: Optional[str | Path] = None,
    dry_run: bool = True,
    configuration_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    requested_datasets = tuple(sorted({cell.dataset for cell in cells}))
    selected_flow_map = {
        dataset: tuple(
            sorted(
                {
                    cell.scenario
                    for cell in cells
                    if cell.dataset == dataset
                }
            )
        )
        for dataset in requested_datasets
    }
    normalized = validate_cell_plan(
        cells,
        datasets=requested_datasets,
        source_seeds=tuple(sorted({cell.source_seed for cell in cells})),
        scenarios=selected_flow_map,
    )
    selected_flow_map = {
        dataset: tuple(
            scenario
            for scenario in (
                _scenario_label(source, target) for source, target in registered_flows(dataset)
            )
            if scenario in set(selected_flow_map[dataset])
        )
        for dataset in requested_datasets
    }
    is_full_flow_scope = all(
        tuple(selected_flow_map[dataset])
        == tuple(
            _scenario_label(source, target)
            for source, target in registered_flows(dataset)
        )
        for dataset in requested_datasets
    )
    metadata_by_dataset = metadata_by_dataset or {}
    configuration_provenance = configuration_provenance or {}
    for dataset in requested_datasets:
        if dataset in metadata_by_dataset:
            for cell in normalized:
                if cell.dataset == dataset:
                    validate_cell_metadata(cell, metadata_by_dataset[dataset])
    rotation_rate_unverified = {}
    for dataset in requested_datasets:
        metadata = metadata_by_dataset.get(dataset, {})
        rotation_rate_unverified[dataset] = bool(
            dataset == "FD"
            and not any(
                metadata.get(name) is not None
                for name in (
                    "rotation_frequency_hz",
                    "operating_frequency_hz",
                    "shaft_frequency_hz",
                    "order_reference_hz",
                )
            )
        )
    return {
        "protocol_version": QUEUE_PROTOCOL_VERSION,
        "status": "planned" if dry_run else "running",
        "output_dir": None if output_dir is None else str(Path(output_dir)),
        "configuration_provenance": {
            str(dataset): dict(values)
            for dataset, values in configuration_provenance.items()
        },
        "datasets": list(requested_datasets),
        "source_seeds": list(sorted({cell.source_seed for cell in normalized})),
        "training_view_seeds": list(sorted({cell.training_view_seed for cell in normalized})),
        "heldout_test_seeds": list(sorted({cell.test_seed for cell in normalized})),
        "test_seeds": list(sorted({cell.test_seed for cell in normalized})),
        "seed_roles_by_dataset": {
            dataset: {
                "source_seeds": sorted(
                    {cell.source_seed for cell in normalized if cell.dataset == dataset}
                ),
                "training_view_seeds": sorted(
                    {
                        cell.training_view_seed
                        for cell in normalized
                        if cell.dataset == dataset
                    }
                ),
                "heldout_test_seeds": sorted(
                    {cell.test_seed for cell in normalized if cell.dataset == dataset}
                ),
            }
            for dataset in requested_datasets
        },
        "variants": list(VARIANTS),
        "evidence_role": "B_no_ssaw_vs_full_heldout_sobol_mechanism",
        "evidence_role_policy": {
            "direction_bank_is_unseen_by_training": True,
            "eligible_coverage_denominator": "confidence_admitted_anchors",
            "metrics": [
                "eligible_coverage",
                "margin_ratio",
                "heldout_flip_rate",
                "heldout_worst_margin",
                "heldout_consistency",
            ],
            "conditional_effects": "report only for label-free active batches with eligible_coverage >= 0.25",
            "overall_coverage_also_required": True,
            "two_by_two_grid": "audit_only",
        },
        "registered_flow_counts": {
            dataset: len(registered_flows(dataset)) for dataset in requested_datasets
        },
        "selected_flow_counts": {
            dataset: len(selected_flow_map[dataset]) for dataset in requested_datasets
        },
        "flows_by_dataset": {
            dataset: list(selected_flow_map[dataset]) for dataset in requested_datasets
        },
        "scenario_scope": (
            "registered_formal_full"
            if is_full_flow_scope
            else "registered_representative_subset"
        ),
        "expected_cells": len(normalized),
        "completed_cells": 0,
        "completed_keys": [],
        # These are the mechanism families used by current production DuSafe.
        # The old SO(3)/sensor names are explicitly moved under
        # ``physical_*`` so a dry-run manifest cannot be read as evidence
        # that production SSAW was trained/evaluated in that family.
        "operator_families": {
            dataset: HELDOUT_SSAW_DIRECTION_FAMILY
            for dataset in requested_datasets
        },
        "training_view_families": {
            dataset: CURRENT_SSAW_DIRECTION_FAMILY
            for dataset in requested_datasets
        },
        "heldout_direction_families": {
            dataset: HELDOUT_SSAW_DIRECTION_FAMILY
            for dataset in requested_datasets
        },
        "physical_operator_families": {
            dataset: OPERATOR_FAMILIES[dataset] for dataset in requested_datasets
        },
        "physical_training_view_families": {
            dataset: TRAINING_VIEW_FAMILIES[dataset]
            for dataset in requested_datasets
        },
        "physical_plausibility_audit": {
            "status": "separate_from_ssaw_mechanism_claim",
            "operator_families": {
                dataset: OPERATOR_FAMILIES[dataset]
                for dataset in requested_datasets
            },
            "training_view_families": {
                dataset: TRAINING_VIEW_FAMILIES[dataset]
                for dataset in requested_datasets
            },
        },
        "heldout_direction_bank": {
            "family": HELDOUT_SSAW_DIRECTION_FAMILY,
            "training_family": CURRENT_SSAW_DIRECTION_FAMILY,
            "num_directions": HELDOUT_SSAW_DIRECTION_COUNT,
            "signs": list(HELDOUT_SSAW_DIRECTION_SIGNS),
            "radius_levels": list(HELDOUT_SSAW_DIRECTION_RADII),
            "candidate_count": HELDOUT_SSAW_CANDIDATE_COUNT,
            "seed_source": "stable_hash(dataset,flow,source_seed,training_seed,test_seed)",
            "seed_disjoint_from_training_sobol_seed": True,
            "seed_disjoint_from_heldout_test_seed": True,
            "seed_overlap_recorded_per_variant": True,
            "candidate_hash_recorded_per_variant": True,
        },
        "physical_metadata_policy": {
            "sampling_rate_must_be_explicit_and_provenanced": True,
            "HHAR_sampling_rate_optional": True,
            "HHAR_without_rate_axis": "cycles_per_sample",
            "HHAR_without_rate_jerk_axis": "per_sample",
            "FD_rotation_frequency_optional": True,
            "FD_rotation_rate_unverified_when_absent": True,
            "FD_order_metric_when_rotation_unverified": None,
            "FD_raw_hz_metrics_without_sampling_rate": None,
        },
        "sampling_rate_provenance": {
            dataset: metadata_by_dataset.get(dataset, {}).get(
                "sampling_rate_provenance",
                metadata_by_dataset.get(dataset, {}).get("sample_rate_provenance"),
            )
            for dataset in requested_datasets
        },
        "rotation_rate_unverified": rotation_rate_unverified,
        "full_no_ssaw_pairing_required": True,
        "process_isolation": "one queue cell per worker process",
        "global_gpu_lock": True,
        "gpu_lock_scope": "per_queue_cell",
        "gpu_lock_wait_policy": "bounded_exponential_backoff",
        "gpu_lock_busy_consumes_attempt": False,
        "hhar_frozen_state_required": "HHAR" in requested_datasets,
        "seed_roles": {
            "source_seed": "source checkpoint construction",
            "training_view_seed": (
                "production spline-residual tta_config.ssaw_sobol_seed"
            ),
            "heldout_test_seed": "separate physical plausibility/test stream",
            "heldout_direction_seed": (
                "independent unseen spline-direction Sobol bank"
            ),
        },
        "training_view_provenance": {
            "families": {
                dataset: CURRENT_SSAW_DIRECTION_FAMILY
                for dataset in requested_datasets
            },
            "config_hash_recorded_per_worker": True,
            "hash_fields": list(TRAINING_VIEW_PROVENANCE_KEYS),
        },
        "heldout_direction_seeds": {
            dataset: sorted(
                {
                    _heldout_sobol_seed(cell)
                    for cell in normalized
                    if cell.dataset == dataset
                }
            )
            for dataset in requested_datasets
        },
        "true_labels_artifact_scope": "offline_f1_and_source_label_accuracy_only",
        "label_leakage_flags": dict(LABEL_LEAKAGE_FLAGS),
        "evaluation_partition_policy": {
            "EEG_HAR_FD": "target_selected_evaluation_descriptive",
            "HHAR_formal_flows": list(HHAR_REPORTED_FLOWS),
            "HHAR_raw_domain_count": 9,
            "HHAR_parameter_selection_data_overlap": True,
            "confirmatory_results": "none: HHAR formal flows are target-selected descriptive",
        },
        "real_nuisance_observed": False,
        "ground_truth_lpr_observed": False,
        "lpr_claimed_online": False,
    }


def _validate_evidence_arrays(
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    clean_logits: torch.Tensor,
    held_out_logits: torch.Tensor,
    clean_features: torch.Tensor,
    held_out_features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    tensors = [
        clean_signal,
        held_out_signal,
        clean_logits,
        held_out_logits,
        clean_features,
        held_out_features,
        labels,
    ]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("evidence arrays must be torch.Tensor instances")
    converted = [value.detach().cpu() for value in tensors]
    clean_signal, held_out_signal, clean_logits, held_out_logits, clean_features, held_out_features, labels = converted
    if clean_signal.ndim != 3 or tuple(clean_signal.shape) != tuple(held_out_signal.shape):
        raise ValueError("clean/held-out signals must match [batch, channels, time]")
    if clean_logits.ndim != 2 or tuple(clean_logits.shape) != tuple(held_out_logits.shape):
        raise ValueError("clean/held-out logits must match [batch, classes]")
    if clean_features.ndim < 2 or tuple(clean_features.shape) != tuple(held_out_features.shape):
        raise ValueError("clean/held-out features must match [batch, ...]")
    batch = clean_signal.size(0)
    if any(value.size(0) != batch for value in (clean_logits, clean_features, labels.reshape(-1, 1))):
        raise ValueError("signals, logits, features, and labels must share batch size")
    labels = labels.reshape(-1).long()
    if labels.numel() != batch:
        raise ValueError("labels must contain one true label per target sample")
    if bool((labels < 0).any()) or bool(
        labels.numel() and labels.max() >= clean_logits.size(1)
    ):
        raise ValueError("labels must be valid non-negative class indices")
    all_values = (clean_signal, held_out_signal, clean_logits, held_out_logits, clean_features, held_out_features)
    if any(value.is_complex() for value in all_values) or not all(torch.isfinite(value).all() for value in all_values):
        raise ValueError("evidence tensors must be finite real values")
    if labels.numel() < 1:
        raise ValueError("evidence cannot be empty")
    return (
        clean_signal.float(),
        held_out_signal.float(),
        clean_logits.float(),
        held_out_logits.float(),
        clean_features.float(),
        held_out_features.float(),
        labels,
    )


@dataclass
class VariantEvidence:
    cell: QueueCell
    source_checkpoint_sha256: str
    clean_signal: torch.Tensor
    held_out_signal: torch.Tensor
    clean_logits: torch.Tensor
    held_out_logits: torch.Tensor
    clean_features: torch.Tensor
    held_out_features: torch.Tensor
    labels: torch.Tensor
    metadata: Mapping[str, Any]
    artifact_path: Optional[str] = None
    # Optional label-free diagnostics from a direction bank sampled with a
    # seed distinct from the training SSAW seed.  They are absent in legacy
    # artifacts and therefore remain backward-compatible.
    heldout_confidence_admitted_mask: Optional[torch.Tensor] = None
    heldout_eligible_mask: Optional[torch.Tensor] = None
    heldout_margin_ratio: Optional[torch.Tensor] = None
    heldout_flip_rate: Optional[torch.Tensor] = None
    heldout_worst_margin: Optional[torch.Tensor] = None
    heldout_consistency: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        values = _validate_evidence_arrays(
            self.clean_signal,
            self.held_out_signal,
            self.clean_logits,
            self.held_out_logits,
            self.clean_features,
            self.held_out_features,
            self.labels,
        )
        (
            self.clean_signal,
            self.held_out_signal,
            self.clean_logits,
            self.held_out_logits,
            self.clean_features,
            self.held_out_features,
            self.labels,
        ) = values
        source_hash = str(self.source_checkpoint_sha256).strip()
        if len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash.lower()):
            raise ValueError("source_checkpoint_sha256 must be a 64-character hex digest")
        self.source_checkpoint_sha256 = source_hash.lower()
        self.metadata = dict(self.metadata)
        validate_cell_metadata(self.cell, self.metadata)
        batch = int(self.labels.numel())
        optional = (
            ("heldout_confidence_admitted_mask", self.heldout_confidence_admitted_mask, True),
            ("heldout_eligible_mask", self.heldout_eligible_mask, True),
            ("heldout_margin_ratio", self.heldout_margin_ratio, False),
            ("heldout_flip_rate", self.heldout_flip_rate, False),
            ("heldout_worst_margin", self.heldout_worst_margin, False),
            ("heldout_consistency", self.heldout_consistency, False),
        )
        present = [value is not None for _, value, _ in optional]
        if any(present) and not all(present):
            raise ValueError(
                "held-out direction diagnostics must provide all optional arrays together"
            )
        for name, value, is_mask in optional:
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            value = value.detach().cpu().reshape(-1)
            if value.numel() != batch:
                raise ValueError(f"{name} must contain one value per evidence sample")
            if is_mask:
                value = value.to(dtype=torch.bool)
            else:
                value = value.to(dtype=torch.float32)
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} must contain finite values")
            setattr(self, name, value)


def save_variant_evidence(evidence: VariantEvidence, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            clean_signal=evidence.clean_signal.numpy(),
            held_out_signal=evidence.held_out_signal.numpy(),
            clean_logits=evidence.clean_logits.numpy(),
            held_out_logits=evidence.held_out_logits.numpy(),
            clean_features=evidence.clean_features.numpy(),
            held_out_features=evidence.held_out_features.numpy(),
            labels=evidence.labels.numpy(),
            source_checkpoint_sha256=np.asarray(
                evidence.source_checkpoint_sha256, dtype="U64"
            ),
            source_seed=np.asarray(evidence.cell.source_seed, dtype=np.int64),
            training_view_seed=np.asarray(evidence.cell.training_view_seed, dtype=np.int64),
            heldout_test_seed=np.asarray(evidence.cell.test_seed, dtype=np.int64),
            **(
                {
                    "heldout_confidence_admitted_mask": evidence.heldout_confidence_admitted_mask.numpy(),
                    "heldout_eligible_mask": evidence.heldout_eligible_mask.numpy(),
                    "heldout_margin_ratio": evidence.heldout_margin_ratio.numpy(),
                    "heldout_flip_rate": evidence.heldout_flip_rate.numpy(),
                    "heldout_worst_margin": evidence.heldout_worst_margin.numpy(),
                    "heldout_consistency": evidence.heldout_consistency.numpy(),
                }
                if evidence.heldout_confidence_admitted_mask is not None
                else {}
            ),
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_variant_evidence(
    path: str | Path,
    *,
    cell: QueueCell,
    source_checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> VariantEvidence:
    with np.load(Path(path), allow_pickle=False) as bundle:
        required = {
            "clean_signal",
            "held_out_signal",
            "clean_logits",
            "held_out_logits",
            "clean_features",
            "held_out_features",
            "labels",
            "source_checkpoint_sha256",
            "source_seed",
            "training_view_seed",
            "heldout_test_seed",
        }
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"evidence artifact missing arrays: {missing}")
        arrays = {
            name: torch.from_numpy(np.asarray(bundle[name]))
            for name in required
            if name != "source_checkpoint_sha256"
        }
        optional_names = (
            "heldout_confidence_admitted_mask",
            "heldout_eligible_mask",
            "heldout_margin_ratio",
            "heldout_flip_rate",
            "heldout_worst_margin",
            "heldout_consistency",
        )
        for name in optional_names:
            if name in bundle.files:
                arrays[name] = torch.from_numpy(np.asarray(bundle[name]))
        artifact_hash = str(np.asarray(bundle["source_checkpoint_sha256"]).item()).strip().lower()
    if artifact_hash != str(source_checkpoint_sha256).strip().lower():
        raise ValueError("evidence source checkpoint hash disagrees with NPZ artifact")
    if int(arrays["source_seed"].item()) != cell.source_seed:
        raise ValueError("evidence source_seed disagrees with cell")
    if int(arrays["training_view_seed"].item()) != cell.training_view_seed:
        raise ValueError("evidence training_view_seed disagrees with cell")
    if int(arrays["heldout_test_seed"].item()) != cell.test_seed:
        raise ValueError("evidence heldout_test_seed disagrees with cell")
    return VariantEvidence(
        cell=cell,
        source_checkpoint_sha256=source_checkpoint_sha256,
        clean_signal=arrays["clean_signal"],
        held_out_signal=arrays["held_out_signal"],
        clean_logits=arrays["clean_logits"],
        held_out_logits=arrays["held_out_logits"],
        clean_features=arrays["clean_features"],
        held_out_features=arrays["held_out_features"],
        labels=arrays["labels"],
        metadata=metadata,
        artifact_path=str(path),
        heldout_confidence_admitted_mask=arrays.get(
            "heldout_confidence_admitted_mask"
        ),
        heldout_eligible_mask=arrays.get("heldout_eligible_mask"),
        heldout_margin_ratio=arrays.get("heldout_margin_ratio"),
        heldout_flip_rate=arrays.get("heldout_flip_rate"),
        heldout_worst_margin=arrays.get("heldout_worst_margin"),
        heldout_consistency=arrays.get("heldout_consistency"),
    )


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> float:
    labels = labels.reshape(-1).long().cpu()
    predictions = predictions.reshape(-1).long().cpu()
    if labels.numel() != predictions.numel() or labels.numel() < 1:
        raise ValueError("F1 labels/predictions must have equal non-zero length")
    scores = []
    for class_index in range(int(num_classes)):
        true_positive = ((labels == class_index) & (predictions == class_index)).sum().item()
        false_positive = ((labels != class_index) & (predictions == class_index)).sum().item()
        false_negative = ((labels == class_index) & (predictions != class_index)).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(sum(scores) / max(len(scores), 1))


def variant_metrics(evidence: VariantEvidence) -> Dict[str, Any]:
    """Compute one variant's predictive, physical, and offline label metrics."""

    values = compute_mechanism_metrics(
        evidence.cell.dataset,
        evidence.clean_signal,
        evidence.held_out_signal,
        clean_logits=evidence.clean_logits,
        held_out_logits=evidence.held_out_logits,
        clean_features=evidence.clean_features,
        held_out_features=evidence.held_out_features,
        metadata=evidence.metadata,
    )
    if evidence.cell.dataset == "FD":
        # Preserve explicit nulls in row JSON when a verified rate/rotation
        # reference is unavailable; aggregation treats these as inapplicable
        # rather than inventing an order or Hz quantity.
        for unavailable_metric in (
            "order_frequency_peak_shift",
            "order_peak_shift",
            "raw_spectral_peak_shift_hz",
            "raw_envelope_spectral_peak_shift_hz",
            "envelope_spectrum_peak_shift",
            "envelope_peak_shift",
        ):
            values.setdefault(unavailable_metric, None)
    clean_prediction = evidence.clean_logits.argmax(dim=-1).cpu()
    held_out_prediction = evidence.held_out_logits.argmax(dim=-1).cpu()
    labels = evidence.labels.cpu()
    num_classes = max(
        int(evidence.clean_logits.size(1)),
        int(labels.max().item()) + 1 if labels.numel() else 0,
    )
    values.update(
        {
            "clean_f1": _macro_f1(labels, clean_prediction, num_classes),
            "heldout_f1": _macro_f1(labels, held_out_prediction, num_classes),
            # Without independent re-annotation, transformed ground-truth LPR
            # is unobservable.  These names deliberately describe the two
            # measurable proxies and never claim LPR.
            "source_label_accuracy_on_view": float((held_out_prediction == labels).double().mean().item()),
            "prediction_label_agreement": float((held_out_prediction == clean_prediction).double().mean().item()),
        }
    )
    if evidence.heldout_confidence_admitted_mask is not None:
        values.update(
            summarize_heldout_direction_diagnostics(
                {
                    "confidence_admitted_mask": evidence.heldout_confidence_admitted_mask,
                    "eligible_mask": evidence.heldout_eligible_mask,
                    "margin_ratio": evidence.heldout_margin_ratio,
                    "heldout_flip_rate": evidence.heldout_flip_rate,
                    "heldout_worst_margin": evidence.heldout_worst_margin,
                    "heldout_consistency": evidence.heldout_consistency,
                }
            )
        )
    else:
        values.update(
            {
                "eligible_coverage": None,
                "margin_ratio": None,
                "heldout_flip_rate": None,
                "heldout_worst_margin": None,
                "heldout_consistency": None,
                "confidence_admitted_count": None,
                "eligible_count": None,
            }
        )
    values["source_checkpoint_sha256"] = evidence.source_checkpoint_sha256
    values["variant"] = evidence.cell.variant
    values["dataset"] = evidence.cell.dataset
    values["scenario"] = evidence.cell.scenario
    values["source_seed"] = int(evidence.cell.source_seed)
    values["training_view_seed"] = int(evidence.cell.training_view_seed)
    values["heldout_test_seed"] = int(evidence.cell.test_seed)
    values["test_seed"] = int(evidence.cell.test_seed)
    values["training_view_family"] = evidence.cell.training_view_family
    values["physical_training_view_family"] = str(
        evidence.metadata.get(
            "physical_training_view_family", evidence.cell.training_view_family
        )
    )
    values["current_training_view_family"] = str(
        evidence.metadata.get(
            "current_training_view_family", CURRENT_SSAW_DIRECTION_FAMILY
        )
    )
    values["heldout_direction_family"] = str(
        evidence.metadata.get(
            "heldout_direction_family", HELDOUT_SSAW_DIRECTION_FAMILY
        )
    )
    values["mechanism_training_view_family"] = str(
        evidence.metadata.get(
            "mechanism_training_view_family", values["current_training_view_family"]
        )
    )
    values["mechanism_heldout_direction_family"] = str(
        evidence.metadata.get(
            "mechanism_heldout_direction_family", values["heldout_direction_family"]
        )
    )
    values["dusafe_variant"] = str(
        evidence.metadata.get("dusafe_variant", "")
    )
    values["training_view_config_sha256"] = str(
        evidence.metadata.get("training_view_config_sha256", "")
    )
    values["heldout_sobol_seed"] = evidence.metadata.get("heldout_sobol_seed")
    values["heldout_direction_seed"] = evidence.metadata.get(
        "heldout_direction_seed", evidence.metadata.get("heldout_sobol_seed")
    )
    values["heldout_sobol_seed_overlap"] = bool(
        evidence.metadata.get("heldout_sobol_seed_overlap", False)
    )
    values["heldout_candidate_count"] = evidence.metadata.get(
        "heldout_candidate_count"
    )
    values["heldout_direction_candidate_count"] = evidence.metadata.get(
        "heldout_direction_candidate_count",
        evidence.metadata.get("heldout_candidate_count"),
    )
    values["heldout_direction_bank_sha256"] = evidence.metadata.get(
        "heldout_direction_bank_sha256"
    )
    if "diag_ssaw_training_participation_rate" in evidence.metadata:
        values["diag_ssaw_training_participation_rate"] = evidence.metadata[
            "diag_ssaw_training_participation_rate"
        ]
    values["held_out_trajectory"] = evidence.cell.trajectory_id
    values["held_out_operator"] = evidence.cell.operator_id
    values["rotation_rate_unverified"] = bool(
        evidence.cell.dataset == "FD"
        and not any(
            evidence.metadata.get(name) is not None
            for name in (
                "rotation_frequency_hz",
                "operating_frequency_hz",
                "shaft_frequency_hz",
                "order_reference_hz",
            )
        )
    )
    values["artifact_path"] = evidence.artifact_path or ""
    partition, selection_overlap, confirmatory = evaluation_partition(
        evidence.cell.dataset, evidence.cell.scenario
    )
    values["target_labels_used_for_updates"] = False
    values["target_labels_used_for_parameter_selection"] = True
    values["parameter_selection_data_overlap"] = selection_overlap
    values["evaluation_partition"] = partition
    values["confirmatory"] = confirmatory
    return values


def validate_variant_row(cell: QueueCell, row: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed if a worker row is not keyed to its serialized cell."""

    if not isinstance(row, Mapping):
        raise TypeError("worker row must be a mapping")
    normalized = dict(row)
    expected = {
        "dataset": cell.dataset,
        "scenario": cell.scenario,
        "source_seed": cell.source_seed,
        "training_view_seed": cell.training_view_seed,
        "test_seed": cell.test_seed,
        "heldout_test_seed": cell.test_seed,
    }
    for name, value in expected.items():
        if name not in normalized:
            raise ValueError(f"worker row missing {name}")
        actual = normalized[name]
        if name in {"source_seed", "training_view_seed", "test_seed", "heldout_test_seed"}:
            if int(actual) != int(value):
                raise ValueError(f"worker row {name} disagrees with cell")
        elif str(actual) != str(value):
            raise ValueError(f"worker row {name} disagrees with cell")
    partition, selection_overlap, confirmatory = evaluation_partition(
        cell.dataset, cell.scenario
    )
    provenance_expected = {
        "target_labels_used_for_updates": False,
        "target_labels_used_for_parameter_selection": True,
        "parameter_selection_data_overlap": selection_overlap,
        "evaluation_partition": partition,
        "confirmatory": confirmatory,
    }
    for name, value in provenance_expected.items():
        if name not in normalized or normalized[name] != value:
            raise ValueError(f"worker row {name} disagrees with evaluation protocol")
    if _canonical_variant(normalized.get("variant")) != cell.variant:
        raise ValueError("worker returned the wrong variant for restoration key")
    if "training_view_family" in normalized and str(
        normalized["training_view_family"]
    ) != cell.training_view_family:
        raise ValueError("worker row training_view_family disagrees with cell")
    if "training_view_config_sha256" in normalized:
        provenance_hash = str(normalized["training_view_config_sha256"]).strip().lower()
        if provenance_hash and (
            len(provenance_hash) != 64
            or any(character not in "0123456789abcdef" for character in provenance_hash)
        ):
            raise ValueError(
                "worker row training_view_config_sha256 must be a 64-character hex digest"
            )
    return normalized


PAIRED_METRIC_COLUMNS = tuple(
    dict.fromkeys(
        (
            *COMMON_PREDICTIVE_METRICS,
            "triad_norm_relative_error",
            "jerk_relative_error",
            "energy_relative_error",
            "dominant_periodic_frequency_shift",
            "dominant_periodic_frequency_shift_cycles_per_sample",
            "dominant_periodic_frequency_shift_hz",
            "relative_bandpower_error",
            "spectral_coherence",
            "dominant_frequency_shift",
            "channel_correlation_distortion",
            "amplitude_envelope_correlation",
            "order_frequency_peak_shift",
            "raw_spectral_peak_shift_hz",
            "normalized_spectral_peak_shift_cycles_per_sample",
            "raw_envelope_spectral_peak_shift_hz",
            "normalized_envelope_spectral_peak_shift_cycles_per_sample",
            "envelope_spectrum_peak_shift",
            "spectral_kurtosis_change",
            "rms_ratio",
            "clean_f1",
            "heldout_f1",
            "source_label_accuracy_on_view",
            "prediction_label_agreement",
            "eligible_coverage",
            "margin_ratio",
            "heldout_flip_rate",
            "heldout_worst_margin",
            "heldout_consistency",
            "confidence_admitted_count",
            "eligible_count",
        )
    )
)


def aggregate_full_no_ssaw(
    records: Sequence[Mapping[str, Any] | VariantEvidence],
) -> list[Dict[str, Any]]:
    """Pair exact Full/no_ssaw cells and report Full-minus-no_ssaw deltas."""

    rows = [
        variant_metrics(record) if isinstance(record, VariantEvidence) else dict(record)
        for record in records
    ]
    grouped: Dict[tuple[Any, ...], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        required = {
            "dataset",
            "scenario",
            "source_seed",
            "training_view_seed",
            "test_seed",
            "heldout_test_seed",
            "variant",
            "source_checkpoint_sha256",
            "held_out_trajectory",
            "held_out_operator",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"variant row missing pairing columns: {missing}")
        if int(row["heldout_test_seed"]) != int(row["test_seed"]):
            raise ValueError("heldout_test_seed disagrees with test_seed")
        if _explicit_bool(row.get("label_leakage", False), field="label_leakage") or _explicit_bool(
            row.get("ground_truth_lpr_observed", False),
            field="ground_truth_lpr_observed",
        ):
            raise ValueError("paired aggregation rejects label leakage or ground-truth LPR claims")
        source_hash = str(row["source_checkpoint_sha256"])
        if len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash.lower()):
            raise ValueError("source_checkpoint_sha256 must be a 64-character hex digest")
        variant = _canonical_variant(row["variant"])
        key = (
            _canonical_dataset(row["dataset"]),
            str(row["scenario"]),
            int(row["source_seed"]),
            int(row["training_view_seed"]),
            int(row["test_seed"]),
            str(row.get("training_view_family", "")),
            str(row.get("training_view_config_sha256", "")),
            str(row["held_out_trajectory"]),
            str(row["held_out_operator"]),
        )
        bucket = grouped.setdefault(key, {})
        if variant in bucket:
            raise ValueError(f"duplicate variant row for pairing key {key}: {variant}")
        bucket[variant] = row
    paired = []
    for key in sorted(grouped):
        bucket = grouped[key]
        if set(bucket) != set(VARIANTS):
            raise ValueError(f"incomplete Full/no_ssaw pair for {key}: {set(bucket)}")
        full = bucket["Full"]
        no_ssaw = bucket["no_ssaw"]
        if str(full["source_checkpoint_sha256"]) != str(no_ssaw["source_checkpoint_sha256"]):
            raise ValueError(f"Full/no_ssaw source checkpoint mismatch for {key}")
        # Current mechanism evidence must use one shared, label-free spline
        # bank.  Legacy rows may omit these fields, but once either paired row
        # declares the new protocol all identity fields are mandatory and
        # equal across Full/no_ssaw.
        mechanism_fields = (
            "current_training_view_family",
            "heldout_direction_family",
            "mechanism_training_view_family",
            "mechanism_heldout_direction_family",
            "heldout_direction_seed",
            "heldout_candidate_count",
            "heldout_direction_candidate_count",
            "heldout_direction_bank_sha256",
            "heldout_sobol_seed_overlap",
        )
        declared_mechanism = any(
            str(row.get("heldout_direction_bank_sha256") or "").strip()
            for row in (full, no_ssaw)
        )
        if declared_mechanism:
            for field in mechanism_fields:
                if field not in full or field not in no_ssaw:
                    raise ValueError(
                        f"Full/no_ssaw mechanism provenance missing {field} for {key}"
                    )
                if full[field] != no_ssaw[field]:
                    raise ValueError(
                        f"Full/no_ssaw held-out direction mismatch for {key}: {field}"
                    )
            if bool(full["heldout_sobol_seed_overlap"]):
                raise ValueError(
                    f"held-out direction seed overlaps a training/test role for {key}"
                )
            bank_hash = str(full["heldout_direction_bank_sha256"] or "").strip().lower()
            if len(bank_hash) != 64 or any(
                character not in "0123456789abcdef" for character in bank_hash
            ):
                raise ValueError(
                    f"invalid held-out direction bank hash for {key}"
                )
        full_rotation_unverified = bool(full.get("rotation_rate_unverified", False))
        no_ssaw_rotation_unverified = bool(no_ssaw.get("rotation_rate_unverified", False))
        if full_rotation_unverified != no_ssaw_rotation_unverified:
            raise ValueError(f"Full/no_ssaw rotation-rate provenance mismatch for {key}")
        output: Dict[str, Any] = {
            "dataset": key[0],
            "scenario": key[1],
            "source_seed": key[2],
            "training_view_seed": key[3],
            "test_seed": key[4],
            "heldout_test_seed": key[4],
            "training_view_family": key[5],
            "training_view_config_sha256": key[6],
            "held_out_trajectory": key[7],
            "held_out_operator": key[8],
            "source_checkpoint_sha256": str(full["source_checkpoint_sha256"]),
            "variants_paired": "Full,no_ssaw",
            "rotation_rate_unverified": full_rotation_unverified,
        }
        for field in (
            "physical_training_view_family",
            "current_training_view_family",
            "heldout_direction_family",
            "mechanism_training_view_family",
            "mechanism_heldout_direction_family",
            "heldout_direction_seed",
            "heldout_sobol_seed_overlap",
            "heldout_candidate_count",
            "heldout_direction_candidate_count",
            "heldout_direction_bank_sha256",
        ):
            if field in full:
                output[field] = full[field]
        for name in (
            "target_labels_used_for_updates",
            "target_labels_used_for_parameter_selection",
            "parameter_selection_data_overlap",
            "evaluation_partition",
            "confirmatory",
        ):
            if name not in full or name not in no_ssaw:
                raise ValueError(f"paired rows lack selection provenance for {key}: {name}")
            if full.get(name) != no_ssaw.get(name):
                raise ValueError(
                    f"Full/no_ssaw selection provenance mismatch for {key}: {name}"
                )
            output[name] = full.get(name)
        for metric in PAIRED_METRIC_COLUMNS:
            if (
                metric not in full
                or metric not in no_ssaw
                or full[metric] is None
                or no_ssaw[metric] is None
            ):
                # Dataset-inapplicable physical metrics are not invented.  The
                # paired row records null for those columns and the applicable
                # mechanism metrics remain exact.
                output[f"full_{metric}"] = None
                output[f"no_ssaw_{metric}"] = None
                output[f"full_minus_no_ssaw_{metric}"] = None
                continue
            full_value = float(full[metric])
            no_value = float(no_ssaw[metric])
            if not math.isfinite(full_value) or not math.isfinite(no_value):
                raise ValueError(f"non-finite paired metric {metric} for {key}")
            output[f"full_{metric}"] = full_value
            output[f"no_ssaw_{metric}"] = no_value
            output[f"full_minus_no_ssaw_{metric}"] = full_value - no_value
        paired.append(output)
    return paired


def _extract_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, Mapping):
        logits = output.get("logits")
        features = output.get("features", output.get("feature", output.get("embedding")))
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        logits, features = output[0], output[1]
    else:
        logits = getattr(output, "logits", None)
        features = getattr(output, "features", getattr(output, "feature", None))
    if logits is None or features is None:
        raise ValueError("model output must expose both logits and features")
    if not isinstance(logits, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise TypeError("model logits/features must be tensors")
    return logits, features


def _predict_logits_features(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the existing PreTrainModel/TTA model without labels."""

    base = getattr(model, "model", model)
    feature_extractor = getattr(base, "feature_extractor", None)
    classifier = getattr(base, "classifier", None)
    with torch.no_grad():
        if feature_extractor is not None and classifier is not None:
            feature_output = feature_extractor(inputs)
            features = feature_output[0] if isinstance(feature_output, (tuple, list)) else feature_output
            logits = classifier(features)
        else:
            output = base(inputs)
            logits, features = _extract_output(output)
    if logits.size(0) != inputs.size(0) or features.size(0) != inputs.size(0):
        raise ValueError("model logits/features batch size mismatch")
    return logits.detach(), features.detach()


def _heldout_sobol_seed(cell: QueueCell) -> int:
    """Derive a stable direction-bank seed disjoint from training/test seeds."""

    material = (
        f"{cell.dataset}|{cell.scenario}|{cell.source_seed}|"
        f"{cell.training_view_seed}|{cell.test_seed}|heldout_sobol"
    ).encode("utf-8")
    candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    candidate %= 2_147_483_647
    if candidate < 1:
        candidate = 1
    # Include the source seed as well.  It is not the production Sobol seed,
    # but keeping all three roles disjoint makes provenance audits and replay
    # checks unambiguous.
    forbidden = {
        int(cell.source_seed),
        int(cell.training_view_seed),
        int(cell.test_seed),
    }
    while candidate in forbidden:
        candidate = (candidate % 2_147_483_646) + 1
    return int(candidate)


def _heldout_direction_bank_sha256(controls_by_view: torch.Tensor) -> str:
    """Hash the sampled spline controls, excluding model/data values.

    The hash is therefore a direction-bank identity rather than a hash of
    candidate logits.  Full and no_ssaw instantiate separate adapters but use
    the same cell seed and bank settings, so equal hashes are a strict paired
    check that their held-out candidate budget is identical.
    """

    if not isinstance(controls_by_view, torch.Tensor):
        raise TypeError("held-out direction controls must be a tensor")
    if controls_by_view.ndim != 3:
        raise ValueError("held-out direction controls must have shape [views,batch,controls]")
    return hashlib.sha256(
        (
            f"{tuple(controls_by_view.shape)}|{controls_by_view.dtype}|".encode(
                "utf-8"
            )
            + controls_by_view.detach().cpu().contiguous().numpy().tobytes()
        )
    ).hexdigest()


def _confidence_admitted_mask(
    trainer: Any,
    raw_logits: torch.Tensor,
    tta_config: Mapping[str, Any],
) -> torch.Tensor:
    """Reproduce fixed-source confidence admission without target labels."""

    if not bool(tta_config.get("enable_confidence_gate", True)):
        return torch.ones(raw_logits.size(0), dtype=torch.bool)
    metadata = getattr(trainer, "source_confidence_metadata", None)
    if not isinstance(metadata, Mapping) or "top1_nll" not in metadata:
        raise RuntimeError(
            "held-out Sobol diagnostics require source confidence metadata"
        )
    source_scores = torch.as_tensor(metadata["top1_nll"], dtype=torch.float32).reshape(-1)
    if source_scores.numel() < 1 or not torch.isfinite(source_scores).all():
        raise ValueError("source confidence metadata has invalid top1_nll")
    keep_fraction = float(tta_config.get("confidence_keep_fraction", 1.0))
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("confidence_keep_fraction must be in (0, 1]")
    threshold = torch.quantile(source_scores, keep_fraction)
    raw_nll = -raw_logits.float().log_softmax(dim=1).amax(dim=1)
    return raw_nll.le(threshold).cpu()


def _heldout_sobol_batch_diagnostics(
    trainer: Any,
    model: torch.nn.Module,
    cell: QueueCell,
    inputs: torch.Tensor,
    raw_logits: torch.Tensor,
    tta_config: Mapping[str, Any],
    generator: Any,
) -> Dict[str, torch.Tensor]:
    """Evaluate an unseen spline direction bank on one post-update batch."""

    from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView

    del cell
    if not isinstance(generator, UnifiedSplineHardView):
        raise TypeError("held-out direction generator must be UnifiedSplineHardView")
    source_loader = getattr(trainer, "src_train_dl", None)
    source_dataset = getattr(source_loader, "dataset", None)
    normalization_stats = getattr(source_dataset, "normalization_stats", None)
    if normalization_stats is None or len(normalization_stats) != 2:
        raise RuntimeError(
            "held-out Sobol diagnostics require fixed source normalization statistics"
        )
    prepared = generator.prepare_view_inputs(
        inputs,
        normalization_mean=normalization_stats[0],
        normalization_std=normalization_stats[1],
    )
    candidate_inputs = torch.as_tensor(prepared["view_inputs"])
    view_count, batch_size = candidate_inputs.shape[:2]
    candidate_logits, _candidate_features = _predict_logits_features(
        model,
        candidate_inputs.reshape(view_count * batch_size, *candidate_inputs.shape[2:]),
    )
    candidate_logits = candidate_logits.reshape(view_count, batch_size, -1).cpu()
    confidence_mask = _confidence_admitted_mask(trainer, raw_logits.cpu(), tta_config)
    return heldout_direction_diagnostics(
        raw_logits.cpu(), candidate_logits, confidence_mask=confidence_mask
    )


def extract_target_stream_evidence(
    trainer: Any,
    model: torch.nn.Module,
    cell: QueueCell,
    *,
    metadata: Mapping[str, Any],
    source_checkpoint_sha256: str,
    tta_config: Optional[Mapping[str, Any]] = None,
) -> VariantEvidence:
    """Extract paired clean/held-out outputs from ``trainer.trg_whole_dl``.

    True labels are copied only to the returned offline artifact.  They are
    not passed to the model or held-out operator.
    """

    case = validate_cell_metadata(cell, metadata)
    del case
    loader = getattr(trainer, "trg_whole_dl", None)
    if loader is None:
        raise ValueError("trainer must expose the target whole-stream loader")
    clean_signals = []
    held_out_signals = []
    clean_logits = []
    held_out_logits = []
    clean_features = []
    held_out_features = []
    labels = []
    heldout_confidence_admitted_masks = []
    heldout_eligible_masks = []
    heldout_margin_ratios = []
    heldout_flip_rates = []
    heldout_worst_margins = []
    heldout_consistencies = []
    heldout_direction_bank_hashes: list[str] = []
    heldout_generator = None
    if tta_config is not None:
        from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView

        heldout_generator = UnifiedSplineHardView(
            num_control_points=int(tta_config.get("spline_control_points", 10)),
            num_directions=int(tta_config.get("spline_num_directions", 4)),
            log_strength=float(tta_config.get("spline_log_strength", 0.2)),
            radius_levels=tuple(
                float(value)
                for value in tta_config.get(
                    "spline_radius_levels", HELDOUT_SSAW_DIRECTION_RADII
                )
            ),
            sobol_seed=_heldout_sobol_seed(cell),
        )
    modules = [model]
    if hasattr(model, "model"):
        modules.append(model.model)
    previous_modes = [module.training for module in modules if hasattr(module, "training")]
    try:
        if hasattr(model, "eval"):
            model.eval()
        for data, target, _index in loader:
            if target is None:
                raise ValueError("held-out evidence requires true target labels for offline F1/source-label-accuracy")
            if isinstance(data, (list, tuple)):
                raise ValueError("held-out evidence requires one tensor target stream")
            if not isinstance(data, torch.Tensor):
                raise TypeError("target stream data must be a torch.Tensor")
            device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
            data_device = data.float().to(device)
            clean_signal = data_device.detach().cpu()
            held_out_signal = apply_held_out_operator(
                cell.dataset,
                clean_signal,
                seed=cell.test_seed,
                trajectory_id=cell.trajectory_id,
                held_out_view_family=cell.held_out_view_family,
            )
            clean_output_logits, clean_output_features = _predict_logits_features(model, data_device)
            held_output_logits, held_output_features = _predict_logits_features(
                model, held_out_signal.to(device)
            )
            clean_signals.append(clean_signal)
            held_out_signals.append(held_out_signal)
            clean_logits.append(clean_output_logits.cpu())
            held_out_logits.append(held_output_logits.cpu())
            clean_features.append(clean_output_features.cpu())
            held_out_features.append(held_output_features.cpu())
            labels.append(torch.as_tensor(target).view(-1).long().cpu())
            if heldout_generator is not None:
                diagnostics = _heldout_sobol_batch_diagnostics(
                    trainer,
                    model,
                    cell,
                    data_device,
                    clean_output_logits,
                    tta_config or {},
                    heldout_generator,
                )
                prepared = heldout_generator.last_view_inputs
                controls = heldout_generator._cached_candidate_controls
                if controls is None:
                    raise RuntimeError(
                        "held-out direction generator did not retain candidate controls"
                    )
                heldout_direction_bank_hashes.append(
                    _heldout_direction_bank_sha256(controls)
                )
                heldout_confidence_admitted_masks.append(
                    diagnostics["confidence_admitted_mask"]
                )
                heldout_eligible_masks.append(diagnostics["eligible_mask"])
                heldout_margin_ratios.append(diagnostics["margin_ratio"])
                heldout_flip_rates.append(diagnostics["heldout_flip_rate"])
                heldout_worst_margins.append(diagnostics["heldout_worst_margin"])
                heldout_consistencies.append(diagnostics["heldout_consistency"])
    finally:
        for module, training in zip([module for module in modules if hasattr(module, "training")], previous_modes):
            module.train(training)
    if not labels:
        raise ValueError("target stream produced no batches")
    evidence_metadata = dict(metadata)
    if heldout_generator is not None:
        if not heldout_direction_bank_hashes:
            raise RuntimeError("held-out direction bank produced no batch hashes")
        evidence_metadata.update(
            {
                "heldout_direction_bank_sha256": hashlib.sha256(
                    "|".join(heldout_direction_bank_hashes).encode("ascii")
                ).hexdigest(),
                "heldout_direction_bank_batch_hashes": list(
                    heldout_direction_bank_hashes
                ),
                "heldout_direction_seed": int(heldout_generator.sobol_seed),
                "heldout_sobol_seed_overlap": bool(
                    int(heldout_generator.sobol_seed)
                    in {
                        int(cell.source_seed),
                        int(cell.training_view_seed),
                        int(cell.test_seed),
                    }
                ),
                "heldout_direction_candidate_count": int(
                    heldout_generator.candidate_count
                ),
                "heldout_direction_bank_spec": {
                    "family": HELDOUT_SSAW_DIRECTION_FAMILY,
                    "num_directions": int(heldout_generator.num_directions),
                    "signs": list(HELDOUT_SSAW_DIRECTION_SIGNS),
                    "radius_levels": list(heldout_generator.radius_levels),
                    "candidate_count": int(heldout_generator.candidate_count),
                },
            }
        )
    return VariantEvidence(
        cell=cell,
        source_checkpoint_sha256=source_checkpoint_sha256,
        clean_signal=torch.cat(clean_signals),
        held_out_signal=torch.cat(held_out_signals),
        clean_logits=torch.cat(clean_logits),
        held_out_logits=torch.cat(held_out_logits),
        clean_features=torch.cat(clean_features),
        held_out_features=torch.cat(held_out_features),
        labels=torch.cat(labels),
        metadata=evidence_metadata,
        heldout_confidence_admitted_mask=(
            torch.cat(heldout_confidence_admitted_masks)
            if heldout_confidence_admitted_masks
            else None
        ),
        heldout_eligible_mask=(
            torch.cat(heldout_eligible_masks) if heldout_eligible_masks else None
        ),
        heldout_margin_ratio=(
            torch.cat(heldout_margin_ratios) if heldout_margin_ratios else None
        ),
        heldout_flip_rate=(
            torch.cat(heldout_flip_rates) if heldout_flip_rates else None
        ),
        heldout_worst_margin=(
            torch.cat(heldout_worst_margins) if heldout_worst_margins else None
        ),
        heldout_consistency=(
            torch.cat(heldout_consistencies) if heldout_consistencies else None
        ),
    )


def adapt_target_stream_without_labels(
    trainer: Any, model: torch.nn.Module
) -> Dict[str, float]:
    """Run online TTA over the target stream without passing true labels.

    ``TTATrainer.evaluate`` is a reporting helper and includes labels in its
    model-input dictionary.  That is unsuitable for a label-free evidence
    queue even though DuSafe currently ignores the field.  This loop keeps
    the existing trainer loader and TTA model, but supplies only signal data
    and sample indices; labels are read later by
    :func:`extract_target_stream_evidence` solely for offline metrics.
    """

    loader = getattr(trainer, "trg_whole_dl", None)
    if loader is None:
        raise ValueError("trainer must expose the target whole-stream loader")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    batch_logs: list[Dict[str, float]] = []
    for data, _target, trg_idx in loader:
        if isinstance(data, (list, tuple)):
            raise ValueError("label-free held-out evidence requires one tensor target stream")
        if not isinstance(data, torch.Tensor):
            raise TypeError("target stream data must be a torch.Tensor")
        inputs = data.float().to(device)
        metadata = {
            "trg_idx": trg_idx.detach().cpu().tolist()
            if torch.is_tensor(trg_idx)
            else trg_idx
        }
        # No labels or target metadata are passed into the online algorithm.
        model({"data": inputs, "meta": metadata})
        log = getattr(model, "_last_batch_log", {})
        if isinstance(log, Mapping):
            numeric = {
                str(key): float(value)
                for key, value in log.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and math.isfinite(float(value))
            }
            if numeric:
                batch_logs.append(numeric)
    if not batch_logs:
        summary: Dict[str, float] = {}
    else:
        common_keys = set.intersection(*(set(row) for row in batch_logs))
        summary = {
            key: float(sum(row[key] for row in batch_logs) / len(batch_logs))
            for key in sorted(common_keys)
        }
    # Keep the diagnostics label-free and compatible with the main trainer's
    # naming convention so the queue can select a mechanism flow from Full
    # raw ``diag_ssaw_training_participation_rate`` without re-running labels.
    setattr(trainer, "last_batch_log_summary", summary)
    return summary


def load_hhar_frozen_overrides(frozen_dir: str | Path) -> Dict[str, Any]:
    """Load and validate HHAR's completed frozen state without tuning it."""

    root = Path(frozen_dir)
    manifest_path = root / "manifest.json"
    state_path = root / "state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        raise RuntimeError("HHAR frozen manifest/state is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not bool(state.get("completed")):
        raise RuntimeError("HHAR frozen state is not complete")
    if manifest.get("target_labels_used_for_selection") is not True:
        raise RuntimeError("HHAR frozen state must declare target-label selection")
    if tuple(manifest.get("evaluation_flows", ())) != tuple(HHAR_REPORTED_FLOWS):
        raise RuntimeError("HHAR frozen state does not match the formal five flows")
    signature = state.get("signature") or {}
    if tuple(signature.get("evaluation_flows", ())) != tuple(HHAR_REPORTED_FLOWS):
        raise RuntimeError("HHAR frozen-state signature has a different flow protocol")
    if manifest.get("confirmatory") is not False:
        raise RuntimeError("HHAR target-selected evaluation cannot be confirmatory")
    config = dict(state.get("tta_config") or {})
    required = {
        "learning_rate",
        "steps",
        "batch_size",
        "ssaw_auxiliary_weight",
        "ssaw_risk_temperature",
        "ssaw_kl_scale",
        "ssaw_strength",
    }
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(f"HHAR frozen config is incomplete: {missing}")
    return config


def wait_for_hhar_frozen_state(
    frozen_dir: str | Path,
    *,
    wait: bool,
    poll_seconds: float = 60.0,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    while True:
        try:
            return load_hhar_frozen_overrides(frozen_dir)
        except RuntimeError:
            if not wait:
                raise
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                raise TimeoutError("timed out waiting for HHAR frozen state")
            time.sleep(max(0.1, float(poll_seconds)))


def classify_worker_failure(returncode: int, output: str = "") -> str:
    text = str(output).lower()
    if "out of memory" in text or "cuda out of memory" in text:
        return "oom"
    if int(returncode) < 0:
        return "native_crash"
    # Windows native access violations and stack overflows are represented as
    # large unsigned return codes rather than negative signals.
    if int(returncode) in {0xC0000005, 0xC00000FD, 0xC0000409}:
        return "native_crash"
    return "failed"


def worker_command(
    cell: QueueCell,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    python_executable: Optional[str] = None,
) -> list[str]:
    executable = python_executable or sys.executable
    return [
        executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_heldout_ssaw_queue.py"),
        "--worker-cell-json",
        str(Path(config_path)),
        "--worker-output-dir",
        str(Path(output_dir)),
        "--worker-key",
        cell.key_string,
    ]


@dataclass
class QueueExecution:
    output_dir: Path
    cells: tuple[QueueCell, ...]
    metadata_by_dataset: Mapping[str, Mapping[str, Any]]
    configuration_provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    completed_keys: set[tuple[str, str, int, int, int, str]] = field(default_factory=set)
    failures: list[Dict[str, Any]] = field(default_factory=list)

    def manifest(self, *, status: str, current: Optional[QueueCell] = None) -> Dict[str, Any]:
        payload = queue_manifest(
            self.cells,
            metadata_by_dataset=self.metadata_by_dataset,
            output_dir=self.output_dir,
            dry_run=status == "planned",
            configuration_provenance=self.configuration_provenance,
        )
        payload.update(
            {
                "status": status,
                "completed_cells": len(self.completed_keys),
                "completed_keys": ["|".join(map(str, key)) for key in sorted(self.completed_keys)],
                "current_key": None if current is None else current.key_string,
                "failures": list(self.failures),
            }
        )
        return payload

    def publish(self, *, status: str, current: Optional[QueueCell] = None) -> None:
        atomic_write_json(self.manifest(status=status, current=current), self.output_dir / "manifest.json")


def _source_checkpoint_hash(trainer: Any, source_model: Any) -> str:
    """Return the canonical pre-adaptation model-state digest.

    Checkpoint files also contain serialization metadata, so their byte digest
    is not comparable with the canonical model-state digest published by the
    main-table and controlled-safety protocols.  The held-out protocol uses the
    state digest for cross-experiment source identity and records the file
    digest separately as provenance.
    """

    return state_dict_sha256(source_model)


def _source_checkpoint_file_hash(trainer: Any) -> Optional[str]:
    cache_path = trainer._pretrain_cache_path() if hasattr(trainer, "_pretrain_cache_path") else None
    if cache_path and Path(cache_path).is_file():
        return file_sha256(cache_path)
    return None


def _variant_tta_config(
    tta_config: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    """Apply the reviewed production Full/no-SSAW class switches.

    The old queue used ``ablation_mode=no_ssaw``.  That only disabled view
    generation after the trainer had selected the spline-residual class, and
    it allowed stale semantic-router keys to remain in the signed profile.
    Keep the source profile untouched and choose the two public production
    variants explicitly before constructing the model.
    """

    values = dict(tta_config)
    canonical = _canonical_variant(variant)
    values.update(
        {
            "dusafe_variant": (
                "spline_residual" if canonical == "Full" else "confidence_raw"
            ),
            "enable_ssaw": canonical == "Full",
            "enable_source_semantic_router": False,
        }
    )
    return values


def run_worker_cell(
    cell: QueueCell,
    *,
    data_path: str | Path,
    device: str,
    backbone: str,
    pretrain_cache_dir: str | Path,
    source_config: Mapping[str, Any],
    tta_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Run one real trainer/checkpoint/target-stream cell.

    Imports are local so queue planning/tests remain CPU-only and do not
    initialize the trainer or a CUDA context.
    """

    if str(device).strip().lower() != "cpu" and not str(device).strip().lower().startswith("cuda"):
        raise ValueError("device must be cpu or cuda:<index>")
    if "ssaw_sobol_seed" not in tta_config:
        raise ValueError("worker TTA config must declare ssaw_sobol_seed")
    actual_training_view_seed = int(tta_config["ssaw_sobol_seed"])
    if actual_training_view_seed == cell.test_seed:
        raise ValueError(
            "frozen tta_config.ssaw_sobol_seed must differ from heldout_test_seed"
        )
    if actual_training_view_seed != cell.training_view_seed:
        raise ValueError(
            "worker cell training_view_seed disagrees with frozen tta_config.ssaw_sobol_seed"
        )
    # Lazy imports are important: importing the queue for dry-run must not
    # pull the online algorithm or instantiate a device.
    from scripts.supplementary_utils import build_trainer, cleanup_trainer, create_tta_model

    effective_tta_config = _variant_tta_config(tta_config, cell.variant)
    trainer = build_trainer(
        data_path=str(data_path),
        device=str(device),
        dataset=cell.dataset,
        da_method="DuSafe",
        backbone=str(backbone),
        exp_name="heldout_ssaw_queue",
        seed=cell.test_seed,
        source_seed=cell.source_seed,
        pretrain_cache_dir=str(pretrain_cache_dir),
        # The variant is selected through effective_tta_config below.  Do not
        # use the legacy ablation preset: it resolves through the retired
        # semantic-router switches and cannot select confidence_raw.
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        trainer.source_hparams.update(dict(source_config))
        trainer.set_runtime_hparams(effective_tta_config)
        adapted, source_model = create_tta_model(
            trainer,
            cell.source,
            cell.target,
            run_seed=cell.test_seed,
        )
        # Hash the source checkpoint before any target-time update can mutate
        # the shared model object.  Prefer the durable pretrain-cache file;
        # the state-dict fallback is therefore also a pre-adaptation hash.
        source_hash = _source_checkpoint_hash(trainer, source_model)
        source_file_hash = _source_checkpoint_file_hash(trainer)
        # Run target-time updates through a label-free input path.  The
        # trainer's reporting evaluate() helper carries labels for metrics,
        # so it is deliberately not used here.
        adaptation_diagnostics = adapt_target_stream_without_labels(
            trainer, adapted
        )
        # Pairing provenance describes the common Full direction protocol,
        # not the no-SSAW branch switch.  Both rows must therefore hash the
        # same spline settings while the row separately records its selected
        # public variant.
        provenance_config = dict(effective_tta_config)
        provenance_config.update(
            {
                "dusafe_variant": "spline_residual",
                "enable_ssaw": True,
                "enable_source_semantic_router": False,
            }
        )
        provenance = training_view_provenance(
            cell.dataset, provenance_config
        )
        evidence_metadata = dict(metadata)
        evidence_metadata.update(
            {
                f"diag_{key}": value
                for key, value in adaptation_diagnostics.items()
            }
        )
        evidence_metadata.update(
            {
                "training_view_family": provenance["training_view_family"],
                "physical_training_view_family": provenance[
                    "physical_training_view_family"
                ],
                "current_training_view_family": provenance[
                    "current_training_view_family"
                ],
                "heldout_direction_family": provenance[
                    "heldout_direction_family"
                ],
                "mechanism_training_view_family": CURRENT_SSAW_DIRECTION_FAMILY,
                "mechanism_heldout_direction_family": HELDOUT_SSAW_DIRECTION_FAMILY,
                "physical_operator_family": cell.held_out_view_family,
                "training_view_config_sha256": provenance["sha256"],
                "dusafe_variant": effective_tta_config["dusafe_variant"],
                "heldout_sobol_seed": _heldout_sobol_seed(cell),
                "heldout_direction_seed": _heldout_sobol_seed(cell),
                "heldout_sobol_seed_overlap": False,
                "heldout_candidate_count": int(
                    2
                    * int(effective_tta_config.get("spline_num_directions", 4))
                    * len(
                        tuple(
                            effective_tta_config.get(
                                "spline_radius_levels", (1.0, 0.5, 0.25)
                            )
                        )
                    )
                ),
                "heldout_direction_protocol": (
                    "post_update_label_free_unseen_spline_amplitude_sobol_directions; "
                    "eligible=confidence_admitted AND preserving AND "
                    "0<heldout_margin<raw_margin"
                ),
            }
        )
        evidence = extract_target_stream_evidence(
            trainer,
            adapted,
            cell,
            metadata=evidence_metadata,
            source_checkpoint_sha256=source_hash,
            tta_config=effective_tta_config,
        )
        artifact_path = Path(output_dir) / "cells" / f"{cell_file_stem(cell)}.npz"
        save_variant_evidence(evidence, artifact_path)
        evidence.artifact_path = str(artifact_path)
        row = variant_metrics(evidence)
        row["artifact_path"] = str(artifact_path)
        row["source_checkpoint_file_sha256"] = source_file_hash
        row["label_leakage"] = False
        row["true_labels_artifact"] = True
        return row
    finally:
        cleanup_trainer(trainer, adapted, source_model, close_summary=True)


def run_subprocess_cell(
    cell: QueueCell,
    *,
    command_config_path: str | Path,
    output_dir: str | Path,
    log_path: str | Path,
    max_attempts: int = 3,
    gpu_lock_path: Optional[str | Path] = None,
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Execute one worker with retries and durable OOM/native-crash evidence."""

    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be positive")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, int(max_attempts) + 1):
        command = worker_command(cell, config_path=command_config_path, output_dir=output_dir)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nATTEMPT {attempt}/{max_attempts} COMMAND {json.dumps(command)}\n")
            log.flush()
            context = (
                wait_for_gpu_experiment_lock(gpu_lock_path)
                if gpu_lock_path
                else _NullContext()
            )
            with context:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            log.write(f"RETURN_CODE {completed.returncode}\n")
        output_tail = ""
        try:
            output_tail = log_path.read_text(encoding="utf-8")[-4000:]
        except OSError:
            pass
        if completed.returncode == 0:
            return True, None
        failure_kind = classify_worker_failure(completed.returncode, output_tail)
        failure = {
            "cell_key": cell.key_string,
            "attempt": attempt,
            "max_attempts": int(max_attempts),
            "returncode": int(completed.returncode),
            "failure_kind": failure_kind,
            "retryable": attempt < int(max_attempts),
            "log": str(log_path),
            "output_tail": output_tail,
        }
        if attempt >= int(max_attempts):
            return False, failure
    return False, {"cell_key": cell.key_string, "failure_kind": "failed"}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


# Keep the historical held-out queue export, but use the exact lock primitive
# used by the main-table runner, physical/baseline queues, and supervisor.  The
# old implementation used an advisory byte-range lock on Windows; that did not
# conflict with the repository-wide O_EXCL lock and could allow two GPU stages
# to run concurrently.  This alias deliberately preserves the call sites and
# makes every queue contend on the same recoverable lock semantics.
GlobalGpuLock = GPUExperimentLock


def restore_completed_keys(manifest_path: str | Path, cells: Sequence[QueueCell]) -> set[tuple[str, str, int, int, int, str]]:
    """Read only completed keys with a durable, matching row artifact.

    A manifest key alone is not sufficient for resumption: a process can be
    interrupted after publishing the manifest but before writing its cell
    JSON.  Such a key must be rerun rather than silently skipped and removed
    from the final Full/no_ssaw pair.
    """

    path = Path(manifest_path)
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = {cell.key for cell in cells}
    restored = set()
    for raw in payload.get("completed_keys", []):
        parts = str(raw).split("|")
        if len(parts) not in {5, 6}:
            continue
        try:
            source_part = parts[2][3:] if parts[2].startswith("src") else parts[2]
            if len(parts) == 6:
                view_part = parts[3][4:] if parts[3].startswith("view") else parts[3]
                seed_part = parts[4][4:] if parts[4].startswith("seed") else parts[4]
                variant_part = parts[5]
                key = (
                    parts[0],
                    parts[1],
                    int(source_part),
                    int(view_part),
                    int(seed_part),
                    variant_part,
                )
            else:
                # Legacy manifests did not encode the SSAW view seed and are
                # intentionally not restorable into the strict new queue.
                continue
        except ValueError:
            continue
        if key in known:
            cell = next(cell for cell in cells if cell.key == key)
            artifact = path.parent / "cells" / f"{cell_file_stem(cell)}.json"
            try:
                artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                artifact_payload.get("completed") is True
                and str(artifact_payload.get("cell_key")) == cell.key_string
                and isinstance(artifact_payload.get("row"), Mapping)
            ):
                restored.add(key)
    return restored


def execute_queue_plan(
    execution: QueueExecution,
    *,
    dry_run: bool = True,
    worker: Optional[Callable[[QueueCell], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run/inject workers with resumable atomic state; dry-run never executes."""

    execution.output_dir.mkdir(parents=True, exist_ok=True)
    execution.publish(status="planned" if dry_run else "running")
    if dry_run:
        return execution.manifest(status="planned")
    if worker is None:
        raise ValueError("a worker callback is required for in-process queue execution")
    for cell in execution.cells:
        if cell.key in execution.completed_keys:
            continue
        execution.publish(status="running", current=cell)
        try:
            row = validate_variant_row(cell, worker(cell))
            execution.completed_keys.add(cell.key)
            atomic_write_json(
                {
                    "cell_key": cell.key_string,
                    "row": row,
                    "completed": True,
                },
                    execution.output_dir / "cells" / f"{cell_file_stem(cell)}.json",
            )
        except Exception as exc:
            execution.failures.append(
                {
                    "cell_key": cell.key_string,
                    "failure_kind": classify_worker_failure(1, repr(exc)),
                    "error": repr(exc),
                }
            )
            execution.publish(status="failed", current=cell)
            return execution.manifest(status="failed", current=cell)
        execution.publish(status="running")
    status = "complete" if len(execution.completed_keys) == len(execution.cells) else "partial"
    execution.publish(status=status)
    return execution.manifest(status=status)


__all__ = [
    "DATASETS",
    "DEFAULT_TEST_SEED",
    "DEFAULT_TRAINING_VIEW_SEED",
    "GlobalGpuLock",
    "LABEL_LEAKAGE_FLAGS",
    "OPERATOR_FAMILIES",
    "OperatorSpec",
    "PAIRED_METRIC_COLUMNS",
    "QUEUE_PROTOCOL_VERSION",
    "QueueCell",
    "QueueExecution",
    "SOURCE_SEEDS",
    "TRAINING_VIEW_FAMILIES",
    "VARIANTS",
    "VariantEvidence",
    "aggregate_full_no_ssaw",
    "adapt_target_stream_without_labels",
    "apply_held_out_operator",
    "atomic_write_json",
    "build_queue_cells",
    "cell_file_stem",
    "cell_key",
    "default_training_view_seed",
    "evaluation_partition",
    "training_view_provenance",
    "classify_worker_failure",
    "execute_queue_plan",
    "extract_target_stream_evidence",
    "file_sha256",
    "load_hhar_frozen_overrides",
    "load_variant_evidence",
    "make_held_out_operator",
    "operator_spec",
    "queue_manifest",
    "registered_flows",
    "restore_completed_keys",
    "run_subprocess_cell",
    "run_worker_cell",
    "save_variant_evidence",
    "state_dict_sha256",
    "validate_cell_metadata",
    "validate_cell_plan",
    "variant_metrics",
    "validate_variant_row",
    "wait_for_hhar_frozen_state",
    "worker_command",
]
