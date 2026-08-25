"""Index-stable corruptions for the HAR 12->16 replication panel.

This mirrors the Sleep-EDF cross-dataset protocol at the level of relative
duration and gain.  It does not reuse SSAW Sobol directions or spline curves.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
import torch


DATASET = "HAR"
SCENARIO = "12->16"
TARGET_SAMPLES = 110
WINDOW_SAMPLES = 128
SAMPLING_RATE_HZ = 50
CORRUPTION_FRACTION = 0.5
CORRUPTION_SEED = 314159
GEOMETRY_SEED = 334159
SEVERITIES = ("s3", "s6")
CORRUPTIONS = (
    "blackout",
    "signal_freeze",
    "smooth_gain_drift",
    "localized_attenuation",
)

_BLACKOUT_FRACTION = {"s3": 0.10, "s6": 0.30}
_FREEZE_FRACTION = {"s3": 0.20, "s6": 0.60}
_GAIN_DRIFT_RATIO = {"s3": 1.35, "s6": 2.00}
_LOCAL_ATTENUATION = {
    "s3": {"duration_fraction": 0.10, "minimum_gain": 0.65},
    "s6": {"duration_fraction": 0.30, "minimum_gain": 0.20},
}


def _uint64_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _mask_score(index: int, seed: int = CORRUPTION_SEED) -> int:
    return _uint64_hash(f"{DATASET}|{SCENARIO}|{seed}|mask|{int(index)}")


def corrupted_indices(
    total_samples: int = TARGET_SAMPLES,
    fraction: float = CORRUPTION_FRACTION,
    seed: int = CORRUPTION_SEED,
) -> tuple[int, ...]:
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    count = int(round(total_samples * float(fraction)))
    ranked = sorted(
        range(total_samples),
        key=lambda index: (_mask_score(index, seed), index),
    )
    return tuple(sorted(ranked[:count]))


CORRUPTED_INDICES = frozenset(corrupted_indices())


def corruption_mask_sha256() -> str:
    mask = np.zeros(TARGET_SAMPLES, dtype=np.uint8)
    mask[list(CORRUPTED_INDICES)] = 1
    return hashlib.sha256(mask.tobytes()).hexdigest()


def exact_index_stable_mask_fn(fraction: float, seed: int):
    if not math.isclose(float(fraction), CORRUPTION_FRACTION):
        raise ValueError(
            f"This protocol fixes corruption_fraction={CORRUPTION_FRACTION}"
        )
    if int(seed) != CORRUPTION_SEED:
        raise ValueError(f"This protocol fixes corruption_seed={CORRUPTION_SEED}")

    def make_mask(data, labels, indices, step, total_steps):
        del data, labels, step, total_steps
        index_tensor = torch.as_tensor(indices, dtype=torch.int64).view(-1)
        if index_tensor.numel() and (
            int(index_tensor.min()) < 0
            or int(index_tensor.max()) >= TARGET_SAMPLES
        ):
            raise ValueError("target index falls outside the registered HAR stream")
        return torch.tensor(
            [int(index) in CORRUPTED_INDICES for index in index_tensor.tolist()],
            dtype=torch.bool,
        )

    return make_mask


def _geometry_unit(index: int, key: str) -> float:
    value = _uint64_hash(
        f"{DATASET}|{SCENARIO}|{GEOMETRY_SEED}|{int(index)}|{key}"
    )
    return value / float(2**64 - 1)


def _validate(
    inputs: torch.Tensor,
    severity: str,
    indices: Iterable[int],
) -> list[int]:
    if inputs.ndim != 3:
        raise ValueError(f"expected [batch, channels, time], got {tuple(inputs.shape)}")
    if inputs.size(-1) != WINDOW_SAMPLES:
        raise ValueError(
            f"HAR protocol expects T={WINDOW_SAMPLES}, got {inputs.size(-1)}"
        )
    if str(severity) not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; expected {SEVERITIES}")
    index_list = [int(index) for index in indices]
    if len(index_list) != inputs.size(0):
        raise ValueError("one target index is required per input sample")
    return index_list


def _sample_count(fraction: float) -> int:
    return int(round(WINDOW_SAMPLES * float(fraction)))


def _centered_interval(
    index: int,
    key: str,
    length: int,
    maximum_length: int,
) -> tuple[int, int]:
    half_maximum = maximum_length // 2
    available = WINDOW_SAMPLES - maximum_length
    center = half_maximum + int(round(_geometry_unit(index, key) * available))
    start = center - length // 2
    return start, start + length


def blackout(
    inputs: torch.Tensor,
    severity: str,
    indices: Iterable[int],
) -> torch.Tensor:
    index_list = _validate(inputs, severity, indices)
    output = inputs.clone()
    length = _sample_count(_BLACKOUT_FRACTION[str(severity)])
    maximum_length = _sample_count(_BLACKOUT_FRACTION["s6"])
    for row, index in enumerate(index_list):
        start, end = _centered_interval(
            index, "blackout_center", length, maximum_length
        )
        output[row, :, start:end] = 0.0
    return output


def signal_freeze(
    inputs: torch.Tensor,
    severity: str,
    indices: Iterable[int],
) -> torch.Tensor:
    _validate(inputs, severity, indices)
    output = inputs.clone()
    length = _sample_count(_FREEZE_FRACTION[str(severity)])
    pivot = WINDOW_SAMPLES - length
    output[..., pivot:] = output[..., pivot - 1 : pivot].expand_as(
        output[..., pivot:]
    )
    return output


def smooth_gain_drift(
    inputs: torch.Tensor,
    severity: str,
    indices: Iterable[int],
) -> torch.Tensor:
    index_list = _validate(inputs, severity, indices)
    ratio = float(_GAIN_DRIFT_RATIO[str(severity)])
    unit_time = torch.linspace(
        0.0,
        1.0,
        WINDOW_SAMPLES,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    signs = torch.tensor(
        [
            1.0 if _geometry_unit(index, "log_gain_sign") >= 0.5 else -1.0
            for index in index_list
        ],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    gain = (
        signs[:, None] * math.log(ratio) * unit_time[None, :]
    ).exp().unsqueeze(1)
    return inputs * gain


def _localized_gain(
    *,
    length: int,
    minimum_gain: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ramp = max(2, length // 4)
    plateau = length - 2 * ramp
    if plateau < 1:
        raise ValueError("localized attenuation interval is too short")
    phase = torch.linspace(0.0, math.pi, ramp, dtype=dtype, device=device)
    rise = 0.5 * (1.0 - phase.cos())
    shape = torch.cat(
        (rise, torch.ones(plateau, dtype=dtype, device=device), rise.flip(0))
    )
    return 1.0 - (1.0 - float(minimum_gain)) * shape


def localized_attenuation(
    inputs: torch.Tensor,
    severity: str,
    indices: Iterable[int],
) -> torch.Tensor:
    index_list = _validate(inputs, severity, indices)
    spec = _LOCAL_ATTENUATION[str(severity)]
    length = _sample_count(float(spec["duration_fraction"]))
    maximum_length = _sample_count(
        float(_LOCAL_ATTENUATION["s6"]["duration_fraction"])
    )
    local_gain = _localized_gain(
        length=length,
        minimum_gain=float(spec["minimum_gain"]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    output = inputs.clone()
    for row, index in enumerate(index_list):
        start, end = _centered_interval(
            index,
            "localized_attenuation_center",
            length,
            maximum_length,
        )
        output[row, :, start:end] *= local_gain.view(1, -1)
    return output


CORRUPTION_REGISTRY: Dict[
    str,
    Callable[[torch.Tensor, str, Iterable[int]], torch.Tensor],
] = {
    "blackout": blackout,
    "signal_freeze": signal_freeze,
    "smooth_gain_drift": smooth_gain_drift,
    "localized_attenuation": localized_attenuation,
}


def resolve_severity(corruption: str, severity: str) -> str:
    if str(corruption) not in CORRUPTION_REGISTRY:
        raise KeyError(f"unknown corruption {corruption!r}")
    if str(severity) not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return str(severity)


def physical_corruption_metadata(
    corruption: str,
    severity: str,
) -> Mapping[str, object]:
    severity = resolve_severity(corruption, severity)
    if corruption == "blackout":
        samples = _sample_count(_BLACKOUT_FRACTION[severity])
        parameters = {
            "blackout_fraction": samples / WINDOW_SAMPLES,
            "blackout_fraction_requested": _BLACKOUT_FRACTION[severity],
            "duration_fraction_actual": samples / WINDOW_SAMPLES,
            "duration_samples": samples,
            "duration_seconds": samples / SAMPLING_RATE_HZ,
            "minimum_gain": 0.0,
        }
    elif corruption == "signal_freeze":
        samples = _sample_count(_FREEZE_FRACTION[severity])
        parameters = {
            "frozen_fraction": samples / WINDOW_SAMPLES,
            "frozen_fraction_requested": _FREEZE_FRACTION[severity],
            "duration_fraction_actual": samples / WINDOW_SAMPLES,
            "duration_samples": samples,
            "duration_seconds": samples / SAMPLING_RATE_HZ,
        }
    elif corruption == "smooth_gain_drift":
        ratio = _GAIN_DRIFT_RATIO[severity]
        parameters = {
            "family": "independent_monotonic_log_gain",
            "full_window": True,
            "positive_endpoint_gain": ratio,
            "negative_endpoint_gain": 1.0 / ratio,
            "gain_envelope": [1.0 / ratio, ratio],
            "ssaw_sobol_or_spline_reused": False,
        }
    else:
        spec = _LOCAL_ATTENUATION[severity]
        samples = _sample_count(float(spec["duration_fraction"]))
        ramp = max(2, samples // 4)
        plateau = samples - 2 * ramp
        parameters = {
            "family": "cosine_ramp_local_attenuation",
            "duration_fraction": samples / WINDOW_SAMPLES,
            "duration_fraction_requested": spec["duration_fraction"],
            "duration_fraction_actual": samples / WINDOW_SAMPLES,
            "duration_samples": samples,
            "duration_seconds": samples / SAMPLING_RATE_HZ,
            "minimum_gain": spec["minimum_gain"],
            "ramp_plateau_ramp_samples": [ramp, plateau, ramp],
            "claim_limit": "controlled_proxy_for_contact_degradation",
        }
    return {
        "corruption": str(corruption),
        "severity_name": severity,
        "normalized_severity": 0.5 if severity == "s3" else 1.0,
        "physical_parameters": parameters,
    }


def _extract_primary(data):
    return data[0] if isinstance(data, (tuple, list)) else data


def _replace_primary(data, new_primary):
    if isinstance(data, tuple):
        items = list(data)
        items[0] = new_primary
        return tuple(items)
    if isinstance(data, list):
        items = list(data)
        items[0] = new_primary
        return items
    return new_primary


class IndexStableBatchTransformLoader:
    """Apply registered corruption geometry by immutable target index."""

    def __init__(
        self,
        base_loader,
        transform_fn,
        severity,
        sample_mask_fn=None,
        meta=None,
        transform_seed=None,
    ):
        del transform_seed
        self.base_loader = base_loader
        self.transform_fn = transform_fn
        self.severity = str(severity)
        self.sample_mask_fn = sample_mask_fn
        self.dataset = base_loader.dataset
        self.batch_size = getattr(base_loader, "batch_size", None)
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        total_steps = len(self)
        for step, batch in enumerate(self.base_loader):
            data, labels, indices = batch
            primary = _extract_primary(data)
            expected = torch.tensor(
                [
                    int(index) in CORRUPTED_INDICES
                    for index in torch.as_tensor(indices).view(-1).tolist()
                ],
                dtype=torch.bool,
            )
            if self.sample_mask_fn is not None:
                observed = torch.as_tensor(
                    self.sample_mask_fn(data, labels, indices, step, total_steps),
                    dtype=torch.bool,
                ).view(-1)
                if not torch.equal(observed.cpu(), expected):
                    raise RuntimeError("corruption mask deviated from registered mask")
            output = primary.clone()
            if expected.any():
                device_mask = expected.to(primary.device)
                selected_indices = torch.as_tensor(indices).view(-1)[expected]
                output[device_mask] = self.transform_fn(
                    output[device_mask],
                    self.severity,
                    selected_indices.tolist(),
                )
            self.meta["corruption_mask"] = expected.tolist()
            self.meta["corruption_severity"] = self.severity
            self.meta["corruption_transform_seed"] = GEOMETRY_SEED
            yield _replace_primary(data, output), labels, indices
