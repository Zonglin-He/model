"""Continuous held-out corruption transforms for the SSAW evidence panel.

These transforms are deliberately separate from the historical
``mild/moderate/severe`` registry.  A run must select one of the pre-registered
``s0`` ... ``s6`` points from :mod:`configs.ssaw_evaluation_protocol`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Tuple

import torch

from configs.ssaw_evaluation_protocol import PHYSICAL_SEVERITY_GRIDS, SeverityPoint


def _validate_input(inputs: torch.Tensor) -> None:
    if inputs.ndim != 3:
        raise ValueError(
            f"Expected a [batch, channels, time] tensor, got {tuple(inputs.shape)}"
        )
    if inputs.size(-1) < 2:
        raise ValueError("Physical corruptions require at least two time samples")


def resolve_severity(corruption: str, severity) -> SeverityPoint:
    try:
        points = PHYSICAL_SEVERITY_GRIDS[str(corruption)]
    except KeyError as exc:
        raise KeyError(f"Unknown physical corruption {corruption!r}") from exc
    if isinstance(severity, SeverityPoint):
        if severity not in points:
            raise ValueError(
                f"Severity point {severity.name!r} is not registered for {corruption}"
            )
        return severity
    name = str(severity)
    for point in points:
        if point.name == name:
            return point
    raise ValueError(
        f"Unknown severity {severity!r} for {corruption}; expected s0...s6"
    )


def _parameter(corruption: str, severity, key: str):
    point = resolve_severity(corruption, severity)
    try:
        return point.parameters[key]
    except KeyError as exc:
        raise KeyError(f"{corruption}/{point.name} has no parameter {key!r}") from exc


def signal_freeze(inputs: torch.Tensor, severity) -> torch.Tensor:
    _validate_input(inputs)
    fraction = float(_parameter("signal_freeze", severity, "frozen_fraction"))
    if fraction == 0.0:
        return inputs.clone()
    output = inputs.clone()
    freeze_length = max(1, int(round(output.size(-1) * fraction)))
    pivot = max(1, output.size(-1) - freeze_length)
    output[..., pivot:] = output[..., pivot - 1 : pivot].expand_as(
        output[..., pivot:]
    )
    return output


def blackout(inputs: torch.Tensor, severity) -> torch.Tensor:
    _validate_input(inputs)
    fraction = float(_parameter("blackout", severity, "blackout_fraction"))
    if fraction == 0.0:
        return inputs.clone()
    output = inputs.clone()
    length = max(1, int(round(output.size(-1) * fraction)))
    maximum_start = output.size(-1) - length
    for sample_index in range(output.size(0)):
        start = int(
            torch.randint(maximum_start + 1, (1,), device=output.device).item()
        )
        output[sample_index, :, start : start + length] = 0.0
    return output


def attenuation(inputs: torch.Tensor, severity) -> torch.Tensor:
    _validate_input(inputs)
    gain = float(_parameter("attenuation", severity, "gain"))
    return inputs.clone() if gain == 1.0 else inputs * gain


def amplitude_drift(inputs: torch.Tensor, severity) -> torch.Tensor:
    _validate_input(inputs)
    end_gain = float(_parameter("amplitude_drift", severity, "end_gain"))
    if end_gain == 1.0:
        return inputs.clone()
    gain = torch.linspace(
        1.0,
        end_gain,
        inputs.size(-1),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    return inputs * gain.view(1, 1, -1)


_PACKET_COUNTS = (0, 1, 2, 4, 8, 12, 16)


def packet_loss(inputs: torch.Tensor, severity) -> torch.Tensor:
    """Zero separated acquisition packets with an auditable missing budget."""

    _validate_input(inputs)
    point = resolve_severity("packet_loss", severity)
    fraction = float(point.parameters["missing_fraction"])
    if fraction == 0.0:
        return inputs.clone()
    level = int(point.name[1:])
    packet_count = min(_PACKET_COUNTS[level], inputs.size(-1))
    total_missing = max(1, int(round(inputs.size(-1) * fraction)))
    packet_count = min(packet_count, total_missing)
    base_length, remainder = divmod(total_missing, packet_count)
    lengths = [base_length + int(index < remainder) for index in range(packet_count)]
    # Allocate disjoint cells and place one packet inside each cell.  A random
    # offset within the cell gives reproducible variation under fork_rng while
    # preventing overlap from silently reducing the requested missing budget.
    boundaries = torch.linspace(
        0, inputs.size(-1), packet_count + 1, device=inputs.device
    ).round().long()
    output = inputs.clone()
    for sample_index in range(output.size(0)):
        for packet_index, length in enumerate(lengths):
            cell_start = int(boundaries[packet_index].item())
            cell_end = int(boundaries[packet_index + 1].item())
            available = max(length, cell_end - cell_start)
            maximum_offset = max(0, available - length)
            offset = (
                0
                if maximum_offset == 0
                else int(
                    torch.randint(
                        maximum_offset + 1, (1,), device=output.device
                    ).item()
                )
            )
            start = min(cell_start + offset, output.size(-1) - length)
            output[sample_index, :, start : start + length] = 0.0
    return output


def saturation(inputs: torch.Tensor, severity) -> torch.Tensor:
    _validate_input(inputs)
    clip_std = _parameter("saturation", severity, "clip_std")
    if clip_std is None:
        return inputs.clone()
    center = inputs.mean(dim=-1, keepdim=True)
    scale = inputs.std(dim=-1, keepdim=True).clamp_min(1e-6) * float(clip_std)
    return torch.maximum(torch.minimum(inputs, center + scale), center - scale)


PHYSICAL_CORRUPTION_REGISTRY: Dict[str, Callable[[torch.Tensor, object], torch.Tensor]] = {
    "signal_freeze": signal_freeze,
    "blackout": blackout,
    "attenuation": attenuation,
    "amplitude_drift": amplitude_drift,
    "packet_loss": packet_loss,
    "saturation": saturation,
}


def physical_corruption_metadata(corruption: str, severity) -> Mapping[str, object]:
    point = resolve_severity(corruption, severity)
    return {
        "corruption": str(corruption),
        "severity_name": point.name,
        "normalized_severity": float(point.normalized),
        "physical_parameters": dict(point.parameters),
    }
