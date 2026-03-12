import math

import torch


SEVERITY_LEVELS = ("mild", "moderate", "severe")


def _validate_input(x: torch.Tensor, severity: str):
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"Unknown severity '{severity}'. Expected one of {SEVERITY_LEVELS}.")


def _num_channels(c: int, severity: str, severe_ratio: float):
    if severity == "mild":
        return 1
    if severity == "moderate":
        return int(math.ceil(c * 0.3))
    return int(math.ceil(c * severe_ratio))


def signal_freeze(x: torch.Tensor, severity: str):
    _validate_input(x, severity)
    ratios = {"mild": 0.10, "moderate": 0.30, "severe": 0.60}
    out = x.clone()
    freeze_len = max(1, int(round(out.size(-1) * ratios[severity])))
    pivot = max(1, out.size(-1) - freeze_len)
    out[..., pivot:] = out[..., pivot - 1 : pivot].expand_as(out[..., pivot:])
    return out


def channel_dropout(x: torch.Tensor, severity: str):
    _validate_input(x, severity)
    out = x.clone()
    batch, channels, _ = out.shape
    num_drop = min(channels, _num_channels(channels, severity, 0.6))
    for batch_idx in range(batch):
        chosen = torch.randperm(channels, device=out.device)[:num_drop]
        out[batch_idx, chosen, :] = 0.0
    return out


def amplitude_drift(x: torch.Tensor, severity: str):
    _validate_input(x, severity)
    end = {"mild": 1.2, "moderate": 1.5, "severe": 2.0}[severity]
    ramp = torch.linspace(1.0, end, x.size(-1), device=x.device, dtype=x.dtype)
    return x * ramp.view(1, 1, -1)


def piecewise_scaling(x: torch.Tensor, severity: str):
    _validate_input(x, severity)
    ranges = {
        "mild": (0.8, 1.2),
        "moderate": (0.6, 1.4),
        "severe": (0.3, 1.8),
    }
    low, high = ranges[severity]
    out = x.clone()
    batch, _, steps = out.shape
    boundaries = torch.linspace(0, steps, steps=5, device=out.device).round().long()
    scales = torch.empty(batch, 4, device=out.device, dtype=out.dtype).uniform_(low, high)
    for segment_idx in range(4):
        start = int(boundaries[segment_idx].item())
        end = int(boundaries[segment_idx + 1].item())
        out[:, :, start:end] = out[:, :, start:end] * scales[:, segment_idx].view(batch, 1, 1)
    return out


def burst_noise(x: torch.Tensor, severity: str):
    _validate_input(x, severity)
    settings = {
        "mild": (0.05, 3.0),
        "moderate": (0.10, 5.0),
        "severe": (0.20, 8.0),
    }
    ratio, mult = settings[severity]
    out = x.clone()
    batch, channels, steps = out.shape
    num_steps = max(1, int(round(steps * ratio)))
    channel_std = out.std(dim=-1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(out) * (channel_std * mult)
    for batch_idx in range(batch):
        chosen = torch.randperm(steps, device=out.device)[:num_steps]
        out[batch_idx, :, chosen] = out[batch_idx, :, chosen] + noise[batch_idx, :, chosen]
    return out


def sensor_disconnect(x: torch.Tensor, severity: str):
    _validate_input(x, severity)
    out = x.clone()
    batch, channels, _ = out.shape
    num_drop = min(channels, _num_channels(channels, severity, 0.5))
    for batch_idx in range(batch):
        chosen = torch.randperm(channels, device=out.device)[:num_drop]
        out[batch_idx, chosen, :] = 0.0
    return out


CORRUPTION_REGISTRY = {
    "signal_freeze": signal_freeze,
    "channel_dropout": channel_dropout,
    "amplitude_drift": amplitude_drift,
    "piecewise_scaling": piecewise_scaling,
    "burst_noise": burst_noise,
    "sensor_disconnect": sensor_disconnect,
}
