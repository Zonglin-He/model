import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

from dataloader.augmentations import magnitude_warp, time_warp
from dataloader.corruption_transforms import amplitude_drift


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def gaussian_noise_view(x: torch.Tensor, noise_scale: float = 0.15) -> torch.Tensor:
    channel_std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(x) * (channel_std * noise_scale)
    return x + noise


def magnitude_warp_view(x: torch.Tensor, sigma: float = 0.2, knot: int = 8) -> torch.Tensor:
    out = magnitude_warp(x.detach().cpu(), sigma=sigma, knot=knot)
    return torch.as_tensor(out, device=x.device, dtype=x.dtype)


def time_warp_view(x: torch.Tensor, sigma: float = 0.2, knot: int = 8) -> torch.Tensor:
    out = time_warp(x.detach().cpu(), sigma=sigma, knot=knot)
    return torch.as_tensor(out, device=x.device, dtype=x.dtype)


def pgd_entropy_attack(model, x: torch.Tensor, eps: float = 0.1, steps: int = 10) -> torch.Tensor:
    x_adv = x.clone().detach()
    alpha = eps / max(1, steps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        feats, _ = model.feature_extractor(x_adv)
        logits = model.classifier(feats)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
        grad = torch.autograd.grad(entropy, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = (x + delta).detach()
    return x_adv


def build_view_bank(tta_model, raw_x: torch.Tensor, pgd_eps: float = 0.1, pgd_steps: int = 10) -> Dict[str, torch.Tensor]:
    model = tta_model.model
    return {
        "original": raw_x,
        "gaussian_noise": gaussian_noise_view(raw_x),
        "magnitude_warp": magnitude_warp_view(raw_x),
        "time_warp": time_warp_view(raw_x),
        "pgd_entropy": pgd_entropy_attack(model, raw_x, eps=pgd_eps, steps=pgd_steps),
        "amplitude_drift_ref": amplitude_drift(raw_x, "moderate"),
        "ssaw": tta_model.get_adversarial_view(raw_x, model),
    }


def signal_energy_ratios(signal: torch.Tensor, low_frac: float = 0.1, high_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    power = torch.fft.rfft(signal, dim=-1).abs().pow(2)
    bins = power.size(-1)
    low_bins = max(1, int(math.ceil(bins * low_frac)))
    high_bins = max(1, int(math.ceil(bins * high_frac)))
    total = power.sum(dim=-1).clamp_min(1e-8)
    low = power[..., :low_bins].sum(dim=-1) / total
    high = power[..., -high_bins:].sum(dim=-1) / total
    return low.detach().cpu().numpy(), high.detach().cpu().numpy()


def total_variation(signal: torch.Tensor) -> np.ndarray:
    return signal.diff(dim=-1).abs().mean(dim=(-1, -2)).detach().cpu().numpy()


def second_diff_energy(signal: torch.Tensor) -> np.ndarray:
    second = signal.diff(dim=-1, n=2)
    return second.pow(2).mean(dim=(-1, -2)).detach().cpu().numpy()


def mse_to_original(signal: torch.Tensor, raw_x: torch.Tensor) -> np.ndarray:
    return (signal - raw_x).pow(2).mean(dim=(-1, -2)).detach().cpu().numpy()


def feature_distance(model, signal: torch.Tensor, raw_x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        feats, _ = model.feature_extractor(signal)
        raw_feats, _ = model.feature_extractor(raw_x)
        return (1.0 - F.cosine_similarity(F.normalize(feats, dim=1), F.normalize(raw_feats, dim=1), dim=1)).detach().cpu().numpy()


def try_import_dtw():
    try:
        from tslearn.metrics import dtw

        return dtw
    except Exception:
        return None


def fallback_dtw(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dp = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1])


def dtw_to_original(signal: torch.Tensor, raw_x: torch.Tensor) -> np.ndarray:
    dtw_impl = try_import_dtw() or fallback_dtw
    signal_np = signal.mean(dim=1).detach().cpu().numpy()
    raw_np = raw_x.mean(dim=1).detach().cpu().numpy()
    return np.asarray([dtw_impl(cur, ref) for cur, ref in zip(signal_np, raw_np)], dtype=np.float32)


def mean_power_spectrum(signal: torch.Tensor) -> np.ndarray:
    power = torch.fft.rfft(signal, dim=-1).abs().pow(2)
    return power.mean(dim=(0, 1)).detach().cpu().numpy()
