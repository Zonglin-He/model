"""CPU-only held-out SSAW mechanism and physical-invariant evaluation.

This module is an offline evidence layer.  It deliberately does not import
the production SSAW implementation, trainer, corruption registry, or any
label-preservation/nuisance estimator.  A caller supplies a clean tensor, a
held-out operator/trajectory output, and an algorithm adapter.  The manifest
records the declared split so a training view cannot be silently scored as a
held-out view.

Tensor convention
-----------------
Signals use ``[batch, channels, time]``.  Logits use ``[batch, classes]`` and
features use ``[batch, ...]``; feature dimensions after the batch axis are
flattened per sample.  All calculations are CPU-only and use natural-log
divergences.

The physical quantities are operational definitions rather than claims about
real nuisance distributions.  In particular, this module never estimates or
reports an LPR (label-preservation rate).
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F


HELD_OUT_PROTOCOL_VERSION = "ssaw_heldout_mechanism_v1"
SUPPORTED_DATASETS = ("EEG", "HAR", "FD", "HHAR")

COMMON_PREDICTIVE_METRICS = (
    "js_divergence",
    "kl_clean_to_heldout",
    "kl_heldout_to_clean",
    "prediction_flip_rate",
    "margin_degradation",
    "confidence_drop",
    "feature_cosine_distance",
)

DATASET_MECHANISM_PROTOCOL: Dict[str, Dict[str, Any]] = {
    "EEG": {
        "training_view_family": "window_constant_channel_gain",
        "held_out_operators": (
            "smooth_channel_gain_drift",
            "local_channel_attenuation",
        ),
        "physical_metrics": (
            "relative_bandpower_error",
            "spectral_coherence",
            "dominant_frequency_shift",
            "channel_correlation_distortion",
            "amplitude_envelope_correlation",
        ),
        "required_metadata": ("sampling_rate_hz", "sampling_rate_provenance"),
        "frequency_axis": "hertz",
        "sampling_rate_optional": False,
        "rotation_rate_unverified": False,
    },
    "HAR": {
        "training_view_family": "window_constant_bounded_so3",
        "held_out_operators": ("smooth_so3_orientation_trajectory",),
        "physical_metrics": (
            "triad_norm_relative_error",
            "jerk_relative_error",
            "energy_relative_error",
            "dominant_periodic_frequency_shift",
        ),
        "required_metadata": ("sampling_rate_hz", "sampling_rate_provenance"),
        "frequency_axis": "hertz",
        "sampling_rate_optional": False,
        "rotation_rate_unverified": False,
    },
    "HHAR": {
        "training_view_family": "window_constant_bounded_so3",
        "held_out_operators": ("smooth_so3_orientation_trajectory",),
        "physical_metrics": (
            "triad_norm_relative_error",
            "jerk_relative_error",
            "energy_relative_error",
            "dominant_periodic_frequency_shift",
        ),
        "required_metadata": (),
        "frequency_axis": "cycles_per_sample_when_rate_unavailable",
        "sampling_rate_optional": True,
        "rotation_rate_unverified": False,
    },
    "FD": {
        "training_view_family": "window_constant_sensor_gain",
        "held_out_operators": (
            "smooth_sensor_response_drift",
            "bounded_filter_response_drift",
        ),
        "physical_metrics": (
            "raw_spectral_peak_shift_hz",
            "normalized_spectral_peak_shift_cycles_per_sample",
            "raw_envelope_spectral_peak_shift_hz",
            "normalized_envelope_spectral_peak_shift_cycles_per_sample",
            "order_frequency_peak_shift",
            "spectral_kurtosis_change",
            "rms_ratio",
        ),
        "required_metadata": (),
        "frequency_axis": "hertz_when_rate_available_else_cycles_per_sample",
        "sampling_rate_optional": True,
        "rotation_rate_unverified": True,
    },
}

# These are protocol defaults, not inferred properties of a recording.  A
# caller may replace them with ``bands_hz`` in metadata.  The highest band is
# deliberately below the common 100-Hz Nyquist limit used by the synthetic
# tests; validation still checks the actual Nyquist rate.
DEFAULT_EEG_BANDS_HZ: Dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

_EPS = 1.0e-12


@dataclass(frozen=True)
class HeldOutCase:
    """One explicitly separated training-view/held-out evaluation cell.

    ``training_seed`` belongs to source training or checkpoint construction;
    ``test_seed`` belongs only to held-out trajectory/operator sampling.  The
    two seeds must differ so a manifest cannot accidentally collapse the split
    into one random stream.  ``held_out_trajectory`` and
    ``held_out_operator`` are identifiers, not fabricated nuisance values.
    """

    dataset: str
    training_view_family: str
    held_out_view_family: str
    held_out_trajectory: str
    held_out_operator: str
    training_seed: int
    test_seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    algorithm: str = "unspecified"

    @property
    def trajectory_id(self) -> str:
        """Alias used by manifest consumers."""

        return self.held_out_trajectory

    @property
    def operator_id(self) -> str:
        """Alias used by manifest consumers."""

        return self.held_out_operator

    @property
    def held_out_seed(self) -> int:
        """Alias clarifying that ``test_seed`` drives held-out sampling."""

        return self.test_seed


def _canonical_dataset(dataset: str) -> str:
    value = str(dataset).strip().upper()
    if value == "MFD":
        value = "FD"
    if value not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}"
        )
    return value


def _non_empty_text(name: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"{name} must be a non-empty identifier")
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        raise ValueError(f"{name} must be a non-empty identifier")
    return text


def _as_finite_number(name: str, value: Any, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        requirement = "finite and > 0" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}")
    return number


def _reject_unfounded_claims(metadata: Mapping[str, Any]) -> None:
    """Reject metadata that would turn synthetic views into false claims."""

    forbidden_true = (
        "real_nuisance",
        "real_nuisance_observed",
        "real_distribution_match",
        "lpr",
        "label_preservation_rate",
        "lpr_estimated",
    )
    for key in forbidden_true:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, str):
            is_true = value.strip().lower() in {"true", "yes", "observed", "estimated"}
        else:
            is_true = bool(value)
        if is_true:
            raise ValueError(
                f"Metadata field {key!r} would make an unsupported real-nuisance/LPR claim"
            )


def _metadata_mapping(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is None:
        raise ValueError("held-out mechanism metadata is required")
    if not isinstance(metadata, Mapping):
        raise TypeError("held-out mechanism metadata must be a mapping")
    _reject_unfounded_claims(metadata)
    return metadata


def _sampling_rate_provenance(metadata: Mapping[str, Any]) -> str:
    for key in ("sampling_rate_provenance", "sample_rate_provenance", "fs_provenance"):
        if key in metadata:
            value = str(metadata[key]).strip()
            if value and value.lower() not in {"unknown", "unverified", "guessed", "inferred"}:
                return value
            break
    raise ValueError(
        "sampling rate requires non-empty verifiable sampling_rate_provenance"
    )


def _sampling_rate_optional(metadata: Mapping[str, Any]) -> Optional[float]:
    for key in ("sampling_rate_hz", "sample_rate_hz", "fs"):
        if key in metadata:
            if metadata[key] is None:
                return None
            value = _as_finite_number(key, metadata[key], positive=True)
            _sampling_rate_provenance(metadata)
            return value
    return None


def _sampling_rate(metadata: Mapping[str, Any]) -> float:
    value = _sampling_rate_optional(metadata)
    if value is not None:
        return value
    raise ValueError(
        "held-out mechanism metadata requires verifiable sampling_rate_hz "
        "and sampling_rate_provenance (or explicit fs aliases)"
    )


def _rotation_rate_optional(metadata: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "rotation_frequency_hz",
        "operating_frequency_hz",
        "shaft_frequency_hz",
        "order_reference_hz",
    ):
        if key in metadata:
            if metadata[key] is None:
                return None
            value = _as_finite_number(key, metadata[key], positive=True)
            provenance = str(metadata.get("rotation_rate_provenance", "")).strip()
            if not provenance or provenance.lower() in {"unknown", "unverified", "guessed", "inferred"}:
                raise ValueError(
                    "rotation frequency requires verifiable rotation_rate_provenance"
                )
            return value
    return None


def _rotation_rate(metadata: Mapping[str, Any]) -> float:
    value = _rotation_rate_optional(metadata)
    if value is not None:
        return value
    raise ValueError(
        "FD order metrics require verified rotation_frequency_hz; "
        "an order axis cannot be invented from a sampling rate"
    )


def _validate_signal_pair(
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    *,
    name: str = "signal",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(clean_signal, torch.Tensor) or not isinstance(
        held_out_signal, torch.Tensor
    ):
        raise TypeError(f"{name} tensors must be torch.Tensor instances")
    if clean_signal.device.type != "cpu" or held_out_signal.device.type != "cpu":
        raise ValueError(f"{name} evaluation is CPU-only; GPU tensors are rejected")
    if clean_signal.ndim != 3 or held_out_signal.ndim != 3:
        raise ValueError(
            f"{name} tensors must have shape [batch, channels, time], got "
            f"{tuple(clean_signal.shape)} and {tuple(held_out_signal.shape)}"
        )
    if tuple(clean_signal.shape) != tuple(held_out_signal.shape):
        raise ValueError(
            f"{name} clean/held-out shapes must match, got "
            f"{tuple(clean_signal.shape)} and {tuple(held_out_signal.shape)}"
        )
    batch, channels, time = clean_signal.shape
    if batch < 1 or channels < 1 or time < 4:
        raise ValueError(
            f"{name} requires positive batch/channels and at least four time samples"
        )
    if clean_signal.is_complex() or held_out_signal.is_complex():
        raise TypeError(f"{name} tensors must be real-valued")
    clean = clean_signal.to(dtype=torch.float64)
    held_out = held_out_signal.to(dtype=torch.float64)
    if not torch.isfinite(clean).all() or not torch.isfinite(held_out).all():
        raise ValueError(f"{name} tensors must contain only finite values")
    return clean, held_out


def _validate_logits_pair(
    clean_logits: torch.Tensor, held_out_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(clean_logits, torch.Tensor) or not isinstance(
        held_out_logits, torch.Tensor
    ):
        raise TypeError("logits must be torch.Tensor instances")
    if clean_logits.device.type != "cpu" or held_out_logits.device.type != "cpu":
        raise ValueError("prediction metrics are CPU-only; GPU logits are rejected")
    if clean_logits.ndim != 2 or held_out_logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if tuple(clean_logits.shape) != tuple(held_out_logits.shape):
        raise ValueError("clean/held-out logits shapes must match")
    if clean_logits.size(0) < 1 or clean_logits.size(1) < 2:
        raise ValueError("logits require at least one sample and two classes")
    if clean_logits.is_complex() or held_out_logits.is_complex():
        raise TypeError("logits must be real-valued")
    clean = clean_logits.to(dtype=torch.float64)
    held_out = held_out_logits.to(dtype=torch.float64)
    if not torch.isfinite(clean).all() or not torch.isfinite(held_out).all():
        raise ValueError("logits must contain only finite values")
    return clean, held_out


def _validate_features_pair(
    clean_features: torch.Tensor, held_out_features: torch.Tensor, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(clean_features, torch.Tensor) or not isinstance(
        held_out_features, torch.Tensor
    ):
        raise TypeError("features must be torch.Tensor instances")
    if clean_features.device.type != "cpu" or held_out_features.device.type != "cpu":
        raise ValueError("feature metrics are CPU-only; GPU features are rejected")
    if clean_features.ndim < 2 or held_out_features.ndim < 2:
        raise ValueError("features must have shape [batch, ...]")
    if clean_features.size(0) != batch or held_out_features.size(0) != batch:
        raise ValueError("feature batch size must match logits and signals")
    if tuple(clean_features.shape) != tuple(held_out_features.shape):
        raise ValueError("clean/held-out feature shapes must match")
    if clean_features.is_complex() or held_out_features.is_complex():
        raise TypeError("features must be real-valued")
    clean = clean_features.to(dtype=torch.float64).reshape(batch, -1)
    held_out = held_out_features.to(dtype=torch.float64).reshape(batch, -1)
    if clean.size(1) < 1:
        raise ValueError("features must contain at least one value per sample")
    if not torch.isfinite(clean).all() or not torch.isfinite(held_out).all():
        raise ValueError("features must contain only finite values")
    return clean, held_out


def _safe_mean(value: torch.Tensor) -> float:
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError("metric computation produced an empty or non-finite tensor")
    return float(value.mean().item())


def prediction_metrics(
    clean_logits: torch.Tensor,
    held_out_logits: torch.Tensor,
    clean_features: torch.Tensor,
    held_out_features: torch.Tensor,
) -> Dict[str, float]:
    """Compute predictive and representation changes for one held-out view.

    KL is reported in both directions and JS is symmetric.  Prediction margin
    is the top-1 minus top-2 softmax probability; ``margin_degradation`` is
    clean margin minus held-out margin, so positive values indicate a drop.
    ``confidence_drop`` is the analogous top-1 probability difference.
    """

    clean_logits, held_out_logits = _validate_logits_pair(
        clean_logits, held_out_logits
    )
    clean_features, held_out_features = _validate_features_pair(
        clean_features, held_out_features, clean_logits.size(0)
    )
    clean_probability = F.softmax(clean_logits, dim=-1).clamp_min(_EPS)
    held_out_probability = F.softmax(held_out_logits, dim=-1).clamp_min(_EPS)
    midpoint = ((clean_probability + held_out_probability) * 0.5).clamp_min(_EPS)
    kl_clean = (clean_probability * (clean_probability.log() - midpoint.log())).sum(-1)
    kl_held_out = (
        held_out_probability
        * (held_out_probability.log() - midpoint.log())
    ).sum(-1)
    js = 0.5 * (kl_clean + kl_held_out)
    kl_clean_to_held_out = (
        clean_probability
        * (clean_probability.log() - held_out_probability.log())
    ).sum(-1)
    kl_held_out_to_clean = (
        held_out_probability
        * (held_out_probability.log() - clean_probability.log())
    ).sum(-1)
    clean_sorted = torch.sort(clean_probability, dim=-1, descending=True).values
    held_out_sorted = torch.sort(held_out_probability, dim=-1, descending=True).values
    clean_margin = clean_sorted[:, 0] - clean_sorted[:, 1]
    held_out_margin = held_out_sorted[:, 0] - held_out_sorted[:, 1]
    clean_confidence = clean_sorted[:, 0]
    held_out_confidence = held_out_sorted[:, 0]
    clean_prediction = clean_probability.argmax(dim=-1)
    held_out_prediction = held_out_probability.argmax(dim=-1)
    cosine = F.cosine_similarity(clean_features, held_out_features, dim=-1, eps=_EPS)
    values = {
        "js_divergence": _safe_mean(js),
        "kl_clean_to_heldout": _safe_mean(kl_clean_to_held_out),
        "kl_heldout_to_clean": _safe_mean(kl_held_out_to_clean),
        "prediction_flip_rate": _safe_mean(
            (clean_prediction != held_out_prediction).to(torch.float64)
        ),
        "margin_degradation": _safe_mean(clean_margin - held_out_margin),
        "confidence_drop": _safe_mean(clean_confidence - held_out_confidence),
        "feature_cosine_distance": _safe_mean(1.0 - cosine),
    }
    # Short aliases are useful to CSV consumers while the canonical names in
    # the manifest remain explicit about direction and units.
    values["kl_divergence"] = values["kl_clean_to_heldout"]
    values["js"] = values["js_divergence"]
    values["kl"] = values["kl_clean_to_heldout"]
    values["prediction_flip"] = values["prediction_flip_rate"]
    values["margin_drop"] = values["margin_degradation"]
    values["feature_cosine"] = 1.0 - values["feature_cosine_distance"]
    return values


def heldout_direction_diagnostics(
    raw_logits: torch.Tensor,
    candidate_logits_by_view: torch.Tensor,
    *,
    confidence_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate unseen Sobol-direction candidates without target labels.

    ``candidate_logits_by_view`` has shape ``[views, batch, classes]``.  A
    candidate is eligible when it preserves the raw pseudo-label, has a
    positive pseudo-class margin, and reduces that margin relative to the raw
    anchor.  ``eligible_mask`` is intersected with the frozen confidence
    admission mask; coverage is therefore reported among confidence-admitted
    anchors rather than silently over all rejected samples.  The returned
    tensors are per-sample CPU evidence and contain no true labels.
    """

    if not isinstance(raw_logits, torch.Tensor) or not isinstance(
        candidate_logits_by_view, torch.Tensor
    ):
        raise TypeError("held-out direction logits must be tensors")
    if raw_logits.device.type != "cpu" or candidate_logits_by_view.device.type != "cpu":
        raise ValueError("held-out direction diagnostics are CPU-only")
    if raw_logits.ndim != 2 or candidate_logits_by_view.ndim != 3:
        raise ValueError("raw logits must be [batch, classes] and candidates [views, batch, classes]")
    if candidate_logits_by_view.size(1) != raw_logits.size(0):
        raise ValueError("candidate and raw logits batch sizes must match")
    if candidate_logits_by_view.size(2) != raw_logits.size(1):
        raise ValueError("candidate and raw logits class counts must match")
    if raw_logits.size(0) < 1 or raw_logits.size(1) < 2 or candidate_logits_by_view.size(0) < 1:
        raise ValueError("held-out directions require non-empty batch/views and at least two classes")
    raw = raw_logits.detach().to(dtype=torch.float64)
    candidates = candidate_logits_by_view.detach().to(dtype=torch.float64)
    if not torch.isfinite(raw).all() or not torch.isfinite(candidates).all():
        raise ValueError("held-out direction logits must be finite")
    batch = raw.size(0)
    if confidence_mask is None:
        admitted = torch.ones(batch, dtype=torch.bool)
    else:
        admitted = torch.as_tensor(confidence_mask, dtype=torch.bool, device="cpu").reshape(-1)
        if admitted.numel() != batch:
            raise ValueError("confidence_mask must contain one value per raw sample")

    raw_labels = raw.argmax(dim=1)
    target = raw.gather(1, raw_labels[:, None]).squeeze(1)
    other = raw.masked_fill(
        F.one_hot(raw_labels, raw.size(1)).bool(), float("-inf")
    ).amax(dim=1)
    raw_margin = target - other
    candidate_target = candidates.gather(
        2, raw_labels[None, :, None].expand(candidates.size(0), -1, 1)
    ).squeeze(2)
    candidate_other = candidates.masked_fill(
        F.one_hot(raw_labels, raw.size(1)).bool().unsqueeze(0), float("-inf")
    ).amax(dim=2)
    candidate_margin = candidate_target - candidate_other
    preserving = candidates.argmax(dim=2).eq(raw_labels[None, :])
    valid = preserving & candidate_margin.gt(0.0) & candidate_margin.lt(raw_margin[None, :])
    candidate_exists = valid.any(dim=0)
    eligible = admitted & candidate_exists
    selected_margin = candidate_margin.masked_fill(~valid, float("inf")).amin(dim=0)
    selected_margin = torch.where(candidate_exists, selected_margin, raw_margin)
    ratio = selected_margin / raw_margin.clamp_min(1.0e-12)
    # Rows outside the eligible set carry a finite neutral value; the summary
    # masks them before averaging and NPZ validation can remain fail-closed.
    ratio = torch.where(eligible, ratio, torch.ones_like(ratio))
    return {
        "confidence_admitted_mask": admitted,
        "eligible_mask": eligible,
        "margin_ratio": ratio,
        "heldout_flip_rate": (~preserving).to(torch.float64).mean(dim=0),
        "heldout_worst_margin": candidate_margin.amin(dim=0),
        "heldout_consistency": preserving.to(torch.float64).mean(dim=0),
        "raw_margin": raw_margin,
    }


def summarize_heldout_direction_diagnostics(
    diagnostics: Mapping[str, torch.Tensor],
) -> Dict[str, Optional[float]]:
    """Reduce :func:`heldout_direction_diagnostics` to auditable scalars."""

    required = {
        "confidence_admitted_mask",
        "eligible_mask",
        "margin_ratio",
        "heldout_flip_rate",
        "heldout_worst_margin",
        "heldout_consistency",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"held-out direction diagnostics missing arrays: {missing}")
    admitted = torch.as_tensor(diagnostics["confidence_admitted_mask"], dtype=torch.bool).reshape(-1)
    eligible = torch.as_tensor(diagnostics["eligible_mask"], dtype=torch.bool).reshape(-1)
    if admitted.numel() != eligible.numel() or admitted.numel() < 1:
        raise ValueError("held-out direction masks must share a non-empty batch")
    admitted_count = int(admitted.sum().item())
    eligible_count = int((eligible & admitted).sum().item())
    ratio = torch.as_tensor(diagnostics["margin_ratio"], dtype=torch.float64).reshape(-1)
    flip = torch.as_tensor(diagnostics["heldout_flip_rate"], dtype=torch.float64).reshape(-1)
    worst = torch.as_tensor(diagnostics["heldout_worst_margin"], dtype=torch.float64).reshape(-1)
    consistency = torch.as_tensor(diagnostics["heldout_consistency"], dtype=torch.float64).reshape(-1)
    if any(value.numel() != admitted.numel() for value in (ratio, flip, worst, consistency)):
        raise ValueError("held-out direction metric arrays must share a batch")
    if any(not torch.isfinite(value).all() for value in (ratio, flip, worst, consistency)):
        raise ValueError("held-out direction metric arrays must be finite")
    output: Dict[str, Optional[float]] = {
        "eligible_coverage": (
            float(eligible_count / admitted_count) if admitted_count else None
        ),
        "margin_ratio": (
            float(ratio[eligible & admitted].mean().item())
            if eligible_count
            else None
        ),
        "heldout_flip_rate": float(flip.mean().item()),
        "heldout_worst_margin": float(worst.mean().item()),
        "heldout_consistency": float(consistency.mean().item()),
        "confidence_admitted_count": float(admitted_count),
        "eligible_count": float(eligible_count),
    }
    return output


def _spectrum(signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    time = signal.size(-1)
    centered = signal - signal.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        time, periodic=False, dtype=signal.dtype, device=signal.device
    )
    transformed = torch.fft.rfft(centered * window, dim=-1)
    power = transformed.abs().square()
    frequency_bins = torch.fft.rfftfreq(
        time, d=1.0, dtype=signal.dtype, device=signal.device
    )
    return power, frequency_bins


def _peak_frequency_bins(signal: torch.Tensor) -> torch.Tensor:
    power, frequency_bins = _spectrum(signal)
    if power.size(-1) < 2:
        raise ValueError("frequency metrics require at least two FFT bins")
    non_dc = power[..., 1:]
    index = non_dc.argmax(dim=-1) + 1
    return frequency_bins[index]


def _peak_frequency(signal: torch.Tensor, sampling_rate_hz: float) -> torch.Tensor:
    return _peak_frequency_bins(signal) * sampling_rate_hz


def _frequency_shift(
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    sampling_rate_hz: Optional[float],
) -> float:
    clean_peak = _peak_frequency_bins(clean_signal)
    held_out_peak = _peak_frequency_bins(held_out_signal)
    if sampling_rate_hz is not None:
        clean_peak = clean_peak * sampling_rate_hz
        held_out_peak = held_out_peak * sampling_rate_hz
    return _safe_mean((clean_peak - held_out_peak).abs())


def _relative_error(reference: torch.Tensor, observed: torch.Tensor) -> float:
    reference_scale = reference.abs().mean(dim=-1).clamp_min(_EPS)
    error = (observed - reference).abs().mean(dim=-1) / reference_scale
    return _safe_mean(error)


def _triad_magnitude(signal: torch.Tensor) -> torch.Tensor:
    channels = signal.size(1)
    if channels < 3 or channels % 3 != 0:
        raise ValueError(
            "HAR/HHAR signals require channels grouped into complete 3-axis triads"
        )
    triads = signal.reshape(signal.size(0), channels // 3, 3, signal.size(-1))
    return triads.square().sum(dim=2).sqrt()


def har_physical_invariants(
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    *,
    metadata: Mapping[str, Any],
) -> Dict[str, float]:
    """Return three-axis magnitude, jerk, energy, and periodic-frequency checks."""

    clean_signal, held_out_signal = _validate_signal_pair(clean_signal, held_out_signal)
    metadata = _metadata_mapping(metadata)
    sampling_rate_hz = _sampling_rate_optional(metadata)
    derivative_scale = 1.0 if sampling_rate_hz is None else sampling_rate_hz
    clean_magnitude = _triad_magnitude(clean_signal)
    held_out_magnitude = _triad_magnitude(held_out_signal)
    clean_jerk = torch.diff(clean_magnitude, dim=-1) * derivative_scale
    held_out_jerk = torch.diff(held_out_magnitude, dim=-1) * derivative_scale
    clean_energy = clean_magnitude.square().mean(dim=-1)
    held_out_energy = held_out_magnitude.square().mean(dim=-1)
    # Average triads before frequency selection so a sensor with more triads
    # cannot receive more weight solely because of channel count.
    clean_periodic = clean_magnitude.mean(dim=1)
    held_out_periodic = held_out_magnitude.mean(dim=1)
    result = {
        "triad_norm_relative_error": _relative_error(
            clean_magnitude, held_out_magnitude
        ),
        "jerk_relative_error": _relative_error(clean_jerk, held_out_jerk),
        "energy_relative_error": _relative_error(clean_energy, held_out_energy),
        "dominant_periodic_frequency_shift": _frequency_shift(
            clean_periodic, held_out_periodic, sampling_rate_hz
        ),
        "dominant_periodic_frequency_shift_cycles_per_sample": _frequency_shift(
            clean_periodic, held_out_periodic, None
        ),
    }
    if sampling_rate_hz is not None:
        result["dominant_periodic_frequency_shift_hz"] = result[
            "dominant_periodic_frequency_shift"
        ]
    result["three_axis_magnitude_relative_error"] = result[
        "triad_norm_relative_error"
    ]
    result["dominant_frequency_shift"] = result[
        "dominant_periodic_frequency_shift"
    ]
    return result


def _parse_bands(
    metadata: Mapping[str, Any], sampling_rate_hz: float, time: int
) -> tuple[tuple[str, float, float], ...]:
    raw = metadata.get("bands_hz", metadata.get("eeg_bands_hz", DEFAULT_EEG_BANDS_HZ))
    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = [(f"band_{index}", value) for index, value in enumerate(raw)]
    else:
        raise TypeError("EEG bands_hz must be a mapping or sequence of [low, high]")
    if not items:
        raise ValueError("EEG bands_hz must contain at least one band")
    nyquist = sampling_rate_hz / 2.0
    frequency_resolution = sampling_rate_hz / float(time)
    parsed = []
    for name, interval in items:
        if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
            raise ValueError(f"EEG band {name!r} must be [low_hz, high_hz]")
        if len(interval) != 2:
            raise ValueError(f"EEG band {name!r} must contain exactly two bounds")
        low = _as_finite_number(f"{name}.low_hz", interval[0])
        high = _as_finite_number(f"{name}.high_hz", interval[1])
        if low < 0.0 or high <= low or high > nyquist:
            raise ValueError(
                f"EEG band {name!r} must satisfy 0 <= low < high <= Nyquist"
            )
        if high - low < frequency_resolution:
            raise ValueError(
                f"EEG band {name!r} is narrower than one FFT bin at the declared rate"
            )
        parsed.append((str(name), low, high))
    return tuple(parsed)


def _relative_bandpower(
    signal: torch.Tensor,
    sampling_rate_hz: float,
    bands: Sequence[tuple[str, float, float]],
) -> torch.Tensor:
    power, frequency_bins = _spectrum(signal)
    non_dc_power = power[..., 1:]
    non_dc_frequency = frequency_bins[1:] * sampling_rate_hz
    total = non_dc_power.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    values = []
    for _name, low, high in bands:
        mask = (non_dc_frequency >= low) & (non_dc_frequency < high)
        if not bool(mask.any()):
            raise ValueError(
                f"EEG band [{low}, {high}) has no FFT bin at the declared rate/length"
            )
        values.append(non_dc_power[..., mask].sum(dim=-1, keepdim=True) / total)
    return torch.cat(values, dim=-1)


def _pearson(clean: torch.Tensor, held_out: torch.Tensor) -> torch.Tensor:
    clean_centered = clean - clean.mean(dim=-1, keepdim=True)
    held_out_centered = held_out - held_out.mean(dim=-1, keepdim=True)
    numerator = (clean_centered * held_out_centered).sum(dim=-1)
    clean_norm = clean_centered.square().sum(dim=-1).sqrt()
    held_out_norm = held_out_centered.square().sum(dim=-1).sqrt()
    denominator = clean_norm * held_out_norm
    regular = numerator / denominator.clamp_min(_EPS)
    both_constant = (clean_norm <= _EPS) & (held_out_norm <= _EPS)
    # Correlation is undefined for two constant vectors.  Treating any two
    # constant envelopes as perfectly pattern-preserving avoids turning a
    # benign gain change into an arbitrary zero correlation; amplitude change
    # is already represented by the dedicated gain-sensitive invariants.
    equal_constant = both_constant
    return torch.where(equal_constant, torch.ones_like(regular), regular.clamp(-1.0, 1.0))


def _channel_correlation(signal: torch.Tensor) -> torch.Tensor:
    centered = signal - signal.mean(dim=-1, keepdim=True)
    covariance = centered @ centered.transpose(-1, -2)
    scale = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(_EPS).sqrt()
    correlation = covariance / (scale.unsqueeze(-1) * scale.unsqueeze(-2)).clamp_min(_EPS)
    return correlation.clamp(-1.0, 1.0)


def _coherence(signal: torch.Tensor) -> torch.Tensor:
    """Estimate pairwise magnitude-squared coherence using two time blocks."""

    batch, channels, time = signal.shape
    if channels < 2:
        # EEG's registered repository shape is one channel.  There is no
        # cross-channel pair in that case; return an empty-pair-equivalent
        # zero change rather than fabricating self-coherence as evidence.
        return signal.new_zeros((batch, channels, channels))
    segment_length = max(4, min(64, time // 2))
    segments = time // segment_length
    if segments < 2:
        # The public validator currently requires at least four samples.  For
        # a four-to-seven-sample input, a correlation fallback avoids silently
        # returning NaN while retaining a bounded coherence-like quantity.
        result = signal.new_zeros((batch, channels, channels))
        correlation = _channel_correlation(signal).abs()
        return correlation
    trimmed = signal[..., : segments * segment_length]
    blocks = trimmed.reshape(batch, channels, segments, segment_length)
    blocks = blocks - blocks.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        segment_length, periodic=False, dtype=signal.dtype, device=signal.device
    )
    transformed = torch.fft.rfft(blocks * window, dim=-1)
    cross = transformed.unsqueeze(2) * transformed.conj().unsqueeze(1)
    cross_mean = cross.mean(dim=3)
    power = transformed.abs().square().mean(dim=2)
    coherence = cross_mean.abs().square() / (
        power.unsqueeze(2) * power.unsqueeze(1)
    ).clamp_min(_EPS)
    # Frequency zero represents the removed DC component and can dominate a
    # short window; exclude it when an actual spectral estimate is available.
    return coherence[..., 1:].mean(dim=-1).clamp(0.0, 1.0)


def eeg_physical_invariants(
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    *,
    metadata: Mapping[str, Any],
) -> Dict[str, float]:
    """Return bandpower, coherence, spectral, correlation, and envelope checks."""

    clean_signal, held_out_signal = _validate_signal_pair(clean_signal, held_out_signal)
    metadata = _metadata_mapping(metadata)
    sampling_rate_hz = _sampling_rate(metadata)
    bands = _parse_bands(metadata, sampling_rate_hz, clean_signal.size(-1))
    clean_bandpower = _relative_bandpower(clean_signal, sampling_rate_hz, bands)
    held_out_bandpower = _relative_bandpower(held_out_signal, sampling_rate_hz, bands)
    bandpower_error = (clean_bandpower - held_out_bandpower).abs().mean()
    clean_coherence = _coherence(clean_signal)
    held_out_coherence = _coherence(held_out_signal)
    clean_correlation = _channel_correlation(clean_signal)
    held_out_correlation = _channel_correlation(held_out_signal)
    off_diagonal = ~torch.eye(
        clean_signal.size(1), dtype=torch.bool, device=clean_signal.device
    )
    correlation_difference = (clean_correlation - held_out_correlation).abs()
    correlation_distortion = (
        correlation_difference[..., off_diagonal].mean()
        if bool(off_diagonal.any())
        else clean_signal.new_zeros(())
    )
    clean_envelope = _analytic_envelope(clean_signal)
    held_out_envelope = _analytic_envelope(held_out_signal)
    envelope_correlation = _pearson(
        clean_envelope.reshape(-1, clean_envelope.size(-1)),
        held_out_envelope.reshape(-1, held_out_envelope.size(-1)),
    ).mean()
    result = {
        "relative_bandpower_error": _safe_mean(bandpower_error),
        "spectral_coherence": _safe_mean(
            (clean_coherence - held_out_coherence).abs()
        ),
        "dominant_frequency_shift": _frequency_shift(
            clean_signal, held_out_signal, sampling_rate_hz
        ),
        "channel_correlation_distortion": _safe_mean(correlation_distortion),
        "amplitude_envelope_correlation": _safe_mean(envelope_correlation),
    }
    result["coherence_change"] = result["spectral_coherence"]
    result["envelope_correlation"] = result["amplitude_envelope_correlation"]
    result["envelope_correlation_drop"] = 1.0 - result[
        "amplitude_envelope_correlation"
    ]
    return result


def _analytic_envelope(signal: torch.Tensor) -> torch.Tensor:
    time = signal.size(-1)
    transformed = torch.fft.fft(signal, dim=-1)
    multiplier = torch.zeros(time, dtype=transformed.dtype, device=signal.device)
    multiplier[0] = 1.0
    if time % 2 == 0:
        multiplier[time // 2] = 1.0
        multiplier[1 : time // 2] = 2.0
    else:
        multiplier[1 : (time + 1) // 2] = 2.0
    return torch.fft.ifft(transformed * multiplier, dim=-1).abs()


def _spectral_kurtosis(signal: torch.Tensor) -> torch.Tensor:
    power, _frequency_bins = _spectrum(signal)
    values = power[..., 1:]
    mean = values.mean(dim=-1)
    centered = values - mean.unsqueeze(-1)
    variance = centered.square().mean(dim=-1)
    fourth = centered.pow(4).mean(dim=-1)
    return torch.where(
        variance > _EPS, fourth / variance.square().clamp_min(_EPS) - 3.0, torch.zeros_like(variance)
    )


def fd_physical_invariants(
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    *,
    metadata: Mapping[str, Any],
) -> Dict[str, float]:
    """Return order/envelope peaks, spectral kurtosis, and RMS checks."""

    clean_signal, held_out_signal = _validate_signal_pair(clean_signal, held_out_signal)
    metadata = _metadata_mapping(metadata)
    sampling_rate_hz = _sampling_rate_optional(metadata)
    rotation_frequency_hz = _rotation_rate_optional(metadata)
    clean_average = clean_signal.mean(dim=1)
    held_out_average = held_out_signal.mean(dim=1)
    clean_peak_normalized = _peak_frequency_bins(clean_average)
    held_out_peak_normalized = _peak_frequency_bins(held_out_average)
    clean_envelope = _analytic_envelope(clean_signal).mean(dim=1)
    held_out_envelope = _analytic_envelope(held_out_signal).mean(dim=1)
    clean_envelope_peak_normalized = _peak_frequency_bins(clean_envelope)
    held_out_envelope_peak_normalized = _peak_frequency_bins(held_out_envelope)
    clean_kurtosis = _spectral_kurtosis(clean_signal)
    held_out_kurtosis = _spectral_kurtosis(held_out_signal)
    clean_rms = clean_signal.square().mean().sqrt().clamp_min(_EPS)
    held_out_rms = held_out_signal.square().mean().sqrt()
    normalized_peak_shift = _safe_mean(
        (clean_peak_normalized - held_out_peak_normalized).abs()
    )
    normalized_envelope_peak_shift = _safe_mean(
        (clean_envelope_peak_normalized - held_out_envelope_peak_normalized).abs()
    )
    result = {
        "normalized_spectral_peak_shift_cycles_per_sample": normalized_peak_shift,
        "normalized_envelope_spectral_peak_shift_cycles_per_sample": normalized_envelope_peak_shift,
        "spectral_kurtosis_change": _safe_mean(
            (held_out_kurtosis - clean_kurtosis).abs()
        ),
        "rms_ratio": float((held_out_rms / clean_rms).item()),
    }
    if sampling_rate_hz is not None:
        raw_peak_shift = _safe_mean(
            (clean_peak_normalized - held_out_peak_normalized).abs()
        ) * sampling_rate_hz
        raw_envelope_peak_shift = _safe_mean(
            (clean_envelope_peak_normalized - held_out_envelope_peak_normalized).abs()
        ) * sampling_rate_hz
        result.update(
            {
                "raw_spectral_peak_shift_hz": raw_peak_shift,
                "raw_envelope_spectral_peak_shift_hz": raw_envelope_peak_shift,
                # Compatibility aliases retain their units in the manifest;
                # they are omitted when no verified sampling rate exists.
                "envelope_spectrum_peak_shift": raw_envelope_peak_shift,
                "envelope_peak_shift": raw_envelope_peak_shift,
            }
        )
        if rotation_frequency_hz is not None:
            order_shift = raw_peak_shift / rotation_frequency_hz
            result["order_frequency_peak_shift"] = order_shift
            result["order_peak_shift"] = order_shift
    result["spectral_kurtosis_delta"] = _safe_mean(
        held_out_kurtosis - clean_kurtosis
    )
    result["rms_relative_error"] = abs(result["rms_ratio"] - 1.0)
    return result


def validate_split_metadata(case: HeldOutCase | Mapping[str, Any]) -> HeldOutCase:
    """Validate and normalize training/held-out identifiers.

    Mapping input accepts ``trajectory_id``/``operator_id`` and
    ``held_out_trajectory``/``held_out_operator`` aliases for JSON manifests.
    The function rejects equal training/test seeds and any metadata that claims
    a measured nuisance or an LPR.
    """

    if isinstance(case, HeldOutCase):
        normalized = case
    elif isinstance(case, Mapping):
        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in case:
                    return case[name]
            return default

        normalized = HeldOutCase(
            dataset=pick("dataset"),
            training_view_family=pick("training_view_family", "training_view"),
            held_out_view_family=pick("held_out_view_family", "test_view_family"),
            held_out_trajectory=pick(
                "held_out_trajectory", "held_out_trajectory_id", "trajectory_id"
            ),
            held_out_operator=pick(
                "held_out_operator", "held_out_operator_id", "operator_id"
            ),
            training_seed=pick("training_seed", "source_seed"),
            test_seed=pick("test_seed", "held_out_seed", "stream_seed"),
            metadata=pick("metadata", default={}),
            algorithm=pick("algorithm", "algorithm_name", default="unspecified"),
        )
    else:
        raise TypeError("case must be HeldOutCase or a mapping")
    dataset = _canonical_dataset(normalized.dataset)
    training_view = _non_empty_text(
        "training_view_family", normalized.training_view_family
    )
    held_out_view = _non_empty_text(
        "held_out_view_family", normalized.held_out_view_family
    )
    trajectory = _non_empty_text(
        "held_out_trajectory", normalized.held_out_trajectory
    )
    operator = _non_empty_text("held_out_operator", normalized.held_out_operator)
    if training_view == held_out_view:
        raise ValueError(
            "training_view_family and held_out_view_family must be distinct"
        )
    if trajectory == training_view or operator == training_view:
        raise ValueError(
            "held-out trajectory/operator identifiers cannot reuse the training view"
        )
    try:
        training_seed_float = float(normalized.training_seed)
        test_seed_float = float(normalized.test_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("training_seed and test_seed must be integers") from exc
    if (
        not math.isfinite(training_seed_float)
        or not math.isfinite(test_seed_float)
        or not training_seed_float.is_integer()
        or not test_seed_float.is_integer()
    ):
        raise ValueError("training_seed and test_seed must be integers")
    training_seed = int(training_seed_float)
    test_seed = int(test_seed_float)
    if training_seed == test_seed:
        raise ValueError(
            "training_seed and test_seed must be separated for held-out evaluation"
        )
    metadata = _metadata_mapping(normalized.metadata)
    if "training_seed" in metadata and int(metadata["training_seed"]) != training_seed:
        raise ValueError("metadata training_seed disagrees with case")
    if "test_seed" in metadata and int(metadata["test_seed"]) != test_seed:
        raise ValueError("metadata test_seed disagrees with case")
    if "held_out_trajectory" in metadata and str(metadata["held_out_trajectory"]) != trajectory:
        raise ValueError("metadata held_out_trajectory disagrees with case")
    if "held_out_operator" in metadata and str(metadata["held_out_operator"]) != operator:
        raise ValueError("metadata held_out_operator disagrees with case")
    # Dataset-specific metadata requirements are checked here as well as in
    # invariant functions, so a manifest cannot advertise an invalid cell.
    # HHAR may not have a common Hz rate after row-window conversion; FD may
    # lack both a verified rate and a verified rotation reference.  Their
    # sample-axis metrics remain valid and are labelled in the manifest.
    if dataset in {"EEG", "HAR"}:
        _sampling_rate(metadata)
    else:
        _sampling_rate_optional(metadata)
    if dataset == "FD":
        _rotation_rate_optional(metadata)
    return HeldOutCase(
        dataset=dataset,
        training_view_family=training_view,
        held_out_view_family=held_out_view,
        held_out_trajectory=trajectory,
        held_out_operator=operator,
        training_seed=training_seed,
        test_seed=test_seed,
        metadata=dict(metadata),
        algorithm=_non_empty_text("algorithm", normalized.algorithm),
    )


def validate_case(case: HeldOutCase | Mapping[str, Any]) -> HeldOutCase:
    """Alias for the public fail-closed case validator."""

    normalized = validate_split_metadata(case)
    expected = DATASET_MECHANISM_PROTOCOL[normalized.dataset]
    # A custom held-out operator is allowed, but it must not be called the
    # training family.  The explicit family separation is the preregistered
    # condition; unknown operator names remain auditable in the manifest.
    if normalized.held_out_view_family == expected["training_view_family"]:
        raise ValueError(
            f"{normalized.dataset} held-out view family reuses its training family"
        )
    return normalized


def compute_mechanism_metrics(
    dataset: str,
    clean_signal: torch.Tensor,
    held_out_signal: torch.Tensor,
    *,
    clean_logits: torch.Tensor,
    held_out_logits: torch.Tensor,
    clean_features: torch.Tensor,
    held_out_features: torch.Tensor,
    metadata: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute common predictive and dataset-specific physical metrics."""

    dataset = _canonical_dataset(dataset)
    clean_signal, held_out_signal = _validate_signal_pair(clean_signal, held_out_signal)
    values = prediction_metrics(
        clean_logits,
        held_out_logits,
        clean_features,
        held_out_features,
    )
    if dataset in {"HAR", "HHAR"}:
        values.update(
            har_physical_invariants(
                clean_signal, held_out_signal, metadata=metadata
            )
        )
    elif dataset == "EEG":
        values.update(
            eeg_physical_invariants(
                clean_signal, held_out_signal, metadata=metadata
            )
        )
    else:
        values.update(
            fd_physical_invariants(
                clean_signal, held_out_signal, metadata=metadata
            )
        )
    return {name: float(value) for name, value in values.items()}


def mechanism_protocol_manifest() -> Dict[str, Any]:
    """Return the serializable protocol without run-specific claims."""

    datasets = {}
    for name, spec in DATASET_MECHANISM_PROTOCOL.items():
        datasets[name] = {
            **spec,
            "held_out_operators": list(spec["held_out_operators"]),
            "physical_metrics": list(spec["physical_metrics"]),
            "required_metadata": list(spec["required_metadata"]),
        }
    return {
        "protocol_version": HELD_OUT_PROTOCOL_VERSION,
        "tensor_convention": "signals [batch, channels, time]; logits [batch, classes]; features [batch, ...]",
        "datasets": datasets,
        "predictive_metrics": list(COMMON_PREDICTIVE_METRICS),
        "divergence_log_base": "natural",
        "split_requirements": {
            "training_view_and_held_out_view_must_differ": True,
            "training_seed_and_test_seed_must_differ": True,
            "held_out_trajectory_and_operator_are_declared_identifiers": True,
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
        "claims": {
            "real_nuisance_observed": False,
            "real_distribution_match_estimated": False,
            "lpr_estimated": False,
            "labels_used_for_mechanism_metrics": False,
            "synthetic_or_declared_operator_only": True,
        },
    }


def build_manifest(
    case: HeldOutCase | Mapping[str, Any],
    *,
    metrics: Optional[Mapping[str, Any]] = None,
    input_shape: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build a run manifest with explicit split and non-claims."""

    normalized = validate_case(case)
    if input_shape is not None:
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("input_shape must be a positive [batch, channels, time] shape")
    else:
        shape = None
    serializable_metrics = None
    if metrics is not None:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        serializable_metrics = {
            str(key): (
                None
                if value is None
                else _as_finite_number(str(key), value)
            )
            for key, value in metrics.items()
        }
    manifest = mechanism_protocol_manifest()
    manifest.update(
        {
            "case": {
                "dataset": normalized.dataset,
                "algorithm": normalized.algorithm,
                "training_view_family": normalized.training_view_family,
                "held_out_view_family": normalized.held_out_view_family,
                "held_out_trajectory": normalized.held_out_trajectory,
                "held_out_operator": normalized.held_out_operator,
                "training_seed": normalized.training_seed,
                "test_seed": normalized.test_seed,
                "metadata": dict(normalized.metadata),
            },
            "input_shape": list(shape) if shape is not None else None,
            "metrics": serializable_metrics,
            "rotation_rate_unverified": bool(
                normalized.dataset == "FD"
                and _rotation_rate_optional(normalized.metadata) is None
            ),
            "sampling_rate_provenance": normalized.metadata.get(
                "sampling_rate_provenance",
                normalized.metadata.get("sample_rate_provenance"),
            ),
            "frequency_axis": (
                "cycles_per_sample"
                if normalized.dataset in {"HHAR", "FD"}
                and _sampling_rate_optional(normalized.metadata) is None
                else "hertz"
            ),
        }
    )
    return manifest


def _call_operator(
    operator: Callable[..., torch.Tensor], inputs: torch.Tensor, seed: int
) -> torch.Tensor:
    if not callable(operator):
        raise TypeError("held-out operator must be callable")
    try:
        signature = inspect.signature(operator)
        parameters = signature.parameters
        accepts_seed_keyword = "seed" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        positional_capacity = sum(
            parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        accepts_seed_keyword = False
        positional_capacity = 1
    if accepts_seed_keyword:
        output = operator(inputs, seed=seed)
    elif positional_capacity >= 2:
        output = operator(inputs, seed)
    else:
        output = operator(inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("held-out operator must return a torch.Tensor")
    if output.device.type != "cpu":
        raise ValueError("held-out operator returned a non-CPU tensor")
    return output


def _extract_algorithm_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, Mapping):
        logits = output.get("logits")
        features = output.get("features", output.get("feature", output.get("embedding")))
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        logits, features = output[0], output[1]
    else:
        logits = getattr(output, "logits", None)
        features = getattr(output, "features", getattr(output, "feature", None))
    if logits is None or features is None:
        raise ValueError(
            "algorithm output must expose both logits and features; "
            "a logits-only result cannot support the required cosine metric"
        )
    return logits, features


class HeldOutMechanismRunner:
    """Unified CPU runner for EEG, HAR, FD, and HHAR algorithm outputs.

    The adapter may return ``{"logits": ..., "features": ...}``, a two-tuple,
    or an object with those attributes.  ``run_case`` applies a separately
    supplied held-out operator under the declared test seed; it does not train
    an algorithm or infer a nuisance/LPR quantity.
    """

    def __init__(
        self,
        algorithm: Callable[[torch.Tensor], Any],
        *,
        algorithm_name: str = "unspecified",
        device: str = "cpu",
    ) -> None:
        if str(device).strip().lower() != "cpu":
            raise ValueError("HeldOutMechanismRunner is CPU-only; GPU execution is disabled")
        if not callable(algorithm):
            raise TypeError("algorithm must be callable")
        self.algorithm = algorithm
        self.algorithm_name = _non_empty_text("algorithm_name", algorithm_name)

    def _predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cpu":
            raise ValueError("runner inputs must be CPU torch tensors")
        with torch.no_grad():
            output = self.algorithm(inputs)
        logits, features = _extract_algorithm_output(output)
        return logits, features

    def evaluate(
        self,
        case: HeldOutCase | Mapping[str, Any],
        clean_inputs: torch.Tensor,
        held_out_inputs: torch.Tensor,
    ) -> Dict[str, Any]:
        normalized = validate_case(case)
        if normalized.algorithm == "unspecified":
            normalized = HeldOutCase(
                **{
                    **asdict(normalized),
                    "algorithm": self.algorithm_name,
                }
            )
        clean_inputs, held_out_inputs = _validate_signal_pair(
            clean_inputs, held_out_inputs
        )
        clean_logits, clean_features = self._predict(clean_inputs)
        held_out_logits, held_out_features = self._predict(held_out_inputs)
        metrics = compute_mechanism_metrics(
            normalized.dataset,
            clean_inputs,
            held_out_inputs,
            clean_logits=clean_logits,
            held_out_logits=held_out_logits,
            clean_features=clean_features,
            held_out_features=held_out_features,
            metadata=normalized.metadata,
        )
        return {
            "manifest": build_manifest(
                normalized, metrics=metrics, input_shape=clean_inputs.shape
            ),
            "metrics": metrics,
        }

    def run_case(
        self,
        case: HeldOutCase | Mapping[str, Any],
        clean_inputs: torch.Tensor,
        held_out_operator: Callable[..., torch.Tensor],
    ) -> Dict[str, Any]:
        normalized = validate_case(case)
        if not isinstance(clean_inputs, torch.Tensor) or clean_inputs.device.type != "cpu":
            raise ValueError("runner inputs must be CPU torch tensors")
        # fork_rng isolates operator randomness from the caller.  ``devices``
        # is empty by construction, so this path cannot initialize CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(normalized.test_seed)
            held_out_inputs = _call_operator(
                held_out_operator, clean_inputs.clone(), normalized.test_seed
            )
        return self.evaluate(normalized, clean_inputs, held_out_inputs)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON manifest/result atomically without changing external state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = [
    "COMMON_PREDICTIVE_METRICS",
    "DATASET_MECHANISM_PROTOCOL",
    "DEFAULT_EEG_BANDS_HZ",
    "HELD_OUT_PROTOCOL_VERSION",
    "HeldOutCase",
    "HeldOutMechanismRunner",
    "build_manifest",
    "compute_mechanism_metrics",
    "eeg_physical_invariants",
    "fd_physical_invariants",
    "har_physical_invariants",
    "mechanism_protocol_manifest",
    "prediction_metrics",
    "heldout_direction_diagnostics",
    "summarize_heldout_direction_diagnostics",
    "validate_case",
    "validate_split_metadata",
    "write_json",
]
