"""Pre-registered evaluation protocol for the SSAW physical-view branch.

The online method does not import this module.  It defines only the held-out
evaluation panel so tuning code cannot silently change corruption severities
or the evidence required to attribute an effect to SSAW.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple


PROTOCOL_VERSION = "ssaw_physical_evaluation_v1"
PRIMARY_CORRUPTIONS = (
    "signal_freeze",
    "blackout",
    "attenuation",
    "amplitude_drift",
    "packet_loss",
    "saturation",
)


@dataclass(frozen=True)
class SeverityPoint:
    """One physically interpretable point on a corruption-specific curve."""

    name: str
    normalized: float
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetPhysicalProtocol:
    """Training-view and held-out mechanism checks for one dataset."""

    dataset: str
    training_view_family: str
    held_out_view_families: Tuple[str, ...]
    physical_invariants: Tuple[str, ...]
    realism_claim_limit: str = "physically_plausible_not_real_distribution_match"


def _points(parameter_name: str, values) -> Tuple[SeverityPoint, ...]:
    if len(values) != 7:
        raise ValueError("Each physical curve must contain exactly seven points")
    return tuple(
        SeverityPoint(
            name=f"s{index}",
            normalized=index / 6.0,
            parameters={parameter_name: value},
        )
        for index, value in enumerate(values)
    )


PHYSICAL_SEVERITY_GRIDS: Dict[str, Tuple[SeverityPoint, ...]] = {
    # Fraction of the acquisition window held at its last valid value.
    "signal_freeze": _points(
        "frozen_fraction", (0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60)
    ),
    # Fraction of one contiguous acquisition window replaced by zero.
    "blackout": _points(
        "blackout_fraction", (0.0, 0.025, 0.05, 0.10, 0.15, 0.225, 0.30)
    ),
    # Multiplicative sensor response.  Lower values are more severe.
    "attenuation": _points(
        "gain", (1.0, 0.90, 0.80, 0.65, 0.50, 0.35, 0.20)
    ),
    # Linear gain at the end of the window; the initial gain is always one.
    "amplitude_drift": _points(
        "end_gain", (1.0, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00)
    ),
    # Total missing fraction.  Packet count is a deterministic function of
    # the level in the held-out transform implementation.
    "packet_loss": _points(
        "missing_fraction", (0.0, 0.025, 0.05, 0.10, 0.15, 0.225, 0.30)
    ),
    # Symmetric clipping threshold in per-channel standard deviations.  None
    # is the identity point and avoids pretending that an arbitrary large
    # finite threshold is exactly clean.
    "saturation": _points(
        "clip_std", (None, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75)
    ),
}


DATASET_PHYSICAL_PROTOCOLS: Dict[str, DatasetPhysicalProtocol] = {
    "EEG": DatasetPhysicalProtocol(
        dataset="EEG",
        training_view_family="window_constant_channel_gain",
        held_out_view_families=(
            "smooth_channel_gain_drift",
            "local_channel_attenuation",
        ),
        physical_invariants=(
            "relative_bandpower_error",
            "spectral_coherence",
            "dominant_frequency_shift",
            "channel_correlation_distortion",
            "amplitude_envelope_correlation",
        ),
    ),
    "HAR": DatasetPhysicalProtocol(
        dataset="HAR",
        training_view_family="window_constant_bounded_so3",
        held_out_view_families=("smooth_so3_orientation_trajectory",),
        physical_invariants=(
            "triad_norm_relative_error",
            "jerk_relative_error",
            "energy_relative_error",
            "dominant_periodic_frequency_shift",
        ),
    ),
    "FD": DatasetPhysicalProtocol(
        dataset="FD",
        training_view_family="window_constant_sensor_gain",
        held_out_view_families=(
            "smooth_sensor_response_drift",
            "bounded_filter_response_drift",
        ),
        physical_invariants=(
            "order_frequency_peak_shift",
            "envelope_spectrum_peak_shift",
            "spectral_kurtosis_change",
            "rms_ratio",
        ),
    ),
    "HHAR": DatasetPhysicalProtocol(
        dataset="HHAR",
        training_view_family="window_constant_bounded_so3",
        held_out_view_families=("smooth_so3_orientation_trajectory",),
        physical_invariants=(
            "triad_norm_relative_error",
            "jerk_relative_error",
            "energy_relative_error",
            "dominant_periodic_frequency_shift",
        ),
    ),
}


def canonical_dataset(dataset: str) -> str:
    """Map the paper's MFD label to the repository's FD identifier."""

    value = str(dataset).upper()
    return "FD" if value == "MFD" else value


def get_dataset_physical_protocol(dataset: str) -> DatasetPhysicalProtocol:
    key = canonical_dataset(dataset)
    try:
        return DATASET_PHYSICAL_PROTOCOLS[key]
    except KeyError as exc:
        raise KeyError(f"No SSAW physical protocol registered for {dataset!r}") from exc


def validate_protocol() -> None:
    """Fail closed if a severity curve or dataset protocol drifts."""

    if tuple(PHYSICAL_SEVERITY_GRIDS) != PRIMARY_CORRUPTIONS:
        raise ValueError("Physical corruption order does not match the primary panel")
    for corruption, points in PHYSICAL_SEVERITY_GRIDS.items():
        if len(points) != 7:
            raise ValueError(f"{corruption} must contain seven severity points")
        normalized = tuple(float(point.normalized) for point in points)
        if normalized[0] != 0.0 or normalized[-1] != 1.0:
            raise ValueError(f"{corruption} must span normalized severity [0, 1]")
        if any(right <= left for left, right in zip(normalized, normalized[1:])):
            raise ValueError(f"{corruption} normalized severity must be increasing")
        if tuple(point.name for point in points) != tuple(f"s{i}" for i in range(7)):
            raise ValueError(f"{corruption} severity names must be s0...s6")
    expected_datasets = {"EEG", "HAR", "FD", "HHAR"}
    if set(DATASET_PHYSICAL_PROTOCOLS) != expected_datasets:
        raise ValueError("Dataset physical protocol registry is incomplete")
    for dataset, protocol in DATASET_PHYSICAL_PROTOCOLS.items():
        if protocol.dataset != dataset:
            raise ValueError(f"Dataset protocol key mismatch for {dataset}")
        if not protocol.held_out_view_families:
            raise ValueError(f"{dataset} has no held-out view family")
        if not protocol.physical_invariants:
            raise ValueError(f"{dataset} has no physical invariants")


def protocol_manifest() -> Dict[str, Any]:
    """Return a JSON-serializable, signed-run-ready protocol description."""

    validate_protocol()
    return {
        "version": PROTOCOL_VERSION,
        "primary_corruptions": list(PRIMARY_CORRUPTIONS),
        "severity_grids": {
            corruption: [asdict(point) for point in points]
            for corruption, points in PHYSICAL_SEVERITY_GRIDS.items()
        },
        "datasets": {
            dataset: asdict(protocol)
            for dataset, protocol in DATASET_PHYSICAL_PROTOCOLS.items()
        },
        "selection_metric": "post_adaptation_macro_f1_on_declared_development_split",
        "primary_predictive_risk": "samplewise_0_1_error",
        "full_no_ssaw_pairing": (
            "same_source_checkpoint_stream_order_batching_corruption_mask_and_seed"
        ),
        "target_labels_used_online": False,
        "realism_claim_without_real_reference": "physically_plausible_only",
        "clean_noninferiority_margin": "must_be_declared_from_source_variation_before_run",
    }


validate_protocol()
