"""CPU synthetic coverage for the held-out SSAW mechanism panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_heldout_ssaw_mechanism import run_bundle
from ssaw_evaluation.heldout_mechanism import (
    HeldOutCase,
    HeldOutMechanismRunner,
    build_manifest,
    compute_mechanism_metrics,
    eeg_physical_invariants,
    fd_physical_invariants,
    har_physical_invariants,
    heldout_direction_diagnostics,
    mechanism_protocol_manifest,
    prediction_metrics,
    summarize_heldout_direction_diagnostics,
    validate_case,
)


def _signals(batch: int = 2, channels: int = 6, time: int = 256) -> torch.Tensor:
    axis = torch.arange(time, dtype=torch.float64) / 128.0
    waves = []
    for channel in range(channels):
        frequency = 3.0 + channel * 0.7
        waves.append(torch.sin(2.0 * torch.pi * frequency * axis))
    base = torch.stack(waves, dim=0)
    return base.unsqueeze(0).repeat(batch, 1, 1)


def _case(dataset: str, metadata: dict) -> HeldOutCase:
    return HeldOutCase(
        dataset=dataset,
        training_view_family="window_constant_gain",
        held_out_view_family="smooth_held_out_trajectory",
        held_out_trajectory="trajectory_test_01",
        held_out_operator="operator_test_01",
        training_seed=11,
        test_seed=29,
        metadata=metadata,
        algorithm="synthetic",
    )


def test_prediction_metrics_are_zero_for_identical_cpu_outputs():
    logits = torch.tensor([[2.0, 0.0, -1.0], [-1.0, 1.5, 0.0]])
    features = torch.tensor([[1.0, 2.0], [0.5, -2.0]])
    metrics = prediction_metrics(logits, logits.clone(), features, features.clone())
    for key in (
        "js_divergence",
        "kl_clean_to_heldout",
        "kl_heldout_to_clean",
        "prediction_flip_rate",
        "margin_degradation",
        "confidence_drop",
        "feature_cosine_distance",
    ):
        assert metrics[key] == pytest.approx(0.0, abs=1e-10)


def test_heldout_sobol_direction_metrics_use_confidence_admitted_denominator():
    raw = torch.tensor([[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]])
    candidates = torch.stack(
        (
            raw,
            torch.tensor([[2.0, 1.0, -1.0], [-1.0, 2.0, 0.0]]),
        )
    )
    diagnostics = heldout_direction_diagnostics(
        raw,
        candidates,
        confidence_mask=torch.tensor([True, False]),
    )
    metrics = summarize_heldout_direction_diagnostics(diagnostics)
    assert metrics["eligible_coverage"] == pytest.approx(1.0)
    assert metrics["margin_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["heldout_flip_rate"] == pytest.approx(0.0)
    assert metrics["confidence_admitted_count"] == pytest.approx(1.0)


def test_prediction_metrics_capture_flip_and_representation_change():
    clean_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    held_out_logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
    clean_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    held_out_features = -clean_features
    metrics = prediction_metrics(
        clean_logits, held_out_logits, clean_features, held_out_features
    )
    assert metrics["prediction_flip_rate"] == pytest.approx(1.0)
    assert metrics["feature_cosine_distance"] == pytest.approx(2.0)
    assert metrics["js_divergence"] > 0.0
    assert metrics["kl_divergence"] == metrics["kl_clean_to_heldout"]


@pytest.mark.parametrize("dataset", ["HAR", "HHAR"])
def test_har_and_hhar_invariants_cover_triads_jerk_energy_and_frequency(dataset):
    clean = _signals(channels=6)
    held_out = clean * torch.linspace(1.0, 1.2, clean.size(-1)).view(1, 1, -1)
    metadata = (
        {}
        if dataset == "HHAR"
        else {
            "sampling_rate_hz": 50.0,
            "sampling_rate_provenance": "UCI_HAR_protocol",
        }
    )
    metrics = har_physical_invariants(
        clean,
        held_out,
        metadata=metadata,
    )
    assert {
        "triad_norm_relative_error",
        "jerk_relative_error",
        "energy_relative_error",
        "dominant_periodic_frequency_shift",
    }.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["triad_norm_relative_error"] > 0.0


def test_eeg_invariants_cover_bandpower_coherence_spectrum_correlation_and_envelope():
    clean = _signals(channels=4)
    held_out = clean * 0.8
    metrics = eeg_physical_invariants(
        clean,
        held_out,
        metadata={
            "sampling_rate_hz": 100.0,
            "sampling_rate_provenance": "repository_EEG_protocol",
            "bands_hz": {
                "low": (1.0, 8.0),
                "high": (8.0, 30.0),
            },
        },
    )
    assert {
        "relative_bandpower_error",
        "spectral_coherence",
        "dominant_frequency_shift",
        "channel_correlation_distortion",
        "amplitude_envelope_correlation",
    }.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["amplitude_envelope_correlation"] == pytest.approx(1.0, abs=1e-5)


def test_fd_invariants_cover_order_envelope_kurtosis_and_rms():
    clean = _signals(channels=2)
    held_out = clean * 1.5
    metrics = fd_physical_invariants(
        clean,
        held_out,
        metadata={},
    )
    assert {
        "normalized_spectral_peak_shift_cycles_per_sample",
        "normalized_envelope_spectral_peak_shift_cycles_per_sample",
        "spectral_kurtosis_change",
        "rms_ratio",
    }.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["rms_ratio"] == pytest.approx(1.5, rel=1e-5)
    assert "order_frequency_peak_shift" not in metrics
    assert "raw_spectral_peak_shift_hz" not in metrics


def test_unified_compute_dispatches_all_four_datasets():
    signal = _signals(channels=6)
    logits = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
    features = signal.mean(dim=-1)
    common = {
        "clean_signal": signal,
        "held_out_signal": signal.clone(),
        "clean_logits": logits,
        "held_out_logits": logits.clone(),
        "clean_features": features,
        "held_out_features": features.clone(),
    }
    metadata_by_dataset = {
        "EEG": {
            "sampling_rate_hz": 100.0,
            "sampling_rate_provenance": "repository_EEG_protocol",
            "bands_hz": {"low": (1.0, 8.0), "high": (8.0, 30.0)},
        },
        "HAR": {
            "sampling_rate_hz": 50.0,
            "sampling_rate_provenance": "UCI_HAR_protocol",
        },
        "HHAR": {},
        "FD": {},
    }
    for dataset in ("EEG", "HAR", "HHAR", "FD"):
        metadata = metadata_by_dataset[dataset]
        result = compute_mechanism_metrics(dataset, metadata=metadata, **common)
        assert result["js_divergence"] == pytest.approx(0.0, abs=1e-10)
        assert all(np.isfinite(value) for value in result.values())


def test_runner_applies_separate_seeded_held_out_operator_on_cpu():
    signal = _signals(channels=3)

    def algorithm(inputs):
        features = inputs.mean(dim=-1)
        logits = torch.cat((features[:, :1], -features[:, :1]), dim=1)
        return {"logits": logits, "features": features}

    runner = HeldOutMechanismRunner(algorithm, algorithm_name="synthetic")
    case = _case(
        "HAR",
        {
            "sampling_rate_hz": 50.0,
            "sampling_rate_provenance": "UCI_HAR_protocol",
        },
    )
    result = runner.run_case(case, signal, lambda inputs, seed: inputs * 0.5)
    assert result["manifest"]["case"]["test_seed"] == 29
    assert result["manifest"]["claims"]["lpr_estimated"] is False
    assert result["metrics"]["feature_cosine_distance"] > 0.0


def test_manifest_protocol_explicitly_disclaims_real_nuisance_and_lpr():
    protocol = mechanism_protocol_manifest()
    assert set(protocol["datasets"]) == {"EEG", "HAR", "FD", "HHAR"}
    assert protocol["claims"]["real_nuisance_observed"] is False
    assert protocol["claims"]["lpr_estimated"] is False
    manifest = build_manifest(
        _case(
            "EEG",
            {
                "sampling_rate_hz": 100.0,
                "sampling_rate_provenance": "repository_EEG_protocol",
            },
        )
    )
    assert manifest["split_requirements"]["training_seed_and_test_seed_must_differ"]


def test_manifest_marks_unverified_fd_rotation_and_hhar_sample_axis():
    fd_manifest = build_manifest(_case("FD", {}))
    assert fd_manifest["rotation_rate_unverified"] is True
    assert fd_manifest["physical_metadata_policy"]["FD_order_metric_when_rotation_unverified"] is None
    hhar_manifest = build_manifest(_case("HHAR", {}))
    assert hhar_manifest["frequency_axis"] == "cycles_per_sample"


def test_fail_closed_shape_metadata_and_split_validation():
    clean = _signals(channels=3)
    with pytest.raises(ValueError, match="sampling_rate"):
        eeg_physical_invariants(clean[:, :2], clean[:, :2], metadata={})
    fd_metrics = fd_physical_invariants(
        clean[:, :1], clean[:, :1], metadata={}
    )
    assert "order_frequency_peak_shift" not in fd_metrics
    with pytest.raises(ValueError, match="provenance"):
        eeg_physical_invariants(
            clean[:, :2],
            clean[:, :2],
            metadata={"sampling_rate_hz": 100.0},
        )
    with pytest.raises(ValueError, match="shapes must match"):
        prediction_metrics(
            torch.zeros(2, 2),
            torch.zeros(1, 2),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
        )
    with pytest.raises(ValueError, match="separated"):
        validate_case(
            HeldOutCase(
                dataset="HAR",
                training_view_family="same",
                held_out_view_family="held_out",
                held_out_trajectory="trajectory",
                held_out_operator="operator",
                training_seed=1,
                test_seed=1,
                metadata={
                    "sampling_rate_hz": 50,
                    "sampling_rate_provenance": "UCI_HAR_protocol",
                },
            )
        )
    with pytest.raises(ValueError, match="real-nuisance/LPR"):
        validate_case(
            _case(
                "HAR",
                {
                    "sampling_rate_hz": 50,
                    "sampling_rate_provenance": "UCI_HAR_protocol",
                    "real_nuisance_observed": True,
                },
            )
        )


def test_npz_runner_writes_cpu_json_result(tmp_path: Path):
    signal = _signals(channels=3)
    logits = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
    features = signal.mean(dim=-1)
    input_path = tmp_path / "bundle.npz"
    output_path = tmp_path / "result.json"
    np.savez(
        input_path,
        clean_signal=signal.numpy(),
        held_out_signal=(signal * 0.9).numpy(),
        clean_logits=logits.numpy(),
        held_out_logits=logits.numpy(),
        clean_features=features.numpy(),
        held_out_features=(features * 0.9).numpy(),
    )
    case = _case(
        "HAR",
        {
            "sampling_rate_hz": 50.0,
            "sampling_rate_provenance": "UCI_HAR_protocol",
        },
    )
    result = run_bundle(input_path, case=case, output_path=output_path)
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["protocol_version"] == result["manifest"]["protocol_version"]
    assert payload["metrics"]["confidence_drop"] == pytest.approx(
        result["metrics"]["confidence_drop"]
    )
