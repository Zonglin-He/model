import math

import numpy as np
import pytest

from configs.ssaw_evaluation_protocol import (
    DATASET_PHYSICAL_PROTOCOLS,
    PHYSICAL_SEVERITY_GRIDS,
    PRIMARY_CORRUPTIONS,
    canonical_dataset,
    get_dataset_physical_protocol,
    protocol_manifest,
    validate_protocol,
)
from utils.probability_metrics import (
    aurc_eaurc,
    classwise_expected_calibration_error,
    expected_calibration_error,
    multiclass_brier,
    multiclass_nll,
    predictive_risk_coverage,
    selective_macro_f1_curve,
    summarize_probability_metrics,
)


def test_physical_protocol_is_complete_monotone_and_held_out():
    validate_protocol()
    assert tuple(PHYSICAL_SEVERITY_GRIDS) == PRIMARY_CORRUPTIONS
    for points in PHYSICAL_SEVERITY_GRIDS.values():
        assert len(points) == 7
        assert [point.name for point in points] == [f"s{i}" for i in range(7)]
        assert [point.normalized for point in points] == sorted(
            point.normalized for point in points
        )
        assert points[0].normalized == 0.0
        assert points[-1].normalized == 1.0
    assert set(DATASET_PHYSICAL_PROTOCOLS) == {"EEG", "HAR", "FD", "HHAR"}
    for protocol in DATASET_PHYSICAL_PROTOCOLS.values():
        assert protocol.training_view_family not in protocol.held_out_view_families
        assert protocol.physical_invariants
        assert protocol.realism_claim_limit.startswith("physically_plausible")
    assert canonical_dataset("MFD") == "FD"
    assert get_dataset_physical_protocol("MFD").dataset == "FD"
    manifest = protocol_manifest()
    assert manifest["target_labels_used_online"] is False
    assert "same_source_checkpoint" in manifest["full_no_ssaw_pairing"]


def test_probability_metrics_known_binary_multiclass_case():
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
    )
    expected_nll = -np.mean(np.log([0.9, 0.2, 0.7, 0.4]))
    assert multiclass_nll(labels, probabilities) == pytest.approx(expected_nll)
    expected_brier = np.mean([0.02, 1.28, 0.18, 0.72])
    assert multiclass_brier(labels, probabilities) == pytest.approx(expected_brier)
    curve = predictive_risk_coverage(labels, probabilities)
    np.testing.assert_allclose(curve["coverage"], [0.25, 0.5, 0.75, 1.0])
    # Confidence ordering: correct, wrong, correct, wrong.
    np.testing.assert_allclose(curve["risk"], [0.0, 0.5, 1 / 3, 0.5])
    aurc = aurc_eaurc(labels, probabilities)
    assert aurc["aurc"] == pytest.approx(np.mean([0.0, 0.5, 1 / 3, 0.5]))
    assert aurc["eaurc"] >= 0.0
    summary = summarize_probability_metrics(labels, probabilities, calibration_bins=5)
    assert summary["accuracy"] == 0.5
    assert summary["nll"] == pytest.approx(expected_nll)
    assert summary["brier"] == pytest.approx(expected_brier)


def test_perfect_predictions_have_zero_metrics_and_eaurc():
    labels = np.array([0, 1, 2])
    probabilities = np.eye(3)
    assert multiclass_nll(labels, probabilities) == 0.0
    assert multiclass_brier(labels, probabilities) == 0.0
    assert expected_calibration_error(labels, probabilities, bins=3) == 0.0
    assert classwise_expected_calibration_error(labels, probabilities, bins=3) == 0.0
    values = aurc_eaurc(labels, probabilities)
    assert values == {"aurc": 0.0, "oracle_aurc": 0.0, "eaurc": 0.0}
    selective = selective_macro_f1_curve(labels, probabilities, coverages=(1.0,))
    assert selective["macro_f1"][0] == 1.0


@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([0], [[0.8, 0.3]]),
        ([2], [[0.5, 0.5]]),
        ([0, 1], [[0.5, 0.5]]),
        ([0], [[math.nan, math.nan]]),
    ],
)
def test_probability_validation_fails_closed(labels, probabilities):
    with pytest.raises(ValueError):
        multiclass_nll(labels, probabilities)
