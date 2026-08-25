"""Unit checks for current-v2 audit metrics and protocol labels."""

from types import SimpleNamespace

import torch
from torch import nn

from scripts.run_current_v2_audit import (
    _band_energy_ratio,
    _physical_rows,
    _supported_method_status,
)


class _FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 4)

    def forward(self, inputs):
        return self.projection(inputs.flatten(1))


def _adapter():
    physical = SimpleNamespace(
        antithetic=True,
        last_warp_curve=torch.full((1, 2, 1, 8), 1.05),
        last_metadata={"antithetic": True},
    )
    return SimpleNamespace(
        source_semantic_feature_extractor=_FeatureExtractor(),
        ssaw=physical,
        _last_gate_log={"pseudo_labels": torch.tensor([0, 1])},
    )


def test_current_v2_spectrum_bands_are_complementary_for_low_frequency_signal():
    time = torch.linspace(0.0, 1.0, 64)
    signal = time.sin().view(1, 1, -1)
    low = _band_energy_ratio(signal, low=True)
    high = _band_energy_ratio(signal, low=False)
    assert low.item() > high.item()


def test_current_v2_rows_keep_antithetic_views_and_semantic_distance():
    raw = torch.randn(2, 1, 8)
    positive = raw * 1.05
    views = torch.stack((positive, 2.0 * raw - positive))
    rows = _physical_rows(
        raw=raw,
        views=views,
        adapter=_adapter(),
        labels=torch.tensor([0, 1]),
        indices=torch.tensor([10, 11]),
        metadata={"dataset": "EEG"},
    )
    assert len(rows) == 4
    assert {row["view_role"] for row in rows} == {
        "antithetic_positive",
        "antithetic_reflection",
    }
    assert all(row["source_semantic_distance"] >= 0.0 for row in rows)
    assert all(row["raw_correct_posthoc"] is True for row in rows[:2])


def test_current_v2_does_not_claim_unregistered_baselines_are_runnable():
    status = {row["method"]: row for row in _supported_method_status()}
    assert status["NoAdap"]["status"] == "runnable"
    for method in ("Tent", "EATA", "SAR", "ACCUPOfficial"):
        assert status[method]["status"] == "unavailable_current_registry"
