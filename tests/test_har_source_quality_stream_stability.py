from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_har_source_quality_stream_stability import (
    NUM_DECILES,
    SOURCE_SEEDS,
    VARIANTS,
    _aggregate_deciles,
    _carry_source_states_to_deciles,
    _decile_indices,
    _evaluate_loader,
)


def test_decile_indices_assign_110_samples_to_ten_equal_ordered_blocks():
    assignments = _decile_indices(110)

    assert assignments.tolist()[:12] == [1] * 11 + [2]
    assert assignments.tolist()[-11:] == [10] * 11
    assert np.bincount(assignments, minlength=NUM_DECILES + 1)[1:].tolist() == [
        11
    ] * NUM_DECILES


@pytest.mark.parametrize("sample_count", [0, 9, 109, 111])
def test_decile_indices_fail_closed_for_non_registered_shapes(sample_count):
    with pytest.raises(ValueError, match="equal non-empty deciles"):
        _decile_indices(sample_count)


def test_source_state_is_carried_only_after_completed_deployment_batches():
    carried = _carry_source_states_to_deciles(
        sample_count=110,
        update_end_positions=[0, 48, 96, 110],
        source_f1_values=[1.0, 0.9, 0.8, 0.7],
    )

    # Decile endpoints are 11,22,...,110.  The state after 48 samples is not
    # available at endpoint 44, and the state after 96 is not available at 88.
    assert carried == [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7]


def _complete_decile_frame() -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(VARIANTS):
        for decile in range(1, NUM_DECILES + 1):
            for source_seed in SOURCE_SEEDS:
                base = 0.40 + 0.01 * method_index + 0.001 * decile
                rows.append(
                    {
                        "method": method,
                        "decile": decile,
                        "source_seed": source_seed,
                        "admission_coverage": base + 0.01 * source_seed,
                        "local_prequential_macro_f1": base + 0.02 * source_seed,
                        "cumulative_prequential_macro_f1": base
                        + 0.03 * source_seed,
                        "source_calibration_f1": base + 0.04 * source_seed,
                    }
                )
    return pd.DataFrame(rows)


def test_decile_aggregation_has_one_complete_three_seed_row_per_panel_point():
    raw = _complete_decile_frame()
    aggregated = _aggregate_deciles(raw)

    assert len(aggregated) == len(VARIANTS) * NUM_DECILES
    assert set(aggregated["method"]) == set(VARIANTS)
    assert set(aggregated["decile"]) == set(range(1, NUM_DECILES + 1))
    assert (aggregated["source_seed_count"] == len(SOURCE_SEEDS)).all()

    selected = aggregated[
        (aggregated["method"] == "Raw TTA")
        & (aggregated["decile"] == 1)
    ].iloc[0]
    assert selected["admission_coverage_mean"] == pytest.approx(0.411)
    assert selected["admission_coverage_std"] == pytest.approx(0.01)


@pytest.mark.parametrize("violation", ["duplicate", "missing", "out_of_range"])
def test_decile_aggregation_rejects_protocol_violations(violation):
    raw = _complete_decile_frame()
    if violation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        expected = "duplicate method-decile-source_seed"
    elif violation == "missing":
        raw = raw.drop(index=0).reset_index(drop=True)
        expected = "lacks exactly one row per source seed"
    else:
        raw.loc[0, "admission_coverage"] = 1.01
        expected = "outside"

    with pytest.raises(RuntimeError, match=expected):
        _aggregate_deciles(raw)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.classifier = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        return self.classifier(self.feature_extractor(inputs))


class _TinyAdapter:
    def __init__(self):
        self.model = _TinyModel()


def test_source_loader_audit_preserves_all_process_rng_streams():
    adapter = _TinyAdapter()
    samples = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 0])
    indices = torch.arange(3)
    loader = DataLoader(
        TensorDataset(samples, labels, indices), batch_size=2, shuffle=False
    )
    torch.manual_seed(123)
    np.random.seed(456)
    random.seed(789)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    score, count = _evaluate_loader(adapter, loader, num_classes=2)

    assert 0.0 <= score <= 1.0
    assert count == 3
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before
