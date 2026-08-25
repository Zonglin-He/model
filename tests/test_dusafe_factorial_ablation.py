import pandas as pd
import pytest
import torch

from ablation_runners.dusafe_factorial import (
    FACTORIAL_RUNNER_SPECS,
    RUNNER_BY_BITS,
    get_factorial_runner,
)
from scripts.run_dusafe_factorial_ablation import (
    bundle_effect_rows,
    factorial_effect_rows,
    synergy_summary,
)
from scripts.dusafe_factorial_runner_common import (
    row_key,
    tensor_state_sha256,
    validate_rows,
)


def _factorial_frame(values):
    rows = []
    for bits, runner in RUNNER_BY_BITS.items():
        rows.append(
            {
                "dataset": "HAR",
                "scenario": "2_to_11",
                "source_seed": 1,
                "stream_seed": 42,
                "runner": runner,
                "factor_ssaw": bits[0],
                "factor_confidence": bits[1],
                "factor_semantic": bits[2],
                "f1": values[bits],
            }
        )
    return pd.DataFrame(rows)


def test_factorial_runner_matrix_covers_every_binary_cell():
    assert len(FACTORIAL_RUNNER_SPECS) == 8
    assert set(RUNNER_BY_BITS) == {
        (w, c, s) for w in (0, 1) for c in (0, 1) for s in (0, 1)
    }
    for name, spec in FACTORIAL_RUNNER_SPECS.items():
        runner = get_factorial_runner(name)
        assert runner is spec.runner_class
        assert runner.factor_ssaw is spec.ssaw
        assert runner.factor_confidence is spec.confidence
        assert runner.factor_semantic is spec.semantic


def test_factorial_effects_recover_pair_and_triple_interactions():
    # y = 1 + 2W + 3C + 5S + 7WC + 11WS + 13CS + 17WCS
    values = {}
    for w, c, s in RUNNER_BY_BITS:
        values[(w, c, s)] = (
            1
            + 2 * w
            + 3 * c
            + 5 * s
            + 7 * w * c
            + 11 * w * s
            + 13 * c * s
            + 17 * w * c * s
        )
    effects = factorial_effect_rows(_factorial_frame(values)).set_index(
        "effect"
    )
    assert effects.loc["W×C|S0", "value"] == pytest.approx(7.0)
    assert effects.loc["W×C|S1", "value"] == pytest.approx(24.0)
    assert effects.loc["C×S|W0", "value"] == pytest.approx(13.0)
    assert effects.loc["C×S|W1", "value"] == pytest.approx(30.0)
    assert effects.loc["W×C×S", "value"] == pytest.approx(17.0)


def test_synergy_requires_full_to_beat_both_component_bundles():
    values = {bits: 0.50 for bits in RUNNER_BY_BITS}
    values[(0, 0, 0)] = 0.50
    values[(1, 0, 0)] = 0.55
    values[(0, 1, 1)] = 0.56
    values[(1, 1, 1)] = 0.65
    summary = synergy_summary(_factorial_frame(values)).iloc[0]
    assert bool(summary["full_is_best_mean"])
    assert bool(summary["positive_bundle_interaction"])
    assert summary["bundle_interaction"] == pytest.approx(0.04)
    assert summary["strict_dominance_cells"] == 1


def test_bundle_effects_recover_a_by_b_interaction():
    values = {bits: 0.50 for bits in RUNNER_BY_BITS}
    values[(0, 0, 0)] = 0.50
    values[(1, 0, 0)] = 0.55
    values[(0, 1, 1)] = 0.56
    values[(1, 1, 1)] = 0.65
    effects = bundle_effect_rows(_factorial_frame(values)).set_index("effect")
    assert effects.loc["B|A0", "value"] == pytest.approx(0.05)
    assert effects.loc["B|A1", "value"] == pytest.approx(0.09)
    assert effects.loc["A|B0", "value"] == pytest.approx(0.06)
    assert effects.loc["A|B1", "value"] == pytest.approx(0.10)
    assert effects.loc["A×B", "value"] == pytest.approx(0.04)


def test_factorial_resume_key_uses_independent_source_seed():
    rows = [
        {
            "dataset": "HAR",
            "scenario": "2_to_11",
            "source_seed": seed,
            "stream_seed": 42,
            "runner": "full",
        }
        for seed in (1, 2, 3)
    ]
    validated = validate_rows(
        rows,
        runner="full",
        dataset="HAR",
        scenarios={"2_to_11"},
        source_seeds={1, 2, 3},
        stream_seed=42,
    )
    assert len({row_key(row) for row in validated}) == 3
    legacy = dict(rows[0])
    legacy.pop("stream_seed")
    legacy["test_time_seed"] = 1
    with pytest.raises(KeyError):
        row_key(legacy)


def test_factorial_checkpoint_hash_tracks_source_state_exactly():
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2)
    first = tensor_state_sha256(model)
    assert len(first) == 64
    assert first == tensor_state_sha256(model)
    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    assert tensor_state_sha256(model) != first
