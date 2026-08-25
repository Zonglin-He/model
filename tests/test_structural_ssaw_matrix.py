import pandas as pd
import pytest

from scripts.run_structural_ssaw_matrix import paired_summary


def row(runner, seed, f1):
    return {
        "dataset": "HAR",
        "scenario": "2->11",
        "source_seed": 1,
        "test_time_seed": seed,
        "runner": runner,
        "f1": f1,
    }


def test_component_gain_is_full_minus_structural_ablation():
    frame = pd.DataFrame(
        [
            row("full_components", 1, 0.90),
            row("no_entire_ssaw", 1, 0.85),
            row("full_components", 2, 0.80),
            row("no_entire_ssaw", 2, 0.82),
        ]
    )
    summary = paired_summary(frame).set_index("runner")
    assert summary.loc[
        "no_entire_ssaw", "component_gain_f1"
    ] == pytest.approx(0.015)
    assert summary.loc[
        "no_entire_ssaw", "component_helped_cells"
    ] == 1
    assert summary.loc[
        "no_entire_ssaw", "component_hurt_cells"
    ] == 1


def test_component_gain_accepts_an_explicit_reference_runner():
    frame = pd.DataFrame(
        [
            row("simplified_full_components", 1, 0.91),
            row("simplified_no_entire_ssaw", 1, 0.88),
        ]
    )
    summary = paired_summary(
        frame, reference_runner="simplified_full_components"
    ).set_index("runner")
    assert summary.loc[
        "simplified_no_entire_ssaw", "component_gain_f1"
    ] == pytest.approx(0.03)
    assert summary.loc[
        "simplified_no_entire_ssaw", "reference_runner"
    ] == "simplified_full_components"
