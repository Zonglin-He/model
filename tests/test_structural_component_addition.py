import pandas as pd
import pytest

from scripts.run_structural_component_addition import (
    ADDITION_STAGES,
    addition_summary,
)


def test_addition_summary_uses_adjacent_dedicated_runners():
    rows = []
    values = {
        "addition_raw_entropy": (0.50, 0.60),
        "addition_confidence": (0.70, 0.60),
        "addition_source_semantic": (0.75, 0.65),
        "addition_full_ssaw": (0.80, 0.60),
    }
    for runner, scores in values.items():
        for seed, score in enumerate(scores, start=1):
            rows.append(
                {
                    "dataset": "HAR",
                    "scenario": "2->11",
                    "source_seed": 1,
                    "test_time_seed": seed,
                    "runner": runner,
                    "f1": score,
                }
            )
    summary = addition_summary(pd.DataFrame(rows)).set_index("runner")
    assert set(summary["dataset"]) == {"HAR"}
    assert summary.loc[
        "addition_confidence", "incremental_f1_gain"
    ] == pytest.approx(0.10)
    assert summary.loc[
        "addition_source_semantic", "incremental_f1_gain"
    ] == pytest.approx(0.05)
    assert summary.loc[
        "addition_full_ssaw", "incremental_f1_gain"
    ] == pytest.approx(0.0)


def test_addition_summary_never_pairs_different_datasets():
    rows = []
    for dataset, offset in (("HAR", 0.0), ("EEG", 0.2)):
        for stage, _ in ADDITION_STAGES:
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": "0->1",
                    "source_seed": 1,
                    "test_time_seed": 1,
                    "runner": stage,
                    "f1": 0.5 + offset,
                }
            )
    summary = addition_summary(pd.DataFrame(rows))
    assert len(summary) == 8
    assert set(summary["dataset"]) == {"HAR", "EEG"}
