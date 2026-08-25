from __future__ import annotations

import pandas as pd
import torch

from algorithms.dusafe_two_factor_ablation import (
    TWO_FACTOR_BITS,
    TWO_FACTOR_RUNNERS,
    TwoFactorBaseline,
    TwoFactorSSAWOnly,
)
from scripts.run_dusafe_replacement_ablation import (
    STUDIES,
    _publish_two_factor_paths,
)


def test_current_two_factor_registry_is_exact_2x2():
    assert set(TWO_FACTOR_BITS.values()) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert tuple(STUDIES["two_factor"]["runners"]) == tuple(TWO_FACTOR_RUNNERS)
    assert STUDIES["two_factor"]["full_runner"] == "F11_full"


def test_cells_without_a_accept_every_sample():
    for runner_class in (TwoFactorBaseline, TwoFactorSSAWOnly):
        runner = object.__new__(runner_class)
        selected = runner._confidence_admission_mask(
            torch.tensor([0.0, 1.0, float("inf")]),
            torch.tensor([0, 1, 2]),
        )
        assert selected.tolist() == [True, True, True]


def test_two_path_table_reuses_one_full_endpoint(tmp_path):
    f1 = {
        "F00_baseline": 0.70,
        "F10_baseline_plus_a_confidence": 0.80,
        "F01_baseline_plus_b_ssaw": 0.75,
        "F11_full": 0.90,
    }
    frame = pd.DataFrame(
        [
            {
                "status": "ok",
                "dataset": "HAR",
                "scenario": "12->16",
                "source_seed": 3,
                "runner": runner,
                "f1": value,
            }
            for runner, value in f1.items()
        ]
    )
    _publish_two_factor_paths(frame, tmp_path)
    row = pd.read_csv(tmp_path / "two_path_f1_table.csv").iloc[0]
    assert row["baseline_plus_a_plus_b_f1"] == row["baseline_plus_b_plus_a_f1"]
    assert abs(row["a_after_baseline"] - 0.10) < 1e-12
    assert abs(row["b_after_a"] - 0.10) < 1e-12
    assert abs(row["b_after_baseline"] - 0.05) < 1e-12
    assert abs(row["a_after_b"] - 0.15) < 1e-12
    assert abs(row["interaction"] - 0.05) < 1e-12
