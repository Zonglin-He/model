from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_ssaw_direction_bank_calibration import (
    PROFILE_KEY,
    candidate_profile,
    select_candidate,
    summarize_panel_b,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "future_macro_f1__confidence_only": [0.8, 0.9],
            "future_macro_f1__hard_ssaw": [0.7, 1.0],
            "heldout_flip_rate__confidence_only": [0.02, 0.04],
            "heldout_flip_rate__hard_ssaw": [0.01, 0.03],
            "heldout_worst_margin__confidence_only": [0.2, 0.3],
            "heldout_worst_margin__hard_ssaw": [0.25, 0.35],
            "heldout_consistency__confidence_only": [0.98, 0.96],
            "heldout_consistency__hard_ssaw": [0.99, 0.97],
            "ssaw_training_participation_rate__hard_ssaw": [0.5, 0.75],
        }
    )


def test_candidate_profile_changes_only_selected_flow_fields(tmp_path):
    base = {
        "profiles": {
            PROFILE_KEY: {
                "batch_size": 48,
                "learning_rate": 0.0001,
                "steps": 1,
                "ssaw_auxiliary_weight": 1.0,
            },
            "HHAR:3->8": {"steps": 1, "ssaw_auxiliary_weight": 1.0},
        }
    }
    candidate = candidate_profile(
        base, auxiliary_weight=2.0, log_strength=0.3, steps=2
    )
    assert candidate["profiles"][PROFILE_KEY]["ssaw_auxiliary_weight"] == 2.0
    assert candidate["profiles"][PROFILE_KEY]["spline_log_strength"] == 0.3
    assert candidate["profiles"][PROFILE_KEY]["steps"] == 2
    assert candidate["profiles"]["HHAR:3->8"] == base["profiles"]["HHAR:3->8"]
    json.dumps(candidate)


def test_selection_requires_non_degraded_f1_and_positive_mechanism():
    summary = summarize_panel_b(_panel())
    assert summary["flip_reduction"] == pytest.approx(0.01)
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "confidence_flip_rate": 0.03,
                "relative_flip_reduction": 0.2,
                "worst_margin_gain": 0.01,
                "consistency_gain": 0.01,
                "ssaw_training_participation": 0.5,
                "ssaw_auxiliary_weight": 1.0,
                "spline_log_strength": 0.2,
                "steps": 1,
                "flip_reduction": 0.006,
                "future_f1_delta": 0.0,
            },
            {
                "candidate_id": "b",
                "confidence_flip_rate": 0.03,
                "relative_flip_reduction": 0.3,
                "worst_margin_gain": 0.01,
                "consistency_gain": 0.01,
                "ssaw_training_participation": 0.5,
                "ssaw_auxiliary_weight": 2.0,
                "spline_log_strength": 0.2,
                "steps": 1,
                "flip_reduction": 0.009,
                "future_f1_delta": -0.001,
            },
        ]
    )
    assert select_candidate(rows)["candidate_id"] == "a"
    rows.loc[rows["candidate_id"] == "b", "future_f1_delta"] = 0.001
    assert select_candidate(rows)["candidate_id"] == "b"
