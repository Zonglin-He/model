from __future__ import annotations

import pandas as pd

from scripts.run_har_guarded_candidate_five_flow import (
    ABLATION_VARIANTS,
    BASELINE_PROFILE,
    HAR_FLOWS,
    _profile,
    _select_profile,
)


def test_har_protocol_has_exactly_the_five_registered_flows_and_one_profile():
    assert HAR_FLOWS == (
        ("2", "11"),
        ("6", "23"),
        ("7", "13"),
        ("9", "18"),
        ("12", "16"),
    )
    profile = _profile("steps_2", {"profile_id": "old", **BASELINE_PROFILE}, steps=2)
    assert profile["profile_id"] == "steps_2"
    assert profile["steps"] == 2
    assert profile["learning_rate"] == BASELINE_PROFILE["learning_rate"]
    assert len(ABLATION_VARIANTS) == 6


def test_profile_selection_preserves_full_f1_before_maximizing_ssaw_delta():
    summary = pd.DataFrame(
        [
            {
                "profile_id": "best_f1",
                "full_f1_mean": 0.900,
                "no_ssaw_f1_mean": 0.901,
                "full_minus_no_ssaw_mean": -0.001,
                "positive_pair_fraction": 0.4,
            },
            {
                "profile_id": "near_best_positive",
                "full_f1_mean": 0.897,
                "no_ssaw_f1_mean": 0.892,
                "full_minus_no_ssaw_mean": 0.005,
                "positive_pair_fraction": 0.8,
            },
            {
                "profile_id": "large_delta_but_bad_f1",
                "full_f1_mean": 0.880,
                "no_ssaw_f1_mean": 0.860,
                "full_minus_no_ssaw_mean": 0.020,
                "positive_pair_fraction": 1.0,
            },
        ]
    )
    selected = _select_profile(summary)
    assert selected["profile_id"] == "near_best_positive"
    assert selected["selected_full_minus_no_ssaw_mean"] == 0.005
