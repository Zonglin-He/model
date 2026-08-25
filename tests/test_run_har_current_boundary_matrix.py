from __future__ import annotations

import pandas as pd

from scripts.run_har_current_boundary_matrix import (
    AUDIT_PROFILE,
    EXECUTION_ORDER,
    FLOWS,
    INNER_STEPS,
    MATRIX_RUNNERS,
    _audit_summary,
    _duplicate_invariant,
    _promotion_decision,
)


def test_protocol_is_three_flows_seed1_two_steps_and_four_variants():
    assert FLOWS == (("6", "23"), ("9", "18"), ("12", "16"))
    assert INNER_STEPS == 2
    assert MATRIX_RUNNERS == (
        "N2_confidence_raw",
        "Fixed_KL_current_B4",
        "CurrentBoundary_KL",
        "CurrentBoundary_Dup",
    )
    assert EXECUTION_ORDER[:2] == (
        "N2_confidence_raw",
        "CurrentBoundary_Dup",
    )
    assert AUDIT_PROFILE["steps"] == 1
    assert AUDIT_PROFILE["enable_adaptation"] is False
    assert AUDIT_PROFILE["record_current_boundary_candidates"] is True


def test_no_training_audit_reports_spearman_and_conditional_reach():
    candidates = pd.DataFrame(
        {
            "scenario": ["6->23"] * 4,
            "source_valid": [True, True, True, False],
            "current_label_preserving": [True, True, True, True],
            "source_percentile": [0.2, 0.5, 0.9, 1.0],
            "source_percentile_delta": [0.1, 0.2, 0.3, 0.4],
            "probability_gap_reduction": [0.1, 0.2, 0.3, -0.1],
        }
    )
    samples = pd.DataFrame(
        {
            "scenario": ["6->23"] * 3,
            "raw_source_supported": [True, True, False],
            "raw_source_percentile": [0.8, 0.95, 0.7],
            "source_frontier_reach": [True, True, False],
            "current_boundary_reach": [True, False, True],
            "cap_hit": [False, False, True],
            "no_reach": [False, True, False],
            "selected_alpha": [0.1, 0.0, 0.3],
        }
    )
    observed = _audit_summary(candidates, samples).iloc[0]
    assert observed["source_percentile_gap_reduction_spearman"] == 1.0
    assert observed["source_frontier_reached_current_boundary_fraction"] == 0.5
    assert observed["current_boundary_reach_rate"] == 0.5
    assert observed["no_reach_rate"] == 0.5


def _sample_rows(runner: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["6->23", "6->23"],
            "runner": [runner, runner],
            "batch_index": [0, 0],
            "sample_index": [0, 1],
            "pseudo_label": [1, 2],
            "prediction": [1, 2],
            "post_update_prediction": [1, 2],
            "pre_final_update_prediction": [1, 2],
            "admitted": [True, False],
            "selected": [True, False],
        }
    )


def test_duplicate_invariant_requires_exact_f1_trajectory_and_zero_loss():
    raw = pd.DataFrame(
        [
            {"scenario": "6->23", "runner": "N2_confidence_raw", "f1": 0.9},
            {"scenario": "6->23", "runner": "CurrentBoundary_Dup", "f1": 0.9},
        ]
    )
    # The function expects all three registered flows; duplicate the same exact
    # evidence with distinct scenario labels.
    raw = pd.concat(
        [raw.assign(scenario=scenario) for scenario in ("6->23", "9->18", "12->16")],
        ignore_index=True,
    )
    samples = []
    batches = []
    for scenario in ("6->23", "9->18", "12->16"):
        for runner in ("N2_confidence_raw", "CurrentBoundary_Dup"):
            samples.append(_sample_rows(runner).assign(scenario=scenario))
            batches.append(
                pd.DataFrame(
                    {
                        "scenario": [scenario],
                        "runner": [runner],
                        "ssaw_consistency_loss": [0.0],
                        "ssaw_weighted_consistency_loss": [0.0],
                    }
                )
            )
    observed = _duplicate_invariant(
        raw, pd.concat(samples, ignore_index=True), pd.concat(batches, ignore_index=True)
    )
    assert observed["status"] == "passed"
    assert observed["compared_samples"] == 6


def test_promotion_requires_f1_gathered_boundary_and_stable_radius():
    rows = []
    for scenario in ("6->23", "9->18", "12->16"):
        rows.extend(
            (
                {
                    "scenario": scenario,
                    "runner": "N2_confidence_raw",
                    "f1": 0.90,
                    "heldout_stable_radius_mean": 0.10,
                },
                {
                    "scenario": scenario,
                    "runner": "CurrentBoundary_KL",
                    "f1": 0.905,
                    "heldout_stable_radius_mean": 0.11,
                },
            )
        )
    batches = pd.DataFrame(
        {
            "scenario": ["6->23", "9->18", "12->16"],
            "runner": ["CurrentBoundary_KL"] * 3,
            "current_boundary_gathered_violation_count": [0.0, 0.0, 0.0],
            "current_boundary_final_coverage": [0.2, 0.3, 0.1],
        }
    )
    observed = _promotion_decision(
        pd.DataFrame(rows), batches, {"status": "passed"}
    )
    assert observed["status"] == "passed"
    assert observed["heldout_stable_radius_mean_delta"] > 0.0
