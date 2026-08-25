import pandas as pd
import pytest

from scripts.run_ssaw_internal_ablation import (
    paired_summary,
    required_jobs,
    sanitized_tta_config,
    validate_rows,
)


def test_required_jobs_cover_both_atomic_ssaw_variants():
    jobs = required_jobs([("2", "11"), ("6", "23")], [1, 2, 3])
    assert len(jobs) == 2 * 3 * 2
    assert {ablation for _, _, ablation in jobs} == {
        "full",
        "no_ssaw",
    }


def test_required_jobs_can_run_a_targeted_paired_subset():
    jobs = required_jobs(
        [("2", "11")],
        [1, 2, 3],
        ("full", "no_ssaw"),
    )
    assert len(jobs) == 6
    assert {job[2] for job in jobs} == {
        "full",
        "no_ssaw",
    }


def test_deleted_signal_parameter_is_not_replayed_from_historical_state():
    state = {
        "tta_config": {
            "learning_rate": 1e-3,
            "signal_anomaly_quantile": 0.995,
            "ssaw_num_candidates": 8,
            "ssaw_selection_rule": "minimum_harder_entropy",
            "ssaw_enable_physical_warp": False,
            "ssaw_require_label_preservation": False,
        }
    }
    assert sanitized_tta_config(state) == {"learning_rate": 1e-3}


def test_result_rows_must_match_fixed_source_protocol():
    row = {
        "dataset": "HAR",
        "scenario": "2->11",
        "source_seed": 2,
        "test_time_seed": 1,
        "ablation": "full",
    }
    with pytest.raises(ValueError, match="seed mismatch"):
        validate_rows(
            [row],
            dataset="HAR",
            scenario_names={"2->11"},
            source_seed=1,
            test_time_seeds={1, 2, 3},
        )


def test_summary_uses_cellwise_paired_f1_difference():
    frame = pd.DataFrame(
        [
            {
                "dataset": "HAR",
                "scenario": "2->11",
                "source_seed": 1,
                "test_time_seed": 1,
                "ablation": "full",
                "f1": 0.90,
            },
            {
                "dataset": "HAR",
                "scenario": "2->11",
                "source_seed": 1,
                "test_time_seed": 1,
                "ablation": "no_ssaw",
                "f1": 0.80,
            },
        ]
    )
    summary = paired_summary(frame).set_index("ablation")
    assert summary.loc["full", "paired_f1_delta_vs_full"] == 0.0
    assert summary.loc[
        "no_ssaw", "paired_f1_delta_vs_full"
    ] == pytest.approx(-0.10)
