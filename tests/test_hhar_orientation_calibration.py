import argparse

import pandas as pd
import pytest

from scripts.calibrate_hhar_orientation_source import (
    PREREGISTERED_AUXILIARY_WEIGHTS,
    PREREGISTERED_LEARNING_RATES,
    PREREGISTERED_STEPS,
    PREREGISTERED_STRENGTHS,
    SECOND_STAGE_COORDINATE_STAGES,
    SOURCE_DOMAINS,
    SOURCE_SEEDS,
    _completed_keys,
    _select,
    _select_adaptation_profile,
    _stage_status_frame,
    adaptation_profile_rows,
    coordinate_profile_rows,
    validate_candidate_strengths,
)


def test_hhar_orientation_grid_is_physical_and_pre_registered():
    assert SOURCE_DOMAINS == tuple(str(value) for value in range(9))
    assert SOURCE_SEEDS == (1, 2, 3)
    assert PREREGISTERED_STRENGTHS == (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)
    assert validate_candidate_strengths("0,1,2,4,6,8") == list(
        PREREGISTERED_STRENGTHS
    )
    with pytest.raises(ValueError, match="pre-registered"):
        validate_candidate_strengths("0,3")
    with pytest.raises(ValueError, match="0-degree"):
        validate_candidate_strengths("1,2,4")


def test_stage2_coordinate_descent_registers_broad_non_cartesian_grid():
    assert set(PREREGISTERED_AUXILIARY_WEIGHTS) >= {1.0, 4.0, 8.0, 12.0}
    assert set(PREREGISTERED_LEARNING_RATES) >= {1e-4, 1e-3}
    assert set(PREREGISTERED_STEPS) >= {1, 2, 4, 8}
    assert [len(coordinate_profile_rows(stage)) for stage in SECOND_STAGE_COORDINATE_STAGES] == [
        5,
        5,
        4,
    ]
    # Coordinate descent has fewer profiles than the full Cartesian product.
    assert len(adaptation_profile_rows()) < (
        len(PREREGISTERED_AUXILIARY_WEIGHTS)
        * len(PREREGISTERED_LEARNING_RATES)
        * len(PREREGISTERED_STEPS)
    )


def test_orientation_selection_uses_worst_source_cell_and_f1_constraint():
    rows = []
    for strength, flip, kl, semantic, raw_f1, view_f1 in (
        (0.0, 0.0, 0.0, 0.0, 0.80, 0.80),
        (4.0, 0.005, 0.01, 0.02, 0.80, 0.795),
        # One source cell violates F1 even though its mean would pass.
        (6.0, 0.005, 0.01, 0.02, 0.80, 0.70),
    ):
        for source_domain in SOURCE_DOMAINS:
            rows.append(
                {
                    "source_domain": source_domain,
                    "source_seed": 1,
                    "strength_deg": strength,
                    "split": "source_test",
                    "label_flip_rate": flip,
                    "kl_mean": kl,
                    "semantic_distance_mean": semantic,
                    "raw_source_f1": raw_f1,
                    "view_source_f1_mean": view_f1,
                }
            )
    selected = _select(
        pd.DataFrame(rows),
        max_label_flip=0.01,
        max_kl=0.02,
        max_semantic_distance=0.03,
        max_f1_drop=0.01,
    )
    assert selected["selected_strength_deg"] == 4.0
    assert selected["target_labels_used"] is False
    assert "f1" in selected["selection_rule"]


def test_stage2_selection_is_paired_source_only_and_deterministic():
    rows = []
    for profile, clean_delta, corrupt_delta, ce_delta, unsafe_delta in (
        ("coord_auxiliary_weight_a1", 0.002, 0.001, 0.004, 0.0),
        ("coord_auxiliary_weight_a8", 0.004, 0.002, 0.005, 0.0),
    ):
        for domain in SOURCE_DOMAINS:
            for condition, f1_delta in (
                ("clean", clean_delta),
                ("signal_freeze_moderate", corrupt_delta),
            ):
                rows.append(
                    {
                        "coordinate_stage": "auxiliary_weight",
                        "source_domain": domain,
                        "source_seed": 1,
                        "profile": profile,
                        "condition": condition,
                        "auxiliary_weight": 1.0 if profile.endswith("a1") else 8.0,
                        "learning_rate": 1e-4,
                        "steps": 1,
                        "full_source_f1": 0.7,
                        "no_ssaw_source_f1": 0.7 - f1_delta,
                        "full_no_ssaw_f1_delta": f1_delta,
                        "full_no_ssaw_next_ce_delta": ce_delta,
                        "full_no_ssaw_unsafe_update_rate_delta": unsafe_delta,
                    }
                )
    selected = _select_adaptation_profile(
        pd.DataFrame(rows),
        max_clean_f1_drop=0.002,
        max_corruption_f1_drop=0.01,
        max_next_ce_delta=0.01,
        max_unsafe_update_delta=0.0,
    )
    assert selected["selected_profile"] == "coord_auxiliary_weight_a8"
    assert selected["target_labels_used"] is False


def test_resume_requires_completed_status_and_profile_cell_presence():
    rows = pd.DataFrame(
        [
            {
                "source_domain": "0",
                "source_seed": 1,
                "strength_deg": 4.0,
                "status": "completed",
            },
            {
                "source_domain": "1",
                "source_seed": 1,
                "strength_deg": 4.0,
                "status": "running",
            },
        ]
    )
    assert _completed_keys(rows) == {("0", 1, 4.0)}
    status = _stage_status_frame(
        [("auxiliary_weight", 4.0, "0", 1, "p", "clean")]
    )
    assert status.iloc[0]["status"] == "pending"
