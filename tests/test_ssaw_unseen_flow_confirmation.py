from __future__ import annotations

import pandas as pd

from scripts.run_ssaw_unseen_flow_confirmation import (
    DEFAULT_PROFILE,
    build_plan,
    summarize,
)


def test_confirmation_plan_is_frozen_five_flow_three_seed(tmp_path):
    plan = build_plan(
        profile_path=DEFAULT_PROFILE,
        output_dir=tmp_path / "out",
        device="cpu",
    )
    assert plan["expected_cells"] == 15
    assert plan["evaluation_flows"] == ["5->0", "6->1", "7->4", "8->3", "0->2"]
    assert plan["source_seeds"] == [0, 1, 2]
    assert plan["confirmatory"] is True
    assert plan["parameter_selection_data_overlap"] is False
    assert all("--evaluation-role" in cell["command"] for cell in plan["cells"])
    assert all("confirmatory_v1" in cell["command"] for cell in plan["cells"])


def test_confirmation_summary_applies_preregistered_checks():
    rows = []
    variants = (
        "confidence_only",
        "matched_raw_duplicate",
        "random_eligible_spline",
        "hard_ssaw",
    )
    for flow_index, flow in enumerate(("5->0", "6->1", "7->4", "8->3", "0->2")):
        for seed in (0, 1, 2):
            for variant in variants:
                stable = {
                    "confidence_only": 0.10,
                    "matched_raw_duplicate": 0.101,
                    "random_eligible_spline": 0.102,
                    "hard_ssaw": 0.11,
                }[variant]
                flip = 0.10 if variant != "hard_ssaw" else 0.09
                rows.append(
                    {
                        "dataset": "HHAR",
                        "scenario": flow,
                        "source_seed": seed,
                        "variant": variant,
                        "future_macro_f1": 0.8,
                        "heldout_flip_rate": flip,
                        "heldout_worst_margin": 0.4 + (0.01 if variant == "hard_ssaw" else 0.0),
                        "heldout_consistency": 1.0 - flip,
                        "heldout_stable_radius_sum": stable * 10.0,
                        "heldout_stable_radius_admitted_count": 10.0,
                        "heldout_sample_count": 10.0,
                        "heldout_cap_stable_ray_successes": 70.0 if variant == "hard_ssaw" else 60.0,
                        "heldout_cap_stable_ray_total": 80.0,
                        "ssaw_training_participation_rate": 0.8 if variant == "hard_ssaw" else 0.0,
                    }
                )
    units, effects, report = summarize(pd.DataFrame(rows))
    assert len(units) == 60
    assert len(effects) == 15
    assert report["decision"] == "supports_stable_neighborhood_expansion"
    assert report["positive_flow_count"] == 5
    assert report["positive_seed_count"] == 3
