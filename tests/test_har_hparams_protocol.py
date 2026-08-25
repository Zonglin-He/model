import argparse

import pandas as pd
import pytest

from scripts.calibrate_har_hparams_source import (
    CONDITIONS,
    _completed_keys,
    _profile_rows,
    _select,
)


def _grid_args():
    return argparse.Namespace(
        strengths=[4.0],
        auxiliary_weights=[1.0, 4.0, 6.0, 8.0, 12.0],
        kl_scales=[0.05],
    )


def test_source_grid_protocol_has_independent_source_seed_cells():
    profiles = _profile_rows(_grid_args())
    assert len(profiles) == 6  # 5 SSAW weights plus the no-SSAW control.
    expected = 5 * 3 * len(profiles) * len(CONDITIONS)
    assert expected == 180

    rows = pd.DataFrame(
        [
            {
                "source_domain": "2",
                "profile": "full_s4_a1_k0.05",
                "condition": "clean",
                "source_seed": seed,
                "stream_seed": 42,
            }
            for seed in (1, 2)
        ]
    )
    keys = _completed_keys(rows)
    assert ("2", "full_s4_a1_k0.05", "clean", 1) in keys
    assert ("2", "full_s4_a1_k0.05", "clean", 2) in keys
    assert len(keys) == 2

    # A legacy file without source_seed is not a valid v3 resume state.
    legacy = rows.drop(columns="source_seed")
    assert _completed_keys(legacy) == set()


def test_selection_aggregates_source_domains_and_source_seeds():
    rows = []
    domains = ("2", "6", "7", "9", "12")
    profiles = {
        "full_a": (0.810, 0.705, 0.02),
        "full_b": (0.820, 0.706, 0.04),
    }
    for domain in domains:
        for source_seed in (1, 2, 3):
            for condition, baseline in (("clean", 0.800), ("signal_freeze_moderate", 0.700)):
                rows.append(
                    {
                        "source_domain": domain,
                        "source_seed": source_seed,
                        "profile": "no_ssaw",
                        "variant": "no_ssaw",
                        "strength": 0.0,
                        "auxiliary_weight": 0.0,
                        "kl_scale": 0.0,
                        "condition": condition,
                        "f1": baseline,
                    }
                )
                for profile, (clean_f1, corrupt_f1, loss_ratio) in profiles.items():
                    rows.append(
                        {
                            "source_domain": domain,
                            "source_seed": source_seed,
                            "profile": profile,
                            "variant": "full",
                            "strength": 1.0,
                            "auxiliary_weight": 0.1,
                            "kl_scale": 0.05,
                            "condition": condition,
                            "f1": clean_f1 if condition == "clean" else corrupt_f1,
                            "diag_raw_ce_loss": 0.1,
                            "diag_ssaw_weighted_consistency_loss": 0.1 * loss_ratio,
                            "diag_ssaw_realized_consistency_ratio": loss_ratio,
                            "diag_ssaw_label_flip_rate": 0.01,
                            "diag_ssaw_training_participation_rate": 0.9,
                            "diag_ssaw_admitted_participation_rate": 1.0,
                        }
                    )

    summary, selection = _select(pd.DataFrame(rows))
    assert set(CONDITIONS).issubset(summary.columns)
    assert selection["selected_profile"] == "full_b"
    assert selection["selection_rule"] == (
        "dual_f1_floor_and_ratio_band_then_max_min_f1"
    )
    assert selection["selected_clean_delta"] == pytest.approx(0.020)
    assert selection["selected_corruption_delta"] == pytest.approx(0.006)
    assert selection["selected_loss_ratio_mean"] == pytest.approx(0.04)
    assert selection["target_data_used"] is False
