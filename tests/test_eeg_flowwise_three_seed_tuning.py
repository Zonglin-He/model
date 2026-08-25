from __future__ import annotations

from scripts.run_eeg_flowwise_three_seed_tuning import STAGES, select_winner


def _row(value, full, delta, status="complete"):
    return {
        "status": status,
        "candidate_value": value,
        "full_f1_mean": full,
        "full_minus_no_ssaw_mean": delta,
    }


def test_selection_prioritizes_three_seed_full_then_delta_within_tolerance():
    winner = select_winner(
        [
            _row(1, 0.7000, 0.0000),
            _row(2, 0.6995, 0.0100),
            _row(3, 0.6980, 0.1000),
        ],
        f1_tolerance_pp=0.10,
    )
    assert winner["candidate_value"] == 2
    assert winner["stage_max_full_f1_mean"] == 0.7000


def test_failed_candidates_are_never_selected():
    winner = select_winner(
        [_row(1, 0.5, 0.5, status="failed"), _row(2, 0.4, 0.0)],
        f1_tolerance_pp=0.10,
    )
    assert winner["candidate_value"] == 2


def test_ssaw_weight_grid_is_strictly_positive():
    grid = dict(STAGES)["ssaw_auxiliary_weight"]
    assert min(grid) > 0.0
