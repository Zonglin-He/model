from scripts.run_eeg_ssaw_weight_sweep_queue import (
    parse_float_csv,
    parse_int_csv,
    select_weights,
)


def test_parse_grids_are_positive_and_unique():
    assert parse_float_csv("0.1,0.3,1,2,4") == (0.1, 0.3, 1.0, 2.0, 4.0)
    assert parse_int_csv("0,1,2") == (0, 1, 2)


def test_selection_uses_delta_inside_full_f1_tolerance_then_smaller_weight():
    records = []
    for scenario in ("0->11", "12->5", "7->18", "16->1", "9->14"):
        records.extend(
            [
                {
                    "scenario": scenario,
                    "weight": 0.1,
                    "full_f1": 0.8000,
                    "no_ssaw_f1": 0.7990,
                    "full_minus_no_ssaw": 0.0010,
                },
                {
                    "scenario": scenario,
                    "weight": 1.0,
                    "full_f1": 0.7995,
                    "no_ssaw_f1": 0.7970,
                    "full_minus_no_ssaw": 0.0025,
                },
                {
                    "scenario": scenario,
                    "weight": 2.0,
                    "full_f1": 0.7900,
                    "no_ssaw_f1": 0.7800,
                    "full_minus_no_ssaw": 0.0100,
                },
            ]
        )
    selected, report = select_weights(records, f1_tolerance_pp=0.10)
    assert set(selected.values()) == {1.0}
    assert len(report) == 5
