from __future__ import annotations

import pandas as pd

from scripts.select_paper_representative_flows import score_flows, select_flows


def _rows() -> pd.DataFrame:
    rows = []
    flows = {
        "EEG": ("0->11", "12->5", "7->18", "16->1", "9->14"),
        "HAR": ("2->11", "6->23", "7->13", "9->18", "12->16"),
        "HHAR": ("0->6", "1->6", "2->7", "3->8", "4->5"),
    }
    for dataset, scenarios in flows.items():
        for flow_index, scenario in enumerate(scenarios):
            for source_seed in (1, 2, 3):
                for batch_index in (0, 1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "source_seed": source_seed,
                            "variant": "full",
                            "batch_index": batch_index,
                            "ssaw_training_participation_rate": 0.5,
                            "ssaw_admitted_participation_rate": 0.8,
                            "ssaw_weighted_consistency_loss": 0.01
                            * (flow_index + 1),
                            "raw_ce_loss": 0.1,
                        }
                    )
    return pd.DataFrame(rows)


def test_selection_uses_signal_density_and_returns_one_of_five_flows():
    scores = score_flows(_rows())
    selected = select_flows(scores)
    assert selected == {"EEG": "9->14", "HAR": "12->16", "HHAR": "4->5"}
    assert len(scores) == 15
    assert scores["source_seed_count"].eq(3).all()


def test_signal_density_is_bounded_by_training_coverage():
    scores = score_flows(_rows())
    assert (scores["signal_density"] >= 0).all()
    assert (scores["signal_density"] <= scores["training_coverage"]).all()
