"""CPU-only tests for the strict horizon clustered analyzer."""

import hashlib

import pytest

import pandas as pd

from scripts.analyze_full_no_ssaw_horizon_queue import (
    inferential_summary,
    validate_unit_panel,
)
from scripts.run_full_no_ssaw_horizon_queue import _expected_key_set


def _panel() -> pd.DataFrame:
    rows = []
    for key in sorted(_expected_key_set()):
        parts = key.split("|")
        dataset, scenario = parts[:2]
        values = dict(part.split("=", 1) for part in parts[2:])
        source_domain = scenario.split("->", 1)[0]
        source_seed = int(values["source_seed"])
        partition = "target_selected_evaluation"
        overlap = True
        condition = (
            "clean"
            if values["corruption"] == "none"
            else f"{values['corruption']}:{values['severity']}"
        )
        checkpoint_hash = hashlib.sha256(
            f"{dataset}|{source_domain}|{source_seed}".encode()
        ).hexdigest()
        effect = 0.001 * source_seed
        rows.append(
            {
                "endpoint_key": key,
                "dataset": dataset,
                "scenario": scenario,
                "source_domain": source_domain,
                "source_seed": source_seed,
                "stream_seed": int(values["stream_seed"]),
                "horizon": int(values["horizon"]),
                "condition": condition,
                "evaluation_partition": partition,
                "parameter_selection_data_overlap": overlap,
                "target_labels_used_for_updates": False,
                "target_labels_used_for_parameter_selection": True,
                "target_labels_used_for_metrics": True,
                "source_checkpoint_hash": checkpoint_hash,
                "state_equivalence_failures": 0,
                "full_vs_no_ssaw_f1_delta_mean": effect,
                "full_vs_no_ssaw_true_label_nll_improvement_mean": effect / 2,
                "full_vs_no_ssaw_f1_beneficial_fraction": 1.0,
                "full_vs_no_ssaw_f1_harmful_fraction": 0.0,
                "full_vs_no_ssaw_nll_beneficial_fraction": 1.0,
                "full_vs_no_ssaw_nll_harmful_fraction": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_exact_2340_endpoint_grid_and_clustered_inference():
    panel = validate_unit_panel(_panel())
    assert len(panel) == 2340
    inference = inferential_summary(panel, replicates=100, seed=17)
    assert len(inference) == 96
    assert inference["cluster_mean"].gt(0).all()
    assert not inference["confirmatory"].any()
    assert inference["cluster_signflip_p_holm_confirmatory"].isna().all()


def test_selection_overlap_and_duplicate_endpoint_fail_closed():
    panel = _panel()
    panel.loc[0, "parameter_selection_data_overlap"] = False
    with pytest.raises(ValueError, match="selection-overlap"):
        validate_unit_panel(panel)

    panel = _panel()
    panel.loc[1, "endpoint_key"] = panel.loc[0, "endpoint_key"]
    with pytest.raises(ValueError, match="duplicated, missing, or unexpected"):
        validate_unit_panel(panel)
