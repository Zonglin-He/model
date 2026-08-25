from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_full_no_ssaw_paired_uncertainty import (
    PairedAnalysisError,
    _coerce_paired_format,
    analyze,
)


def _frame() -> pd.DataFrame:
    rows = []
    for dataset_index, dataset in enumerate(("EEG", "FD", "HAR", "HHAR")):
        for flow_index in range(5):
            for seed in (0, 1, 2):
                no_ssaw = 0.7 + dataset_index * 0.01 + flow_index * 0.001
                delta = (flow_index - 2) * 0.001
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": f"{flow_index}->{flow_index + 1}",
                        "source_seed": seed,
                        "stream_seed": 42,
                        "full_f1": no_ssaw + delta,
                        "no_ssaw_f1": no_ssaw,
                        "full_minus_no_ssaw": delta,
                        "source_model_sha256": f"hash-{dataset}-{flow_index}-{seed}",
                        "target_labels_used_for_online_decision": False,
                        "target_labels_used_for_parameter_selection": True,
                    }
                )
    return pd.DataFrame(rows)


def test_flow_cluster_analysis_has_expected_units_and_ties():
    per_flow, dataset, overall = analyze(_frame())
    assert len(per_flow) == 20
    assert len(dataset) == 4
    assert overall["paired_cells"] == 60
    assert overall["flow_clusters"] == 20
    assert overall["positive_flows"] == 8
    assert overall["tie_flows"] == 4
    assert overall["negative_flows"] == 8
    assert overall["paired_delta_mean"] == pytest.approx(0.0)


def test_analysis_rejects_duplicate_paired_cell():
    frame = _frame()
    with pytest.raises(PairedAnalysisError, match="duplicate"):
        analyze(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_v4_long_form_is_coerced_to_paired_cells():
    paired = _frame()
    rows = []
    for row in paired.to_dict("records"):
        shared = {
            key: row[key]
            for key in (
                "dataset",
                "scenario",
                "source_seed",
                "stream_seed",
                "source_model_sha256",
                "target_labels_used_for_online_decision",
                "target_labels_used_for_parameter_selection",
            )
        }
        rows.append({**shared, "runner": "hard_ssaw", "f1": row["full_f1"]})
        rows.append(
            {**shared, "runner": "confidence_only", "f1": row["no_ssaw_f1"]}
        )
    converted = _coerce_paired_format(pd.DataFrame(rows))
    assert len(converted) == 60
    assert set(converted.columns) >= {
        "full_f1",
        "no_ssaw_f1",
        "full_minus_no_ssaw",
    }
    _, _, overall = analyze(converted)
    assert overall["paired_delta_mean"] == pytest.approx(0.0)
