from __future__ import annotations

import pandas as pd
import pytest

from scripts.finalize_core_online_ablation import PROTOCOL, RUNNERS, finalize


def _frame() -> pd.DataFrame:
    rows = []
    for dataset in ("HAR", "HHAR"):
        for flow in range(5):
            for seed in (0, 1, 2):
                for index, runner in enumerate(RUNNERS):
                    rows.append(
                        {
                            "status": "ok", "protocol": PROTOCOL,
                            "dataset": dataset, "scenario": f"{flow}->{flow + 1}",
                            "source_seed": seed, "stream_seed": 42,
                            "runner": runner,
                            "source_model_sha256": f"hash-{dataset}-{flow}-{seed}",
                            "f1": 0.7 + index * 0.01,
                            "target_labels_used_for_online_decision": False,
                            "target_labels_used_for_parameter_selection": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_core_finalizer_validates_120_cells_and_three_contributions():
    outputs = finalize(_frame())
    dataset, overall = outputs[1], outputs[2]
    contrast_overall, manifest = outputs[6], outputs[7]
    assert len(dataset) == 8
    assert len(overall) == 4
    assert manifest["cells"] == 120
    effects = contrast_overall.set_index("contrast")["paired_delta_mean"]
    assert effects["confidence_vs_raw"] == pytest.approx(0.01)
    assert effects["random_view_vs_confidence"] == pytest.approx(0.01)
    assert effects["hard_selection_vs_random"] == pytest.approx(0.01)


def test_core_finalizer_rejects_missing_cell():
    with pytest.raises(RuntimeError, match="expected 120"):
        finalize(_frame().iloc[:-1])
