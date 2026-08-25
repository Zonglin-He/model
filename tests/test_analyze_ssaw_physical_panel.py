from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_ssaw_physical_panel import (
    _holm_adjust,
    _paired_signflip_p,
    analyze,
    paired_auc,
    paired_cells,
    physical_auc_per_unit,
    summarize_panel,
)


def _panel():
    rows = []
    for cluster_index in range(3):
        for corruption in ("blackout", "attenuation"):
            for severity_index, severity in enumerate(np.linspace(0.0, 1.0, 7)):
                no_ssaw = 0.80 - 0.20 * severity - 0.01 * cluster_index
                delta = 0.0 if severity_index == 0 else 0.02
                for variant, f1 in (("no_ssaw", no_ssaw), ("full", no_ssaw + delta)):
                    rows.append(
                        {
                            "dataset": "HHAR",
                            "scenario": f"{cluster_index}->{cluster_index + 1}",
                            "corruption": corruption,
                            "severity_name": f"s{severity_index}",
                            "normalized_severity": severity,
                            "source_seed": 1,
                            "stream_seed": 42,
                            "corruption_seed": 7,
                            "source_model_sha256": f"checkpoint-{cluster_index}",
                            "variant": variant,
                            "f1": f1,
                        }
                    )
    return pd.DataFrame(rows)


def test_paired_panel_auc_and_cluster_summary():
    frame = _panel()
    pairs = paired_cells(frame)
    assert len(pairs) == 3 * 2 * 7
    assert set(pairs.loc[pairs.normalized_severity > 0, "full_minus_no_ssaw_f1"].round(8)) == {0.02}
    auc = physical_auc_per_unit(frame)
    auc_pairs = paired_auc(auc)
    # A linear curve with zero delta at s0 and +.02 at s1...s6.
    assert set(auc_pairs["full_minus_no_ssaw_physical_auc"].round(8)) == {
        round(0.02 * (1.0 - 1.0 / 12.0), 8)
    }
    summary = summarize_panel(pairs, auc_pairs, replicates=500, seed=3)
    assert summary.loc[0, "clean_full_minus_no_ssaw_f1"] == pytest.approx(0.0)
    assert summary.loc[0, "mean_physical_full_minus_no_ssaw_f1"] == pytest.approx(0.02)
    assert summary.loc[0, "physical_positive_cluster_fraction"] == 1.0
    assert summary.loc[0, "physical_cluster_ci95_low"] == pytest.approx(0.02)
    assert 0.0 < summary.loc[0, "physical_cluster_signflip_p_raw"] <= 1.0
    assert (
        summary.loc[0, "physical_cluster_signflip_p_holm"]
        >= summary.loc[0, "physical_cluster_signflip_p_raw"]
    )


def test_cluster_signflip_and_holm_are_bounded_and_deterministic():
    frame = pd.DataFrame(
        {
            "source_model_sha256": [f"checkpoint-{index}" for index in range(8)],
            "effect": [0.1] * 8,
        }
    )
    first = _paired_signflip_p(frame, "effect", replicates=2000, seed=7)
    second = _paired_signflip_p(frame, "effect", replicates=2000, seed=7)
    assert first == second
    assert 0.0 < first < 0.05
    adjusted = _holm_adjust([0.01, 0.04, 0.03])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_duplicate_or_unpaired_cells_fail_closed():
    frame = _panel()
    with pytest.raises(ValueError, match="Duplicate"):
        paired_cells(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="exactly one"):
        paired_cells(frame.iloc[1:].copy())


def test_end_to_end_writes_signed_outputs(tmp_path: Path):
    input_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "analysis"
    _panel().to_csv(input_path, index=False)
    summary = analyze(input_path, output_dir, replicates=200, seed=9)
    assert len(summary) == 1
    for name in (
        "paired_physical_cells.csv",
        "physical_auc_per_unit.csv",
        "paired_physical_auc.csv",
        "physical_panel_summary.csv",
        "manifest.json",
    ):
        assert (output_dir / name).is_file()
