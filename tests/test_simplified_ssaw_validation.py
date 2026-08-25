import pandas as pd
import pytest

from scripts.run_simplified_ssaw_validation import (
    build_comparison,
    comparison_summary,
    row_key,
)


def _row(ablation, f1):
    return {
        "dataset": "HAR",
        "scenario": "2->11",
        "source_seed": 1,
        "test_time_seed": 1,
        "ablation": ablation,
        "f1": f1,
        "coverage": 0.8,
        "accepted_pseudo_label_accuracy": 0.9,
        "unsafe_update_rate": 0.1,
        "wrong_rejection_recall": 0.2,
        "correct_false_rejection_rate": 0.05,
    }


def test_row_key_uses_fixed_source_and_test_time_seed():
    assert row_key(_row("full", 0.9)) == ("HAR", "2->11", 1, 1)


def test_comparison_is_cellwise_and_preserves_legacy_references(tmp_path):
    output_dir = tmp_path / "new"
    legacy_dir = tmp_path / "legacy"
    (output_dir / "HAR").mkdir(parents=True)
    (legacy_dir / "HAR").mkdir(parents=True)
    pd.DataFrame([_row("full", 0.92)]).to_csv(
        output_dir / "HAR" / "raw.csv", index=False
    )
    pd.DataFrame(
        [
            _row("full", 0.90),
            _row("random_smooth_warp", 0.91),
            _row("no_source_supported_selection", 0.89),
            _row("random_no_source_support", 0.905),
            _row("no_ssaw", 0.87),
        ]
    ).to_csv(legacy_dir / "HAR" / "raw.csv", index=False)

    comparison = build_comparison(
        output_dir=output_dir,
        legacy_dir=legacy_dir,
        datasets=["HAR"],
    )
    assert comparison.loc[0, "delta_f1_vs_legacy_ranked_source_full"] == pytest.approx(
        0.02
    )
    summary = comparison_summary(comparison)
    row = summary[
        summary["reference"].eq("legacy_random_only")
        & summary["dataset"].eq("HAR")
    ].iloc[0]
    assert row["paired_f1_delta"] == pytest.approx(0.01)
    assert row["paired_wins"] == 1
