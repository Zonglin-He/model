import json
import sys

import pandas as pd
import pytest

from scripts.analyze_controlled_safety import main


def _write_panel(root, *, include_post=True):
    rows = []
    aurc_rows = []
    for method, f1, post_aurc in (
        ("DuSafe", 0.8, 0.1),
        ("Tent", 0.7, 0.2),
    ):
        signature = f"signed-{method}"
        base = {
            "dataset": "HAR",
            "scenario": "12->16",
            "method": method,
            "variant": "full",
            "corruption": "signal_freeze",
            "severity": "moderate",
            "source_seed": 1,
            "stream_seed": 42,
            "corruption_seed": 1,
            "protocol_signature": signature,
        }
        rows.append(
            {
                **base,
                "f1": f1,
                "coverage": 0.5,
                "accepted_pseudo_label_accuracy": 0.9,
                "corruption_rejection_recall": 0.8,
                "clean_correct_false_rejection_rate": 0.1,
                "unsafe_update_rate": 0.05,
            }
        )
        aurc_rows.append(
            {
                **base,
                "risk_policy": "common_pre_update_top1_nll",
                "aurc": 0.9,
            }
        )
        if include_post:
            aurc_rows.append(
                {
                    **base,
                    "risk_policy": "common_post_update_top1_nll",
                    "aurc": post_aurc,
                }
            )
    pd.DataFrame(rows).to_csv(root / "summary_raw.csv", index=False)
    pd.DataFrame(aurc_rows).to_csv(
        root / "predictive_aurc_per_source_seed.csv", index=False
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "finalize_only": True,
                "requested_missing_job_count": 0,
                "requested_job_count": 2,
            }
        ),
        encoding="utf-8",
    )


def test_analysis_uses_post_update_aurc_only(tmp_path, monkeypatch):
    _write_panel(tmp_path, include_post=True)
    monkeypatch.setattr(sys, "argv", ["analyze", "--input_dir", str(tmp_path)])
    main()
    summary = pd.read_csv(tmp_path / "paired_method_summary.csv")
    assert summary.loc[summary["method"].eq("DuSafe"), "aurc_mean"].item() == 0.1
    assert (
        tmp_path / "paired_safety_analysis_policy.txt"
    ).read_text(encoding="utf-8").strip().endswith(
        "common_post_update_top1_nll"
    )


def test_analysis_rejects_missing_post_update_aurc(tmp_path, monkeypatch):
    _write_panel(tmp_path, include_post=False)
    monkeypatch.setattr(sys, "argv", ["analyze", "--input_dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="No AURC rows"):
        main()
