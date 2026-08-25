from types import SimpleNamespace

import pandas as pd
import pytest

from configs.har_frozen_profile import (
    FROZEN_HAR_TTA_PARAMS,
    validate_frozen_har_profile,
)
from scripts.aggregate_fd_gate_calibration_extended import (
    _read_status as read_fd_status,
    _select_quantile,
)
from scripts.aggregate_har_final_panel import (
    PRIMARY_METRICS,
    source_seed_inference,
)
from scripts.run_fd_gate_calibration_extended import (
    candidate_label,
    cell_key as fd_cell_key,
    expected_cell_keys,
)
from scripts.run_har_final_panel import (
    REQUIRED_SAFETY_METRICS,
    VARIANTS,
    _expected_keys,
    _metadata,
    atomic_write_json,
    build_command,
    summary_matches,
)


def test_har_final_panel_has_30_paired_jobs_and_60_variant_rows(tmp_path):
    keys = _expected_keys((1, 2, 3), 42, 1)
    assert len(keys) == 30
    assert len(set(keys)) == 30
    assert len(keys) * len(VARIANTS) == 60

    args = SimpleNamespace(
        controlled_script=tmp_path / "controlled.py",
        data_path=tmp_path / "data",
        device="cuda",
        backbone="CNN",
        registry="production",
        stream_seed=42,
        corruption_seed=1,
        pretrain_cache_dir=tmp_path / "cache",
        ssaw_auxiliary_weight=None,
    )
    command = build_command(
        args,
        flow="2->11",
        condition="clean",
        source_seed=1,
        output_dir=tmp_path / "cell",
    )
    assert command[command.index("--variants") + 1] == "full,no_ssaw"
    assert command[command.index("--corruption_fraction") + 1] == "0.0"

    args.ssaw_auxiliary_weight = 0.1
    weighted_command = build_command(
        args,
        flow="2->11",
        condition="clean",
        source_seed=1,
        output_dir=tmp_path / "weighted_cell",
    )
    assert weighted_command[weighted_command.index("--override") + 1] == (
        "ssaw_auxiliary_weight=0.1"
    )

    args.ssaw_auxiliary_weight = 2.5
    with pytest.raises(ValueError, match="frozen auxiliary weight"):
        build_command(
            args,
            flow="2->11",
            condition="clean",
            source_seed=1,
            output_dir=tmp_path / "invalid_weight_cell",
        )


def test_har_formal_profile_is_frozen_and_ssaw_is_active():
    observed = validate_frozen_har_profile()
    assert observed == FROZEN_HAR_TTA_PARAMS
    assert observed["dusafe_variant"] == "spline_residual"
    assert observed["steps"] == 2
    assert observed["spline_log_strength"] > 0.0
    assert observed["ssaw_auxiliary_weight"] > 0.0


def test_har_clean_cell_accepts_only_undefined_corruption_recall(tmp_path):
    metadata = _metadata(
        flow="2->11",
        condition="clean",
        source_seed=1,
        stream_seed=42,
        corruption_seed=1,
    )
    atomic_write_json(tmp_path / "cell_metadata.json", metadata)
    rows = []
    for variant in VARIANTS:
        row = {
            "dataset": "HAR",
            "scenario": "2->11",
            "method": "DuSafe",
            "variant": variant,
            "corruption": "signal_freeze",
            "severity": "moderate",
            "source_seed": 1,
            "stream_seed": 42,
            "corruption_seed": 1,
            "f1": 0.9,
            "coverage": 0.8,
            "accepted_pseudo_label_accuracy": 0.95,
            "corruption_rejection_recall": float("nan"),
            "clean_correct_false_rejection_rate": 0.1,
            "unsafe_update_rate": 0.05,
        }
        assert set(REQUIRED_SAFETY_METRICS).issubset(row)
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "summary_raw.csv", index=False)
    assert summary_matches(tmp_path, "2->11", "clean", 1, 42, 1)


def test_har_inference_averages_fixed_flows_before_testing_source_seeds():
    rows = []
    seed_deltas = {1: 0.01, 2: 0.0, 3: 0.0}
    for source_seed, f1_delta in seed_deltas.items():
        for flow in ("a", "b"):
            row = {
                "flow": flow,
                "condition": "clean",
                "source_seed": source_seed,
            }
            for metric in PRIMARY_METRICS:
                row[f"paired_{metric}_delta"] = (
                    float("nan")
                    if metric == "corruption_rejection_recall"
                    else f1_delta if metric == "f1" else 0.0
                )
            rows.append(row)

    seed_summary, inference = source_seed_inference(pd.DataFrame(rows))
    assert len(seed_summary) == 3
    assert set(seed_summary["n_flows"]) == {2}
    assert seed_summary.loc[
        seed_summary["source_seed"].eq(1), "paired_f1_delta_mean"
    ].item() == pytest.approx(0.01)

    f1 = inference[
        inference["condition"].eq("clean") & inference["metric"].eq("f1")
    ].iloc[0]
    assert f1["n_source_seeds"] == 3
    assert f1["mean_delta"] == pytest.approx(0.01 / 3.0)
    assert f1["exact_sign_flip_pvalue"] == pytest.approx(1.0)

    undefined = inference[
        inference["condition"].eq("clean")
        & inference["metric"].eq("corruption_rejection_recall")
    ].iloc[0]
    assert undefined["n_source_seeds"] == 0
    assert pd.isna(undefined["mean_delta"])


def test_fd_extended_grid_has_96_cells_and_preregistered_selection():
    keys = expected_cell_keys()
    assert len(keys) == 96
    assert len(set(keys)) == 96

    rows = []
    profiles = {
        0.95: (0.9000, 0.8000, 0.20, 0.40),
        0.975: (0.8995, 0.7990, 0.15, 0.35),
        0.99: (0.8990, 0.7985, 0.10, 0.30),
        # Ineligible because corrupted F1 is more than 0.002 below q=.95.
        1.0: (0.9000, 0.7970, 0.00, 0.20),
    }
    for quantile, (clean_f1, corrupt_f1, clean_fpr, corrupt_fpr) in profiles.items():
        rows.extend(
            [
                {
                    "candidate_label": candidate_label(quantile),
                    "confidence_keep_fraction": quantile,
                    "condition": "clean",
                    "source_domain": 0,
                    "source_seed": 1,
                    "f1": clean_f1,
                    "coverage": 1.0 - clean_fpr,
                    "clean_correct_false_rejection_rate": clean_fpr,
                },
                {
                    "candidate_label": candidate_label(quantile),
                    "confidence_keep_fraction": quantile,
                    "condition": "signal_freeze_moderate",
                    "source_domain": 0,
                    "source_seed": 1,
                    "f1": corrupt_f1,
                    "coverage": 1.0 - corrupt_fpr,
                    "clean_correct_false_rejection_rate": corrupt_fpr,
                },
            ]
        )
    selected, audit = _select_quantile(pd.DataFrame(rows), tolerance=0.002)
    assert selected == 0.99
    assert audit["target_labels_used_for_selection"] is False


def test_fd_status_reader_preserves_leading_zero_candidate_label(tmp_path):
    key = fd_cell_key(0.95, 0, "clean", 1, 42, 1)
    pd.DataFrame(
        [
            {
                "candidate_label": "095",
                "confidence_keep_fraction": 0.95,
                "source_domain": 0,
                "calibration_flow": "0->0",
                "condition": "clean",
                "source_seed": 1,
                "stream_seed": 42,
                "corruption_seed": 1,
                "output_dir": str(tmp_path / "cell"),
                "status": "completed",
            }
        ]
    ).to_csv(tmp_path / "cell_status.csv", index=False)

    rows = read_fd_status(tmp_path, (key,))

    assert set(rows) == {key}
    assert rows[key]["candidate_label"] == "095"
