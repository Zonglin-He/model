import argparse
import sys

import pandas as pd
import pytest
import torch

from configs.benchmark_baselines import get_benchmark_hparams_class
from scripts.run_controlled_safety_benchmark import (
    PROBABILITY_RECORD_SCHEMA,
    REQUIRED_SAFETY_METRICS,
    admission_risk_score,
    attach_probability_records,
    common_predictive_risk_coverage,
    corruption_conditioned_probability_metrics,
    read_csv_records,
    risk_coverage,
    safety_job_key,
    safety_protocol_signature,
    safety_record_name,
    sample_record_matches,
    main as safety_main,
    try_safety_job_key,
    write_native_risk_artifacts,
    write_common_predictive_risk_artifacts,
)


def test_safety_contract_contains_all_required_joint_metrics():
    assert REQUIRED_SAFETY_METRICS == (
        "coverage",
        "accepted_pseudo_label_accuracy",
        "corruption_rejection_recall",
        "clean_correct_false_rejection_rate",
        "unsafe_update_rate",
    )


def test_binary_baseline_mask_does_not_fabricate_risk_coverage():
    records = pd.DataFrame(
        {
            "correct": [True, False],
            "admitted": [True, False],
        }
    )
    assert risk_coverage(
        records, {"risk_coverage_status": "not_available_no_continuous_score"}
    ) == []


def test_resume_reader_accepts_zero_byte_native_curve(tmp_path):
    path = tmp_path / "risk_coverage_raw.csv"
    path.touch()
    assert read_csv_records(path) == []


def test_safety_signature_changes_with_runtime_override(tmp_path):
    args = argparse.Namespace(
        registry="benchmark",
        backbone="CNN",
        data_path=str(tmp_path / "data"),
        pretrain_cache_dir=str(tmp_path / "pretrain"),
        overrides={},
        corruption_fraction=0.5,
        eata_fisher_samples=2000,
    )
    default = safety_protocol_signature(args, "HAR", "DuSafe", "full")
    args.overrides = {"confidence_keep_fraction": 0.95}
    calibrated = safety_protocol_signature(args, "HAR", "DuSafe", "full")
    assert default != calibrated


def test_signed_sample_record_is_required_for_resume(tmp_path):
    key = (
        "HAR", "12->16", "DuSafe", "full", "signal_freeze",
        "moderate", 1, 42, 1,
    )
    path = tmp_path / safety_record_name(key)
    row = {
        "dataset": key[0],
        "scenario": key[1],
        "method": key[2],
        "variant": key[3],
        "corruption": key[4],
        "severity": key[5],
        "source_seed": key[6],
        "stream_seed": key[7],
        "corruption_seed": key[8],
        "protocol_signature": "signed-v1",
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    assert safety_job_key(row) == key
    assert sample_record_matches(path, key, "signed-v1")
    assert not sample_record_matches(path, key, "signed-v2")
    assert try_safety_job_key({"dataset": "HAR"}) is None


def test_sample_record_filename_includes_scenario_to_prevent_flow_overwrite():
    prefix = ("HHAR",)
    suffix = ("DuSafe", "full", "blackout", "s3", 1, 42, 7)
    first = safety_record_name((*prefix, "0->6", *suffix))
    second = safety_record_name((*prefix, "1->6", *suffix))
    assert first != second
    assert "0__6" in first
    assert "1__6" in second


def test_complete_probability_records_and_standard_metrics_are_attached():
    records = pd.DataFrame(
        {
            "label": [0, 1],
            "prediction": [0, 1],
            "corrupted": [True, False],
            "sample_index": [3, 4],
        }
    )
    post_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    pre_logits = torch.tensor([[0.0, 3.0], [0.0, 3.0]])
    enriched, summary = attach_probability_records(
        records,
        post_logits,
        pre_logits,
        torch.tensor([0, 1]),
        calibration_bins=5,
    )
    assert enriched["probability_record_schema"].eq(
        PROBABILITY_RECORD_SCHEMA
    ).all()
    assert {
        "post_update_logit_0",
        "post_update_logit_1",
        "post_update_probability_0",
        "post_update_probability_1",
        "pre_final_update_logit_0",
        "pre_final_update_probability_1",
    }.issubset(enriched.columns)
    assert summary["post_update_accuracy"] == 1.0
    assert summary["pre_final_update_accuracy"] == 0.5
    assert summary["post_update_brier"] < summary["pre_final_update_brier"]
    assert summary["probability_class_count"] == 2
    conditioned = corruption_conditioned_probability_metrics(
        enriched, calibration_bins=5
    )
    assert conditioned["corrupted_sample_count"] == 1
    assert conditioned["clean_sample_count"] == 1
    assert conditioned["corrupted_post_update_macro_f1"] == 0.5
    assert conditioned["clean_post_update_macro_f1"] == 0.5


def test_finalize_only_verifies_signed_summary_without_running_gpu(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "safety"
    records_dir = output_dir / "sample_records"
    records_dir.mkdir(parents=True)
    signature_args = argparse.Namespace(
        registry="benchmark",
        backbone="CNN",
        data_path=str(tmp_path / "data"),
        pretrain_cache_dir=str(tmp_path / "pretrain"),
        overrides={},
        corruption_fraction=0.5,
        eata_fisher_samples=2000,
    )
    signature = safety_protocol_signature(
        signature_args, "HAR", "DuSafe", "full"
    )
    key = (
        "HAR", "12->16", "DuSafe", "full", "signal_freeze",
        "moderate", 1, 42, 1,
    )
    meta = {
        "dataset": key[0],
        "scenario": key[1],
        "method": key[2],
        "variant": key[3],
        "corruption": key[4],
        "severity": key[5],
        "source_seed": key[6],
        "stream_seed": key[7],
        "corruption_seed": key[8],
        "protocol_signature": signature,
    }
    pd.DataFrame(
        [
            {
                **meta,
                "f1": 0.8,
                "coverage": 0.5,
                "accepted_pseudo_label_accuracy": 0.9,
                "corruption_rejection_recall": 0.8,
                "clean_correct_false_rejection_rate": 0.1,
                "admission_corruption_rejection_recall": 0.8,
                "admission_clean_correct_false_rejection_rate": 0.1,
                "admitted_corruption_rate": 0.2,
                "unsafe_update_rate": 0.05,
            }
        ]
    ).to_csv(output_dir / "summary_raw.csv", index=False)
    pd.DataFrame(
        [
            {
                **meta,
                "risk_coverage_status": "available",
                "risk_score_policy": "source score",
                "risk_score_components": "nll",
                "admission_risk_score": 0.1,
                "pre_final_update_confidence": 0.9,
                "pre_final_update_correct": True,
                "confidence": 0.8,
                "correct": True,
            }
        ]
    ).to_csv(records_dir / safety_record_name(key), index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safety",
            "--data_path",
            str(tmp_path / "data"),
            "--registry",
            "benchmark",
            "--datasets",
            "HAR",
            "--methods",
            "DuSafe",
            "--corruptions",
            "signal_freeze",
            "--severities",
            "moderate",
            "--source_seeds",
            "1",
            "--stream_seeds",
            "42",
            "--corruption_seed",
            "1",
            "--pretrain_cache_dir",
            str(tmp_path / "pretrain"),
            "--output_dir",
            str(output_dir),
            "--finalize_only",
        ],
    )
    assert safety_main() == 0
    manifest = pd.read_json(output_dir / "manifest.json", typ="series")
    assert manifest["requested_missing_job_count"] == 0


def test_common_predictive_curve_is_available_without_native_gate_score():
    records = pd.DataFrame(
        {
            "pre_final_update_confidence": [0.9, 0.8, 0.2],
            "pre_final_update_correct": [True, False, False],
        }
    )
    rows = common_predictive_risk_coverage(
        records,
        {"method": "Tent", "risk_coverage_status": "not_available_no_continuous_score"},
    )
    assert len(rows) == 3
    assert rows[-1]["coverage"] == 1.0
    assert rows[-1]["selective_risk"] == pytest.approx(2 / 3)
    assert {row["risk_policy"] for row in rows} == {
        "common_pre_update_top1_nll"
    }


def test_predictive_curve_parses_false_strings_and_supports_post_update():
    records = pd.DataFrame(
        {
            "confidence": [0.9, 0.1],
            "correct": ["True", "False"],
        }
    )
    rows = common_predictive_risk_coverage(
        records, {"method": "Tent"}, stage="post_update"
    )
    assert rows[-1]["selective_risk"] == pytest.approx(0.5)
    assert rows[-1]["risk_policy"] == "common_post_update_top1_nll"


def test_common_predictive_artifacts_rebuild_from_partitioned_records(tmp_path):
    records_dir = tmp_path / "sample_records"
    records_dir.mkdir()
    meta = {
        "dataset": "HAR",
        "scenario": "2->11",
        "method": "Tent",
        "variant": "full",
        "corruption": "signal_freeze",
        "severity": "moderate",
        "source_seed": 1,
        "stream_seed": 42,
        "corruption_seed": 1,
        "protocol_signature": "test-signature",
    }
    pd.DataFrame(
        {
            **{key: [value, value] for key, value in meta.items()},
            "pre_final_update_confidence": [0.9, 0.1],
            "pre_final_update_correct": [True, False],
            "confidence": [0.8, 0.2],
            "correct": [True, False],
        }
    ).to_csv(records_dir / "job.csv", index=False)
    curves, aurc = write_common_predictive_risk_artifacts(
        records_dir, tmp_path
    )
    assert len(curves) == 4
    assert len(aurc) == 2
    assert set(pd.read_csv(tmp_path / "predictive_aurc_per_source_seed.csv")["method"]) == {
        "Tent"
    }


def test_native_curve_rebuild_skips_empty_records_and_recovers_dusafe(tmp_path):
    records_dir = tmp_path / "sample_records"
    records_dir.mkdir()
    (records_dir / "empty.csv").touch()
    meta = {
        "dataset": "HAR",
        "scenario": "2->11",
        "method": "DuSafe",
        "variant": "full",
        "corruption": "signal_freeze",
        "severity": "moderate",
        "source_seed": 1,
        "stream_seed": 42,
        "corruption_seed": 1,
        "protocol_signature": "test-signature",
    }
    pd.DataFrame(
        {
            **{key: [value, value] for key, value in meta.items()},
            "risk_coverage_status": ["available", "available"],
            "risk_score_policy": ["source score", "source score"],
            "risk_score_components": ["nll", "nll"],
            "admission_risk_score": [0.1, 0.9],
            "pre_final_update_correct": ["True", "False"],
        }
    ).to_csv(records_dir / "dusafe.csv", index=False)
    curves, aurc = write_native_risk_artifacts(records_dir, tmp_path)
    assert len(curves) == 2
    assert len(aurc) == 1


def test_dusafe_semantic_risk_uses_pre_update_prediction():
    records = pd.DataFrame(
        {
            "source_semantic_prediction": [1, 0],
            # The first sample changes only after an admitted update.
            "pre_final_update_prediction": [1, 0],
            "prediction": [0, 0],
            "raw_top1_nll": [0.1, 0.2],
        }
    )
    score, components = admission_risk_score(
        records,
        confidence_enabled=False,
        confidence_threshold=float("nan"),
        semantic_enabled=True,
    )
    assert components == ["fixed_source_semantic_disagreement"]
    assert score.tolist() == [0.0, 0.0]


def test_eata_benchmark_profile_requires_fisher_injection():
    for dataset in ("EEG", "HAR", "FD"):
        hparams = get_benchmark_hparams_class(dataset)().alg_hparams["EATA"]
        assert hparams["fisher_enabled"] is True
        assert hparams["fisher_samples"] == 2000
