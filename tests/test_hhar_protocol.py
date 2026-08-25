import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from configs.data_model_configs import (
    HHAR,
    get_dataset_class,
    supports_ssaw_orientation,
    validate_scenario,
)
from configs.tta_hparams_new import get_hparams_class
from scripts.audit_hhar_schema import audit_hhar_dataset, audit_payload
from scripts.convert_hhar_adatime import convert_hhar
from scripts.hhar_protocol import (
    HHAR_DOMAIN_IDS,
    HHAR_LABEL_MAP,
    HHAR_USERS,
    source_normalization_manifest,
    validate_source_normalization_manifest,
)
from scripts.run_hhar_experiment_queue import build_queue, prerequisite_complete


EXPECTED_HHAR_FLOWS = (
    ("0", "6"),
    ("1", "6"),
    ("2", "7"),
    ("3", "8"),
    ("4", "5"),
    ("5", "0"),
    ("6", "1"),
    ("7", "4"),
    ("8", "3"),
    ("0", "2"),
)


def test_hhar_config_is_exactly_the_adatime_protocol():
    assert get_dataset_class("HHAR") is HHAR
    assert HHAR.num_classes == 6
    assert HHAR.input_channels == 3
    assert HHAR.sequence_len == 128
    assert HHAR.class_names == [
        "bike",
        "sit",
        "stairsdown",
        "stairsup",
        "stand",
        "walk",
    ]
    assert HHAR.scenarios == list(EXPECTED_HHAR_FLOWS)
    assert len(HHAR.scenarios) == 10
    assert supports_ssaw_orientation("HHAR")
    assert validate_scenario("HHAR", "0", "6") == ("0", "6")
    with pytest.raises(ValueError, match="Invalid HHAR scenario"):
        validate_scenario("HHAR", "6", "8")


def test_hhar_hparams_are_source_only_safe_defaults_not_tuned_claims():
    profile = get_hparams_class("HHAR")()
    assert profile.alg_hparams["DuSafe"]["normalization_reference"] == "source"
    assert profile.alg_hparams["NoAdap"]["normalization_reference"] == "source"
    assert profile.alg_hparams["DuSafe"]["spline_log_strength"] == 0.2
    assert "enable_source_semantic_router" not in profile.alg_hparams["DuSafe"]
    assert profile.source_train_params["batch_size"] == 16
    assert profile.train_params["batch_size"] == 48


def test_source_normalization_manifest_forbids_target_fit():
    manifest = source_normalization_manifest()
    validate_source_normalization_manifest(manifest)
    assert manifest["raw_windows_unstandardized"] is True
    assert manifest["normalization_variant"] == "fixed-source"
    assert manifest["notebook_sample_equivalent"] is False
    assert manifest["label_categories"] == list(HHAR_LABEL_MAP)
    assert manifest["window_grouping"] == (
        "global user+label groups, non-overlapping 128/128 windows"
    )
    assert manifest["runtime_scaler_ddof"] == 1
    assert manifest["scaler_fit_split"] == "train"
    assert manifest["target_scaler_fit_forbidden"] is True
    assert manifest["target_labels_used_for_scaler"] is False
    invalid = dict(manifest)
    invalid["scaler_fit_split"] = "target_test"
    with pytest.raises(ValueError, match="source train"):
        validate_source_normalization_manifest(invalid)


def test_missing_hhar_files_are_reported_without_fabrication(tmp_path):
    report = audit_hhar_dataset(tmp_path)
    assert report["status"] == "missing"
    assert len(report["missing_files"]) == 18
    assert report["errors"] == []


def test_audit_rejects_nan_payload():
    payload = {
        "samples": torch.full((1, 3, 128), float("nan")),
        "labels": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="NaN"):
        audit_payload(payload, domain="0", split="train", path=Path("fixture.pt"))


def _write_hhar_fixture(root: Path) -> None:
    raw_dir = root / "activity_recognition" / "Activity recognition exp"
    raw_dir.mkdir(parents=True)
    fields = [
        "Index",
        "Arrival_Time",
        "Creation_Time",
        "x",
        "y",
        "z",
        "User",
        "Model",
        "Device",
        "gt",
    ]
    path = raw_dir / "Phones_accelerometer.csv"
    index = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for user in HHAR_USERS:
            # Interleave labels to verify global user+label grouping rather
            # than contiguous label-run windowing.
            for _ in range(512):
                for label in HHAR_LABEL_MAP:
                    writer.writerow(
                        {
                            "Index": index,
                            "Arrival_Time": index,
                            "Creation_Time": index,
                            "x": 1.0,
                            "y": 2.0,
                            "z": 9.0,
                            "User": user,
                            "Model": "samsungold",
                            "Device": "samsungold_1",
                            "gt": label,
                        }
                    )
                    index += 1
        # A non-selected phone and a dropped incomplete/null row must not
        # enter any window or reset a user+label group.
        writer.writerow(
            {
                "Index": index,
                "Arrival_Time": index,
                "Creation_Time": index,
                "x": 99.0,
                "y": 99.0,
                "z": 99.0,
                "User": "a",
                "Model": "nexus4",
                "Device": "nexus4_1",
                "gt": "bike",
            }
        )
        writer.writerow(
            {
                "Index": index + 1,
                "Arrival_Time": index + 1,
                "Creation_Time": index + 1,
                "x": "",
                "y": 99.0,
                "z": 99.0,
                "User": "a",
                "Model": "samsungold",
                "Device": "samsungold_1",
                "gt": "null",
            }
        )


def test_streaming_converter_writes_cpu_raw_windows_and_manifest(tmp_path):
    _write_hhar_fixture(tmp_path)
    output = tmp_path / "processed"
    manifest = convert_hhar(
        hhar_root=tmp_path,
        output_dir=output,
        chunk_size=211,
    )
    assert manifest["normalization_applied_by_converter"] is False
    assert manifest["split"] == {
        "test_size": 0.30,
        "stratified": True,
        "random_state": 1,
    }
    assert manifest["model_filter"] == ["samsungold"]
    assert manifest["device_filter"] == []
    assert manifest["rows_seen"] == 9 * 6 * 512 + 2
    assert manifest["rows_dropped_na"] == 1
    assert manifest["rows_kept_after_filters"] == 9 * 6 * 512
    assert set(output.glob("train_*.pt"))
    assert set(output.glob("test_*.pt"))
    report = audit_hhar_dataset(output)
    assert report["status"] == "ok"
    assert len(report["files"]) == 18
    for row in report["files"]:
        assert row["shape"][1:] == [3, 128]
        assert set(row["labels_present"]) == set(range(6))
        assert row["normalization_applied"] is False
    persisted = json.loads(
        (output / "source_normalization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    validate_source_normalization_manifest(persisted)
    resumed = convert_hhar(
        hhar_root=tmp_path,
        output_dir=output,
        chunk_size=211,
    )
    assert resumed["source_windows_per_domain"] == {
        domain: 24 for domain in HHAR_DOMAIN_IDS
    }


def test_hhar_queue_is_exactly_ten_serial_jobs_and_cpu_by_default(tmp_path):
    jobs = build_queue(data_path=tmp_path / "data")
    assert len(jobs) == 10
    assert [job["scenario"] for job in jobs] == [
        f"{source}->{target}" for source, target in EXPECTED_HHAR_FLOWS
    ]
    assert all(job["device"] == "cpu" for job in jobs)
    assert all(not job["uses_gpu"] for job in jobs)
    assert prerequisite_complete(tmp_path / "missing-status.json") is False
