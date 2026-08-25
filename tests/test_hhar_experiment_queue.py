"""CPU-only protocol tests for the formal HHAR queue."""

import json
from pathlib import Path

import pytest

from scripts.run_hhar_experiment_queue import (
    FORMAL_CORRUPTIONS,
    FORMAL_FLOWS,
    FORMAL_METHODS,
    FORMAL_SEVERITIES,
    atomic_write_json,
    build_formal_plan,
    calibration_runtime_overrides,
    execute_plan,
    is_oom_text,
    is_retryable_failure,
    reject_target_selected_config,
    validate_smoke_output,
    validate_source_only_calibration_manifests,
)


def _complete_reviewer_status(path: Path) -> None:
    atomic_write_json(path, {"phase": "complete"})


def test_formal_plan_freezes_flows_seeds_methods_and_safety_grid(tmp_path):
    plan = build_formal_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "results",
        device="cpu",
    )

    assert plan["flows"] == [f"{source}->{target}" for source, target in FORMAL_FLOWS]
    assert plan["flow_count"] == 10
    assert plan["source_seeds"] == [1, 2, 3]
    assert plan["stream_seed"] == 42
    assert plan["methods"] == list(FORMAL_METHODS)
    assert plan["target_labels_used_for_selection"] is False
    assert plan["safety_corruptions"] == list(FORMAL_CORRUPTIONS)
    assert plan["safety_severities"] == list(FORMAL_SEVERITIES)
    stages = {stage["id"]: stage for stage in plan["stages"]}
    assert stages["controlled_safety"]["command_count"] == 10
    assert stages["controlled_safety"]["flow_count"] == 10
    assert set(stages) == {
        "schema_audit",
        "orientation_calibration",
        "smoke",
        "main_table",
        "controlled_safety",
        "full_no_ssaw",
        "factorial",
        "overhead",
    }
    assert all(stage["status"] != "blocked_unsupported" for stage in stages.values())
    assert stages["orientation_calibration"]["status"] == "planned"
    orientation_command = stages["orientation_calibration"]["command"]
    assert orientation_command[1].endswith("calibrate_hhar_orientation_source.py")
    assert "--skip-stage2" not in orientation_command
    assert stages["smoke"]["expected_cells"] == 11
    smoke_command = stages["smoke"]["command"]
    assert "0->6" in smoke_command
    assert ",".join(FORMAL_METHODS) in smoke_command
    assert stages["main_table"]["expected_cells"] == 330
    assert stages["controlled_safety"]["expected_cells"] == 3960
    assert stages["full_no_ssaw"]["expected_cells"] == 60
    assert stages["factorial"]["expected_cells"] == 240
    assert stages["overhead"]["expected_cells"] == 22
    assert stages["overhead"]["flow_count"] == 1
    assert stages["smoke"]["uses_gpu"] is False


def test_formal_plan_rejects_target_selected_configuration(tmp_path):
    with pytest.raises(ValueError, match="target-selected"):
        build_formal_plan(
            data_path=tmp_path / "data",
            target_labels_used_for_selection=True,
        )
    with pytest.raises(ValueError, match="target labels"):
        reject_target_selected_config(
            selection_config={"selection": {"target_labels_used_for_selection": True}}
        )


def test_cpu_dry_run_writes_atomic_manifest_without_launching_commands(tmp_path):
    prerequisite = tmp_path / "reviewer.json"
    _complete_reviewer_status(prerequisite)
    output = tmp_path / "queue"
    plan = build_formal_plan(data_path=tmp_path / "data", output_dir=output)

    returncode, payload = execute_plan(
        plan,
        output_dir=output,
        prerequisite_status=prerequisite,
        dry_run=True,
    )

    assert returncode == 0
    assert payload["status"] == "dry_run"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["dry_run"]
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["target_labels_used_for_selection"] is False
    assert not list(output.glob(".*.tmp"))


def test_incomplete_reviewer_queue_blocks_even_cpu_dry_run(tmp_path):
    prerequisite = tmp_path / "reviewer.json"
    atomic_write_json(prerequisite, {"phase": "running"})
    output = tmp_path / "queue"
    plan = build_formal_plan(data_path=tmp_path / "data", output_dir=output)

    returncode, payload = execute_plan(
        plan,
        output_dir=output,
        prerequisite_status=prerequisite,
        dry_run=True,
    )

    assert returncode == 2
    assert payload["status"] == "blocked_waiting_for_reviewer_queue"
    assert payload["dry_run"] is True


def test_oom_classifier_is_cpu_safe():
    assert is_oom_text("RuntimeError: CUDA out of memory")
    assert is_oom_text("CUDNN_STATUS_ALLOC_FAILED")
    assert not is_oom_text("normal nonzero subprocess exit")
    assert is_retryable_failure(1, "", False)
    assert is_retryable_failure(-1073741819, "", False)
    assert not is_retryable_failure(2, "ValueError: invalid override", False)


def test_calibration_validator_fails_closed_and_accepts_source_only_profiles(tmp_path):
    orientation = tmp_path / "manifest.json"
    tta = tmp_path / "selected_profile.json"
    orientation.write_text(
        json.dumps(
            {
                "dataset": "HHAR",
                "status": "complete",
                "target_labels_used": False,
                "target_labels_used_for_selection": False,
                "target_metrics_used": False,
                "target_data_used": False,
                "target_transfer_flows_excluded": True,
                "source_splits": ["source_train", "source_test"],
                "controlled_corruption": {"type": "signal_freeze"},
                "orientation_selection": {
                    "selected_strength_deg": 4.0,
                    "target_labels_used": False,
                },
                "second_stage": {
                    "status": "complete",
                    "selected_profile": {
                        "auxiliary_weight": 1.0,
                        "learning_rate": 1e-4,
                        "steps": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "selected_strength.json").write_text(
        json.dumps(
            {
                "dataset": "HHAR",
                "selected_strength_deg": 4.0,
                "target_labels_used": False,
                "target_metrics_used": False,
            }
        ),
        encoding="utf-8",
    )
    tta.write_text(
        json.dumps(
            {
                "orientation": {
                    "dataset": "HHAR",
                    "selected_strength_deg": 4.0,
                    "target_labels_used": False,
                },
                "adaptation": {
                    "selected_profile": "full_a1_lr0.0001_steps1",
                    "auxiliary_weight": 1.0,
                    "learning_rate": 1e-4,
                    "steps": 1,
                    "selection_rule": "paired source clean and controlled-corruption F1 floors",
                },
                "target_labels_used": False,
                "target_data_used": False,
            }
        ),
        encoding="utf-8",
    )
    assert validate_source_only_calibration_manifests(orientation, tta) == (True, [])
    assert calibration_runtime_overrides(tta) == {
        "ssaw_strength": 4.0,
        "ssaw_auxiliary_weight": 1.0,
        "learning_rate": 1e-4,
        "steps": 1,
        "ssaw_sigma": 0.0,
        "normalization_reference": "source",
    }
    tta.write_text(
        tta.read_text(encoding="utf-8").replace(
            '"target_data_used": false', '"target_data_used": true'
        ),
        encoding="utf-8",
    )
    valid, errors = validate_source_only_calibration_manifests(orientation, tta)
    assert valid is False
    assert any("target" in error for error in errors)


def test_smoke_validator_requires_exact_eleven_ok_cells(tmp_path):
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    columns = ["dataset", "scenario", "method", "source_seed", "stream_seed", "status", "is_oom"]
    rows = [
        {
            "dataset": "HHAR",
            "scenario": "0->6",
            "method": method,
            "source_seed": "1",
            "stream_seed": "42",
            "status": "ok",
            "is_oom": "False",
        }
        for method in FORMAL_METHODS
    ]
    import csv

    with (smoke / "per_source_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    assert validate_smoke_output(smoke) == (True, [])
    rows[-1]["status"] = "failed"
    with (smoke / "per_source_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    valid, errors = validate_smoke_output(smoke)
    assert valid is False
    assert any("failed/OOM" in error for error in errors)
