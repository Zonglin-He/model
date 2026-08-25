"""CPU-only checks for the HHAR clean main-table queue."""

import json

import pandas as pd
import pytest

from scripts.run_hhar_five_flow_main_table_queue import (
    BASELINE_METHODS,
    DUSAFE_METHODS,
    EXPECTED_CELL_COUNT,
    FORMAL_FLOWS,
    METHODS,
    SOURCE_SEEDS,
    STREAM_SEED,
    build_plan,
    load_frozen_tta_config,
    merge_group_outputs,
    run_queue,
)


def _frozen_files(tmp_path):
    config = {"learning_rate": 3e-4, "steps": 8, "enable_ssaw": True}
    state = {
        "completed": True,
        "phase": "complete",
        "tta_config": config,
        "single_flow_protocol": {
            "evaluation_flows": list(FORMAL_FLOWS),
            "evaluation_partition": "target_selected_evaluation",
            "parameter_selection_data_overlap": True,
            "confirmatory": False,
            "target_labels_used_for_selection": True,
        },
    }
    manifest = {
        "status": "complete",
        "phase": "complete",
        "tuning_complete": True,
        "current_tta_config": config,
        "evaluation_flows": list(FORMAL_FLOWS),
        "evaluation_partition": "target_selected_evaluation",
        "parameter_selection_data_overlap": True,
        "confirmatory": False,
        "target_labels_used_for_selection": True,
    }
    state_path = tmp_path / "state.json"
    manifest_path = tmp_path / "manifest.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return state_path, manifest_path


def _group_frame(methods, signature=""):
    rows = []
    for scenario in FORMAL_FLOWS:
        for method in methods:
            for source_seed in SOURCE_SEEDS:
                row = {
                    "status": "ok",
                    "dataset": "HHAR",
                    "scenario": scenario,
                    "method": method,
                    "source_seed": source_seed,
                    "stream_seed": STREAM_SEED,
                    "run_signature": signature,
                    "source_model_sha256": f"model-{source_seed}",
                    "source_checkpoint_file_sha256": f"file-{source_seed}",
                    "f1": 0.5,
                    "accuracy": 0.5,
                    "auroc": 0.5,
                }
                if method == "EATA":
                    row.update(
                        {
                            "fisher_enabled": True,
                            "fisher_cache_path": "fisher.pt",
                            "fisher_cache_hash": "fisher-hash",
                            "fisher_source_checkpoint_sha256": f"model-{source_seed}",
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def test_plan_is_five_flow_165_cells_and_separates_overrides(tmp_path):
    state, manifest = _frozen_files(tmp_path)
    plan = build_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "out",
        frozen_state=state,
        frozen_manifest=manifest,
    )
    assert plan["flows"] == list(FORMAL_FLOWS)
    assert plan["methods"] == list(METHODS)
    assert plan["expected_cells"] == EXPECTED_CELL_COUNT == 165
    assert plan["expected_groups"] == EXPECTED_CELL_COUNT == 165
    assert len(plan["groups"]) == EXPECTED_CELL_COUNT
    baseline = [group for group in plan["groups"] if group["method"] in BASELINE_METHODS]
    dusafe = [group for group in plan["groups"] if group["method"] in DUSAFE_METHODS]
    assert len(baseline) == 150
    assert len(dusafe) == 15
    assert all("--override" not in group["command"] for group in baseline)
    assert all(
        group["command"].count("--override") == len(plan["frozen_tta_config"])
        for group in dusafe
    )
    assert plan["evaluation_partition"] == "target_selected_evaluation"
    assert plan["selection_overlap"] is True
    assert plan["confirmatory"] is False
    from scripts.run_hhar_five_flow_main_table_queue import validate_cells

    assert validate_cells(plan["groups"])["passed"] is True


def test_frozen_state_and_manifest_must_agree_on_profile(tmp_path):
    state, manifest = _frozen_files(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["current_tta_config"]["steps"] = 9
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="tta_config differ"):
        load_frozen_tta_config(state, manifest)


def test_incomplete_early_freeze_is_rejected(tmp_path):
    state, manifest = _frozen_files(tmp_path)
    state_payload = json.loads(state.read_text(encoding="utf-8"))
    state_payload["phase"] = "tuning"
    state_payload["completed"] = False
    state.write_text(json.dumps(state_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state.phase|state.completed"):
        load_frozen_tta_config(state, manifest)


def test_merge_validates_exact_165_keys_and_shared_source_hashes():
    baseline = _group_frame(BASELINE_METHODS)
    dusafe = _group_frame(DUSAFE_METHODS)
    merged = merge_group_outputs(baseline, dusafe)
    assert len(merged) == EXPECTED_CELL_COUNT
    assert set(merged["evaluation_partition"]) == {"target_selected_evaluation"}
    assert set(merged["selection_overlap"]) == {True}
    assert set(merged["confirmatory"]) == {False}


def test_cpu_dry_run_writes_status_without_child_process(tmp_path):
    state, manifest = _frozen_files(tmp_path)
    plan = build_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "out",
        frozen_state=state,
        frozen_manifest=manifest,
    )
    code, status = run_queue(plan, dry_run=True)
    assert code == 0
    assert status["status"] == "dry_run"
    assert status["expected_cells"] == 165
    assert status["expected_groups"] == 165
    assert (tmp_path / "out" / "manifest.json").is_file()
    assert not list((tmp_path / "out").glob(".*.tmp"))


def test_native_crash_and_oom_are_retryable_without_gpu():
    from scripts.run_hhar_five_flow_main_table_queue import is_retryable_failure

    assert is_retryable_failure(0xC0000005, "", False)
    assert is_retryable_failure(-1073741819, "", False)
    assert is_retryable_failure(1, "MemoryError: cannot allocate memory", False)
    assert not is_retryable_failure(2, "invalid override", False)
