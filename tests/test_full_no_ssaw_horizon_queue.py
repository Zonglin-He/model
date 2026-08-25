"""CPU-only protocol tests for the process-isolated horizon queue."""

import json
from pathlib import Path

import pytest

from scripts.run_full_no_ssaw_horizon_queue import (
    FORMAL_CORRUPTIONS,
    FORMAL_SEVERITIES,
    HORIZONS,
    GPU_LOCK_PATH,
    SOURCE_SEEDS,
    build_queue,
    execute_queue,
    expected_cell_count,
    expected_stream_cell_count,
    frozen_hhar_provenance,
    is_retryable_failure,
    _gpu_lock,
    validate_cells,
)
from scripts.run_full_main_table import GPUExperimentLock


def test_formal_grid_count_and_key_set_are_complete(tmp_path):
    plan = build_queue(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "queue",
        pretrain_cache_dir=tmp_path / "cache",
        hhar_frozen_state=None,
    )
    assert expected_stream_cell_count() == 780
    assert expected_cell_count() == 2340
    assert plan["expected_cell_count"] == 2340
    assert len(plan["cells"]) == 780
    assert validate_cells(plan["cells"])["passed"] is True
    keys = {cell["key"] for cell in plan["cells"]}
    assert len(keys) == len(plan["cells"])
    assert set(plan["source_seeds"]) == set(SOURCE_SEEDS)
    assert set(plan["horizons"]) == set(HORIZONS)
    assert plan["corruptions"] == ["clean", *FORMAL_CORRUPTIONS]
    assert plan["severities"] == list(FORMAL_SEVERITIES)
    assert all(
        cell["target_labels_used_for_updates"] is False
        and cell["target_labels_used_for_parameter_selection"] is True
        and cell["target_labels_used_for_metrics"] is True
        and cell["evaluation_partition"] == "target_selected_evaluation"
        for cell in plan["cells"]
    )
    assert all(cell["horizons"] == [1, 3, 5] for cell in plan["cells"])
    assert all(len(cell["expected_endpoint_keys"]) == 3 for cell in plan["cells"])


def test_representative_horizon_subset_builds_har_six_stream_cells(tmp_path):
    plan = build_queue(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "queue",
        pretrain_cache_dir=tmp_path / "cache",
        hhar_frozen_state=None,
        datasets=("HAR",),
        scenarios={"HAR": ("12->16",)},
        conditions=("clean", "signal_freeze:severe"),
        source_seeds=(1, 2, 3),
        horizons=(1, 3, 5),
    )
    assert len(plan["cells"]) == 6
    assert plan["expected_stream_cell_count"] == 6
    assert plan["expected_cell_count"] == 18
    assert plan["scenario_scope"] == "registered_representative_subset"
    assert plan["conditions"] == ["clean", "signal_freeze:severe"]
    assert all(cell["scenario"] == "12->16" for cell in plan["cells"])
    assert all(cell["horizons"] == [1, 3, 5] for cell in plan["cells"])
    assert validate_cells(
        plan["cells"],
        datasets=("HAR",),
        scenarios={"HAR": ("12->16",)},
        conditions=("clean", "signal_freeze:severe"),
        source_seeds=(1, 2, 3),
        horizons=(1, 3, 5),
    )["passed"] is True


def test_representative_horizon_filter_rejects_unregistered_condition(tmp_path):
    with pytest.raises(ValueError, match="registered corruption"):
        build_queue(
            data_path=tmp_path / "data",
            output_dir=tmp_path / "queue",
            pretrain_cache_dir=tmp_path / "cache",
            hhar_frozen_state=None,
            datasets=("HAR",),
            scenarios={"HAR": ("12->16",)},
            conditions=("signal_freeze:extreme",),
        )


def test_hhar_frozen_config_provenance_is_fingerprinted_and_forwarded(tmp_path):
    frozen = tmp_path / "selected_profile.json"
    frozen.write_text(
        json.dumps(
            {
                "profile_id": "hhar-source-only-v1",
                "target_labels_used": False,
                "target_data_used": False,
                "orientation": {"selected_strength_deg": 4.0},
                "adaptation": {
                    "auxiliary_weight": 1.0,
                    "learning_rate": 1e-4,
                    "steps": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = frozen_hhar_provenance(frozen)
    assert provenance["status"] == "ready"
    assert provenance["sha256"]
    assert provenance["target_labels_used_for_selection"] is False
    plan = build_queue(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "queue",
        pretrain_cache_dir=tmp_path / "cache",
        hhar_frozen_state=frozen,
    )
    assert plan["hhar_frozen_state"]["sha256"] == provenance["sha256"]
    hhar_cell = next(cell for cell in plan["cells"] if cell["dataset"] == "HHAR")
    assert "--hhar-frozen-config" in hhar_cell["command"]
    assert str(frozen.resolve()) in hhar_cell["command"]

    changed = json.loads(frozen.read_text(encoding="utf-8"))
    changed["target_data_used"] = True
    frozen.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="target labels/data"):
        frozen_hhar_provenance(frozen)


def test_completed_hhar_five_flow_state_and_commands_share_caches(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "completed": True,
                "signature": {
                    "target_labels_used_for_selection": True,
                    "evaluation_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
                    "parameter_selection_data_overlap": True,
                    "confirmatory": False,
                },
                "tta_config": {
                    "ssaw_strength": 4.0,
                    "ssaw_auxiliary_weight": 8.0,
                    "learning_rate": 3e-4,
                    "steps": 8,
                    "batch_size": 48,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
                "parameter_selection_data_overlap": True,
                "confirmatory": False,
            }
        ),
        encoding="utf-8",
    )
    provenance = frozen_hhar_provenance(state)
    assert provenance["selection_mode"] == "target_selected_five_flow_f1"
    assert provenance["parameter_selection_data_overlap"] is True
    assert provenance["confirmatory"] is False
    plan = build_queue(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "queue",
        pretrain_cache_dir=tmp_path / "cache",
        hhar_frozen_state=state,
    )
    hhar_flow = next(
        cell
        for cell in plan["cells"]
        if cell["dataset"] == "HHAR"
        and cell["scenario"] == "0->6"
        and cell["source_seed"] == 2
    )
    assert hhar_flow["evaluation_partition"] == "target_selected_evaluation"
    assert hhar_flow["parameter_selection_data_overlap"] is True
    command = hhar_flow["command"]
    assert command[command.index("--source-seed") + 1] == "2"
    assert command[command.index("--corruption-seed") + 1] == "1"
    assert command[command.index("--horizons") + 1] == "1,3,5"
    assert "hhar_formal" in command[command.index("--pretrain-cache-dir") + 1]
    assert GPU_LOCK_PATH.name == ".current_experiment_gpu.lock"


def test_cpu_dry_run_publishes_atomic_manifest_without_launching_cells(tmp_path):
    output = tmp_path / "queue"
    plan = build_queue(
        data_path=tmp_path / "data",
        output_dir=output,
        pretrain_cache_dir=tmp_path / "cache",
        hhar_frozen_state=None,
    )
    returncode, payload = execute_queue(
        plan,
        output_dir=output,
        dry_run=True,
    )
    assert returncode == 0
    assert payload["status"] == "dry_run"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_cell_count"] == 2340
    assert not list(output.glob(".*.tmp"))
    assert not list((output / "logs").glob("*.log"))


def test_native_crash_and_oom_are_retryable_without_being_gpu_specific():
    assert is_retryable_failure(0xC0000005, "", False)
    assert is_retryable_failure(-1073741819, "", False)
    assert is_retryable_failure(1, "MemoryError: cannot allocate memory", True)


def test_horizon_gpu_lock_uses_waiting_wrapper_per_cell(tmp_path, monkeypatch):
    lock_path = tmp_path / "gpu.lock"
    events = []

    class WaitingContext:
        def __enter__(self):
            events.append(("enter", lock_path))
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("exit", lock_path))
            return False

    monkeypatch.setattr(
        "scripts.run_full_no_ssaw_horizon_queue.wait_for_gpu_experiment_lock",
        lambda path: WaitingContext(),
    )
    with _gpu_lock(lock_path):
        events.append(("body", lock_path))
    assert [event[0] for event in events] == ["enter", "body", "exit"]
