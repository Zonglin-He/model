"""CPU-only protocol tests for the serial SSAW supervisor."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_ssaw_protocol_supervisor as supervisor


def test_direct_script_cli_resolves_project_package(tmp_path):
    script = Path(supervisor.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Fail-closed serial supervisor" in completed.stdout


def _args(tmp_path: Path, *extra: str):
    parser = supervisor.build_parser()
    args = parser.parse_args(
        [
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "supervisor"),
            "--poll-seconds",
            "0.1",
            "--replicates",
            "100",
            *extra,
        ]
    )
    return supervisor.normalize_args(args, parser)


def test_stage_order_and_current_python_are_frozen(tmp_path):
    args = _args(tmp_path)
    specs = supervisor.build_stage_specs(args)

    assert tuple(spec.name for spec in specs) == supervisor.STAGE_ORDER
    assert specs[0].name == "hhar_coupling_analyzer"
    assert specs[0].command[1].endswith("analyze_hhar_coupling_factorial.py")
    assert specs[0].command[specs[0].command.index("--input") + 1].endswith(
        "hhar_ssaw_f1_delta_v1\\coupling_factorial_single_flow\\raw.csv"
    )
    assert specs[0].command[specs[0].command.index("--output-dir") + 1].endswith(
        "hhar_ssaw_f1_delta_v1\\coupling_factorial_single_flow\\analysis"
    )
    current_python = str(Path(sys.executable).resolve())
    assert all(spec.command[0] == current_python for spec in specs)
    assert all(Path(spec.command[1]).is_absolute() for spec in specs)
    heldout = specs[1]
    assert heldout.command[heldout.command.index("--metadata-json") + 1] == str(
        Path(args.metadata_json).resolve()
    )
    assert "--no-dry-run" in specs[3].command
    assert "--wait-for-core" in specs[5].command
    baseline_finalizer = specs[-3]
    assert baseline_finalizer.command[
        baseline_finalizer.command.index("--baseline-input-dir") + 1
    ].endswith(
        "baseline_physical_reference\\raw"
    )
    evidence = specs[-2]
    assert evidence.name == "evidence_synthesizer"
    assert evidence.command[1].endswith("synthesize_ssaw_evidence.py")
    assert evidence.command[
        evidence.command.index("--physical-dir") + 1
    ].endswith("physical_panel\\final")
    assert evidence.command[
        evidence.command.index("--heldout-dir") + 1
    ].endswith("ssaw_heldout_mechanism_v1\\analysis")
    assert evidence.command[
        evidence.command.index("--horizon-dir") + 1
    ].endswith("full_no_ssaw_horizon_queue\\analysis")
    assert evidence.command[
        evidence.command.index("--baseline-dir") + 1
    ].endswith("baseline_physical_reference\\final_panel")
    assert evidence.command[
        evidence.command.index("--coupling-dir") + 1
    ].endswith("hhar_ssaw_f1_delta_v1\\coupling_factorial_single_flow\\analysis")
    assert evidence.command[
        evidence.command.index("--output-dir") + 1
    ].endswith("ssaw_evidence_v1\\evidence_ledger")
    overhead = specs[-1]
    assert overhead.name == "compute_overhead"
    assert overhead.command[1].endswith("run_compute_overhead_v2.py")
    assert overhead.command[overhead.command.index("--device") + 1] == "cuda"
    assert overhead.command[overhead.command.index("--registry") + 1] == "benchmark"
    assert overhead.command[overhead.command.index("--datasets") + 1] == "EEG,HAR,FD,HHAR"
    assert overhead.command[overhead.command.index("--profiles") + 1] == "default"
    assert overhead.command[overhead.command.index("--source-seed") + 1] == "1"
    assert overhead.command[overhead.command.index("--stream-seed") + 1] == "42"
    assert "--queue" in overhead.command
    assert overhead.command[overhead.command.index("--max-attempts") + 1] == "3"
    assert overhead.command[overhead.command.index("--output-dir") + 1].endswith(
        "compute_overhead_formal_v4"
    )
    lock_path = Path(args.gpu_lock_path).resolve()
    assert overhead.command[overhead.command.index("--gpu-lock-path") + 1] == str(
        lock_path
    )
    assert lock_path not in overhead.input_paths


def test_hhar_coupling_analysis_completion_gate_checks_protocol_counts_and_csvs(tmp_path):
    root = tmp_path / "coupling-analysis"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "hhar_coupling_factorial_clustered_analysis_v2_single_flow",
                "expected_cells": 120,
                "validated_cells": 120,
                "paired_flow_seed_units": 15,
                "files": {
                    "validated_cells": "validated_cells.csv",
                    "paired_effects": "paired_effects.csv",
                    "clustered_inference": "clustered_inference.csv",
                },
            }
        ),
        encoding="utf-8",
    )
    for name in ("validated_cells.csv", "paired_effects.csv", "clustered_inference.csv"):
        (root / name).write_text("x\n", encoding="utf-8")
    assert supervisor.validate_hhar_coupling_analysis(root) == (True, "ready")

    bad = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bad["validated_cells"] = 119
    (root / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    ready, reason = supervisor.validate_hhar_coupling_analysis(root)
    assert not ready
    assert "120" in reason


def test_evidence_synthesizer_completion_gate_is_fail_closed(tmp_path):
    root = tmp_path / "evidence-ledger"
    root.mkdir()
    (root / "evidence_ledger.csv").write_text("endpoint\nphysical_mean_f1\n", encoding="utf-8")
    manifest = {
        "protocol_version": supervisor.EVIDENCE_LEDGER_PROTOCOL_VERSION,
        "status": "complete",
        "component_errors": {},
        "ledger_rows": 1,
        "confirmatory_partition": None,
        "confirmatory_rows": 0,
        "decision": {
            "recommendation": "descriptive_only",
            "confirmatory_evidence_present": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert supervisor.validate_evidence_synthesizer(root) == (True, "ready")

    for key, value in (
        ("component_errors", {"A_physical_f1": "bad"}),
        ("decision", {"recommendation": "inconclusive"}),
        ("protocol_version", "wrong"),
    ):
        invalid = dict(manifest)
        invalid[key] = value
        manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
        ready, _reason = supervisor.validate_evidence_synthesizer(root)
        assert not ready

    invalid = dict(manifest)
    invalid["ledger_rows"] = 2
    manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
    ready, reason = supervisor.validate_evidence_synthesizer(root)
    assert not ready
    assert "row count" in reason


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_compute_overhead_fixture(root: Path) -> None:
    expected_keys = sorted(supervisor._expected_compute_overhead_keys())
    merged_rows = []
    cell_status_rows = []
    cells = []
    for ordinal, key in enumerate(expected_keys, 1):
        dataset, scenario, method, variant, profile, source_seed, stream_seed = key
        name = f"cell-{ordinal:04d}"
        cells.append(
            {
                "name": name,
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "variant": variant,
                "profile": profile,
                "source_seed": source_seed,
                "stream_seed": stream_seed,
            }
        )
        cell_status_rows.append(
            {
                **cells[-1],
                "cell": name,
                "status": "ok",
                "attempts": 1,
                "return_code": 0,
                "error": "",
            }
        )
        source_hash = f"source-{dataset}-{scenario}"
        row = {
            **cells[-1],
            "status": "ok",
            "hardware": "GPU-test",
            "target_selected_descriptive": True,
            "evaluation_partition": "target_selected_evaluation",
            "parameter_selection_data_overlap": True,
            "selection_overlap": True,
            "confirmatory": False,
            "source_checkpoint_sha256": source_hash,
            "source_checkpoint_file_sha256": f"file-{source_hash}",
            "oom_fallback": False,
            "oom_history": "",
        }
        if method == "EATA":
            row.update(
                {
                    "fisher_enabled": True,
                    "fisher_cache_path": f"cache/{dataset}/{scenario.split('->', 1)[0]}",
                    "fisher_cache_hash": f"fisher-{dataset}-{scenario.split('->', 1)[0]}",
                    "fisher_cache_bytes": 10,
                    "fisher_samples": 10,
                    "fisher_batches": 1,
                    "fisher_source_checkpoint_sha256": source_hash,
                    "fisher_parameter_count": 1,
                }
            )
        merged_rows.append(row)

    queue_status = {
        "status": "complete",
        "protocol": supervisor.COMPUTE_OVERHEAD_PROTOCOL_VERSION,
        "expected_cells": 240,
        "observed_rows": 240,
        "missing_cells": [],
        "errors": [],
        "cell_failures": [],
        "candidate_view_curve": {"status": "not_applicable"},
        "gpu_lock_required": True,
        "gpu_lock_acquired": True,
    }
    finalization = {
        "status": "complete",
        "protocol": supervisor.COMPUTE_OVERHEAD_PROTOCOL_VERSION,
        "expected_cells": 240,
        "observed_rows": 240,
        "missing_cells": [],
        "errors": [],
        "candidate_view_curve": {"status": "not_applicable"},
    }
    manifest = {
        "protocol": supervisor.COMPUTE_OVERHEAD_PROTOCOL_VERSION,
        "protocol_version": supervisor.COMPUTE_OVERHEAD_PROTOCOL_VERSION,
        "queue_status": queue_status,
        "datasets": list(supervisor.COMPUTE_OVERHEAD_DATASETS),
        "formal_scenarios": {
            dataset: list(flows)
            for dataset, flows in supervisor.COMPUTE_OVERHEAD_SCENARIOS.items()
        },
        "methods": list(supervisor.COMPUTE_OVERHEAD_METHODS),
        "method_variants": [
            list(item) for item in supervisor.COMPUTE_OVERHEAD_METHOD_VARIANTS
        ],
        "profiles": ["default"],
        "source_seeds": [1],
        "stream_seed": 42,
        "expected_cells": 240,
        "expected_cell_count": 240,
        "device": "cuda",
        "algorithm_registry": "benchmark",
        "same_hardware_required": True,
        "target_selected_descriptive": True,
        "evaluation_partition": "target_selected_evaluation",
        "parameter_selection_data_overlap": True,
        "selection_overlap": True,
        "confirmatory": False,
        "gpu_lock_path": str(supervisor.DEFAULT_GPU_LOCK_PATH.resolve()),
        "gpu_lock_required": True,
        "gpu_lock_acquired": True,
        "hardware": "GPU-test",
        "candidate_view_curve": {"status": "not_applicable"},
        "cells": cells,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "queue_status.json").write_text(json.dumps(queue_status), encoding="utf-8")
    (root / "finalization.json").write_text(json.dumps(finalization), encoding="utf-8")
    _write_csv(root / "cell_status.csv", cell_status_rows)
    _write_csv(root / "method_overhead.csv", merged_rows)


def test_compute_overhead_completion_gate_requires_exact_formal_queue_and_fisher(tmp_path):
    root = tmp_path / "compute-overhead"
    _write_valid_compute_overhead_fixture(root)
    assert supervisor.validate_compute_overhead(root) == (True, "ready")

    rows = list(csv.DictReader((root / "method_overhead.csv").open(encoding="utf-8")))
    rows[0]["source_checkpoint_sha256"] = ""
    _write_csv(root / "method_overhead.csv", rows)
    ready, reason = supervisor.validate_compute_overhead(root)
    assert not ready
    assert "source_checkpoint_sha256" in reason

    _write_valid_compute_overhead_fixture(root)
    rows = list(csv.DictReader((root / "method_overhead.csv").open(encoding="utf-8")))
    eata = next(row for row in rows if row["method"] == "EATA")
    eata["fisher_enabled"] = "False"
    _write_csv(root / "method_overhead.csv", rows)
    ready, reason = supervisor.validate_compute_overhead(root)
    assert not ready
    assert "fisher_enabled" in reason


def test_metadata_waits_for_missing_file_and_rejects_malformed_file(tmp_path):
    path = tmp_path / "metadata.json"
    ready, reason = supervisor.validate_metadata(path)
    assert not ready
    assert "waiting" in reason

    path.write_text("not-json", encoding="utf-8")
    ready, reason = supervisor.validate_metadata(path)
    assert not ready
    assert "invalid" in reason


def test_hhar_tuner_requires_complete_single_five_flow_profile(tmp_path):
    root = tmp_path / "tuner"
    root.mkdir()
    manifest = {
        "status": "complete",
        "target_labels_used_for_selection": True,
        "evaluation_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
        "holdout_evaluation_confirmatory": False,
    }
    state = {"completed": True, "tta_config": {"ssaw_sobol_seed": 1729}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert supervisor.validate_hhar_tuner(root) == (True, "ready")

    manifest["evaluation_flows"] = ["5->0", "6->1", "7->4", "8->3", "0->2"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ready, reason = supervisor.validate_hhar_tuner(root)
    assert not ready
    assert "five-flow" in reason


def test_physical_core_completion_gate_checks_counts_and_final_manifest(tmp_path):
    root = tmp_path / "physical"
    (root / "final").mkdir(parents=True)
    (root / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "phase": "complete",
                "expected_groups": 840,
                "completed_groups": 840,
                "expected_cells": 5040,
                "completed_cells": 5040,
            }
        ),
        encoding="utf-8",
    )
    (root / "final" / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol_version": "ssaw_physical_evaluation_v1",
                "expected_cells": 5040,
                "validated_cells": 5040,
            }
        ),
        encoding="utf-8",
    )
    assert supervisor.validate_physical_core(root) == (True, "ready")

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    status["completed_cells"] = 5039
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
    ready, reason = supervisor.validate_physical_core(root)
    assert not ready
    assert "5040" in reason


def test_analyzer_completion_gates_use_protocol_and_files_without_status_field(tmp_path):
    heldout = tmp_path / "heldout-analysis"
    heldout.mkdir()
    (heldout / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "ssaw_heldout_clustered_analysis_v2_five_formal_flows",
                "paired_units": 60,
            }
        ),
        encoding="utf-8",
    )
    for name in ("paired_units.csv", "confirmatory_inference.csv", "operator_plausibility.csv"):
        (heldout / name).write_text("x\n", encoding="utf-8")
    assert supervisor.validate_heldout_analysis(heldout) == (True, "ready")

    horizon = tmp_path / "horizon-analysis"
    horizon.mkdir()
    (horizon / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "full_no_ssaw_horizon_clustered_analysis_v2_five_formal_flows",
                "horizon_endpoint_cells": 2340,
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "paired_horizon_endpoints.csv",
        "clustered_inference.csv",
        "condition_descriptive.csv",
    ):
        (horizon / name).write_text("x\n", encoding="utf-8")
    assert supervisor.validate_horizon_analysis(horizon) == (True, "ready")


def test_atomic_status_is_valid_json_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "nested" / "status.json"
    supervisor.atomic_write_json({"status": "running", "n": 1}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "running",
        "n": 1,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_resume_rejects_changed_command_or_input_fingerprint(tmp_path):
    args = _args(tmp_path)
    specs = supervisor.build_stage_specs(args)
    status = supervisor.build_status(args, specs)
    prior = {
        "protocol_version": supervisor.PROTOCOL_VERSION,
        "stages": [
            {
                "name": "heldout",
                "status": "completed",
                "command_sha256": "old-command",
                "input_fingerprint": {},
            }
        ],
    }
    supervisor._merge_resume_status(status, prior, specs)
    stage = status["stages"][1]
    assert stage["status"] == "planned"
    assert "stale" in stage["resume_stale_reason"]


def test_stage_input_fingerprint_refreshes_after_prerequisite_wait(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, "--no-dry-run")
    args.status_path = str(tmp_path / "status.json")
    mutable_input = tmp_path / "tuner-manifest.json"
    mutable_input.write_text("before", encoding="utf-8")
    specs = tuple(
        supervisor.StageSpec(
            name,
            (str(Path(sys.executable).resolve()), f"{name}.py"),
            tmp_path / name,
            tmp_path / "logs" / f"{name}.log",
            (mutable_input,),
        )
        for name in supervisor.STAGE_ORDER
    )
    status = supervisor.build_status(args, specs)
    initial_fingerprint = status["stages"][0]["input_fingerprint"]

    prerequisite_calls = {"count": 0}

    def prerequisite_state(_args):
        prerequisite_calls["count"] += 1
        ready = prerequisite_calls["count"] >= 2
        return ready, {
            name: {"status": "ready" if ready else "waiting", "reason": "test"}
            for name in ("hhar_tuner", "physical_core", "physical_metadata")
        }

    monkeypatch.setattr(supervisor, "_prerequisite_state", prerequisite_state)

    def fake_sleep(_seconds):
        mutable_input.write_text("after", encoding="utf-8")

    monkeypatch.setattr(supervisor.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda _command, **_kwargs: SimpleNamespace(returncode=0),
    )
    for name in supervisor.STAGE_ORDER:
        monkeypatch.setitem(
            supervisor.VALIDATORS, name, lambda _path: (True, "ready")
        )

    assert supervisor.run_supervisor(args, status=status, specs=specs) == 0
    published = json.loads(Path(args.status_path).read_text(encoding="utf-8"))
    refreshed = published["stages"][0]["input_fingerprint"]
    assert refreshed != initial_fingerprint
    assert refreshed == supervisor._stage_input_fingerprint((mutable_input,))


def test_supervisor_runs_only_in_declared_order_and_stops_on_failure(tmp_path, monkeypatch):
    args = _args(tmp_path, "--no-dry-run")
    args.status_path = str(tmp_path / "status.json")
    specs = tuple(
        supervisor.StageSpec(
            name,
            (str(Path(sys.executable).resolve()), f"{name}.py"),
            tmp_path / name,
            tmp_path / "logs" / f"{name}.log",
            (),
        )
        for name in supervisor.STAGE_ORDER
    )
    status = supervisor.build_status(args, specs)
    monkeypatch.setattr(
        supervisor,
        "_prerequisite_state",
        lambda _args: (True, {name: {"status": "ready", "reason": "ready"} for name in ("hhar_tuner", "physical_core", "physical_metadata")}),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setitem(supervisor.VALIDATORS, "hhar_coupling_analyzer", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "heldout", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "heldout_analyzer", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "horizon", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "horizon_analyzer", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "baseline", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "baseline_finalizer", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "evidence_synthesizer", lambda _path: (True, "ready"))
    monkeypatch.setitem(supervisor.VALIDATORS, "compute_overhead", lambda _path: (True, "ready"))

    assert supervisor.run_supervisor(args, status=status, specs=specs) == 0
    assert calls == [f"{name}.py" for name in supervisor.STAGE_ORDER]
    published = json.loads(Path(args.status_path).read_text(encoding="utf-8"))
    assert published["status"] == "complete"
    assert all(item["status"] == "completed" for item in published["stages"])


def test_supervisor_failure_is_fail_closed_and_does_not_call_later_stages(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, "--no-dry-run")
    args.status_path = str(tmp_path / "status.json")
    specs = tuple(
        supervisor.StageSpec(
            name,
            (str(Path(sys.executable).resolve()), f"{name}.py"),
            tmp_path / name,
            tmp_path / "logs" / f"{name}.log",
            (),
        )
        for name in supervisor.STAGE_ORDER
    )
    status = supervisor.build_status(args, specs)
    monkeypatch.setattr(
        supervisor,
        "_prerequisite_state",
        lambda _args: (True, {name: {"status": "ready", "reason": "ready"} for name in ("hhar_tuner", "physical_core", "physical_metadata")}),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[1])
        return SimpleNamespace(returncode=23 if len(calls) == 2 else 0)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    for name in supervisor.STAGE_ORDER:
        monkeypatch.setitem(supervisor.VALIDATORS, name, lambda _path: (True, "ready"))

    result = supervisor.run_supervisor(args, status=status, specs=specs)
    assert result == 23
    assert calls == [f"{name}.py" for name in supervisor.STAGE_ORDER[:2]]
    published = json.loads(Path(args.status_path).read_text(encoding="utf-8"))
    assert published["status"] == "failed"
    assert published["failure"]["stage"] == supervisor.STAGE_ORDER[1]
    assert published["stages"][2]["status"] == "planned"
