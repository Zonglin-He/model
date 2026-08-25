from __future__ import annotations

import json
from pathlib import Path

from scripts.wait_and_finalize_four_dataset_main_table import (
    EXPECTED_HHAR_CELLS,
    EXPECTED_HHAR_FLOWS,
    EXPECTED_METHODS,
    EXPECTED_SOURCE_SEEDS,
    EXPECTED_STREAM_SEED,
    HHAR_QUEUE_PROTOCOL,
    inspect_hhar_contract,
    run_waiter,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_hhar_contract(root: Path, *, complete: bool = True, protocol: str = HHAR_QUEUE_PROTOCOL) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    phase = "complete" if complete else "running"
    status = {
        "protocol_version": protocol,
        "status": "complete" if complete else "running",
        "phase": phase,
        "completed_cells": EXPECTED_HHAR_CELLS if complete else 12,
        "expected_cells": EXPECTED_HHAR_CELLS,
        "flows": list(EXPECTED_HHAR_FLOWS),
        "methods": list(EXPECTED_METHODS),
        "source_seeds": list(EXPECTED_SOURCE_SEEDS),
        "stream_seed": EXPECTED_STREAM_SEED,
        "selection_overlap": True,
        "confirmatory": False,
        "evaluation_partition": "target_selected_evaluation",
        "failures": [],
    }
    manifest = {
        "protocol_version": protocol,
        "status": "complete" if complete else "running",
        "phase": phase,
        "raw_rows": EXPECTED_HHAR_CELLS if complete else 12,
        "successful_rows": EXPECTED_HHAR_CELLS if complete else 12,
        "expected_cells": EXPECTED_HHAR_CELLS,
        "flows": list(EXPECTED_HHAR_FLOWS),
        "methods": list(EXPECTED_METHODS),
        "source_seeds": list(EXPECTED_SOURCE_SEEDS),
        "stream_seed": EXPECTED_STREAM_SEED,
        "selection_overlap": True,
        "confirmatory": False,
        "evaluation_partition": "target_selected_evaluation",
        "failures": [],
        "cells": [{} for _ in range(EXPECTED_HHAR_CELLS if complete else 12)],
    }
    _write_json(root / "status.json", status)
    _write_json(root / "manifest.json", manifest)
    return root


def _fake_finalizer(**kwargs) -> int:
    output_dir = Path(kwargs["finalizer_output_dir"])
    _write_json(
        output_dir / "manifest.json",
        {
            "protocol_version": "fixed_source_main_table_v1_five_flows_descriptive",
            "status": "complete",
            "decision_status": "descriptive_only",
            "confirmatory": False,
            "observed_cells": 660,
        },
    )
    return 0


def test_waiter_does_not_run_finalizer_before_strict_completion(tmp_path: Path):
    hhar = _write_hhar_contract(tmp_path / "hhar", complete=False)
    calls = []

    def should_not_run(**kwargs):
        calls.append(kwargs)
        return 0

    code, status = run_waiter(
        hhar_input_dir=hhar,
        legacy_input_dir=tmp_path / "legacy",
        finalizer_output_dir=tmp_path / "final",
        waiter_output_dir=tmp_path / "waiter",
        poll_seconds=0,
        max_polls=1,
        finalizer_runner=should_not_run,
    )
    assert code == 1
    assert status["phase"] == "waiting"
    assert calls == []
    assert status["hhar_contract"]["errors"]


def test_contract_requires_both_v2_documents_and_165_cells(tmp_path: Path):
    hhar = _write_hhar_contract(tmp_path / "hhar", complete=True, protocol="old_protocol")
    ready, details = inspect_hhar_contract(hhar)
    assert ready is False
    assert any("protocol" in error for error in details["errors"])
    # A complete status alone cannot make a stale manifest ready.
    (hhar / "manifest.json").write_text(
        json.dumps({
            "protocol_version": HHAR_QUEUE_PROTOCOL,
            "status": "complete",
            "raw_rows": 164,
            "expected_cells": EXPECTED_HHAR_CELLS,
            "selection_overlap": True,
            "confirmatory": False,
            "cells": [{} for _ in range(164)],
        }),
        encoding="utf-8",
    )
    ready, details = inspect_hhar_contract(hhar)
    assert ready is False
    assert any("cell count" in error for error in details["errors"])


def test_waiter_runs_once_and_resumes_from_complete_output(tmp_path: Path):
    hhar = _write_hhar_contract(tmp_path / "hhar", complete=True)
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _fake_finalizer(**kwargs)

    options = dict(
        hhar_input_dir=hhar,
        legacy_input_dir=tmp_path / "legacy",
        finalizer_output_dir=tmp_path / "final",
        waiter_output_dir=tmp_path / "waiter",
        poll_seconds=0,
        max_polls=1,
        finalizer_runner=fake,
    )
    code, status = run_waiter(**options)
    assert code == 0
    assert status["phase"] == "complete"
    assert len(calls) == 1
    code, resumed = run_waiter(**options)
    assert code == 0
    assert resumed["phase"] == "complete"
    assert len(calls) == 1
    persisted = json.loads((tmp_path / "waiter" / "status.json").read_text(encoding="utf-8"))
    assert persisted["gpu_lock_acquired"] is False
    assert persisted["torch_imported"] is False


def test_waiter_persists_finalizer_failure_return_code(tmp_path: Path):
    hhar = _write_hhar_contract(tmp_path / "hhar", complete=True)

    def failed(**kwargs):
        return 17

    code, status = run_waiter(
        hhar_input_dir=hhar,
        legacy_input_dir=tmp_path / "legacy",
        finalizer_output_dir=tmp_path / "final",
        waiter_output_dir=tmp_path / "waiter",
        poll_seconds=0,
        max_polls=1,
        finalizer_runner=failed,
    )
    assert code == 17
    assert status["phase"] == "failed"
    assert status["finalizer_returncode"] == 17
    assert (tmp_path / "waiter" / "status.json").is_file()
