"""CPU-only tests for the bounded representative SSAW supervisor."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import run_representative_ssaw_supervisor as supervisor


def _write_rows(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell"])
        for index in range(count):
            writer.writerow([index])


def test_stage_scope_is_bounded_and_serial(tmp_path: Path) -> None:
    specs = supervisor.build_stage_specs(output_dir=tmp_path / "rep")
    assert tuple(spec.name for spec in specs) == supervisor.STAGE_ORDER
    assert [spec.expected.get("cells") for spec in specs if "cells" in spec.expected] == [12, 6, 108, 132]
    physical = next(spec for spec in specs if spec.name == "representative_physical")
    assert "run_ssaw_evidence_queue.py" in physical.command[1]
    assert physical.command[physical.command.index("--datasets") + 1] == "HAR"
    assert physical.command[physical.command.index("--scenarios") + 1] == "12->16"
    assert physical.command[physical.command.index("--severities") + 1] == "s0,s3,s6"
    assert physical.command[physical.command.index("--source-seeds") + 1] == "1,2,3"
    assert physical.command[physical.command.index("--variants") + 1] == "full,no_ssaw"
    serialized = " ".join(" ".join(spec.command) for spec in specs)
    assert "run_ssaw_protocol_supervisor.py" not in serialized
    assert "5040" not in serialized
    manifest = supervisor.build_manifest(specs, tmp_path / "rep")
    assert manifest["gpu_policy"]["supervisor_allocates_cuda"] is False
    assert manifest["gpu_policy"]["max_concurrent_child_processes"] == 1
    assert manifest["scope"]["main_table_included"] is False
    assert manifest["scope"]["formal_physical_core_included"] is False


def test_default_run_is_cpu_only_plan_and_writes_atomic_status(tmp_path: Path) -> None:
    specs = supervisor.build_stage_specs(output_dir=tmp_path / "rep")
    result = supervisor.run_supervisor(
        specs=specs,
        output_dir=tmp_path / "rep",
        execute=False,
    )
    assert result["status"] == "planned"
    manifest = json.loads((tmp_path / "rep" / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "rep" / "status.json").read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == supervisor.PROTOCOL_VERSION
    assert status["protocol_version"] == supervisor.PROTOCOL_VERSION
    assert status["status"] == "planned"
    assert status["completed_stages"] == []
    assert all(row["status"] == "planned" for row in status["stage_status"])


def test_g_validator_requires_exact_12_cells(tmp_path: Path) -> None:
    root = tmp_path / "g"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": supervisor.G_PROTOCOL_VERSION,
                "status": "complete",
                "expected_cells": 12,
                "queue_status": {"status": "complete"},
            }
        ),
        encoding="utf-8",
    )
    _write_rows(root / "cell_status.csv", 12)
    _write_rows(root / "method_overhead.csv", 12)
    assert supervisor.validate_g_output(root) == (True, "ready")
    bad = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bad["expected_cells"] = 240
    (root / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    ready, reason = supervisor.validate_g_output(root)
    assert not ready
    assert "12" in reason or "240" in reason


def test_heldout_validator_requires_six_cells_but_three_pairs(tmp_path: Path) -> None:
    root = tmp_path / "heldout"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": supervisor.HELDOUT_PROTOCOL_VERSION,
                "status": "complete",
                "expected_cells": 6,
                "completed_cells": 6,
            }
        ),
        encoding="utf-8",
    )
    (root / "paired_summary.json").write_text(
        json.dumps({"paired_rows": [{"source_seed": seed} for seed in (1, 2, 3)]}),
        encoding="utf-8",
    )
    assert supervisor.validate_heldout_output(root) == (True, "ready")

    (root / "paired_summary.json").write_text(
        json.dumps({"paired_rows": [{"source_seed": seed} for seed in range(6)]}),
        encoding="utf-8",
    )
    ready, reason = supervisor.validate_heldout_output(root)
    assert not ready
    assert "3 source-seed pairs" in reason


def test_physical_validator_rejects_old_or_full_scope(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    (root / "final").mkdir(parents=True)
    (root / "raw").mkdir(parents=True)
    (root / "status.json").write_text(
        json.dumps(
            {
                "version": "ssaw_evidence_queue_v2_five_flow",
                "phase": "complete",
                "scenario_scope": "registered_formal_full",
                "expected_cells": 5040,
                "completed_cells": 5040,
            }
        ),
        encoding="utf-8",
    )
    ready, reason = supervisor.validate_physical_output(root)
    assert not ready
    assert "protocol" in reason or "108" in reason or "scope" in reason


def test_representative_queue_dry_run_contract() -> None:
    from scripts import run_ssaw_evidence_queue as queue

    selected = queue.groups(
        datasets=("HAR",),
        scenarios={"HAR": ("12->16",)},
        corruptions=("signal_freeze", "blackout", "attenuation", "amplitude_drift", "packet_loss", "saturation"),
        severities=("s0", "s3", "s6"),
    )
    assert len(selected) == 18
    assert len(selected) * 2 * 3 == 108
    assert queue._normalize_variants("full,no_ssaw") == ("full", "no_ssaw")
