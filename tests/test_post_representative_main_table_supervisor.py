from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import run_post_representative_main_table_supervisor as supervisor


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path / "supervisor"),
        representative_status=str(tmp_path / "representative" / "status.json"),
        data_path=str(tmp_path / "data"),
        device="cuda",
        backbone="CNN",
        pretrain_cache_root=str(tmp_path / "cache"),
        eata_fisher_cache_dir=str(tmp_path / "fisher"),
        legacy_main_dir=str(tmp_path / "legacy"),
        result_root=str(tmp_path / "results"),
        hhar_output_dir=str(tmp_path / "hhar"),
        hhar_tuning_dir=str(tmp_path / "tuning"),
        poll_seconds=1.0,
        execute=False,
    )


def test_plan_is_main_table_only_and_uses_concrete_checkpoint_dirs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    roots, commands = supervisor.build_plan(args)
    assert tuple(roots) == supervisor.STAGES
    refresh = commands["fused_refresh"]
    assert refresh[refresh.index("--datasets") + 1] == "EEG,HAR,FD"
    assert refresh[refresh.index("--methods") + 1] == "DuSafe"
    assert refresh[refresh.index("--pretrain-cache-dir") + 1].endswith("optuna_stepwise")
    hhar = commands["hhar_main"]
    assert hhar[hhar.index("--pretrain-cache-dir") + 1].endswith("hhar_formal")
    serialized = " ".join(" ".join(command) for command in commands.values())
    assert "run_ssaw_evidence_queue.py" not in serialized
    assert "run_ssaw_protocol_supervisor.py" not in serialized


def test_default_run_writes_cpu_only_plan(tmp_path: Path) -> None:
    args = _args(tmp_path)
    assert supervisor.run(args) == 0
    manifest = json.loads((tmp_path / "supervisor" / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "supervisor" / "status.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert status["status"] == "planned"
    assert status["scope"]["main_table_rows"] == 660
    assert status["scope"]["non_main_experiments_included"] is False


def test_representative_gate_fails_closed_on_protocol_or_failure(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"protocol_version": "wrong", "status": "complete"}), encoding="utf-8")
    assert supervisor.representative_ready(path)[0] is False
    path.write_text(
        json.dumps(
            {
                "protocol_version": "ssaw_representative_serial_supervisor_v1_har_g_to_evidence",
                "status": "failed",
                "error": "boom",
            }
        ),
        encoding="utf-8",
    )
    try:
        supervisor.representative_ready(path)
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("failed representative queue must raise")
