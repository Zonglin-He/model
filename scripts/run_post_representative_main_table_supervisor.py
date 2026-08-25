"""Resume the full main table after bounded representative evidence finishes.

The supervisor is CPU-only until the representative queue reports ``complete``.
It then runs exactly one child at a time:

1. refresh the 45 EEG/HAR/FD DuSafe cells with fused execution;
2. merge those cells into the existing 495-row three-dataset main table;
3. resume the 165-cell HHAR five-flow main table;
4. finalize the 660-row four-dataset table.

No physical-evidence queue or legacy A--G supervisor is reachable here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "post_representative_main_table_supervisor_v1_fused_refresh"
STAGES = ("representative_gate", "fused_refresh", "merge_legacy", "hhar_main", "finalize")


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except OSError:
        return None


def _command_hash(command: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(command), separators=(",", ":")).encode()).hexdigest()


def _complete_json(path: Path) -> bool:
    payload = _read_json(path)
    return payload is not None and str(payload.get("status", "")).lower() in {"complete", "completed"}


def representative_ready(path: Path) -> tuple[bool, str]:
    payload = _read_json(path)
    if payload is None:
        return False, "representative status is missing"
    if str(payload.get("protocol_version")) != "ssaw_representative_serial_supervisor_v1_har_g_to_evidence":
        return False, "representative protocol mismatch"
    status = str(payload.get("status", "")).lower()
    if status in {"failed", "complete_with_failures"}:
        raise RuntimeError(f"representative queue failed: {payload.get('error')}")
    return status == "complete", status or "not_started"


def _output_valid(stage: str, roots: Mapping[str, Path]) -> tuple[bool, str]:
    if stage == "representative_gate":
        return representative_ready(roots[stage])
    if stage == "fused_refresh":
        count = _csv_rows(roots[stage] / "per_source_seed_results.csv")
        return (count == 45, f"rows={count}, expected 45")
    if stage == "merge_legacy":
        count = _csv_rows(roots[stage] / "per_source_seed_results.csv")
        ready = count == 495 and _complete_json(roots[stage] / "manifest.json")
        return ready, f"rows={count}, expected 495"
    if stage == "hhar_main":
        count = _csv_rows(roots[stage] / "per_source_seed_results.csv")
        ready = count == 165 and _complete_json(roots[stage] / "status.json")
        return ready, f"rows={count}, expected 165"
    if stage == "finalize":
        count = _csv_rows(roots[stage] / "per_source_seed_results.csv")
        ready = count == 660 and _complete_json(roots[stage] / "manifest.json")
        return ready, f"rows={count}, expected 660"
    raise KeyError(stage)


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, list[str]]]:
    python = str(Path(sys.executable).resolve())
    data = _absolute(args.data_path)
    cache_root = _absolute(args.pretrain_cache_root)
    result_root = _absolute(args.result_root)
    roots = {
        "representative_gate": _absolute(args.representative_status),
        "fused_refresh": result_root / "main_table_dusafe_fused_refresh",
        "merge_legacy": result_root / "main_table_source_calibrated_fused",
        "hhar_main": _absolute(args.hhar_output_dir),
        "finalize": result_root / "four_dataset_main_table_final_fused",
    }
    commands = {
        "fused_refresh": [
            python, str(ROOT / "scripts" / "run_full_main_table.py"),
            "--data-path", str(data), "--device", args.device, "--backbone", args.backbone,
            "--datasets", "EEG,HAR,FD", "--methods", "DuSafe",
            "--source-seeds", "1,2,3", "--stream-seed", "42",
            "--output-dir", str(roots["fused_refresh"]),
            "--pretrain-cache-dir", str(cache_root / "optuna_stepwise"),
            "--eata-fisher-cache-dir", str(_absolute(args.eata_fisher_cache_dir)),
            "--run-signature", "dusafe_fused_batch_v1",
            "--override", "dusafe_execution_mode='fused'",
            "--override", "update_transaction_scope='batch'",
            "--override", "record_optimizer_diagnostics=False",
        ],
        "merge_legacy": [
            python, str(ROOT / "scripts" / "refresh_main_table_dusafe_fused.py"),
            "--legacy-input-dir", str(_absolute(args.legacy_main_dir)),
            "--fused-input-dir", str(roots["fused_refresh"]),
            "--output-dir", str(roots["merge_legacy"]),
        ],
        "hhar_main": [
            python, str(ROOT / "scripts" / "run_hhar_five_flow_main_table_queue.py"),
            "--data-path", str(data), "--device", args.device, "--backbone", args.backbone,
            "--output-dir", str(roots["hhar_main"]),
            "--pretrain-cache-dir", str(cache_root / "hhar_formal"),
            "--eata-fisher-cache-dir", str(_absolute(args.eata_fisher_cache_dir)),
            "--hhar-tuning-dir", str(_absolute(args.hhar_tuning_dir)),
            "--max-attempts", "2", "--no-dry-run",
        ],
        "finalize": [
            python, str(ROOT / "scripts" / "finalize_four_dataset_main_table.py"),
            "--legacy-input-dir", str(roots["merge_legacy"]),
            "--hhar-input-dir", str(roots["hhar_main"]),
            "--output-dir", str(roots["finalize"]),
            "--bootstrap-replicates", "5000", "--seed", "20260820",
        ],
    }
    return roots, commands


def _publish(output: Path, status: str, stage_rows: list[dict[str, Any]], **extra: Any) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "updated_at": _utc_now(),
        "stage_order": list(STAGES),
        "stages": stage_rows,
        "scope": {"main_table_rows": 660, "non_main_experiments_included": False},
        **extra,
    }
    _atomic_json(payload, output / "status.json")


def run(args: argparse.Namespace) -> int:
    output = _absolute(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    roots, commands = build_plan(args)
    stage_rows = [
        {
            "name": name,
            "status": "planned",
            "output": str(roots[name]),
            "command_sha256": None if name == "representative_gate" else _command_hash(commands[name]),
        }
        for name in STAGES
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "planned" if not args.execute else "running",
        "stage_order": list(STAGES),
        "commands": commands,
        "outputs": {key: str(value) for key, value in roots.items()},
        "excludes": ["run_ssaw_protocol_supervisor.py", "run_ssaw_evidence_queue.py", "physical_panel"],
    }
    _atomic_json(manifest, output / "manifest.json")
    _publish(output, manifest["status"], stage_rows)
    if not args.execute:
        return 0

    for index, name in enumerate(STAGES):
        row = stage_rows[index]
        try:
            ready, reason = _output_valid(name, roots)
        except RuntimeError as exc:
            row.update(status="failed", reason=str(exc), completed_at=_utc_now())
            _publish(output, "failed", stage_rows, current_stage=name, error=str(exc))
            return 2
        if name == "representative_gate":
            while not ready:
                row.update(status="waiting", reason=reason)
                _publish(output, "waiting", stage_rows, current_stage=name)
                time.sleep(args.poll_seconds)
                try:
                    ready, reason = _output_valid(name, roots)
                except RuntimeError as exc:
                    row.update(status="failed", reason=str(exc), completed_at=_utc_now())
                    _publish(output, "failed", stage_rows, current_stage=name, error=str(exc))
                    return 2
            row.update(status="complete", reason="ready", completed_at=_utc_now())
            _publish(output, "running", stage_rows)
            continue
        if ready:
            row.update(status="complete", reason="resume_valid", completed_at=_utc_now())
            _publish(output, "running", stage_rows)
            continue
        row.update(status="running", reason=reason, started_at=_utc_now())
        _publish(output, "running", stage_rows, current_stage=name)
        log_path = output / "logs" / f"{name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(commands[name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        ready, reason = _output_valid(name, roots)
        if result.returncode != 0 or not ready:
            row.update(status="failed", returncode=result.returncode, reason=reason, completed_at=_utc_now())
            _publish(output, "failed", stage_rows, current_stage=name, error=reason)
            return 2
        row.update(status="complete", returncode=0, reason="validated", completed_at=_utc_now())
        _publish(output, "running", stage_rows)
    _publish(output, "complete", stage_rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", default="results/post_representative_main_table_supervisor")
    parser.add_argument("--representative-status", default="results/representative_ssaw_evidence_v1/status.json")
    parser.add_argument("--data-path", default="data/Dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-root", default="results/pretrain_cache")
    parser.add_argument("--eata-fisher-cache-dir", default="results/eata_fisher_cache/benchmark_fisher")
    parser.add_argument("--legacy-main-dir", default="results/reviewer_queue_v2/main_table_source_calibrated")
    parser.add_argument("--result-root", default="results/reviewer_queue_v2")
    parser.add_argument("--hhar-output-dir", default="results/hhar_five_flow_main_table_v2")
    parser.add_argument("--hhar-tuning-dir", default="results/optuna/hhar_ssaw_f1_delta_v1")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
