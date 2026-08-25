"""Wait for the HHAR clean panel, then run the CPU-only four-dataset finalizer.

The waiter is intentionally standard-library only.  It does not import the
trainer, torch, CUDA, or the GPU-lock implementation.  While the HHAR queue is
running it only reads its JSON status/manifest files and atomically publishes
its own progress.  The finalizer is launched as a separate process only after
both HHAR contract documents pass the strict v2/165-cell completion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
HHAR_QUEUE_PROTOCOL = "hhar_five_flow_main_table_queue_v2_one_cell"
WAITER_PROTOCOL = "fixed_source_main_table_waiter_v1"
EXPECTED_HHAR_CELLS = 165
EXPECTED_HHAR_FLOWS = ("0->6", "1->6", "2->7", "3->8", "4->5")
EXPECTED_METHODS = (
    "NoAdap",
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
    "DuSafe",
)
EXPECTED_SOURCE_SEEDS = (1, 2, 3)
EXPECTED_STREAM_SEED = 42
FINALIZER_PROTOCOLS = {
    "fixed_source_main_table_v1_five_flows_descriptive",
    "fixed_source_main_table_v2_current_dusafe_five_flows_descriptive",
}
DEFAULT_HHAR_INPUT_DIR = ROOT / "results" / "hhar_five_flow_main_table_v2"
DEFAULT_LEGACY_INPUT_DIR = ROOT / "results" / "reviewer_queue_v2" / "main_table_source_calibrated"
DEFAULT_FINALIZER_OUTPUT_DIR = ROOT / "results" / "reviewer_queue_v2" / "four_dataset_main_table_final"
DEFAULT_WAITER_OUTPUT_DIR = ROOT / "results" / "reviewer_queue_v2" / "four_dataset_main_table_finalizer_wait"
DEFAULT_FINALIZER_SCRIPT = ROOT / "scripts" / "finalize_four_dataset_main_table.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"unreadable {path.name}: {exc}"
    if not isinstance(payload, Mapping):
        return None, f"{path.name} is not a JSON object"
    return payload, None


def _append_log(path: Path, message: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now()}] {message}\n")


def _declared_cells(payload: Mapping[str, Any], *, list_key: str) -> int | None:
    value = payload.get("completed_cells")
    if value is None:
        value = payload.get("raw_rows")
    if value is None:
        value = payload.get("successful_rows")
    if value is None and isinstance(payload.get(list_key), list):
        value = len(payload[list_key])
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def inspect_hhar_contract(hhar_input_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    """Return whether both HHAR queue documents pass the strict ready gate."""

    input_dir = Path(hhar_input_dir).resolve()
    status, status_error = _read_json(input_dir / "status.json")
    manifest, manifest_error = _read_json(input_dir / "manifest.json")
    details: dict[str, Any] = {
        "input_dir": str(input_dir),
        "status_path": str((input_dir / "status.json").resolve()),
        "manifest_path": str((input_dir / "manifest.json").resolve()),
        "status_present": status is not None,
        "manifest_present": manifest is not None,
        "errors": [],
    }
    errors: list[str] = []
    if status_error:
        errors.append(status_error)
    if manifest_error:
        errors.append(manifest_error)
    if status is None or manifest is None:
        details["errors"] = errors
        return False, details

    for name, payload in (("status", status), ("manifest", manifest)):
        protocol = str(payload.get("protocol_version", payload.get("protocol", "")))
        if protocol != HHAR_QUEUE_PROTOCOL:
            errors.append(f"{name}.protocol={protocol!r} != {HHAR_QUEUE_PROTOCOL!r}")
        if payload.get("status") != "complete":
            errors.append(f"{name}.status={payload.get('status')!r} != 'complete'")
        if payload.get("phase") is not None and payload.get("phase") != "complete":
            errors.append(f"{name}.phase={payload.get('phase')!r} != 'complete'")
        if payload.get("selection_overlap") is not True:
            errors.append(f"{name}.selection_overlap is not true")
        if payload.get("confirmatory") is not False:
            errors.append(f"{name}.confirmatory is not false")
        if payload.get("evaluation_partition") not in {None, "target_selected_evaluation"}:
            errors.append(f"{name}.evaluation_partition is not target_selected_evaluation")

    status_cells = _declared_cells(status, list_key="cells")
    manifest_cells = _declared_cells(manifest, list_key="cells")
    if status_cells != EXPECTED_HHAR_CELLS:
        errors.append(f"status cell count={status_cells!r} != {EXPECTED_HHAR_CELLS}")
    if manifest_cells != EXPECTED_HHAR_CELLS:
        errors.append(f"manifest cell count={manifest_cells!r} != {EXPECTED_HHAR_CELLS}")
    for name, payload in (("status", status), ("manifest", manifest)):
        for key, expected in (
            ("expected_cells", EXPECTED_HHAR_CELLS),
            ("expected_cell_count", EXPECTED_HHAR_CELLS),
        ):
            if key in payload and payload[key] != expected:
                errors.append(f"{name}.{key}={payload[key]!r} != {expected}")
        failures = payload.get("failures")
        if failures not in (None, [], ()):
            errors.append(f"{name}.failures is non-empty")
    if tuple(str(value) for value in status.get("flows", ())) not in {
        EXPECTED_HHAR_FLOWS,
        (),
    }:
        errors.append("status flows are not the formal five HHAR flows")
    if tuple(str(value) for value in manifest.get("flows", ())) not in {
        EXPECTED_HHAR_FLOWS,
        (),
    }:
        errors.append("manifest flows are not the formal five HHAR flows")
    if "methods" in status and tuple(str(value) for value in status["methods"]) != EXPECTED_METHODS:
        errors.append("status methods differ from the formal 11-method set")
    if "methods" in manifest and tuple(str(value) for value in manifest["methods"]) != EXPECTED_METHODS:
        errors.append("manifest methods differ from the formal 11-method set")
    if "source_seeds" in status and tuple(int(value) for value in status["source_seeds"]) != EXPECTED_SOURCE_SEEDS:
        errors.append("status source seeds differ from 1/2/3")
    if "source_seeds" in manifest and tuple(int(value) for value in manifest["source_seeds"]) != EXPECTED_SOURCE_SEEDS:
        errors.append("manifest source seeds differ from 1/2/3")
    for name, payload in (("status", status), ("manifest", manifest)):
        if "stream_seed" in payload and int(payload["stream_seed"]) != EXPECTED_STREAM_SEED:
            errors.append(f"{name}.stream_seed differs from 42")
    details.update(
        {
            "status": status.get("status"),
            "manifest_status": manifest.get("status"),
            "status_cells": status_cells,
            "manifest_cells": manifest_cells,
            "protocol": HHAR_QUEUE_PROTOCOL,
            "errors": errors,
        }
    )
    return not errors, details


def _final_output_complete(output_dir: Path) -> bool:
    path = Path(output_dir).resolve() / "manifest.json"
    payload, error = _read_json(path)
    if error or payload is None:
        return False
    return (
        payload.get("protocol_version") in FINALIZER_PROTOCOLS
        and payload.get("status") == "complete"
        and payload.get("decision_status") == "descriptive_only"
        and payload.get("confirmatory") is False
        and int(payload.get("observed_cells", -1)) == 660
    )


def _base_status(
    *,
    phase: str,
    hhar_input_dir: Path,
    legacy_input_dir: Path,
    finalizer_output_dir: Path,
    waiter_output_dir: Path,
    poll_seconds: int,
    bootstrap_replicates: int,
    seed: int,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "protocol_version": WAITER_PROTOCOL,
        "status": phase,
        "phase": phase,
        "hhar_queue_protocol": HHAR_QUEUE_PROTOCOL,
        "expected_hhar_cells": EXPECTED_HHAR_CELLS,
        "hhar_input_dir": str(hhar_input_dir),
        "legacy_input_dir": str(legacy_input_dir),
        "finalizer_output_dir": str(finalizer_output_dir),
        "waiter_output_dir": str(waiter_output_dir),
        "poll_seconds": int(poll_seconds),
        "bootstrap_replicates": int(bootstrap_replicates),
        "seed": int(seed),
        "gpu_lock_acquired": False,
        "torch_imported": False,
        "updated_at": _utc_now(),
    }
    payload.update(extra)
    return payload


def _run_finalizer(
    *,
    finalizer_script: Path,
    legacy_input_dir: Path,
    hhar_input_dir: Path,
    finalizer_output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
    log_path: Path,
) -> int:
    command = [
        sys.executable,
        str(finalizer_script.resolve()),
        "--legacy-input-dir",
        str(legacy_input_dir),
        "--hhar-input-dir",
        str(hhar_input_dir),
        "--output-dir",
        str(finalizer_output_dir),
        "--bootstrap-replicates",
        str(int(bootstrap_replicates)),
        "--seed",
        str(int(seed)),
    ]
    _append_log(log_path, "FINALIZER COMMAND " + json.dumps(command, ensure_ascii=False))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    _append_log(log_path, f"FINALIZER RETURN_CODE {result.returncode}")
    return int(result.returncode)


def run_waiter(
    *,
    hhar_input_dir: str | Path = DEFAULT_HHAR_INPUT_DIR,
    legacy_input_dir: str | Path = DEFAULT_LEGACY_INPUT_DIR,
    finalizer_output_dir: str | Path = DEFAULT_FINALIZER_OUTPUT_DIR,
    waiter_output_dir: str | Path = DEFAULT_WAITER_OUTPUT_DIR,
    finalizer_script: str | Path = DEFAULT_FINALIZER_SCRIPT,
    poll_seconds: int = 60,
    bootstrap_replicates: int = 5000,
    seed: int = 20260820,
    max_polls: int | None = None,
    resume: bool = True,
    finalizer_runner: Callable[..., int] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any]]:
    """Poll the HHAR contract and run the finalizer once it is complete."""

    if int(poll_seconds) < 0:
        raise ValueError("poll_seconds must be non-negative")
    if max_polls is not None and int(max_polls) < 1:
        raise ValueError("max_polls must be positive when supplied")
    hhar_dir = Path(hhar_input_dir).resolve()
    legacy_dir = Path(legacy_input_dir).resolve()
    final_dir = Path(finalizer_output_dir).resolve()
    waiter_dir = Path(waiter_output_dir).resolve()
    finalizer_path = Path(finalizer_script).resolve()
    waiter_dir.mkdir(parents=True, exist_ok=True)
    status_path = waiter_dir / "status.json"
    log_path = waiter_dir / "waiter.log"
    previous, _ = _read_json(status_path)
    if resume and _final_output_complete(final_dir):
        status = _base_status(
            phase="complete",
            hhar_input_dir=hhar_dir,
            legacy_input_dir=legacy_dir,
            finalizer_output_dir=final_dir,
            waiter_output_dir=waiter_dir,
            poll_seconds=poll_seconds,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            resumed=True,
            reason="finalizer output is already complete",
        )
        _atomic_write_json(status, status_path)
        _append_log(log_path, "RESUME complete: finalizer output already exists")
        return 0, status
    if previous and previous.get("phase") == "failed" and not resume:
        return int(previous.get("returncode", 2) or 2), dict(previous)

    polls = 0
    while True:
        polls += 1
        ready, details = inspect_hhar_contract(hhar_dir)
        if ready:
            status = _base_status(
                phase="running_finalizer",
                hhar_input_dir=hhar_dir,
                legacy_input_dir=legacy_dir,
                finalizer_output_dir=final_dir,
                waiter_output_dir=waiter_dir,
                poll_seconds=poll_seconds,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
                poll_count=polls,
                hhar_contract=details,
                finalizer_script=str(finalizer_path),
                finalizer_log=str(log_path),
                finalizer_command_started=True,
            )
            _atomic_write_json(status, status_path)
            _append_log(log_path, "HHAR contract ready; starting finalizer")
            runner = finalizer_runner or _run_finalizer
            try:
                if finalizer_runner is None:
                    returncode = runner(
                        finalizer_script=finalizer_path,
                        legacy_input_dir=legacy_dir,
                        hhar_input_dir=hhar_dir,
                        finalizer_output_dir=final_dir,
                        bootstrap_replicates=bootstrap_replicates,
                        seed=seed,
                        log_path=log_path,
                    )
                else:
                    returncode = runner(
                        finalizer_script=finalizer_path,
                        legacy_input_dir=legacy_dir,
                        hhar_input_dir=hhar_dir,
                        finalizer_output_dir=final_dir,
                        bootstrap_replicates=bootstrap_replicates,
                        seed=seed,
                        log_path=log_path,
                    )
            except Exception as exc:  # pragma: no cover - defensive process boundary
                returncode = 1
                _append_log(log_path, f"FINALIZER EXCEPTION {type(exc).__name__}: {exc}")
            if int(returncode) != 0:
                failed = _base_status(
                    phase="failed",
                    hhar_input_dir=hhar_dir,
                    legacy_input_dir=legacy_dir,
                    finalizer_output_dir=final_dir,
                    waiter_output_dir=waiter_dir,
                    poll_seconds=poll_seconds,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed,
                    poll_count=polls,
                    hhar_contract=details,
                    finalizer_returncode=int(returncode),
                    finalizer_log=str(log_path),
                    error="finalizer returned non-zero",
                )
                _atomic_write_json(failed, status_path)
                return int(returncode), failed
            if not _final_output_complete(final_dir):
                failed = _base_status(
                    phase="failed",
                    hhar_input_dir=hhar_dir,
                    legacy_input_dir=legacy_dir,
                    finalizer_output_dir=final_dir,
                    waiter_output_dir=waiter_dir,
                    poll_seconds=poll_seconds,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed,
                    poll_count=polls,
                    hhar_contract=details,
                    finalizer_returncode=0,
                    finalizer_log=str(log_path),
                    error="finalizer returned zero but final manifest failed strict validation",
                )
                _atomic_write_json(failed, status_path)
                return 2, failed
            complete = _base_status(
                phase="complete",
                hhar_input_dir=hhar_dir,
                legacy_input_dir=legacy_dir,
                finalizer_output_dir=final_dir,
                waiter_output_dir=waiter_dir,
                poll_seconds=poll_seconds,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
                poll_count=polls,
                hhar_contract=details,
                finalizer_returncode=0,
                finalizer_log=str(log_path),
            )
            _atomic_write_json(complete, status_path)
            _append_log(log_path, "FINALIZER complete and final manifest passed")
            return 0, complete

        waiting = _base_status(
            phase="waiting",
            hhar_input_dir=hhar_dir,
            legacy_input_dir=legacy_dir,
            finalizer_output_dir=final_dir,
            waiter_output_dir=waiter_dir,
            poll_seconds=poll_seconds,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            poll_count=polls,
            hhar_contract=details,
            finalizer_log=str(log_path),
        )
        _atomic_write_json(waiting, status_path)
        _append_log(log_path, "WAITING " + "; ".join(str(item) for item in details.get("errors", ())))
        if max_polls is not None and polls >= int(max_polls):
            waiting["returncode"] = 1
            waiting["reason"] = "max_polls reached before HHAR contract became ready"
            _atomic_write_json(waiting, status_path)
            return 1, waiting
        sleep_fn(float(poll_seconds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--hhar-input-dir", default=str(DEFAULT_HHAR_INPUT_DIR))
    parser.add_argument("--legacy-input-dir", default=str(DEFAULT_LEGACY_INPUT_DIR))
    parser.add_argument("--finalizer-output-dir", default=str(DEFAULT_FINALIZER_OUTPUT_DIR))
    parser.add_argument("--waiter-output-dir", default=str(DEFAULT_WAITER_OUTPUT_DIR))
    parser.add_argument("--finalizer-script", default=str(DEFAULT_FINALIZER_SCRIPT))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, status = run_waiter(
            hhar_input_dir=args.hhar_input_dir,
            legacy_input_dir=args.legacy_input_dir,
            finalizer_output_dir=args.finalizer_output_dir,
            waiter_output_dir=args.waiter_output_dir,
            finalizer_script=args.finalizer_script,
            poll_seconds=args.poll_seconds,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            max_polls=args.max_polls,
            resume=args.resume,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"[four-dataset finalizer waiter] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": status.get("status"), "phase": status.get("phase"), "returncode": code}, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_HHAR_CELLS",
    "EXPECTED_HHAR_FLOWS",
    "HHAR_QUEUE_PROTOCOL",
    "WAITER_PROTOCOL",
    "inspect_hhar_contract",
    "run_waiter",
]
