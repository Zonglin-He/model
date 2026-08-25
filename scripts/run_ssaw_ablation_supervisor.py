"""Run small SSAW-ablation groups in child processes to bound memory use."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_complete(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        return bool(
            json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "completed_at"
            )
        )
    except (OSError, ValueError, TypeError):
        return False


def without_option(arguments: list[str], option: str) -> list[str]:
    filtered = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument == option:
            skip_value = True
            continue
        if argument.startswith(f"{option}="):
            continue
        filtered.append(argument)
    return filtered


def atomic_status(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--runner",
        default="run_ssaw_internal_ablation.py",
        help="Script filename under scripts/ to execute in isolated workers.",
    )
    parser.add_argument("--restart-delay-seconds", type=float, default=0.5)
    parser.add_argument("--jobs-per-worker", type=int, default=3)
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to the SSAW runner.",
    )
    args = parser.parse_args()
    if args.jobs_per_worker < 1:
        parser.error("--jobs-per-worker must be positive")
    forwarded = list(args.runner_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    forwarded = without_option(forwarded, "--max-jobs")
    forwarded = without_option(forwarded, "--output-dir")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = (ROOT / "scripts" / args.runner).resolve()
    if runner.parent != (ROOT / "scripts").resolve() or not runner.is_file():
        parser.error("--runner must name an existing script under scripts/")
    command = [
        sys.executable,
        str(runner),
        "--output-dir",
        str(output_dir),
        *forwarded,
        "--max-jobs",
        str(args.jobs_per_worker),
    ]
    child_environment = os.environ.copy()
    iteration = 0
    consecutive_failures = 0
    while not is_complete(output_dir):
        iteration += 1
        started_at = utc_now()
        print(f"[Supervisor] {runner.stem} worker {iteration}", flush=True)
        return_code = subprocess.call(
            command, cwd=ROOT, env=child_environment
        )
        consecutive_failures = (
            0 if return_code == 0 else consecutive_failures + 1
        )
        completed = is_complete(output_dir)
        atomic_status(
            output_dir / "supervisor.status.json",
            {
                "supervisor_pid": os.getpid(),
                "iteration": iteration,
                "worker_return_code": int(return_code),
                "consecutive_failures": consecutive_failures,
                "worker_started_at": started_at,
                "worker_finished_at": utc_now(),
                "completed": completed,
                "jobs_per_worker": int(args.jobs_per_worker),
                "command": command,
            },
        )
        if completed:
            print("[Supervisor] SSAW ablation completed.", flush=True)
            return 0
        delay = min(
            30.0,
            args.restart_delay_seconds * (2 ** min(consecutive_failures, 5)),
        )
        time.sleep(delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
