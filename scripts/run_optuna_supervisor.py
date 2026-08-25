"""Restart the stepwise Optuna worker after every isolated trial."""

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


def manifest_is_complete(output_dir: Path) -> bool:
    path = Path(output_dir) / "manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return bool(manifest.get("completed_at"))
    except (OSError, ValueError, TypeError):
        return False


def without_option(arguments: list[str], option: str) -> list[str]:
    """Remove one value-taking option so the supervisor can force it."""
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


def write_status(output_dir: Path, payload: dict) -> None:
    path = Path(output_dir) / "supervisor.status.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--restart-delay-seconds", type=float, default=3.0)
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to run_optuna_stepwise.py.",
    )
    args = parser.parse_args()
    if args.restart_delay_seconds < 0.0:
        parser.error("--restart-delay-seconds must be non-negative")

    forwarded = list(args.runner_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    forwarded = without_option(forwarded, "--max-trials-per-invocation")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_optuna_stepwise.py"),
        "--output-dir",
        str(output_dir),
        *forwarded,
        "--max-trials-per-invocation",
        "1",
    ]
    child_environment = os.environ.copy()
    child_environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    iteration = 0
    consecutive_failures = 0
    while not manifest_is_complete(output_dir):
        iteration += 1
        started_at = utc_now()
        print(
            f"[Supervisor] worker {iteration} starting at {started_at}",
            flush=True,
        )
        return_code = subprocess.call(
            command,
            cwd=ROOT,
            env=child_environment,
        )
        if return_code == 0:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        status = {
            "supervisor_pid": os.getpid(),
            "iteration": iteration,
            "worker_return_code": int(return_code),
            "consecutive_failures": consecutive_failures,
            "worker_started_at": started_at,
            "worker_finished_at": utc_now(),
            "completed": manifest_is_complete(output_dir),
            "command": command,
            "allocator_config": child_environment.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
        }
        write_status(output_dir, status)
        if status["completed"]:
            print("[Supervisor] all requested studies completed.", flush=True)
            return 0
        delay = min(
            30.0,
            args.restart_delay_seconds * (2 ** min(consecutive_failures, 4)),
        )
        print(
            f"[Supervisor] worker exited with {return_code}; "
            f"restarting in {delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
