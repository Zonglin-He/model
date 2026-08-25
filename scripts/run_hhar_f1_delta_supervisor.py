"""Restart the HHAR paired-F1 tuner after each isolated Optuna trial."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def is_complete(output_dir: Path) -> bool:
    manifest = output_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def write_status(output_dir: Path, payload: dict) -> None:
    path = output_dir / "supervisor.status.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--restart-delay-seconds", type=float, default=2.0)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = list(args.runner_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "tune_hhar_ssaw_f1_delta.py"),
        "--output-dir",
        str(output_dir),
        *forwarded,
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    iteration = 0
    failures = 0
    while not is_complete(output_dir):
        iteration += 1
        returncode = subprocess.call(command, cwd=ROOT, env=environment)
        failures = failures + 1 if returncode else 0
        payload = {
            "pid": os.getpid(),
            "iteration": iteration,
            "worker_returncode": int(returncode),
            "consecutive_failures": failures,
            "complete": is_complete(output_dir),
            "command": command,
            "allocator": environment.get("PYTORCH_CUDA_ALLOC_CONF"),
            "updated_at_unix": time.time(),
        }
        write_status(output_dir, payload)
        if payload["complete"]:
            return 0
        if failures >= 5:
            return int(returncode or 1)
        time.sleep(min(30.0, args.restart_delay_seconds * (2 ** min(failures, 4))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
