"""Launch the five-scenario DuSafe Optuna sweep as a detached process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"
        ),
    )
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="Restart an isolated one-trial worker until tuning completes.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to run_optuna_stepwise.py.",
    )
    args = parser.parse_args()
    forwarded = list(args.runner_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "runner.stdout.log"
    stderr_path = output_dir / "runner.stderr.log"
    target = (
        ROOT / "scripts" / "run_optuna_supervisor.py"
        if args.supervise
        else ROOT / "scripts" / "run_optuna_stepwise.py"
    )
    command = [
        sys.executable,
        str(target),
        "--output-dir",
        str(output_dir),
        *(["--"] if args.supervise else []),
        *forwarded,
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creationflags,
        )
    payload = {
        "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "supervised": bool(args.supervise),
    }
    (output_dir / "runner.pid.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
