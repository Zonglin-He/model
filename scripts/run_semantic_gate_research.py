import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"
STATUS_PATH = RESULTS_ROOT / "semantic_gate_research_status.json"
TUNE_LOG = RESULTS_ROOT / "semantic_gate_research_tuning.log"
GATE_LOG = RESULTS_ROOT / "semantic_gate_research_gate.log"


def now():
    return datetime.now().isoformat(timespec="seconds")


def write_status(payload):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(command, log_path):
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                handle.write(line)
                handle.flush()
        finally:
            return_code = process.wait()
    return return_code


def main():
    python = str(ROOT / ".venv311" / "Scripts" / "python.exe")
    data_path = str(ROOT / "data" / "Dataset")

    status = {
        "started_at": now(),
        "stage": "semantic_tuning",
        "semantic_tuning": {
            "started": None,
            "finished": None,
            "exit_code": None,
            "log": str(TUNE_LOG),
        },
        "gate_diagnostics": {
            "started": None,
            "finished": None,
            "exit_code": None,
            "log": str(GATE_LOG),
        },
    }
    write_status(status)

    tune_cmd = [
        python,
        "scripts/tune_semantic_gate_deep.py",
        "--datasets",
        "EEG,HAR,FD",
        "--backbone",
        "CNN",
        "--da-method",
        "ACCUP",
        "--data-path",
        data_path,
        "--device",
        "cuda",
        "--seeds",
        "41,42,43",
        "--save-dir",
        str(ROOT / "results" / "tta_experiments_logs" / "semantic_gate_deep_runs"),
        "--output-dir",
        str(ROOT / "results" / "tta_experiments_logs" / "semantic_gate_deep_summary"),
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache"),
        "--write-overrides",
        "--sem-step",
        "0.05",
        "--sem-points",
        "19",
        "--proto-step",
        "0.1",
        "--proto-points",
        "13",
        "--sem-pass-low",
        "0.20",
        "--sem-pass-high",
        "0.85",
        "--sem-pass-focus-threshold",
        "0.90",
        "--f1-tol",
        "1e-6",
    ]

    status["stage"] = "semantic_tuning"
    status["semantic_tuning"]["started"] = now()
    write_status(status)
    tune_exit = run_command(tune_cmd, TUNE_LOG)
    status["semantic_tuning"]["finished"] = now()
    status["semantic_tuning"]["exit_code"] = tune_exit
    write_status(status)
    if tune_exit != 0:
        raise SystemExit(tune_exit)

    gate_cmd = [
        python,
        "scripts/run_gate_diagnostics.py",
        "--data_path",
        data_path,
        "--device",
        "cuda",
        "--seeds",
        "41,42,43",
        "--backbone",
        "CNN",
    ]

    status["stage"] = "gate_diagnostics"
    status["gate_diagnostics"]["started"] = now()
    write_status(status)
    gate_exit = run_command(gate_cmd, GATE_LOG)
    status["gate_diagnostics"]["finished"] = now()
    status["gate_diagnostics"]["exit_code"] = gate_exit
    status["stage"] = "completed"
    status["completed_at"] = now()
    write_status(status)
    raise SystemExit(gate_exit)


if __name__ == "__main__":
    main()
