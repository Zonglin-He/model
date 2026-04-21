import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"
STATUS_PATH = RESULTS_ROOT / "component_gate_research_status.json"
TUNE_LOG = RESULTS_ROOT / "component_gate_research_tuning.log"
GATE_LOG = RESULTS_ROOT / "component_gate_research_gate.log"


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
        "stage": "tuning",
        "tuning": {
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
        "scripts/tune_component_gates_stepwise.py",
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
        str(ROOT / "results" / "tta_experiments_logs" / "component_gate_tuning_runs"),
        "--output-dir",
        str(ROOT / "results" / "tta_experiments_logs" / "component_gate_tuning_summary"),
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache"),
        "--write-overrides",
        "--adv-sigma-step",
        "0.05",
        "--adv-sigma-points",
        "9",
        "--adv-num-span",
        "16",
        "--adv-num-step",
        "4",
        "--adv-num-max",
        "64",
        "--cons-step",
        "0.1",
        "--cons-points",
        "13",
        "--sem-step",
        "0.1",
        "--sem-points",
        "15",
        "--proto-step",
        "0.1",
        "--proto-points",
        "9",
    ]

    status["stage"] = "tuning"
    status["tuning"]["started"] = now()
    write_status(status)
    tune_exit = run_command(tune_cmd, TUNE_LOG)
    status["tuning"]["finished"] = now()
    status["tuning"]["exit_code"] = tune_exit
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
