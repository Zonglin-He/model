import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"
STATUS_PATH = RESULTS_ROOT / "semantic_har_fd_research_status.json"
TUNE_LOG = RESULTS_ROOT / "semantic_har_fd_tuning.log"
GATE_LOG = RESULTS_ROOT / "semantic_har_fd_gate.log"
ABLATION_LOG = RESULTS_ROOT / "semantic_har_fd_ablation.log"


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
            "output_dir": str(RESULTS_ROOT / "semantic_gate_har_fd_summary"),
        },
        "gate_diagnostics": {
            "started": None,
            "finished": None,
            "exit_code": None,
            "log": str(GATE_LOG),
        },
        "ablation": {
            "started": None,
            "finished": None,
            "exit_code": None,
            "log": str(ABLATION_LOG),
        },
    }
    write_status(status)

    tune_cmd = [
        python,
        "scripts/tune_semantic_gate_deep.py",
        "--datasets",
        "HAR,FD",
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
        str(RESULTS_ROOT / "semantic_gate_har_fd_runs"),
        "--output-dir",
        str(RESULTS_ROOT / "semantic_gate_har_fd_summary"),
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache"),
        "--write-overrides",
        "--param-order",
        "include_warmup_support,warmup_min,sem_thresh,proto_momentum",
        "--sem-step",
        "0.05",
        "--sem-points",
        "37",
        "--proto-step",
        "0.1",
        "--proto-points",
        "29",
        "--warmup-values",
        "1,2,4,8,16,32,64,96,128",
        "--sem-pass-low",
        "0.20",
        "--sem-pass-high",
        "0.85",
        "--sem-pass-focus-threshold",
        "0.85",
        "--f1-tol",
        "1e-6",
    ]

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

    ablation_cmd = [
        python,
        "scripts/run_crossdataset_ablation.py",
        "--data_path",
        data_path,
        "--device",
        "cuda",
        "--seeds",
        "41,42,43",
        "--backbone",
        "CNN",
    ]

    for stage_name, command, log_path in [
        ("semantic_tuning", tune_cmd, TUNE_LOG),
        ("gate_diagnostics", gate_cmd, GATE_LOG),
        ("ablation", ablation_cmd, ABLATION_LOG),
    ]:
        status["stage"] = stage_name
        status[stage_name]["started"] = now()
        write_status(status)
        exit_code = run_command(command, log_path)
        status[stage_name]["finished"] = now()
        status[stage_name]["exit_code"] = exit_code
        write_status(status)
        if exit_code != 0:
            raise SystemExit(exit_code)

    status["stage"] = "completed"
    status["completed_at"] = now()
    write_status(status)


if __name__ == "__main__":
    main()
