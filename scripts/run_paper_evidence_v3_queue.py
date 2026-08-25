"""Run the canonical paper-evidence v3 reruns sequentially and resumably."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL_PATH = ROOT / "configs" / "paper_evidence_protocol_v3.json"
OUTPUT_ROOT = ROOT / "results" / "paper_evidence_v3"
STATUS_PATH = OUTPUT_ROOT / "queue_status.json"
LOCK_PATH = OUTPUT_ROOT / "queue.lock"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _phase_commands() -> list[tuple[str, list[str], Path]]:
    python = sys.executable
    source_profiles = (
        ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"
    )
    source_reference = (
        source_profiles / "validation_seeds_0_1_2" / "paired_raw.csv"
    )
    tta_profiles = ROOT / "configs" / "paper_flow_profiles_v1.json"
    common = [
        "--source-seeds",
        "0,1,2",
        "--profile-root",
        str(source_profiles),
        "--tta-profile-json",
        str(tta_profiles),
        "--source-reference-csv",
        str(source_reference),
        "--data-path",
        str(ROOT / "data" / "Dataset"),
        "--device",
        "cuda:0",
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    ]

    main_output = OUTPUT_ROOT / "main_full_no_ssaw"
    main_command = [
        python,
        str(ROOT / "scripts" / "run_dusafe_replacement_ablation.py"),
        "--study",
        "core",
        "--datasets",
        "EEG,HAR,FD,HHAR",
        "--runners",
        "confidence_only,hard_ssaw",
        "--protocol-override",
        "paper_evidence_v3_canonical_current_dusafe_seed012_main_full_no_ssaw",
        "--output-dir",
        str(main_output),
        *common,
    ]

    core_output = OUTPUT_ROOT / "core_ablation_har_hhar"
    core_command = [
        python,
        str(ROOT / "scripts" / "run_dusafe_replacement_ablation.py"),
        "--study",
        "core",
        "--datasets",
        "HAR,HHAR",
        "--runners",
        "accept_all_raw,confidence_only,random_eligible_spline,hard_ssaw",
        "--protocol-override",
        "paper_evidence_v3_canonical_core_har_hhar_seed012",
        "--output-dir",
        str(core_output),
        *common,
    ]

    safety_output = OUTPUT_ROOT / "safety_har_12_to_16_physical_s3_s6"
    safety_command = [
        python,
        str(ROOT / "scripts" / "run_controlled_safety_benchmark.py"),
        "--data_path",
        str(ROOT / "data" / "Dataset"),
        "--device",
        "cuda:0",
        "--registry",
        "production",
        "--datasets",
        "HAR",
        "--methods",
        "DuSafe",
        "--variants",
        "full,no_ssaw",
        "--scenarios",
        "HAR:12->16",
        "--flow-profile-json",
        str(tta_profiles),
        "--flowwise-source-profile-json",
        str(source_profiles / "selected_profiles.json"),
        "--source-reference-csv",
        str(source_reference),
        "--corruptions",
        "blackout,signal_freeze",
        "--severities",
        "s3,s6",
        "--physical_protocol",
        "--source_seeds",
        "0,1,2",
        "--stream_seeds",
        "42",
        "--corruption_fraction",
        "0.5",
        "--corruption_seed",
        "314159",
        "--pretrain_cache_dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--output_dir",
        str(safety_output),
    ]
    return [
        ("main_full_no_ssaw", main_command, main_output),
        ("core_ablation_har_hhar", core_command, core_output),
        ("safety_har_12_to_16_physical_s3_s6", safety_command, safety_output),
    ]


def main() -> int:
    from scripts.run_final_ssaw_full_no_ssaw_five_flow import (
        production_code_sha256,
    )

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    observed_code_hash = production_code_sha256()
    expected_code_hash = str(protocol["production_code_sha256"])
    if observed_code_hash != expected_code_hash:
        raise RuntimeError(
            "production code changed after protocol freeze: "
            f"{observed_code_hash} != {expected_code_hash}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"queue lock already exists: {LOCK_PATH}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    phases: list[dict[str, object]] = []
    queue = {
        "protocol": protocol["protocol"],
        "status": "running",
        "pid": os.getpid(),
        "started_at": _now(),
        "production_code_sha256": observed_code_hash,
        "protocol_path": str(PROTOCOL_PATH),
        "phases": phases,
    }
    _write_json(STATUS_PATH, queue)
    try:
        environment = os.environ.copy()
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64"
        )
        for name, command, output_dir in _phase_commands():
            phase = {
                "name": name,
                "status": "running",
                "started_at": _now(),
                "command": command,
                "output_dir": str(output_dir),
            }
            phases.append(phase)
            queue["current_phase"] = name
            _write_json(STATUS_PATH, queue)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "queue.log"
            with log_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            phase["returncode"] = int(completed.returncode)
            phase["completed_at"] = _now()
            phase["status"] = (
                "complete" if completed.returncode == 0 else "failed"
            )
            _write_json(STATUS_PATH, queue)
            if completed.returncode != 0:
                queue["status"] = "failed"
                queue["failed_phase"] = name
                queue["completed_at"] = _now()
                _write_json(STATUS_PATH, queue)
                return int(completed.returncode)
        queue["status"] = "complete"
        queue["current_phase"] = None
        queue["completed_at"] = _now()
        _write_json(STATUS_PATH, queue)
        return 0
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
