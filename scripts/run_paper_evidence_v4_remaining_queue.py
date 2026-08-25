"""Run all model-dependent v4 paper experiments after the main rerun.

The queue is resumable at the phase and cell levels.  It waits for the
separately launched four-dataset main rerun, then serializes all remaining GPU
work so production and evidence logging are never benchmarked concurrently.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results" / "paper_evidence_v4"
STATUS_PATH = RESULT_ROOT / "remaining_queue_status.json"
LOCK_PATH = RESULT_ROOT / "remaining_queue.lock"
MAIN_MANIFEST = RESULT_ROOT / "main_full_no_ssaw" / "manifest.json"
CODE_HASH = "1bf2ab908abffcbc5829d3bd8797f9270c983b380e70014bbdcb6f637ed58314"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _main_complete() -> bool:
    manifest = _load_json(MAIN_MANIFEST)
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("completed_cells", -1)) == 120
        and manifest.get("failures") == []
        and manifest.get("production_code_sha256") == CODE_HASH
    )


def _commands() -> list[tuple[str, list[str], Path]]:
    python = sys.executable
    profiles = ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"
    reference = profiles / "validation_seeds_0_1_2" / "paired_raw.csv"
    flow_profiles = ROOT / "configs" / "paper_flow_profiles_v1.json"
    common = [
        "--source-seeds", "0,1,2",
        "--profile-root", str(profiles),
        "--tta-profile-json", str(flow_profiles),
        "--source-reference-csv", str(reference),
        "--data-path", str(ROOT / "data" / "Dataset"),
        "--device", "cuda:0",
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    ]

    core = RESULT_ROOT / "core_ablation_har_hhar"
    core_command = [
        python, str(ROOT / "scripts" / "run_dusafe_replacement_ablation.py"),
        "--study", "core",
        "--datasets", "HAR,HHAR",
        "--runners",
        "accept_all_raw,confidence_only,random_eligible_spline,hard_ssaw",
        "--protocol-override", "paper_evidence_v4_logging_split_core_seed012",
        "--output-dir", str(core),
        *common,
    ]

    safety = RESULT_ROOT / "safety_har_12_to_16_physical_s3_s6"
    safety_command = [
        python, str(ROOT / "scripts" / "run_controlled_safety_benchmark.py"),
        "--data_path", str(ROOT / "data" / "Dataset"),
        "--device", "cuda:0",
        "--registry", "production",
        "--datasets", "HAR",
        "--methods", "DuSafe",
        "--variants", "full,no_ssaw",
        "--override", "dusafe_logging_mode=evidence",
        "--scenarios", "HAR:12->16",
        "--flow-profile-json", str(flow_profiles),
        "--flowwise-source-profile-json", str(profiles / "selected_profiles.json"),
        "--source-reference-csv", str(reference),
        "--corruptions", "blackout,signal_freeze",
        "--severities", "s3,s6",
        "--physical_protocol",
        "--source_seeds", "0,1,2",
        "--stream_seeds", "42",
        "--corruption_fraction", "0.5",
        "--corruption_seed", "314159",
        "--pretrain_cache_dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--output_dir", str(safety),
    ]

    heldout = RESULT_ROOT / "heldout_selection_hhar_4_to_5"
    heldout_command = [
        python, str(ROOT / "scripts" / "run_representative_causal_ablation.py"),
        "--datasets", "HHAR",
        "--source-seeds", "0,1,2",
        "--conditions", "clean",
        "--horizons", "1",
        "--selected-flows-json",
        str(ROOT / "configs" / "paper_representative_flow_selection_secondary_v1.json"),
        "--profile-json", str(flow_profiles),
        "--source-profile-root", str(profiles),
        "--eeg-source-profile-root",
        str(ROOT / "results" / "optuna" / "eeg_ssaw_weight_sweep_v1" / "selected"),
        "--source-reference-csv", str(reference),
        "--data-path", str(ROOT / "data" / "Dataset"),
        "--device", "cuda:0",
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--output-dir", str(heldout),
        "--heldout-bank-tag", "paper_v4_unseen",
        "--execute",
    ]

    confidence = RESULT_ROOT / "confidence_eeg_7_to_18"
    confidence_command = [
        python, str(ROOT / "scripts" / "run_representative_causal_ablation.py"),
        "--datasets", "EEG",
        "--source-seeds", "0,1,2",
        "--conditions", "clean,signal_freeze:moderate",
        # EEG 7->18 has too few deployment batches to form a complete
        # five-batch future window under its frozen flow-specific batch size.
        # Use the next-batch endpoint, which is defined for every batch except
        # the terminal one and matches the causal runner's supported protocol.
        "--horizons", "1",
        "--selected-flows-json",
        str(ROOT / "configs" / "paper_representative_flow_selection_eeg_retuned_v1.json"),
        "--profile-json",
        str(ROOT / "results" / "optuna" / "eeg_flowwise_three_seed_v1" / "paper_flow_profiles_v3_eeg_retuned.json"),
        "--source-profile-root", str(profiles),
        "--eeg-source-profile-root",
        str(ROOT / "results" / "optuna" / "eeg_ssaw_weight_sweep_v1" / "selected"),
        "--source-reference-csv", str(reference),
        "--data-path", str(ROOT / "data" / "Dataset"),
        "--device", "cuda:0",
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--output-dir", str(confidence),
        "--heldout-bank-tag", "paper_v4_unseen",
        "--execute",
    ]

    augmentation = RESULT_ROOT / "augmentation_controls_har_hhar"
    augmentation_command = [
        python, str(ROOT / "scripts" / "run_dusafe_replacement_ablation.py"),
        "--study", "augmentation",
        "--datasets", "HAR,HHAR",
        "--protocol-override", "paper_evidence_v4_augmentation_controls_seed012",
        "--output-dir", str(augmentation),
        *common,
    ]

    efficiency = RESULT_ROOT / "efficiency_all_methods_har_12to16"
    efficiency_command = [
        python, str(ROOT / "scripts" / "run_compute_overhead_v2.py"),
        "--queue",
        "--data-path", str(ROOT / "data" / "Dataset"),
        "--device", "cuda:0",
        "--registry", "benchmark",
        "--datasets", "HAR",
        "--scenarios", "12->16",
        "--methods",
        "NoAdap,Tent,EATA,SAR,ACCUPOfficial,CoTTA,SoTTA,RoTTA,COME,NOTE,DuSafe",
        "--variants", "full,no_ssaw",
        "--profiles", "default",
        "--source-seed", "1",
        "--stream-seed", "42",
        "--pretrain-cache-dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--eata-fisher-cache-dir", str(ROOT / "results" / "eata_fisher_cache"),
        "--flow-profile-json", str(flow_profiles),
        "--output-dir", str(efficiency),
    ]

    final = RESULT_ROOT / "final"
    final_command = [
        python, str(ROOT / "scripts" / "finalize_paper_evidence_v4.py")
    ]
    return [
        ("core_ablation_har_hhar", core_command, core),
        ("safety_har_12_to_16", safety_command, safety),
        ("heldout_selection_hhar_4_to_5", heldout_command, heldout),
        ("confidence_eeg_7_to_18", confidence_command, confidence),
        ("augmentation_controls_har_hhar", augmentation_command, augmentation),
        ("efficiency_all_methods_har_12to16", efficiency_command, efficiency),
        ("finalize_main_core_safety", final_command, final),
    ]


def main() -> int:
    from scripts.run_final_ssaw_full_no_ssaw_five_flow import (
        production_code_sha256,
    )

    observed_hash = production_code_sha256()
    if observed_hash != CODE_HASH:
        raise RuntimeError(f"production code hash changed: {observed_hash}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"remaining queue already exists: {LOCK_PATH}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    state: dict[str, object] = {
        "protocol": "paper_evidence_v4_logging_split_exact_lazy_seed012",
        "status": "waiting_for_main",
        "pid": os.getpid(),
        "production_code_sha256": observed_hash,
        "started_at": _now(),
        "phases": [],
    }
    _write_json(STATUS_PATH, state)
    try:
        while not _main_complete():
            time.sleep(30)
            if production_code_sha256() != CODE_HASH:
                raise RuntimeError("production code changed while waiting")
        state["status"] = "running"
        for name, command, output_dir in _commands():
            phase = {
                "name": name,
                "status": "running",
                "started_at": _now(),
                "command": command,
                "output_dir": str(output_dir),
            }
            state["phases"].append(phase)
            state["current_phase"] = name
            _write_json(STATUS_PATH, state)
            output_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True,max_split_size_mb:64",
            )
            with (output_dir / "queue.log").open("a", encoding="utf-8") as log:
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
            phase["status"] = "complete" if completed.returncode == 0 else "failed"
            _write_json(STATUS_PATH, state)
            if completed.returncode != 0:
                state["status"] = "failed"
                state["failed_phase"] = name
                state["completed_at"] = _now()
                _write_json(STATUS_PATH, state)
                return int(completed.returncode)
        state["status"] = "complete"
        state["current_phase"] = None
        state["completed_at"] = _now()
        _write_json(STATUS_PATH, state)
        return 0
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
