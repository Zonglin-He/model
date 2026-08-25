"""Serialize the long HAR TTA search and EEG coupling ablation on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TTA_PARAMETERS = (
    "batch_size",
    "learning_rate",
    "steps",
    "ssaw_auxiliary_weight",
    "ssaw_strength",
    "ssaw_kl_scale",
    "confidence_keep_fraction",
    "grad_clip",
    "weight_decay",
)
EEG_RUNNERS = (
    "raw_only",
    "confidence_only",
    "semantic_only",
    "dual_gate_only",
    "ssaw_only",
    "ssaw_confidence",
    "ssaw_semantic",
    "full",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def build_commands(args) -> tuple[list[str], list[str]]:
    optuna_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_optuna_supervisor.py"),
        "--output-dir",
        str(Path(args.har_output_dir).resolve()),
        "--restart-delay-seconds",
        "3",
        "--",
        "--data-path",
        str(Path(args.data_path).resolve()),
        "--device",
        args.device,
        "--backbone",
        args.backbone,
        "--datasets",
        "HAR",
        "--passes",
        str(args.passes),
        "--source-seeds",
        "1,2,3",
        "--test-time-seeds",
        "42",
        "--pretrain-cache-dir",
        str(Path(args.pretrain_cache_dir).resolve()),
        "--skip-source",
        "--tta-parameters",
        ",".join(TTA_PARAMETERS),
        "--tta-batch-cap",
        str(args.tta_batch_cap),
        "--min-ssaw-participation",
        str(args.min_ssaw_participation),
    ]
    eeg_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_dusafe_factorial_ablation.py"),
        "--output-root",
        str(Path(args.eeg_output_dir).resolve()),
        "--data-path",
        str(Path(args.data_path).resolve()),
        "--device",
        args.device,
        "--backbone",
        args.backbone,
        "--datasets",
        "EEG",
        "--source-seeds",
        "1,2,3",
        "--stream-seed",
        "42",
        "--runners",
        ",".join(EEG_RUNNERS),
        "--pretrain-cache-dir",
        str(Path(args.pretrain_cache_dir).resolve()),
    ]
    return optuna_command, eeg_command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--data-path", default=str(ROOT / "data" / "Dataset")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--tta-batch-cap", type=int, default=96)
    parser.add_argument("--min-ssaw-participation", type=float, default=0.25)
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--har-output-dir",
        default=str(ROOT / "results" / "optuna" / "har_tta_only_3source_v1"),
    )
    parser.add_argument(
        "--eeg-output-dir",
        default=str(
            ROOT / "results" / "ablation" / "dusafe_bundle_synergy_eeg_v2"
        ),
    )
    parser.add_argument(
        "--status-path",
        default=str(ROOT / "results" / "background" / "har_then_eeg.json"),
    )
    args = parser.parse_args(argv)
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.tta_batch_cap < 1:
        parser.error("--tta-batch-cap must be positive")
    if not 0.0 <= args.min_ssaw_participation <= 1.0:
        parser.error("--min-ssaw-participation must be in [0, 1]")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    status_path = Path(args.status_path).resolve()
    optuna_command, eeg_command = build_commands(args)
    status = {
        "protocol": "HAR TTA-only Optuna followed by EEG coupling ablation v1",
        "queue_pid": os.getpid(),
        "started_at": utc_now(),
        "phase": "har_optuna",
        "source_seeds": [1, 2, 3],
        "source_seed_is_independent_unit": True,
        "stream_seed": 42,
        "stream_seed_is_paired_control": True,
        "source_hyperparameters_tuned": False,
        "target_labels_used_for_har_selection": True,
        "tta_parameters": list(TTA_PARAMETERS),
        "har_command": optuna_command,
        "eeg_command": eeg_command,
    }
    atomic_write_json(status, status_path)
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    har_result = subprocess.run(
        optuna_command, cwd=ROOT, env=environment, check=False
    )
    status["har_return_code"] = int(har_result.returncode)
    status["har_finished_at"] = utc_now()
    if har_result.returncode != 0:
        status["phase"] = "failed_har_optuna"
        atomic_write_json(status, status_path)
        return int(har_result.returncode)

    status["phase"] = "eeg_coupling_ablation"
    atomic_write_json(status, status_path)
    eeg_result = subprocess.run(
        eeg_command, cwd=ROOT, env=environment, check=False
    )
    status["eeg_return_code"] = int(eeg_result.returncode)
    status["eeg_finished_at"] = utc_now()
    status["phase"] = (
        "complete" if eeg_result.returncode == 0 else "failed_eeg_ablation"
    )
    status["completed_at"] = (
        utc_now() if eeg_result.returncode == 0 else None
    )
    atomic_write_json(status, status_path)
    return int(eeg_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
