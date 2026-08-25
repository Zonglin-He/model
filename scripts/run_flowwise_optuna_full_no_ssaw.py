"""Flow-wise source-and-TTA coordinate tuning for paired Full/No-SSAW.

The tuner intentionally uses target Macro-F1 for parameter selection.  It is
therefore a target-selected development protocol, not a label-free or
confirmatory evaluation.  Every flow owns both a source-training profile and
a TTA profile.  This deliberately permits two flows with the same source
domain to select different source checkpoints.  Within one flow/seed/candidate,
Full and No-SSAW must still use the exact same checkpoint.

Each coordinate is exhausted before the next coordinate starts.  Full and
No-SSAW are evaluated from the same source checkpoint in isolated child
processes.  Selection is lexicographic: retain candidates within a small
Full-F1 tolerance of the stage maximum, then maximize Full minus No-SSAW.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence

import optuna
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from scripts.dusafe_factorial_runner_common import current_profiles  # noqa: E402
from scripts.run_final_ssaw_full_no_ssaw_five_flow import (  # noqa: E402
    DEFAULT_CACHE_DIRS,
    DEFAULT_GPU_LOCK,
    PROTOCOL as CELL_PROTOCOL,
    VARIANT_CLASSES,
    production_code_sha256,
)
from scripts.run_optuna_stepwise import (  # noqa: E402
    acquire_run_lock,
    atomic_write_json,
    recover_interrupted_grid_trials,
    storage_for,
)
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402


PROTOCOL = "flowwise_paired_ssaw_source_tta_deadline_tuning_v4_no_semantic"
DATASETS = ("EEG", "HAR", "FD", "HHAR")
VARIANTS = ("no_ssaw", "full")
TUNING_SOURCE_SEED = 1
VALIDATION_SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
F1_TOLERANCE_PP = 0.10
DEFAULT_OUTPUT_DIR = ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"
CELL_RUNNER = ROOT / "scripts" / "run_final_ssaw_full_no_ssaw_five_flow.py"
WORKER_TIMEOUT_SECONDS = 600
_SMALL_BATCHES = (1, 2, 3, 4, 5, 6)


STAGES = (
    # Deadline profile: retain only parameters with prior sensitivity or a
    # direct role in the Full/No-SSAW contrast.  Every coordinate is still
    # exhausted before advancing to the next one.
    ("source_learning_rate_deadline", "source", "pre_learning_rate"),
    ("source_num_epochs_deadline", "source", "num_epochs"),
    ("tta_batch_size_deadline", "tta", "batch_size"),
    ("tta_learning_rate_deadline", "tta", "learning_rate"),
    ("tta_steps_deadline", "tta", "steps"),
    ("tta_ssaw_auxiliary_weight_deadline", "tta", "ssaw_auxiliary_weight"),
    ("tta_spline_log_strength_deadline", "tta", "spline_log_strength"),
)

_DEADLINE_SOURCE_LR = {
    "EEG": (1e-4, 3e-4, 5e-4, 1e-3, 3e-3),
    "HAR": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    "FD": (1e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "HHAR": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
}

_DEADLINE_SOURCE_EPOCHS = {
    "EEG": (160, 240, 320),
    "HAR": (60, 100, 140),
    "FD": (40, 60, 100),
    "HHAR": (60, 100, 140),
}

_DEADLINE_TTA_BATCH = {
    "EEG": (*_SMALL_BATCHES, 8, 16, 48, 96, 192),
    "HAR": (*_SMALL_BATCHES, 8, 16, 24, 48, 96),
    "FD": (*_SMALL_BATCHES, 8, 16, 48, 96, 192),
    "HHAR": (*_SMALL_BATCHES, 8, 16, 24, 48, 96),
}

_DEADLINE_TTA_LR = {
    "EEG": (3e-4, 7.5e-4, 1.5e-3, 3e-3, 5e-3, 1e-2),
    "HAR": (1e-4, 2e-4, 3.325e-4, 5e-4, 1e-3, 2e-3),
    "FD": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4),
    "HHAR": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3),
}

_SOURCE_BATCH_COARSE = {
    "EEG": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192),
    "HAR": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512),
    "FD": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192),
    "HHAR": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512),
}

_TTA_BATCH_COARSE = {
    "EEG": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256),
    "HAR": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192),
    "FD": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256),
    "HHAR": (*_SMALL_BATCHES, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192),
}

_SOURCE_LR_COARSE = (
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
    1e-3, 3e-3, 1e-2, 3e-2,
)

_SOURCE_EPOCHS = (10, 20, 30, 40, 60, 80, 100, 140, 180, 240, 320)
_WEIGHT_DECAY = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

_LR_COARSE = {
    "EEG": (
        1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3,
        1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2,
    ),
    "HAR": (
        1e-5, 3e-5, 5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4,
        2.5e-4, 3e-4, 3.325e-4, 5e-4, 7.5e-4, 1e-3,
        1.5e-3, 2e-3, 3e-3,
    ),
    "FD": (
        1e-7, 3e-7, 1e-6, 2e-6, 3e-6, 5e-6, 7.5e-6,
        1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3,
    ),
    "HHAR": (
        1e-5, 3e-5, 5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4,
        3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3,
    ),
}

_AUXILIARY_WEIGHT = {
    "EEG": (0.0, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3,
            7.5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1),
    "HAR": (0.0, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2,
            5e-2, 7.5e-2, 1e-1, 1.5e-1, 2e-1, 3e-1),
    "FD": (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1,
           3e-1, 1.0, 3.0),
    "HHAR": (0.0, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 7.5e-1,
             1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
}

_GRAD_CLIP = (1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2,
              1e-1, 3e-1, 1.0)
_CONFIDENCE = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98,
               0.985, 0.99, 0.9925, 0.995, 0.9975, 0.999, 1.0)
_STEPS = tuple(range(1, 17)) + (20, 24, 32)
_SPLINE_STRENGTH = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
                    0.18, 0.20, 0.25, 0.30, 0.40, 0.50)
_SPLINE_CONTROL_POINTS = (3, 4, 6, 8, 10, 12, 16)
_SPLINE_DIRECTIONS = (1, 2, 4, 6, 8)


class WorkerFailure(RuntimeError):
    """Structured isolated-worker failure retained in trial provenance."""

    def __init__(self, message: str, details: Mapping[str, object]):
        super().__init__(message)
        self.details = dict(details)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tuner_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), CELL_RUNNER.resolve()):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _flow_slug(flow: Sequence[str]) -> str:
    return f"{flow[0]}_to_{flow[1]}"


def _stable_unique(values: Iterable[object], current: object) -> list[object]:
    result: list[object] = []
    for value in (*values, current):
        if value not in result:
            result.append(value)
    return result


def stage_values(
    dataset: str,
    stage_name: str,
    current_value: int | float,
) -> list[int | float]:
    """Return the explicit grid for one sequential coordinate stage."""

    dataset = str(dataset).upper()
    if stage_name == "source_learning_rate_deadline":
        return _stable_unique(_DEADLINE_SOURCE_LR[dataset], float(current_value))
    if stage_name == "source_num_epochs_deadline":
        return _stable_unique(_DEADLINE_SOURCE_EPOCHS[dataset], int(current_value))
    if stage_name == "tta_batch_size_deadline":
        return _stable_unique(_DEADLINE_TTA_BATCH[dataset], int(current_value))
    if stage_name == "tta_learning_rate_deadline":
        return _stable_unique(_DEADLINE_TTA_LR[dataset], float(current_value))
    if stage_name == "tta_steps_deadline":
        return _stable_unique((1, 2, 4, 8), int(current_value))
    if stage_name == "tta_ssaw_auxiliary_weight_deadline":
        center = float(current_value)
        values = (0.0, center * 0.25, center * 0.5, center, center * 2.0)
        return sorted(set(float(f"{value:.12g}") for value in values))
    if stage_name == "tta_spline_log_strength_deadline":
        return _stable_unique((0.10, 0.20, 0.30), float(current_value))
    if stage_name == "source_learning_rate_coarse":
        return _stable_unique(_SOURCE_LR_COARSE, float(current_value))
    if stage_name == "source_learning_rate_fine":
        center = float(current_value)
        factors = (0.50, 0.65, 0.80, 0.90, 1.0, 1.10, 1.25, 1.50, 2.0)
        values = [float(f"{center * factor:.12g}") for factor in factors]
        return sorted(set(value for value in values if value > 0.0))
    if stage_name == "source_batch_size_coarse":
        return _stable_unique(_SOURCE_BATCH_COARSE[dataset], int(current_value))
    if stage_name == "source_batch_size_dense":
        center = int(current_value)
        lower = max(1, center - 12)
        upper = min(512, center + 12)
        return _stable_unique((*_SMALL_BATCHES, *range(lower, upper + 1)), center)
    if stage_name == "source_num_epochs":
        return _stable_unique(_SOURCE_EPOCHS, int(current_value))
    if stage_name == "source_weight_decay":
        return _stable_unique(_WEIGHT_DECAY, float(current_value))
    if stage_name == "tta_batch_size_coarse":
        return _stable_unique(_TTA_BATCH_COARSE[dataset], int(current_value))
    if stage_name == "tta_batch_size_dense":
        center = int(current_value)
        lower = max(1, center - 12)
        upper = min(256, center + 12)
        return _stable_unique((*_SMALL_BATCHES, *range(lower, upper + 1)), center)
    if stage_name == "tta_learning_rate_coarse":
        return _stable_unique(_LR_COARSE[dataset], float(current_value))
    if stage_name == "tta_learning_rate_fine":
        center = float(current_value)
        factors = (0.50, 0.65, 0.80, 0.90, 1.0, 1.10, 1.25, 1.50, 2.0)
        values = [float(f"{center * factor:.12g}") for factor in factors]
        return sorted(set(value for value in values if value > 0.0))
    if stage_name == "tta_steps":
        return _stable_unique(_STEPS, int(current_value))
    if stage_name == "tta_weight_decay":
        return _stable_unique(_WEIGHT_DECAY, float(current_value))
    if stage_name == "tta_grad_clip":
        return _stable_unique(_GRAD_CLIP, float(current_value))
    if stage_name == "tta_confidence_keep_fraction":
        return _stable_unique(_CONFIDENCE, float(current_value))
    if stage_name == "tta_ssaw_auxiliary_weight":
        return _stable_unique(_AUXILIARY_WEIGHT[dataset], float(current_value))
    if stage_name == "tta_spline_log_strength":
        return _stable_unique(_SPLINE_STRENGTH, float(current_value))
    if stage_name == "tta_spline_control_points":
        return _stable_unique(_SPLINE_CONTROL_POINTS, int(current_value))
    if stage_name == "tta_spline_num_directions":
        return _stable_unique(_SPLINE_DIRECTIONS, int(current_value))
    raise KeyError(stage_name)


def flow_plan(datasets: Sequence[str]) -> list[tuple[str, tuple[str, str]]]:
    plan: list[tuple[str, tuple[str, str]]] = []
    for dataset in datasets:
        flows = tuple(formal_scenario_pairs(dataset))
        if len(flows) != 5:
            raise RuntimeError(f"{dataset}: expected five flows, found {len(flows)}")
        plan.extend((dataset, tuple(map(str, flow))) for flow in flows)
    return plan


def paired_tta_configs(base: Mapping[str, object]) -> dict[str, dict]:
    pair: dict[str, dict] = {}
    for variant in VARIANTS:
        config = deepcopy(dict(base))
        config["dusafe_variant"] = VARIANT_CLASSES[variant]
        config["enable_ssaw"] = variant == "full"
        config["enable_source_semantic_router"] = False
        pair[variant] = config
    return pair


def _cache_dir(dataset: str) -> Path:
    return Path(DEFAULT_CACHE_DIRS[dataset]).resolve()


def worker_spec(
    *,
    args,
    cell_dir: Path,
    dataset: str,
    flow: Sequence[str],
    source_seed: int,
    variant: str,
    source_config: Mapping[str, object],
    tta_config: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol": CELL_PROTOCOL,
        "production_code_sha256": production_code_sha256(),
        "cell_dir": str(cell_dir.resolve()),
        "dataset": dataset,
        "flow": list(flow),
        "source_seed": int(source_seed),
        "stream_seed": int(args.stream_seed),
        "variant": variant,
        "source_config": dict(source_config),
        "tta_config": dict(tta_config),
        "data_path": str(Path(args.data_path).resolve()),
        "device": str(args.device),
        "backbone": str(args.backbone),
        "pretrain_cache_dir": str(_cache_dir(dataset)),
        "gpu_lock_path": str(Path(args.gpu_lock_path).resolve()),
        "max_batches": args.max_batches,
    }


def _run_worker(spec: Mapping[str, object], work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / "worker_spec.json"
    atomic_write_json(spec, spec_path)
    log_path = work_dir / "worker.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, str(CELL_RUNNER), "--worker-spec", str(spec_path)],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=WORKER_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as exc:
        raise WorkerFailure(
            f"worker exceeded {WORKER_TIMEOUT_SECONDS}s deadline timeout",
            {
                "worker_returncode": -1,
                "worker_status": "timeout",
                "worker_error_type": type(exc).__name__,
                "worker_is_oom": False,
                "worker_native_crash": False,
                "worker_timed_out": True,
                "worker_summary_path": str((work_dir / "summary.json").resolve()),
            },
        ) from exc
    summary_path = work_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    if completed.returncode != 0 or summary.get("status") != "ok":
        error = (
            summary.get("error_message")
            or summary.get("error")
            or summary.get("message")
            or summary
        )
        details = {
            "worker_returncode": int(completed.returncode),
            "worker_status": str(summary.get("status", "missing_summary")),
            "worker_error_type": str(summary.get("error_type", "")),
            "worker_is_oom": bool(summary.get("is_oom", False)),
            "worker_native_crash": int(completed.returncode) not in (0, 1),
            "worker_summary_path": str(summary_path.resolve()),
        }
        raise WorkerFailure(
            f"worker failed rc={completed.returncode}: {str(error)[:1500]}",
            details,
        )
    return summary


def evaluate_pair(
    *,
    args,
    trial_dir: Path,
    dataset: str,
    flow: Sequence[str],
    source_seed: int,
    source_config: Mapping[str, object],
    tta_config: Mapping[str, object],
    expected_source_model_sha256: str | None = None,
) -> dict[str, object]:
    configs = paired_tta_configs(tta_config)
    rows: dict[str, dict] = {}
    for variant in VARIANTS:
        cell_dir = trial_dir / variant
        spec = worker_spec(
            args=args,
            cell_dir=cell_dir,
            dataset=dataset,
            flow=flow,
            source_seed=source_seed,
            variant=variant,
            source_config=source_config,
            tta_config=configs[variant],
        )
        rows[variant] = _run_worker(spec, cell_dir)
    source_hashes = {str(row["source_model_sha256"]) for row in rows.values()}
    if len(source_hashes) != 1:
        raise RuntimeError(
            f"paired source checkpoint mismatch: {sorted(source_hashes)}"
        )
    source_paths = {str(row["source_checkpoint_path"]) for row in rows.values()}
    if len(source_paths) != 1:
        raise RuntimeError(
            f"paired source checkpoint path mismatch: {sorted(source_paths)}"
        )
    source_hash = next(iter(source_hashes))
    if (
        expected_source_model_sha256 is not None
        and source_hash != str(expected_source_model_sha256)
    ):
        raise RuntimeError(
            "TTA candidate did not reuse the selected flow checkpoint: "
            f"expected={expected_source_model_sha256}, actual={source_hash}"
        )
    # Source confidence/semantic metadata are context-specific even though
    # the source tensor state is fixed.  Both variants use the same context
    # because the paired configs have the same deployment batch size.
    metadata_context = {
        "source_model_sha256": source_hash,
        "deployment_batch_size": int(tta_config["batch_size"]),
        "confidence_keep_fraction": float(
            tta_config["confidence_keep_fraction"]
        ),
        "stream_seed": int(args.stream_seed),
    }
    full_f1 = float(rows["full"]["f1"])
    no_ssaw_f1 = float(rows["no_ssaw"]["f1"])
    result = {
        "dataset": dataset,
        "scenario": _flow_label(flow),
        "source_seed": int(source_seed),
        "stream_seed": int(args.stream_seed),
        "full_f1": full_f1,
        "no_ssaw_f1": no_ssaw_f1,
        "full_minus_no_ssaw": full_f1 - no_ssaw_f1,
        "source_model_sha256": source_hash,
        "source_checkpoint_path": next(iter(source_paths)),
        "source_metadata_context_sha256": _sha256_json(metadata_context),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
    }
    atomic_write_json(result, trial_dir / "paired_summary.json")
    return result


def select_stage_winner(
    trials: Sequence[optuna.trial.FrozenTrial],
    *,
    parameter: str,
    f1_tolerance_pp: float,
) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.user_attrs.get("full_f1") is not None
        and trial.user_attrs.get("full_minus_no_ssaw") is not None
    ]
    if not completed:
        raise RuntimeError("stage has no completed paired trials")
    maximum_full = max(float(trial.user_attrs["full_f1"]) for trial in completed)
    tolerance = float(f1_tolerance_pp) / 100.0
    eligible = [
        trial
        for trial in completed
        if float(trial.user_attrs["full_f1"]) >= maximum_full - tolerance - 1e-12
    ]
    return max(
        eligible,
        key=lambda trial: (
            float(trial.user_attrs["full_minus_no_ssaw"]),
            float(trial.user_attrs["full_f1"]),
            float(trial.user_attrs.get("ssaw_participation", -math.inf)),
            -float(trial.params[parameter]),
            -int(trial.number),
        ),
    )


def _flow_signature(
    args,
    dataset: str,
    flow: Sequence[str],
    initial_source_config: Mapping[str, object],
    initial_tta_config: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "tuner_code_sha256": _tuner_code_sha256(),
        "production_code_sha256": production_code_sha256(),
        "dataset": dataset,
        "flow": list(flow),
        "tuning_source_seed": int(args.tuning_source_seed),
        "stream_seed": int(args.stream_seed),
        "stage_order": [stage[0] for stage in STAGES],
        "initial_source_config": dict(initial_source_config),
        "initial_tta_config": dict(initial_tta_config),
        "flow_specific_source_checkpoint": True,
        "f1_tolerance_pp": float(args.f1_tolerance_pp),
        "max_batches": args.max_batches,
        "target_labels_used_for_parameter_selection": True,
    }


def _initialize_flow_state(
    path: Path,
    *,
    signature: Mapping[str, object],
    source_config: Mapping[str, object],
    tta_config: Mapping[str, object],
) -> dict:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != dict(signature):
            raise RuntimeError(
                f"stale flow state at {path}; use a new output directory"
            )
        return state
    state = {
        "signature": dict(signature),
        "source_config": dict(source_config),
        "tta_config": dict(tta_config),
        "source_config_sha256": _sha256_json(dict(source_config)),
        "source_checkpoint_sha256": None,
        "source_checkpoint_path": None,
        "source_metadata_context_sha256": None,
        "next_stage_index": 0,
        "history": [],
        "completed": False,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_write_json(state, path)
    return state


def _study_name(dataset: str, flow: Sequence[str], index: int, stage: str) -> str:
    return (
        f"flowwise_{dataset.lower()}_{flow[0]}_to_{flow[1]}_"
        f"{index:02d}_{stage}"
    )


def _completed_candidates(study: optuna.Study, parameter: str) -> set[object]:
    return {
        trial.params[parameter]
        for trial in study.trials
        if parameter in trial.params
        and trial.state
        in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    }


def _publish_live_trial(output_dir: Path, row: Mapping[str, object]) -> None:
    path = output_dir / "live_trials.csv"
    frame = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if not frame.empty and {"study", "trial"}.issubset(frame.columns):
        frame = frame[
            ~(
                frame["study"].eq(row["study"])
                & frame["trial"].eq(int(row["trial"]))
            )
        ]
    frame = pd.concat([frame, pd.DataFrame([dict(row)])], ignore_index=True)
    atomic_write_csv(frame, path, index=False)


def run_stage(
    *,
    args,
    output_dir: Path,
    storage,
    dataset: str,
    flow: Sequence[str],
    flow_dir: Path,
    state: dict,
    stage_index: int,
) -> dict:
    stage_name, scope, parameter = STAGES[stage_index]
    config_key = "source_config" if scope == "source" else "tta_config"
    current = state[config_key][parameter]
    values = stage_values(dataset, stage_name, current)
    if args.smoke:
        values = [current]
    study_name = _study_name(dataset, flow, stage_index, stage_name)
    sampler = optuna.samplers.GridSampler(
        {parameter: values}, seed=1729 + stage_index
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    recover_interrupted_grid_trials(study, max_retries=1)
    trial_root = flow_dir / "trials" / study_name
    trial_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        candidate = trial.suggest_categorical(parameter, values)
        candidate_source = deepcopy(state["source_config"])
        candidate_tta = deepcopy(state["tta_config"])
        if scope == "source":
            candidate_source[parameter] = candidate
        else:
            candidate_tta[parameter] = candidate
        trial_dir = trial_root / f"trial_{trial.number:04d}"
        started = time.time()
        try:
            result = evaluate_pair(
                args=args,
                trial_dir=trial_dir,
                dataset=dataset,
                flow=flow,
                source_seed=args.tuning_source_seed,
                source_config=candidate_source,
                tta_config=candidate_tta,
                expected_source_model_sha256=(
                    state.get("source_checkpoint_sha256")
                    if scope == "tta"
                    else None
                ),
            )
        except WorkerFailure as exc:
            trial.set_user_attr("failure", "paired_worker_failure")
            trial.set_user_attr("failure_message", str(exc)[:2000])
            for name, value in exc.details.items():
                if isinstance(value, (bool, int, float, str)):
                    trial.set_user_attr(name, value)
            raise optuna.TrialPruned(str(exc)[:500]) from exc
        except RuntimeError as exc:
            trial.set_user_attr("failure", "protocol_validation_failure")
            trial.set_user_attr("failure_message", str(exc)[:2000])
            raise optuna.TrialPruned(str(exc)[:500]) from exc
        full_summary = json.loads(
            (trial_dir / "full" / "summary.json").read_text(encoding="utf-8")
        )
        participation = float(
            full_summary.get("diag_ssaw_gathered_training_rate", 0.0)
        )
        attrs = {
            **result,
            "ssaw_participation": participation,
            "elapsed_seconds": time.time() - started,
        }
        for name, value in attrs.items():
            if isinstance(value, (bool, int, float, str)):
                trial.set_user_attr(name, value)
        _publish_live_trial(
            output_dir,
            {
                "study": study_name,
                "trial": trial.number,
                "dataset": dataset,
                "scenario": _flow_label(flow),
                "stage": stage_name,
                "scope": scope,
                "parameter": parameter,
                "candidate": candidate,
                **attrs,
                "finished_at": utc_now(),
            },
        )
        # Optuna tracks Full F1.  Final stage selection additionally uses the
        # paired delta under the explicit F1 non-inferiority tolerance.
        return float(result["full_f1"])

    completed = _completed_candidates(study, parameter)
    remaining = max(0, len(values) - len(completed))
    if args.max_trials_per_invocation is not None:
        remaining = min(remaining, int(args.max_trials_per_invocation))
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    trials_frame = study.trials_dataframe()
    atomic_write_csv(
        trials_frame,
        flow_dir / f"{stage_index:02d}_{stage_name}.csv",
        index=False,
    )
    if len(_completed_candidates(study, parameter)) < len(values):
        return {"complete": False, "new_trials": remaining}
    winner = select_stage_winner(
        study.trials,
        parameter=parameter,
        f1_tolerance_pp=args.f1_tolerance_pp,
    )
    return {
        "complete": True,
        "stage_index": stage_index,
        "stage": stage_name,
        "scope": scope,
        "parameter": parameter,
        "previous_value": current,
        "selected_value": winner.params[parameter],
        "selected_trial": int(winner.number),
        "selected_full_f1": float(winner.user_attrs["full_f1"]),
        "selected_no_ssaw_f1": float(winner.user_attrs["no_ssaw_f1"]),
        "selected_full_minus_no_ssaw": float(
            winner.user_attrs["full_minus_no_ssaw"]
        ),
        "selected_source_model_sha256": str(
            winner.user_attrs["source_model_sha256"]
        ),
        "selected_source_checkpoint_path": str(
            winner.user_attrs["source_checkpoint_path"]
        ),
        "selected_source_metadata_context_sha256": str(
            winner.user_attrs["source_metadata_context_sha256"]
        ),
        "stage_max_full_f1": max(
            float(trial.user_attrs["full_f1"])
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and "full_f1" in trial.user_attrs
        ),
        "f1_tolerance_pp": float(args.f1_tolerance_pp),
        "completed_at": utc_now(),
    }


def _status(output_dir: Path, **updates) -> None:
    path = output_dir / "status.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    )
    payload.update(updates)
    payload["updated_at"] = utc_now()
    atomic_write_json(payload, path)


def _selected_profiles(output_dir: Path, plan) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for dataset, flow in plan:
        state_path = output_dir / "flows" / dataset / _flow_slug(flow) / "state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state.get("completed"):
            continue
        profiles[f"{dataset}:{_flow_label(flow)}"] = {
            "dataset": dataset,
            "flow": list(flow),
            "source_config": state["source_config"],
            "source_config_sha256": state["source_config_sha256"],
            "source_checkpoint_sha256": state["source_checkpoint_sha256"],
            "source_checkpoint_path": state["source_checkpoint_path"],
            "tta_config": state["tta_config"],
            "history": state["history"],
        }
    return profiles


def run_validation(args, output_dir: Path, plan) -> dict[str, object]:
    profiles = _selected_profiles(output_dir, plan)
    if len(profiles) != len(plan):
        raise RuntimeError("cannot validate before all flow profiles are selected")
    validation_source_seeds = tuple(
        int(seed)
        for seed in getattr(
            args, "validation_source_seeds", VALIDATION_SOURCE_SEEDS
        )
    )
    validation_subdir = str(
        getattr(args, "validation_subdir", "validation")
    ).strip()
    validation_root = output_dir / validation_subdir
    validation_root.mkdir(parents=True, exist_ok=True)
    raw_path = validation_root / "paired_raw.csv"
    failure_path = validation_root / "failure_attempts.csv"

    jobs = [
        (dataset, tuple(flow), int(source_seed))
        for dataset, flow in plan
        for source_seed in validation_source_seeds
    ]
    expected_keys = {
        (dataset, _flow_label(flow), source_seed, int(args.stream_seed))
        for dataset, flow, source_seed in jobs
    }

    rows_by_key: dict[tuple[str, str, int, int], dict] = {}
    if raw_path.is_file():
        prior = pd.read_csv(
            raw_path,
            dtype={
                "dataset": str,
                "scenario": str,
                "source_model_sha256": str,
                "source_checkpoint_path": str,
            },
        )
        required = {
            "dataset", "scenario", "source_seed", "stream_seed",
            "full_f1", "no_ssaw_f1", "source_model_sha256",
            "source_checkpoint_path",
        }
        missing = sorted(required - set(prior.columns))
        if missing:
            raise RuntimeError(f"validation resume is missing columns: {missing}")
        for record in prior.to_dict(orient="records"):
            key = (
                str(record["dataset"]),
                str(record["scenario"]),
                int(record["source_seed"]),
                int(record["stream_seed"]),
            )
            if key not in expected_keys:
                raise RuntimeError(f"unexpected validation resume key: {key}")
            if key in rows_by_key:
                raise RuntimeError(f"duplicate validation resume key: {key}")
            rows_by_key[key] = record

    failure_rows = (
        pd.read_csv(failure_path).to_dict(orient="records")
        if failure_path.is_file()
        else []
    )
    max_attempts = 1 + int(args.validation_retries)
    for attempt in range(1, max_attempts + 1):
        pending = [
            job
            for job in jobs
            if (
                job[0], _flow_label(job[1]), job[2], int(args.stream_seed)
            ) not in rows_by_key
        ]
        if not pending:
            break
        for dataset, flow, source_seed in pending:
            key = (
                dataset, _flow_label(flow), source_seed, int(args.stream_seed)
            )
            profile = profiles[f"{dataset}:{_flow_label(flow)}"]
            trial_dir = (
                validation_root
                / dataset
                / _flow_slug(flow)
                / f"source_seed_{source_seed}"
            )
            try:
                result = evaluate_pair(
                    args=args,
                    trial_dir=trial_dir,
                    dataset=dataset,
                    flow=flow,
                    source_seed=source_seed,
                    source_config=profile["source_config"],
                    tta_config=profile["tta_config"],
                    expected_source_model_sha256=(
                        profile["source_checkpoint_sha256"]
                        if source_seed == args.tuning_source_seed
                        else None
                    ),
                )
            except (WorkerFailure, RuntimeError) as exc:
                details = dict(getattr(exc, "details", {}) or {})
                failure_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": _flow_label(flow),
                        "source_seed": source_seed,
                        "stream_seed": int(args.stream_seed),
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:2000],
                        **details,
                        "failed_at": utc_now(),
                    }
                )
                atomic_write_csv(
                    pd.DataFrame(failure_rows), failure_path, index=False
                )
                continue
            rows_by_key[key] = result
            ordered = [rows_by_key[item] for item in sorted(rows_by_key)]
            atomic_write_csv(pd.DataFrame(ordered), raw_path, index=False)
        if attempt < max_attempts:
            time.sleep(2.0)

    frame = pd.DataFrame(
        [rows_by_key[key] for key in sorted(rows_by_key)]
    )
    expected_count = len(jobs)
    if len(frame) != expected_count:
        missing_keys = sorted(expected_keys - set(rows_by_key))
        payload = {
            "status": "incomplete_after_retries",
            "paired_units": int(len(frame)),
            "expected_paired_units": expected_count,
            "missing_keys": [list(key) for key in missing_keys],
            "validation_retries": int(args.validation_retries),
            "failure_attempts": int(len(failure_rows)),
            "updated_at": utc_now(),
        }
        atomic_write_json(payload, validation_root / "summary.json")
        _status(output_dir, status="validation_incomplete", validation=payload)
        raise RuntimeError(
            f"validation incomplete after retries: {len(frame)}/{expected_count}"
        )

    unit_counts = frame.groupby(["dataset", "scenario"])["source_seed"].nunique()
    if not unit_counts.eq(len(validation_source_seeds)).all():
        raise RuntimeError(
            "validation does not contain every requested source seed per flow"
        )
    flow_summary = (
        frame.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            source_seeds=("source_seed", "nunique"),
            full_f1_mean=("full_f1", "mean"),
            full_f1_std=("full_f1", "std"),
            no_ssaw_f1_mean=("no_ssaw_f1", "mean"),
            no_ssaw_f1_std=("no_ssaw_f1", "std"),
            full_minus_no_ssaw_mean=("full_minus_no_ssaw", "mean"),
            positive_pairs=("full_minus_no_ssaw", lambda x: int((x > 0).sum())),
            negative_pairs=("full_minus_no_ssaw", lambda x: int((x < 0).sum())),
        )
    )
    dataset_summary = (
        frame.groupby("dataset", as_index=False)
        .agg(
            paired_units=("full_minus_no_ssaw", "size"),
            full_f1_mean=("full_f1", "mean"),
            no_ssaw_f1_mean=("no_ssaw_f1", "mean"),
            full_minus_no_ssaw_mean=("full_minus_no_ssaw", "mean"),
            positive_pairs=("full_minus_no_ssaw", lambda x: int((x > 0).sum())),
            zero_pairs=("full_minus_no_ssaw", lambda x: int((x == 0).sum())),
            negative_pairs=("full_minus_no_ssaw", lambda x: int((x < 0).sum())),
        )
    )
    atomic_write_csv(flow_summary, validation_root / "flow_summary.csv", index=False)
    atomic_write_csv(
        dataset_summary, validation_root / "dataset_summary.csv", index=False
    )
    payload = {
        "status": "complete",
        "paired_units": int(len(frame)),
        "expected_paired_units": expected_count,
        "source_seeds": list(validation_source_seeds),
        "stream_seed": int(args.stream_seed),
        "validation_retries": int(args.validation_retries),
        "failure_attempts": int(len(failure_rows)),
        "recovered_after_failure": bool(failure_rows),
        "target_labels_used_for_parameter_selection": True,
        "confirmatory": False,
        "completed_at": utc_now(),
    }
    atomic_write_json(payload, validation_root / "summary.json")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gpu-lock-path", default=str(DEFAULT_GPU_LOCK))
    parser.add_argument("--tuning-source-seed", type=int, default=TUNING_SOURCE_SEED)
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    parser.add_argument("--f1-tolerance-pp", type=float, default=F1_TOLERANCE_PP)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-flows", type=int, default=None)
    parser.add_argument("--max-stages", type=int, default=None)
    parser.add_argument("--max-trials-per-invocation", type=int, default=None)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument(
        "--validation-source-seeds",
        default=",".join(map(str, VALIDATION_SOURCE_SEEDS)),
    )
    parser.add_argument("--validation-subdir", default="validation")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(
        item.strip().upper().replace("MFD", "FD")
        for item in args.datasets.split(",")
        if item.strip()
    )
    if not args.datasets or len(args.datasets) != len(set(args.datasets)):
        parser.error("--datasets must be a non-empty unique list")
    try:
        args.validation_source_seeds = tuple(
            int(item.strip())
            for item in args.validation_source_seeds.split(",")
            if item.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --validation-source-seeds: {exc}")
    if (
        not args.validation_source_seeds
        or len(args.validation_source_seeds)
        != len(set(args.validation_source_seeds))
        or min(args.validation_source_seeds) < 0
    ):
        parser.error(
            "--validation-source-seeds must be unique non-negative integers"
        )
    validation_subdir = str(args.validation_subdir).strip()
    if (
        not validation_subdir
        or Path(validation_subdir).is_absolute()
        or validation_subdir in {".", ".."}
        or len(Path(validation_subdir).parts) != 1
    ):
        parser.error("--validation-subdir must be one relative directory name")
    args.validation_subdir = validation_subdir
    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown datasets: {unknown}")
    if args.tuning_source_seed not in VALIDATION_SOURCE_SEEDS:
        parser.error("--tuning-source-seed must be 1, 2, or 3")
    if args.f1_tolerance_pp < 0:
        parser.error("--f1-tolerance-pp must be non-negative")
    if args.validation_retries < 0:
        parser.error("--validation-retries must be non-negative")
    for name in ("max_batches", "max_flows", "max_stages", "max_trials_per_invocation"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(output_dir)
    plan = flow_plan(args.datasets)
    if args.max_flows is not None:
        plan = plan[: int(args.max_flows)]
    manifest = {
        "protocol": PROTOCOL,
        "tuner_code_sha256": _tuner_code_sha256(),
        "production_code_sha256": production_code_sha256(),
        "datasets": list(args.datasets),
        "flows": [f"{dataset}:{_flow_label(flow)}" for dataset, flow in plan],
        "flow_specific_tta_profiles": True,
        "flow_specific_source_training_profiles": True,
        "dataset_shared_source_training_profile": False,
        "flow_specific_source_checkpoints": True,
        "source_checkpoint_independent_unit": "dataset/flow/source_seed",
        "full_no_ssaw_share_checkpoint_within_pair": True,
        "tuning_source_seed": int(args.tuning_source_seed),
        "validation_source_seeds": list(args.validation_source_seeds),
        "validation_subdir": str(args.validation_subdir),
        "stream_seed": int(args.stream_seed),
        "stage_order": [stage[0] for stage in STAGES],
        "coordinate_search": True,
        "selection_rule": (
            "within f1_tolerance_pp of stage-max Full F1, maximize "
            "Full-minus-NoSSAW, then Full F1"
        ),
        "f1_tolerance_pp": float(args.f1_tolerance_pp),
        "target_labels_used_for_parameter_selection": True,
        "confirmatory": False,
        "started_at": utc_now(),
    }
    manifest_path = output_dir / "manifest.json"
    if args.validation_only and args.validation_subdir != "validation":
        custom_validation_dir = output_dir / args.validation_subdir
        custom_validation_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = custom_validation_dir / "run_manifest.json"
    atomic_write_json(manifest, manifest_path)
    storage = storage_for(output_dir / "studies.sqlite3")
    completed_stage_count = 0
    try:
        if not args.validation_only:
            for flow_index, (dataset, flow) in enumerate(plan):
                flow_dir = output_dir / "flows" / dataset / _flow_slug(flow)
                flow_dir.mkdir(parents=True, exist_ok=True)
                source_config, tta_config = current_profiles(dataset)
                signature = _flow_signature(
                    args,
                    dataset,
                    flow,
                    source_config,
                    tta_config,
                )
                state_path = flow_dir / "state.json"
                state = _initialize_flow_state(
                    state_path,
                    signature=signature,
                    source_config=source_config,
                    tta_config=tta_config,
                )
                while state["next_stage_index"] < len(STAGES):
                    if (
                        args.max_stages is not None
                        and completed_stage_count >= args.max_stages
                    ):
                        _status(output_dir, status="partial_by_max_stages")
                        return 0
                    stage_index = int(state["next_stage_index"])
                    stage_name, scope, parameter = STAGES[stage_index]
                    _status(
                        output_dir,
                        status="running_tuning",
                        flow_index=flow_index,
                        flow_count=len(plan),
                        current_dataset=dataset,
                        current_scenario=_flow_label(flow),
                        current_stage=stage_name,
                        current_scope=scope,
                        current_parameter=parameter,
                        completed_flows=sum(
                            1
                            for d, f in plan
                            if (
                                output_dir / "flows" / d / _flow_slug(f) / "state.json"
                            ).is_file()
                            and json.loads(
                                (
                                    output_dir
                                    / "flows"
                                    / d
                                    / _flow_slug(f)
                                    / "state.json"
                                ).read_text(encoding="utf-8")
                            ).get("completed")
                        ),
                    )
                    result = run_stage(
                        args=args,
                        output_dir=output_dir,
                        storage=storage,
                        dataset=dataset,
                        flow=flow,
                        flow_dir=flow_dir,
                        state=state,
                        stage_index=stage_index,
                    )
                    if not result["complete"]:
                        _status(output_dir, status="partial_stage_saved")
                        return 0
                    config_key = (
                        "source_config" if scope == "source" else "tta_config"
                    )
                    state[config_key][parameter] = result["selected_value"]
                    state["source_config_sha256"] = _sha256_json(
                        state["source_config"]
                    )
                    state["source_checkpoint_sha256"] = result[
                        "selected_source_model_sha256"
                    ]
                    state["source_checkpoint_path"] = result[
                        "selected_source_checkpoint_path"
                    ]
                    state["source_metadata_context_sha256"] = result[
                        "selected_source_metadata_context_sha256"
                    ]
                    state["history"].append(result)
                    state["next_stage_index"] = stage_index + 1
                    state["updated_at"] = utc_now()
                    atomic_write_json(state, state_path)
                    completed_stage_count += 1
                state["completed"] = True
                state["updated_at"] = utc_now()
                atomic_write_json(state, state_path)
                profiles = _selected_profiles(output_dir, plan)
                atomic_write_json(profiles, output_dir / "selected_profiles.json")

        if not args.skip_validation:
            _status(output_dir, status="running_three_seed_validation")
            validation = run_validation(args, output_dir, plan)
        else:
            validation = None
        manifest["completed_at"] = utc_now()
        manifest["validation"] = validation
        atomic_write_json(manifest, manifest_path)
        _status(
            output_dir,
            status="complete",
            completed_flows=len(plan),
            flow_count=len(plan),
            validation=validation,
        )
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
