"""Source-only HHAR calibration for the physical SSAW orientation bound.

This runner is deliberately separate from the transfer benchmark.  A cell
trains (or loads) one source checkpoint for one HHAR user and evaluates the
frozen checkpoint on that user's ``source_train`` and ``source_test`` files.
No target user, target loader, target label, or target metric is read by this
module.

The candidate grid is registered before a run.  ``strength_deg`` is the
maximum total axis-angle radius of the SO(3) perturbation, in degrees; it is
not an Euler-axis standard deviation.  ``sigma=0`` disables the gain process,
so the only perturbation in this protocol is a bounded physical orientation.

The output is resumable and auditable:

* ``cells.csv`` contains one row per source-domain/seed/strength/split cell;
* ``cell_status.csv`` is atomically updated after every cell;
* ``selected_profile.json`` records the deterministic source-only rule; and
* ``manifest.json`` records the protocol and target-label exclusion.

After orientation selection, three sequential source-only coordinate stages
freeze the selected orientation and sweep auxiliary weight, learning rate, and
inner steps.  Each candidate is paired with an identical source-test
no-SSAW control on clean and deterministic 50% signal-freeze streams.

The implementation uses the production DuSafe class through the shared
trainer helpers.  It does not modify or fork ``algorithms/dusafe.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import msvcrt
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe import _extract_features  # noqa: E402
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from dataloader.corruption_transforms import CORRUPTION_REGISTRY  # noqa: E402
from dataloader.demo_dataloader import data_generator_demo  # noqa: E402
from optim.optimizer import build_optimizer  # noqa: E402
from scripts.hhar_protocol import HHAR_DOMAIN_IDS  # noqa: E402
from scripts.run_controlled_safety_benchmark import (  # noqa: E402
    deterministic_mask_fn,
)
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    BatchTransformLoader,
    build_trainer,
    cleanup_trainer,
)
from utils.utils import AverageMeter, fix_randomness, starting_logs  # noqa: E402


DATASET = "HHAR"
PROTOCOL = "HHAR source-only SSAW orientation calibration v1"

# These values are part of the protocol, not a runtime tuning range.  A CLI
# subset is accepted for a smoke/resume run, but every requested value must be
# one of these pre-registered physical candidates and 0 degrees must remain in
# the set as the no-orientation control.
PREREGISTERED_STRENGTHS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)
ORIENTATION_STRENGTHS = PREREGISTERED_STRENGTHS
GRID_STRENGTHS = PREREGISTERED_STRENGTHS
SOURCE_DOMAINS = tuple(str(value) for value in HHAR_DOMAIN_IDS)
SOURCE_SEEDS = (1, 2, 3)
ORIENTATION_SIGMA = 0.0
ORIENTATION_TEMPORAL_MODE = "window_constant"

DEFAULT_MAX_LABEL_FLIP = 0.01
DEFAULT_MAX_KL = 0.02
DEFAULT_MAX_SEMANTIC_DISTANCE = 0.03
DEFAULT_MAX_F1_DROP = 0.01
DEFAULT_STREAM_SEED = 42

# Stage 2 keeps the selected physical orientation fixed and calibrates only
# the DuSafe optimization coordinates on the same source-only protocol.  The
# lists are frozen before execution; ``no_ssaw`` is an implicit paired control
# for each full profile and is not a tunable candidate.  The range is broad
# enough to expose under/over-weighted auxiliary updates without importing any
# transfer-label evidence.
PREREGISTERED_AUXILIARY_WEIGHTS = (1.0, 4.0, 8.0, 12.0, 16.0)
PREREGISTERED_LEARNING_RATES = (3e-5, 5e-5, 1e-4, 3e-4, 1e-3)
PREREGISTERED_STEPS = (1, 2, 4, 8)
SECOND_STAGE_AUXILIARY_WEIGHTS = PREREGISTERED_AUXILIARY_WEIGHTS
SECOND_STAGE_LEARNING_RATES = PREREGISTERED_LEARNING_RATES
SECOND_STAGE_STEPS = PREREGISTERED_STEPS
SECOND_STAGE_CONDITIONS = ("clean", "signal_freeze_moderate")
SECOND_STAGE_CORRUPTION = "signal_freeze"
SECOND_STAGE_CORRUPTION_SEVERITY = "moderate"
SECOND_STAGE_CORRUPTION_FRACTION = 0.5
SECOND_STAGE_CORRUPTION_SEED = 1
SECOND_STAGE_COORDINATE_STAGES = (
    "auxiliary_weight",
    "learning_rate",
    "steps",
)
DEFAULT_MIN_CLEAN_F1_DELTA = -0.002
DEFAULT_MIN_CORRUPTION_F1_DELTA = -0.010
DEFAULT_MAX_CLEAN_F1_DROP = -DEFAULT_MIN_CLEAN_F1_DELTA
DEFAULT_MAX_CORRUPTION_F1_DROP = -DEFAULT_MIN_CORRUPTION_F1_DELTA
DEFAULT_MAX_NEXT_CE_DELTA = 0.010
DEFAULT_MAX_UNSAFE_UPDATE_DELTA = 0.005

STATUS_COLUMNS = (
    "source_domain",
    "source_seed",
    "strength_deg",
    "status",
    "rows_written",
    "started_at",
    "completed_at",
    "error",
)
CELL_KEY_COLUMNS = ("source_domain", "source_seed", "strength_deg")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(payload: Mapping, path: str | Path) -> None:
    """Publish a complete JSON manifest/profile in one rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        for attempt in range(20):
            try:
                temporary.replace(destination)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_lock(path: Path):
    """Acquire the repository-wide experiment lock on Windows.

    CPU smoke runs do not acquire this lock.  The fallback is useful when the
    module is imported on a POSIX host for protocol tests; production Windows
    runs use the same lock file as the other experiment runners.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"GPU lock is busy: {path}") from exc
    return handle


def _release_gpu_lock(handle) -> None:
    if handle is None:
        return
    lock_path = Path(handle.name)
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()
        # The other long-running experiment runners use an O_EXCL lock at the
        # same path.  Leaving the advisory-lock file behind after releasing the
        # byte-range lock makes those runners treat an idle GPU as permanently
        # busy (the empty file has no reclaimable owner PID).
        lock_path.unlink(missing_ok=True)


def _float_key(value: float) -> float:
    return round(float(value), 8)


def validate_candidate_strengths(
    values: Iterable[float] | str,
) -> list[float]:
    """Validate a candidate subset against the pre-registered angle grid."""

    if isinstance(values, str):
        values = [
            float(value.strip())
            for value in values.split(",")
            if value.strip()
        ]
    parsed = [_float_key(value) for value in values]
    if not parsed:
        raise ValueError("orientation strength grid must not be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError("orientation strength grid must not contain duplicates")
    allowed = {_float_key(value) for value in PREREGISTERED_STRENGTHS}
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(
            "orientation strengths are not pre-registered physical candidates: "
            f"{unknown}; allowed={list(PREREGISTERED_STRENGTHS)}"
        )
    if 0.0 not in parsed:
        raise ValueError(
            "orientation strength grid must include the 0-degree source control"
        )
    return sorted(parsed)


def validate_source_seeds(values: Iterable[int] | str) -> list[int]:
    if isinstance(values, str):
        values = [int(value.strip()) for value in values.split(",") if value.strip()]
    parsed = [int(value) for value in values]
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("source seeds must be a non-empty set of unique integers")
    if any(value < 0 for value in parsed):
        raise ValueError("source seeds must be non-negative")
    return parsed


def profile_rows(strengths: Iterable[float]) -> list[dict]:
    """Return the pre-registered profiles represented in an output manifest."""

    return [
        {
            "profile": f"orientation_s{float(strength):g}_sigma0",
            "variant": "orientation",
            "strength_deg": float(strength),
            "sigma": ORIENTATION_SIGMA,
            "orientation_definition": (
                "bounded SO(3) axis-angle radius; maximum total angle in degrees"
            ),
        }
        for strength in strengths
    ]


_profile_rows = profile_rows


def adaptation_profile_rows(
    auxiliary_weights: Iterable[float] = PREREGISTERED_AUXILIARY_WEIGHTS,
    learning_rates: Iterable[float] = PREREGISTERED_LEARNING_RATES,
    steps: Iterable[int] = PREREGISTERED_STEPS,
) -> list[dict]:
    """Build a sequential, pre-registered stage-2 profile list.

    The three coordinates are swept one at a time around a frozen anchor,
    rather than taking a Cartesian product.  This keeps the source-only
    paired panel finite while still registering every requested coordinate.
    """

    auxiliary_weights = tuple(float(value) for value in auxiliary_weights)
    learning_rates = tuple(float(value) for value in learning_rates)
    steps = tuple(int(value) for value in steps)
    if not auxiliary_weights or not learning_rates or not steps:
        raise ValueError("stage-2 coordinate grids must not be empty")
    anchor_aux = 8.0 if 8.0 in auxiliary_weights else auxiliary_weights[0]
    anchor_lr = 1e-4 if 1e-4 in learning_rates else learning_rates[0]
    anchor_steps = 1 if 1 in steps else steps[0]
    coordinates = []
    coordinates.extend(
        ("auxiliary_weight", value, anchor_lr, anchor_steps)
        for value in auxiliary_weights
    )
    coordinates.extend(
        ("learning_rate", anchor_aux, value, anchor_steps)
        for value in learning_rates
    )
    coordinates.extend(
        ("steps", anchor_aux, anchor_lr, value)
        for value in steps
    )
    rows = []
    seen = set()
    for coordinate, auxiliary_weight, learning_rate, step_count in coordinates:
        auxiliary_weight = float(auxiliary_weight)
        learning_rate = float(learning_rate)
        step_count = int(step_count)
        if auxiliary_weight <= 0.0:
            raise ValueError("stage-2 auxiliary weights must be positive")
        if learning_rate <= 0.0:
            raise ValueError("stage-2 learning rates must be positive")
        if step_count < 1:
            raise ValueError("stage-2 steps must be positive")
        key = (auxiliary_weight, learning_rate, step_count)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "profile": (
                    f"full_a{auxiliary_weight:g}_lr{learning_rate:g}_"
                    f"steps{step_count}"
                ),
                "variant": "full",
                "coordinate": coordinate,
                "auxiliary_weight": auxiliary_weight,
                "learning_rate": learning_rate,
                "steps": step_count,
            }
        )
    return rows


_adaptation_profile_rows = adaptation_profile_rows


def coordinate_profile_rows(
    stage: str,
    *,
    auxiliary_weight: float = 8.0,
    learning_rate: float = 1e-4,
    steps: int = 1,
) -> list[dict]:
    """Return one coordinate-descent stage with its other coordinates frozen."""

    stage = str(stage)
    if stage not in SECOND_STAGE_COORDINATE_STAGES:
        raise ValueError(
            f"unknown stage {stage!r}; expected {SECOND_STAGE_COORDINATE_STAGES}"
        )
    if stage == "auxiliary_weight":
        values = [
            (
                float(value),
                float(learning_rate),
                int(steps),
            )
            for value in PREREGISTERED_AUXILIARY_WEIGHTS
        ]
    elif stage == "learning_rate":
        values = [
            (
                float(auxiliary_weight),
                float(value),
                int(steps),
            )
            for value in PREREGISTERED_LEARNING_RATES
        ]
    else:
        values = [
            (
                float(auxiliary_weight),
                float(learning_rate),
                int(value),
            )
            for value in PREREGISTERED_STEPS
        ]
    rows = []
    for aux, lr, step_count in values:
        rows.append(
            {
                "profile": (
                    f"coord_{stage}_a{aux:g}_lr{lr:g}_steps{step_count}"
                ),
                "variant": "full",
                "coordinate_stage": stage,
                "coordinate": stage,
                "auxiliary_weight": aux,
                "learning_rate": lr,
                "steps": step_count,
            }
        )
    return rows


def _key_tuple(row: Mapping) -> tuple[str, int, float]:
    return (
        str(row.get("source_domain")),
        int(row.get("source_seed")),
        _float_key(row.get("strength_deg")),
    )


def _completed_keys(existing: pd.DataFrame) -> set[tuple[str, int, float]]:
    """Return only cells with a durable ``completed`` status and cell rows."""

    if existing is None or existing.empty:
        return set()
    required = set(CELL_KEY_COLUMNS) | {"status"}
    if not required.issubset(existing.columns):
        return set()
    completed = existing[existing["status"].astype(str).eq("completed")]
    keys = set()
    for _, row in completed.iterrows():
        try:
            keys.add(_key_tuple(row))
        except (TypeError, ValueError):
            continue
    return keys


def _status_key_set(frame: pd.DataFrame) -> set[tuple[str, int, float]]:
    if frame is None or frame.empty:
        return set()
    if not set(CELL_KEY_COLUMNS).issubset(frame.columns):
        return set()
    grouped: dict[tuple[str, int, float], set[str]] = {}
    for _, row in frame.iterrows():
        try:
            key = _key_tuple(row)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, set()).add(str(row.get("split", "")))
    # A calibration cell is durable only after both source-train and
    # source-test rows are atomically present.  This repairs a crash between
    # the row and status publications instead of silently skipping half a
    # cell on resume.
    return {
        key
        for key, splits in grouped.items()
        if {"source_train", "source_test"}.issubset(splits)
    }


def _expected_keys(
    strengths: Iterable[float] = PREREGISTERED_STRENGTHS,
    source_domains: Iterable[str] = SOURCE_DOMAINS,
    source_seeds: Iterable[int] = SOURCE_SEEDS,
) -> list[tuple[str, int, float]]:
    return [
        (str(domain), int(seed), _float_key(strength))
        for seed in source_seeds
        for domain in source_domains
        for strength in strengths
    ]


def _status_frame(
    expected_keys: Iterable[tuple[str, int, float]],
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize status state and discard rows outside the frozen protocol."""

    old = {}
    if existing is not None and not existing.empty:
        for _, row in existing.iterrows():
            try:
                old[_key_tuple(row)] = {
                    column: row.get(column, "") for column in STATUS_COLUMNS
                }
            except (TypeError, ValueError):
                continue
    rows = []
    for key in expected_keys:
        old_row = old.get(key, {})
        status = str(old_row.get("status", "pending"))
        if status not in {"pending", "running", "completed", "oom", "failed"}:
            status = "pending"
        rows.append(
            {
                "source_domain": key[0],
                "source_seed": key[1],
                "strength_deg": key[2],
                "status": status,
                "rows_written": int(old_row.get("rows_written", 0) or 0),
                "started_at": str(old_row.get("started_at", "") or ""),
                "completed_at": str(old_row.get("completed_at", "") or ""),
                "error": str(old_row.get("error", "") or ""),
            }
        )
    return pd.DataFrame(rows, columns=list(STATUS_COLUMNS))


def _mark_status(
    statuses: pd.DataFrame,
    key: tuple[str, int, float],
    status: str,
    *,
    rows_written: int | None = None,
    error: str = "",
) -> pd.DataFrame:
    statuses = statuses.copy()
    mask = (
        statuses["source_domain"].astype(str).eq(key[0])
        & statuses["source_seed"].astype(int).eq(key[1])
        & statuses["strength_deg"].astype(float).round(8).eq(key[2])
    )
    if not mask.any():
        raise KeyError(f"status key is outside protocol: {key}")
    statuses.loc[mask, "status"] = str(status)
    if rows_written is not None:
        statuses.loc[mask, "rows_written"] = int(rows_written)
    statuses.loc[mask, "error"] = str(error)
    if status == "running":
        statuses.loc[mask, "started_at"] = _utc_now()
        statuses.loc[mask, "completed_at"] = ""
    elif status in {"completed", "oom", "failed"}:
        statuses.loc[mask, "completed_at"] = _utc_now()
    return statuses


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _features(module, inputs: torch.Tensor) -> torch.Tensor:
    values = module(inputs)
    if isinstance(values, (tuple, list)):
        values = values[0]
    return values


def _load_source_only_data(trainer, source_domain: str, eval_batch_size: int):
    """Load only ``train_<source>`` and ``test_<source>``.

    ``TTATrainer.load_data_demo`` necessarily constructs a target loader for
    transfer experiments, so this calibration uses the lower-level loader and
    leaves ``trainer.trg_whole_dl`` unset.  This is the source-label firewall:
    the only labels visible below are source-train/source-test labels.
    """

    trainer.src_train_dl = data_generator_demo(
        trainer.data_path,
        source_domain,
        trainer.dataset_configs,
        trainer.source_hparams,
        "train",
        seed_id=trainer._current_source_seed,
    )
    source_stats = trainer.src_train_dl.dataset.normalization_stats
    trainer.src_test_dl = data_generator_demo(
        trainer.data_path,
        source_domain,
        trainer.dataset_configs,
        trainer.source_hparams,
        "test",
        seed_id=trainer._current_source_seed,
        normalization_stats=source_stats,
    )
    trainer.trg_whole_dl = None
    # Evaluation order and batch context are fixed across candidates.  The
    # source training loader above remains shuffled for source pre-training.
    train_eval = DataLoader(
        trainer.src_train_dl.dataset,
        batch_size=int(eval_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_eval = DataLoader(
        trainer.src_test_dl.dataset,
        batch_size=int(eval_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_eval, test_eval, source_stats


def _create_source_only_model(
    trainer,
    source_domain: str,
    *,
    stream_seed: int,
    eval_batch_size: int,
):
    """Prepare the shared source checkpoint and construct DuSafe once."""

    trainer._current_source_seed = int(trainer.source_seed)
    trainer._current_run_seed = int(stream_seed)
    trainer.set_test_time_seed(int(stream_seed))
    trainer._current_scenario = (str(source_domain), "source_test")
    trainer.dataset_configs._active_scenario = trainer._current_scenario
    # ``set_scenario_hparams`` validates registered transfer pairs and would
    # force a target loader.  The dataset-level composition is identical here
    # and is safe to apply directly because this protocol has no transfer pair.
    trainer.hparams = {
        **trainer._base_alg_hparams,
        **trainer._train_params,
        **trainer._runtime_hparam_overrides,
    }
    trainer._apply_backbone_overrides(trainer.hparams)
    train_eval, test_eval, source_stats = _load_source_only_data(
        trainer, source_domain, eval_batch_size
    )
    trainer.logger, trainer.scenario_log_dir = starting_logs(
        trainer.dataset,
        trainer.da_method,
        trainer.exp_log_dir,
        str(source_domain),
        "source_test",
        0,
    )
    trainer.pre_loss_avg_meters = {}
    trainer.loss_avg_meters = {}
    # AverageMeter dictionaries are accessed by the pre-training loop.
    trainer.pre_loss_avg_meters = __import__(
        "collections"
    ).defaultdict(lambda: AverageMeter())
    trainer.loss_avg_meters = __import__("collections").defaultdict(
        lambda: AverageMeter()
    )
    fix_randomness(int(trainer._current_source_seed))
    _non_adapted, source_model = trainer.pre_train()
    # Source training is complete.  Re-seed the fixed stream before creating
    # SSAW so a cache hit and cache miss have the same candidate trajectories.
    fix_randomness(int(stream_seed))
    optimizer = build_optimizer(trainer.hparams)
    adapter_class = trainer.get_tta_model_class()
    adapter = adapter_class(
        trainer.dataset_configs,
        trainer.hparams,
        source_model,
        optimizer,
    ).to(trainer.device)
    if not hasattr(adapter, "ssaw"):
        raise RuntimeError("HHAR orientation calibration requires DuSafe SSAW")
    if float(adapter.ssaw.sigma) != ORIENTATION_SIGMA:
        raise RuntimeError("HHAR source calibration requires sigma=0")
    if hasattr(adapter, "load_source_normalization_reference"):
        adapter.load_source_normalization_reference(*source_stats)
    if getattr(adapter, "enable_confidence_gate", False):
        if trainer.source_confidence_metadata is None:
            raise RuntimeError("source confidence metadata is unavailable")
        adapter.load_source_confidence_reference(trainer.source_confidence_metadata)
    if getattr(adapter, "enable_source_semantic_gate", False):
        if trainer.source_semantic_metadata is None:
            raise RuntimeError("source semantic metadata is unavailable")
        adapter.load_source_semantic_reference(trainer.source_semantic_metadata)
    return adapter, source_model, train_eval, test_eval


def _evaluate_split(
    adapter,
    loader,
    *,
    split: str,
) -> dict:
    """Evaluate one source split without adapting the model."""

    labels_all: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    view_predictions: list[list[torch.Tensor]] | None = None
    flip_sum = 0.0
    kl_sum = 0.0
    semantic_sum = 0.0
    rms_sum = 0.0
    sample_count = 0
    view_count = None
    source_extractor = adapter.source_semantic_feature_extractor

    with torch.inference_mode():
        for data, labels, _indices in loader:
            if labels is None:
                raise ValueError(f"{split} source labels are required for F1")
            inputs_cpu = data[0] if isinstance(data, (list, tuple)) else data
            model_device = next(adapter.model.parameters()).device
            inputs = inputs_cpu.float().to(model_device)
            labels_cpu = labels.detach().view(-1).cpu().long()
            adapter.ssaw.clear_cached_view()
            adapter.ssaw(
                inputs,
                adapter.model,
                normalization_mean=adapter.source_normalization_mean,
                normalization_std=adapter.source_normalization_std,
            )
            views = adapter.ssaw.last_view_inputs
            metadata = adapter.ssaw.last_metadata
            raw_logits = adapter.ssaw.last_reference_logits
            raw_prediction = raw_logits.argmax(dim=1)
            if views.dim() == 3:
                views = views.unsqueeze(0)
            if view_count is None:
                view_count = int(views.size(0))
                view_predictions = [[] for _ in range(view_count)]
            if int(views.size(0)) != int(view_count):
                raise RuntimeError("SSAW view count changed within one source split")

            logits_by_view = []
            for view in views:
                with adapter.ssaw._preserved_bn_buffers(adapter.model):
                    logits_by_view.append(
                        adapter.model.classifier(
                            _extract_features(adapter.model, view)
                        )
                    )
            view_logits = torch.stack(logits_by_view)
            view_pred = view_logits.argmax(dim=2)
            for view_index in range(view_count):
                view_predictions[view_index].append(
                    view_pred[view_index].detach().cpu()
                )
            labels_all.append(labels_cpu)
            raw_predictions.append(raw_prediction.detach().cpu())

            source_features = F.normalize(
                _features(source_extractor, inputs).flatten(1), dim=1
            )
            flattened_views = views.reshape(-1, *views.shape[2:])
            view_features = F.normalize(
                _features(source_extractor, flattened_views).flatten(1), dim=1
            ).reshape(view_count, inputs.shape[0], -1)
            semantic_distance = 1.0 - (
                view_features * source_features.unsqueeze(0)
            ).sum(dim=-1)
            residual = views - inputs.unsqueeze(0)
            input_rms = inputs.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-8)
            relative_rms = residual.square().mean(dim=(-2, -1)).sqrt() / input_rms
            kl_by_view = torch.as_tensor(
                metadata["selected_kl_by_view"],
                device=inputs.device,
                dtype=inputs.dtype,
            )
            flip_by_view = view_pred.ne(raw_prediction.unsqueeze(0)).float()
            flip_sum += float(flip_by_view.sum().item())
            kl_sum += float(kl_by_view.sum().item())
            semantic_sum += float(semantic_distance.sum().item())
            rms_sum += float(relative_rms.sum().item())
            sample_count += int(inputs.shape[0])

    if not labels_all or view_predictions is None or view_count is None:
        raise RuntimeError(f"No samples in {split} source loader")
    labels = torch.cat(labels_all).numpy()
    raw_pred = torch.cat(raw_predictions).numpy()
    view_f1 = [
        float(
            f1_score(
                labels,
                torch.cat(predictions).numpy(),
                average="macro",
                zero_division=0,
            )
        )
        for predictions in view_predictions
    ]
    raw_f1 = float(
        f1_score(labels, raw_pred, average="macro", zero_division=0)
    )
    raw_accuracy = float((labels == raw_pred).mean())
    view_accuracy = [
        float((labels == torch.cat(predictions).numpy()).mean())
        for predictions in view_predictions
    ]
    denominator = float(max(sample_count * view_count, 1))
    return {
        "split": str(split),
        "samples": int(sample_count),
        "view_count": int(view_count),
        "label_flip_rate": float(flip_sum / denominator),
        "kl_mean": float(kl_sum / denominator),
        "semantic_distance_mean": float(semantic_sum / denominator),
        "relative_rms_mean": float(rms_sum / denominator),
        "raw_source_f1": raw_f1,
        "view_source_f1_mean": float(sum(view_f1) / len(view_f1)),
        "view_source_f1_min": float(min(view_f1)),
        "raw_source_accuracy": raw_accuracy,
        "view_source_accuracy_mean": float(sum(view_accuracy) / len(view_accuracy)),
    }


def _run_cell(
    *,
    args: argparse.Namespace,
    source_domain: str,
    source_seed: int,
    strength: float,
) -> list[dict]:
    """Run one source domain/seed/strength cell; no target loader is built."""

    hparams = get_hparams_class(DATASET)()
    source_config = {
        **hparams.alg_hparams["NoAdap"],
        **hparams.source_train_params,
    }
    tta_config = {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
        # These are protocol invariants, even if a future config drifts.
        "ssaw_sigma": ORIENTATION_SIGMA,
        "ssaw_strength": float(strength),
        "ssaw_temporal_mode": ORIENTATION_TEMPORAL_MODE,
    }
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=DATASET,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="hhar_orientation_source_calibration",
        seed=int(args.stream_seed),
        source_seed=int(source_seed),
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model, train_eval, test_eval = _create_source_only_model(
            trainer,
            str(source_domain),
            stream_seed=int(args.stream_seed),
            eval_batch_size=int(args.eval_batch_size),
        )
        # The constructor already receives this strength; assign it again to
        # make the cell identity explicit and guard against config drift.
        adapter.ssaw.strength = float(strength)
        rows = []
        for split, loader in (("source_train", train_eval), ("source_test", test_eval)):
            metrics = _evaluate_split(adapter, loader, split=split)
            rows.append(
                {
                    "dataset": DATASET,
                    "source_domain": str(source_domain),
                    "source_seed": int(source_seed),
                    "strength_deg": float(strength),
                    "profile": f"orientation_s{float(strength):g}_sigma0",
                    "sigma": ORIENTATION_SIGMA,
                    "view_role": "positive_and_inverse_mean",
                    "target_labels_used": False,
                    **metrics,
                }
            )
        return rows
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)
        del trainer, adapter, source_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _adaptation_variant(
    *,
    args: argparse.Namespace,
    source_domain: str,
    source_seed: int,
    orientation_strength: float,
    profile: Mapping,
    condition: str,
    variant: str,
) -> dict:
    """Run one frozen-source checkpoint on one source-test condition.

    The full and no-SSAW variants are called separately by
    ``_run_profile_cell``.  They therefore share the same source checkpoint
    cache and source-test order but never share an adapted model state.
    """

    hparams = get_hparams_class(DATASET)()
    source_config = {
        **hparams.alg_hparams["NoAdap"],
        **hparams.source_train_params,
    }
    full = str(variant) == "full"
    tta_config = {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
        "ssaw_sigma": ORIENTATION_SIGMA,
        "ssaw_strength": float(orientation_strength),
        "ssaw_temporal_mode": ORIENTATION_TEMPORAL_MODE,
        "ssaw_auxiliary_weight": (
            float(profile["auxiliary_weight"]) if full else 0.0
        ),
        "learning_rate": float(profile["learning_rate"]),
        "steps": int(profile["steps"]),
        "enable_ssaw": bool(full),
    }
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=DATASET,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="hhar_orientation_source_profile_calibration",
        seed=int(args.stream_seed),
        source_seed=int(source_seed),
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model, _train_eval, test_eval = _create_source_only_model(
            trainer,
            str(source_domain),
            stream_seed=int(args.stream_seed),
            eval_batch_size=int(args.eval_batch_size),
        )
        if condition == "clean":
            stream = test_eval
        elif condition == "signal_freeze_moderate":
            stream = BatchTransformLoader(
                test_eval,
                CORRUPTION_REGISTRY[SECOND_STAGE_CORRUPTION],
                SECOND_STAGE_CORRUPTION_SEVERITY,
                sample_mask_fn=deterministic_mask_fn(
                    SECOND_STAGE_CORRUPTION_FRACTION,
                    SECOND_STAGE_CORRUPTION_SEED,
                ),
                meta={
                    "corruption_type": SECOND_STAGE_CORRUPTION,
                    "severity": SECOND_STAGE_CORRUPTION_SEVERITY,
                },
                transform_seed=SECOND_STAGE_CORRUPTION_SEED + 20_000,
            )
        else:
            raise ValueError(f"unknown source-only profile condition: {condition}")
        # ``calculate_metrics`` consumes only this source-test stream.  It is
        # named ``trg_whole_dl`` inside the legacy trainer API, but no target
        # domain is loaded or referenced in this protocol.
        trainer.trg_whole_dl = stream
        metrics = trainer.calculate_metrics(adapter)
        safety = dict(getattr(trainer, "last_safety_summary", {}) or {})
        diagnostics = dict(getattr(trainer, "last_batch_log_summary", {}) or {})
        return {
            "variant": str(variant),
            "condition": str(condition),
            "source_f1": float(metrics[1]),
            "source_accuracy": float(metrics[0]),
            "post_update_ce": float(trainer.loss.item()),
            "next_ce": float(trainer.pre_final_update_loss.item()),
            "unsafe_update_rate": float(
                safety.get("unsafe_update_rate", float("nan"))
            ),
            "coverage": float(safety.get("coverage", float("nan"))),
            "accepted_pseudo_label_accuracy": float(
                safety.get("accepted_pseudo_label_accuracy", float("nan"))
            ),
            "raw_ce_loss": float(diagnostics.get("raw_ce_loss", float("nan"))),
            "ssaw_weighted_consistency_loss": float(
                diagnostics.get(
                    "ssaw_weighted_consistency_loss", float("nan")
                )
            ),
            "target_labels_used": False,
        }
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)
        del trainer, adapter, source_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_profile_cell(
    *,
    args: argparse.Namespace,
    source_domain: str,
    source_seed: int,
    orientation_strength: float,
    profile: Mapping,
    condition: str,
) -> dict:
    """Run paired Full/no-SSAW source-only evidence for one profile cell."""

    full = _adaptation_variant(
        args=args,
        source_domain=source_domain,
        source_seed=source_seed,
        orientation_strength=orientation_strength,
        profile=profile,
        condition=condition,
        variant="full",
    )
    no_ssaw = _adaptation_variant(
        args=args,
        source_domain=source_domain,
        source_seed=source_seed,
        orientation_strength=orientation_strength,
        profile=profile,
        condition=condition,
        variant="no_ssaw",
    )
    return {
        "dataset": DATASET,
        "source_domain": str(source_domain),
        "source_seed": int(source_seed),
        "orientation_strength_deg": float(orientation_strength),
        "profile": str(profile["profile"]),
        "coordinate": str(profile.get("coordinate", "sequential")),
        "auxiliary_weight": float(profile["auxiliary_weight"]),
        "learning_rate": float(profile["learning_rate"]),
        "steps": int(profile["steps"]),
        "condition": str(condition),
        "full_source_f1": float(full["source_f1"]),
        "no_ssaw_source_f1": float(no_ssaw["source_f1"]),
        "full_no_ssaw_f1_delta": float(
            full["source_f1"] - no_ssaw["source_f1"]
        ),
        "full_next_ce": float(full["next_ce"]),
        "no_ssaw_next_ce": float(no_ssaw["next_ce"]),
        "full_no_ssaw_next_ce_delta": float(
            full["next_ce"] - no_ssaw["next_ce"]
        ),
        "full_unsafe_update_rate": float(full["unsafe_update_rate"]),
        "no_ssaw_unsafe_update_rate": float(no_ssaw["unsafe_update_rate"]),
        "full_no_ssaw_unsafe_update_rate_delta": float(
            full["unsafe_update_rate"] - no_ssaw["unsafe_update_rate"]
        ),
        "full_coverage": float(full["coverage"]),
        "no_ssaw_coverage": float(no_ssaw["coverage"]),
        "full_raw_ce_loss": float(full["raw_ce_loss"]),
        "full_ssaw_weighted_consistency_loss": float(
            full["ssaw_weighted_consistency_loss"]
        ),
        "target_labels_used": False,
        "target_data_used": False,
    }


def _select_adaptation_profile(
    frame: pd.DataFrame,
    *,
    max_clean_f1_drop: float = DEFAULT_MAX_CLEAN_F1_DROP,
    max_corruption_f1_drop: float = DEFAULT_MAX_CORRUPTION_F1_DROP,
    max_next_ce_delta: float = DEFAULT_MAX_NEXT_CE_DELTA,
    max_unsafe_update_delta: float = DEFAULT_MAX_UNSAFE_UPDATE_DELTA,
) -> dict:
    """Select a stage-2 profile using paired source-only constraints.

    Source users are averaged within each independent source seed before the
    objective is formed.  This prevents one noisy user cell from becoming an
    implicit veto; worst-cell values are retained for diagnostics and audit.
    """

    if frame is None or frame.empty:
        raise ValueError("cannot select stage-2 profile from an empty frame")
    required = {
        "profile",
        "condition",
        "source_seed",
        "full_no_ssaw_f1_delta",
        "full_no_ssaw_next_ce_delta",
        "full_no_ssaw_unsafe_update_rate_delta",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stage-2 profile frame is missing columns: {missing}")
    # Collapse the nine source users within each independent source seed first.
    # Seeds, not individual users, are the replicated unit for ranking.  The
    # per-cell extrema remain in the summary as diagnostics only.
    work = frame.copy()
    work["_seed"] = work["source_seed"].astype(int)
    work["_condition"] = work["condition"].astype(str)
    seed_condition = (
        work.groupby(["profile", "_seed", "_condition"], as_index=False)
        .agg(
            seed_f1_delta=("full_no_ssaw_f1_delta", "mean"),
            seed_next_ce_delta=("full_no_ssaw_next_ce_delta", "mean"),
            seed_unsafe_delta=(
                "full_no_ssaw_unsafe_update_rate_delta",
                "mean",
            ),
            seed_full_f1=("full_source_f1", "mean"),
        )
    )
    rows = []
    for profile, group in frame.groupby("profile", sort=True):
        clean = group[group["condition"].astype(str).eq("clean")]
        corrupt = group[
            group["condition"].astype(str).eq("signal_freeze_moderate")
        ]
        if clean.empty or corrupt.empty:
            continue
        seed_group = seed_condition[
            seed_condition["profile"].astype(str).eq(str(profile))
        ]
        seed_clean = seed_group[seed_group["_condition"].eq("clean")]
        seed_corrupt = seed_group[
            seed_group["_condition"].eq("signal_freeze_moderate")
        ]
        clean_delta_mean = float(seed_clean["seed_f1_delta"].mean())
        corrupt_delta_mean = float(seed_corrupt["seed_f1_delta"].mean())
        next_ce_mean = float(seed_group["seed_next_ce_delta"].mean())
        unsafe_delta_mean = float(seed_group["seed_unsafe_delta"].mean())
        # Worst user cell is retained as an audit diagnostic, not used as the
        # primary rank or eligibility criterion.
        clean_delta_min = float(clean["full_no_ssaw_f1_delta"].min())
        corrupt_delta_min = float(corrupt["full_no_ssaw_f1_delta"].min())
        next_ce_max = float(group["full_no_ssaw_next_ce_delta"].max())
        unsafe_delta_max = float(
            group["full_no_ssaw_unsafe_update_rate_delta"].max()
        )
        first = group.iloc[0]
        rows.append(
            {
                "profile": str(profile),
                "coordinate": str(first.get("coordinate", "sequential")),
                "auxiliary_weight": float(first["auxiliary_weight"]),
                "learning_rate": float(first["learning_rate"]),
                "steps": int(first["steps"]),
                "clean_f1_delta_mean": clean_delta_mean,
                "corruption_f1_delta_mean": corrupt_delta_mean,
                "next_ce_delta_mean": next_ce_mean,
                "unsafe_update_delta_mean": unsafe_delta_mean,
                "mean_paired_f1_delta": float(
                    (clean_delta_mean + corrupt_delta_mean) / 2.0
                ),
                "clean_f1_delta_min": clean_delta_min,
                "corruption_f1_delta_min": corrupt_delta_min,
                "next_ce_delta_max": next_ce_max,
                "unsafe_update_delta_max": unsafe_delta_max,
                "full_clean_f1_mean": float(seed_clean["seed_full_f1"].mean()),
                "full_clean_f1_min": float(clean["full_source_f1"].min()),
                "full_corruption_f1_min": float(
                    corrupt["full_source_f1"].min()
                ),
                "source_cells": int(len(group)),
                "source_seed_units": int(seed_group["_seed"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("stage-2 profile frame has no complete clean/corruption pairs")
    summary["meets_constraints"] = (
        summary["clean_f1_delta_mean"].ge(-float(max_clean_f1_drop))
        & summary["corruption_f1_delta_mean"].ge(-float(max_corruption_f1_drop))
        & summary["next_ce_delta_mean"].le(float(max_next_ce_delta))
        & summary["unsafe_update_delta_mean"].le(float(max_unsafe_update_delta))
    )
    eligible = summary[summary["meets_constraints"]]
    if not eligible.empty:
        selected = eligible.sort_values(
            [
                "mean_paired_f1_delta",
                "clean_f1_delta_mean",
                "corruption_f1_delta_mean",
                "next_ce_delta_mean",
                "unsafe_update_delta_mean",
                "steps",
                "learning_rate",
                "auxiliary_weight",
            ],
            ascending=[False, False, False, True, True, True, True, True],
        ).iloc[0]
        rule = (
            "paired_source_clean_and_controlled_corruption_f1_floor_then_"
            "min_next_ce_and_safety_tie_break"
        )
        fallback_used = False
    else:
        scales = [
            max(float(max_clean_f1_drop), 1e-12),
            max(float(max_corruption_f1_drop), 1e-12),
            max(float(max_next_ce_delta), 1e-12),
            max(float(max_unsafe_update_delta), 1e-12),
        ]
        violation = (
            ((-summary["clean_f1_delta_mean"] - scales[0]).clip(lower=0.0) / scales[0])
            + ((-summary["corruption_f1_delta_mean"] - scales[1]).clip(lower=0.0) / scales[1])
            + ((summary["next_ce_delta_mean"] - scales[2]).clip(lower=0.0) / scales[2])
            + ((summary["unsafe_update_delta_mean"] - scales[3]).clip(lower=0.0) / scales[3])
        )
        summary["constraint_violation"] = violation
        selected = summary.sort_values(
            [
                "constraint_violation",
                "next_ce_delta_mean",
                "unsafe_update_delta_mean",
                "steps",
                "learning_rate",
            ],
            ascending=[True, True, True, True, True],
        ).iloc[0]
        rule = "fallback_minimize_paired_source_constraint_violation"
        fallback_used = True
    selected_profile = str(selected["profile"])
    return {
        "dataset": DATASET,
        "selected_profile": selected_profile,
        "orientation_strength_deg": float(
            frame["orientation_strength_deg"].iloc[0]
            if "orientation_strength_deg" in frame
            else float("nan")
        ),
        "auxiliary_weight": float(selected["auxiliary_weight"]),
        "learning_rate": float(selected["learning_rate"]),
        "steps": int(selected["steps"]),
        "selection_rule": rule,
        "aggregation": (
            "mean across source users within each source_seed, then mean across "
            "source_seed units; worst source cell retained as diagnostic"
        ),
        "fallback_used": fallback_used,
        "constraints": {
            "max_clean_f1_drop_vs_no_ssaw": float(max_clean_f1_drop),
            "max_corruption_f1_drop_vs_no_ssaw": float(max_corruption_f1_drop),
            "max_next_ce_delta_vs_no_ssaw": float(max_next_ce_delta),
            "max_unsafe_update_rate_delta_vs_no_ssaw": float(
                max_unsafe_update_delta
            ),
        },
        "selected_metrics": {
            key: float(selected[key])
            for key in (
                "mean_paired_f1_delta",
                "clean_f1_delta_mean",
                "corruption_f1_delta_mean",
                "next_ce_delta_mean",
                "unsafe_update_delta_mean",
                "clean_f1_delta_min",
                "corruption_f1_delta_min",
                "next_ce_delta_max",
                "unsafe_update_delta_max",
                "full_clean_f1_min",
                "full_corruption_f1_min",
            )
            if key in selected
        },
        "candidate_summary": summary.to_dict(orient="records"),
        "target_labels_used": False,
        "target_data_used": False,
    }


STAGE_STATUS_COLUMNS = (
    "coordinate_stage",
    "orientation_strength_deg",
    "source_domain",
    "source_seed",
    "profile",
    "condition",
    "status",
    "rows_written",
    "started_at",
    "completed_at",
    "error",
)


def _stage_key(row: Mapping) -> tuple[str, float, str, int, str, str]:
    return (
        str(row.get("coordinate_stage")),
        _float_key(row.get("orientation_strength_deg")),
        str(row.get("source_domain")),
        int(row.get("source_seed")),
        str(row.get("profile")),
        str(row.get("condition")),
    )


def _stage_expected_keys(
    stage: str,
    profiles: Iterable[Mapping],
    source_domains: Iterable[str],
    source_seeds: Iterable[int],
) -> list[tuple[str, float, str, int, str, str]]:
    return [
        (
            str(stage),
            _float_key(profile.get("orientation_strength_deg", 0.0)),
            str(domain),
            int(seed),
            str(profile["profile"]),
            str(condition),
        )
        for profile in profiles
        for seed in source_seeds
        for domain in source_domains
        for condition in SECOND_STAGE_CONDITIONS
    ]


def _stage_status_frame(
    expected_keys: Iterable[tuple[str, float, str, int, str, str]],
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    old = {}
    if existing is not None and not existing.empty:
        for _, row in existing.iterrows():
            try:
                old[_stage_key(row)] = {
                    column: row.get(column, "")
                    for column in STAGE_STATUS_COLUMNS
                }
            except (TypeError, ValueError):
                continue
    rows = []
    for key in expected_keys:
        previous = old.get(key, {})
        status = str(previous.get("status", "pending"))
        if status not in {"pending", "running", "completed", "oom", "failed"}:
            status = "pending"
        rows.append(
            {
                "coordinate_stage": key[0],
                "orientation_strength_deg": key[1],
                "source_domain": key[2],
                "source_seed": key[3],
                "profile": key[4],
                "condition": key[5],
                "status": status,
                "rows_written": int(previous.get("rows_written", 0) or 0),
                "started_at": str(previous.get("started_at", "") or ""),
                "completed_at": str(previous.get("completed_at", "") or ""),
                "error": str(previous.get("error", "") or ""),
            }
        )
    return pd.DataFrame(rows, columns=list(STAGE_STATUS_COLUMNS))


def _mark_stage_status(
    statuses: pd.DataFrame,
    key: tuple[str, float, str, int, str, str],
    status: str,
    *,
    rows_written: int = 0,
    error: str = "",
) -> pd.DataFrame:
    statuses = statuses.copy()
    mask = statuses.apply(lambda row: _stage_key(row) == key, axis=1)
    if not mask.any():
        raise KeyError(f"stage-2 status key is outside protocol: {key}")
    statuses.loc[mask, "status"] = str(status)
    statuses.loc[mask, "rows_written"] = int(rows_written)
    statuses.loc[mask, "error"] = str(error)
    if status == "running":
        statuses.loc[mask, "started_at"] = _utc_now()
        statuses.loc[mask, "completed_at"] = ""
    elif status in {"completed", "oom", "failed"}:
        statuses.loc[mask, "completed_at"] = _utc_now()
    return statuses


def _run_stage2(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    orientation_strength: float,
    source_domains: list[str],
    source_seeds: list[int],
) -> tuple[int, dict | None]:
    """Run the three durable source-only coordinate-descent stages."""

    profile_cells_path = output_dir / "profile_cells.csv"
    profile_status_path = output_dir / "profile_cell_status.csv"
    profile_selection_path = output_dir / "stage2_selected_profiles.json"
    profile_frame = _read_csv(profile_cells_path)
    if not profile_frame.empty and "orientation_strength_deg" in profile_frame:
        profile_frame = profile_frame[
            profile_frame["orientation_strength_deg"]
            .astype(float)
            .round(8)
            .eq(_float_key(orientation_strength))
        ].copy()
    existing_status = _read_csv(profile_status_path)
    status_rows = []
    if existing_status is not None and not existing_status.empty:
        status_rows = existing_status.to_dict(orient="records")
    status_frame_all = pd.DataFrame(status_rows, columns=list(STAGE_STATUS_COLUMNS))
    if not status_frame_all.empty:
        if "orientation_strength_deg" not in status_frame_all:
            # Legacy stage-2 status cannot prove the physical orientation and
            # is therefore never resumable.
            status_frame_all = pd.DataFrame(columns=list(STAGE_STATUS_COLUMNS))
        else:
            status_frame_all = status_frame_all[
                status_frame_all["orientation_strength_deg"]
                .astype(float)
                .round(8)
                .eq(_float_key(orientation_strength))
            ].copy()
    selections: dict[str, dict] = {}
    if profile_selection_path.exists():
        try:
            payload = json.loads(profile_selection_path.read_text(encoding="utf-8"))
            selections.update(payload.get("stages", {}) or {})
        except (OSError, ValueError, json.JSONDecodeError):
            selections = {}
    fixed_auxiliary = 8.0
    fixed_learning_rate = 1e-4
    fixed_steps = 1
    exit_code = 0
    final_selection = None

    for stage in SECOND_STAGE_COORDINATE_STAGES:
        profiles = coordinate_profile_rows(
            stage,
            auxiliary_weight=fixed_auxiliary,
            learning_rate=fixed_learning_rate,
            steps=fixed_steps,
        )
        profiles = [
            {**profile, "orientation_strength_deg": float(orientation_strength)}
            for profile in profiles
        ]
        expected = _stage_expected_keys(
            stage, profiles, source_domains, source_seeds
        )
        old_stage = (
            status_frame_all[
                status_frame_all.get("coordinate_stage", pd.Series(dtype=str))
                .astype(str)
                .eq(stage)
                & status_frame_all.get(
                    "orientation_strength_deg", pd.Series(dtype=float)
                )
                .astype(float)
                .round(8)
                .eq(_float_key(orientation_strength))
            ]
            if not status_frame_all.empty
            else pd.DataFrame()
        )
        statuses = _stage_status_frame(expected, old_stage)
        existing_profile_keys = set()
        if not profile_frame.empty:
            for _, row in profile_frame.iterrows():
                try:
                    existing_profile_keys.add(_stage_key(row))
                except (TypeError, ValueError):
                    continue
        durable = {
            key
            for key, row in zip(expected, statuses.to_dict(orient="records"))
            if str(row.get("status")) == "completed" and key in existing_profile_keys
        }
        statuses.loc[
            statuses.apply(lambda row: _stage_key(row) not in durable, axis=1)
            & statuses["status"].eq("completed"),
            "status",
        ] = "pending"
        status_frame_all = pd.concat(
            [
                status_frame_all[
                    ~status_frame_all.get("coordinate_stage", pd.Series(dtype=str))
                    .astype(str)
                    .eq(stage)
                ] if not status_frame_all.empty else pd.DataFrame(),
                statuses,
            ],
            ignore_index=True,
        )
        atomic_write_csv(status_frame_all, profile_status_path, index=False)

        for profile in profiles:
            for source_seed in source_seeds:
                for source_domain in source_domains:
                    for condition in SECOND_STAGE_CONDITIONS:
                        key = (
                            stage,
                            _float_key(orientation_strength),
                            str(source_domain),
                            int(source_seed),
                            str(profile["profile"]),
                            str(condition),
                        )
                        if key in durable:
                            continue
                        status_frame_all = _mark_stage_status(
                            status_frame_all, key, "running"
                        )
                        atomic_write_csv(
                            status_frame_all, profile_status_path, index=False
                        )
                        try:
                            row = _run_profile_cell(
                                args=args,
                                source_domain=str(source_domain),
                                source_seed=int(source_seed),
                                orientation_strength=float(orientation_strength),
                                profile=profile,
                                condition=str(condition),
                            )
                            row["coordinate_stage"] = stage
                            row["orientation_strength_deg"] = float(
                                orientation_strength
                            )
                            profile_frame = pd.concat(
                                [profile_frame, pd.DataFrame([row])],
                                ignore_index=True,
                            )
                            durable.add(key)
                            status_frame_all = _mark_stage_status(
                                status_frame_all,
                                key,
                                "completed",
                                rows_written=1,
                            )
                            atomic_write_csv(
                                profile_frame,
                                profile_cells_path,
                                index=False,
                            )
                            atomic_write_csv(
                                status_frame_all,
                                profile_status_path,
                                index=False,
                            )
                        except Exception as exc:
                            status = "oom" if _is_oom_error(exc) else "failed"
                            status_frame_all = _mark_stage_status(
                                status_frame_all,
                                key,
                                status,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            atomic_write_csv(
                                status_frame_all,
                                profile_status_path,
                                index=False,
                            )
                            if _is_oom_error(exc) and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            exit_code = 2
                            print(
                                f"[{DATASET} stage2] {status} key={key}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )

        stage_status = status_frame_all[
            status_frame_all["coordinate_stage"].astype(str).eq(stage)
        ]
        stage_complete = bool(
            not stage_status.empty
            and stage_status["status"].astype(str).eq("completed").all()
        )
        if not stage_complete:
            return exit_code or 2, None
        stage_frame = profile_frame[
            profile_frame["coordinate_stage"].astype(str).eq(stage)
        ]
        selection = _select_adaptation_profile(
            stage_frame,
            max_clean_f1_drop=args.max_clean_f1_drop_vs_no_ssaw,
            max_corruption_f1_drop=args.max_corruption_f1_drop_vs_no_ssaw,
            max_next_ce_delta=args.max_next_ce_delta_vs_no_ssaw,
            max_unsafe_update_delta=args.max_unsafe_update_rate_delta_vs_no_ssaw,
        )
        selections[stage] = selection
        fixed_auxiliary = float(selection["auxiliary_weight"])
        fixed_learning_rate = float(selection["learning_rate"])
        fixed_steps = int(selection["steps"])
        final_selection = selection
        _atomic_write_json(
            {
                "status": "running",
                "stages": selections,
                "target_labels_used": False,
                "target_data_used": False,
            },
            profile_selection_path,
        )

    if final_selection is not None:
        _atomic_write_json(
            {
                "status": "complete",
                "stages": selections,
                "selected_profile": final_selection,
                "target_labels_used": False,
                "target_data_used": False,
            },
            profile_selection_path,
        )
    return exit_code, final_selection


def _metric_column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing calibration metric; expected one of {names}")


def _select(
    frame: pd.DataFrame,
    *,
    max_label_flip: float = DEFAULT_MAX_LABEL_FLIP,
    max_kl: float = DEFAULT_MAX_KL,
    max_semantic_distance: float = DEFAULT_MAX_SEMANTIC_DISTANCE,
    max_f1_drop: float = DEFAULT_MAX_F1_DROP,
) -> dict:
    """Select the largest registered angle satisfying source-only bounds.

    Flip/KL/semantic constraints use the worst source domain, source seed, and
    source split.  F1 uses the worst held-out/source-train drop.  Thus a cell
    cannot be hidden by averaging it with another source user.  The 0-degree
    control makes the feasible set non-empty unless the input is malformed.
    """

    if frame is None or frame.empty:
        raise ValueError("cannot select an orientation from an empty cell frame")
    strength_column = _metric_column(frame, "strength_deg", "strength")
    flip_column = _metric_column(frame, "label_flip_rate", "flip_rate")
    kl_column = _metric_column(frame, "kl_mean", "prediction_kl")
    semantic_column = _metric_column(
        frame, "semantic_distance_mean", "semantic_distance"
    )
    raw_f1_column = _metric_column(frame, "raw_source_f1", "raw_f1")
    view_f1_column = _metric_column(
        frame, "view_source_f1_mean", "view_source_f1", "source_f1"
    )
    work = frame.copy()
    work["_strength"] = work[strength_column].astype(float).round(8)
    work["_f1_drop"] = (
        work[raw_f1_column].astype(float) - work[view_f1_column].astype(float)
    )
    grouped_rows = []
    for strength, group in work.groupby("_strength", sort=True):
        grouped_rows.append(
            {
                "strength_deg": float(strength),
                "label_flip_rate_max": float(group[flip_column].astype(float).max()),
                "kl_mean_max": float(group[kl_column].astype(float).max()),
                "semantic_distance_mean_max": float(
                    group[semantic_column].astype(float).max()
                ),
                "raw_source_f1_mean": float(
                    group[raw_f1_column].astype(float).mean()
                ),
                "view_source_f1_mean": float(
                    group[view_f1_column].astype(float).mean()
                ),
                "f1_drop_max": float(group["_f1_drop"].astype(float).max()),
                "f1_delta_min": float(group["_f1_drop"].astype(float).min() * -1.0),
                "source_cells": int(len(group)),
            }
        )
    summary = pd.DataFrame(grouped_rows).sort_values("strength_deg").reset_index(
        drop=True
    )
    summary["meets_constraints"] = (
        summary["label_flip_rate_max"].le(float(max_label_flip))
        & summary["kl_mean_max"].le(float(max_kl))
        & summary["semantic_distance_mean_max"].le(float(max_semantic_distance))
        & summary["f1_drop_max"].le(float(max_f1_drop))
    )
    if summary["meets_constraints"].any():
        selected = summary[summary["meets_constraints"]].sort_values(
            "strength_deg"
        ).iloc[-1]
        rule = (
            "largest_pre_registered_strength_within_source_flip_kl_"
            "semantic_and_f1_constraints"
        )
        fallback_used = False
    else:
        # This should only happen for malformed synthetic input because the
        # zero-degree control is expected to satisfy all finite bounds.
        scales = [
            max(float(max_label_flip), 1e-12),
            max(float(max_kl), 1e-12),
            max(float(max_semantic_distance), 1e-12),
            max(float(max_f1_drop), 1e-12),
        ]
        summary["constraint_violation"] = (
            (summary["label_flip_rate_max"] / scales[0]).clip(lower=0.0)
            + (summary["kl_mean_max"] / scales[1]).clip(lower=0.0)
            + (summary["semantic_distance_mean_max"] / scales[2]).clip(lower=0.0)
            + (summary["f1_drop_max"] / scales[3]).clip(lower=0.0)
        )
        selected = summary.sort_values(
            ["constraint_violation", "strength_deg"], ascending=[True, True]
        ).iloc[0]
        rule = (
            "fallback_minimize_normalized_source_constraint_violation_then_"
            "smallest_strength"
        )
        fallback_used = True
    selected_strength = float(selected["strength_deg"])
    selected_payload = {
        "dataset": DATASET,
        "selected_profile": f"orientation_s{selected_strength:g}_sigma0",
        "selected_strength_deg": selected_strength,
        "sigma": ORIENTATION_SIGMA,
        "orientation_definition": (
            "bounded SO(3) axis-angle radius; maximum total angle in degrees"
        ),
        "selection_rule": rule,
        "fallback_used": fallback_used,
        "constraints": {
            "max_label_flip_rate": float(max_label_flip),
            "max_prediction_kl": float(max_kl),
            "max_semantic_distance": float(max_semantic_distance),
            "max_source_f1_drop": float(max_f1_drop),
        },
        "selected_metrics": {
            key: (
                float(selected[key])
                if pd.api.types.is_number(selected[key])
                else selected[key]
            )
            for key in selected.index
            if key not in {"meets_constraints", "constraint_violation"}
        },
        "candidate_summary": summary.to_dict(orient="records"),
        "target_labels_used": False,
        "target_metrics_used": False,
    }
    return selected_payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read resumable CSV {path}") from exc


def _validate_resume_manifest(existing: Mapping, expected: Mapping) -> None:
    """Reject resume artifacts produced by a different frozen protocol."""

    if not existing:
        return
    checks = (
        ("protocol", existing.get("protocol"), expected.get("protocol")),
        ("dataset", existing.get("dataset"), expected.get("dataset")),
        (
            "source_domains",
            existing.get("source_domains"),
            expected.get("source_domains"),
        ),
        (
            "source_seeds",
            existing.get("source_seeds"),
            expected.get("source_seeds"),
        ),
        ("stream_seed", existing.get("stream_seed"), expected.get("stream_seed")),
        (
            "orientation.candidate_strengths_deg",
            (existing.get("orientation") or {}).get("candidate_strengths_deg"),
            (expected.get("orientation") or {}).get("candidate_strengths_deg"),
        ),
        (
            "orientation.sigma",
            (existing.get("orientation") or {}).get("sigma"),
            (expected.get("orientation") or {}).get("sigma"),
        ),
        (
            "selection_rule.constraints",
            (existing.get("selection_rule") or {}).get("constraints"),
            (expected.get("selection_rule") or {}).get("constraints"),
        ),
    )
    for name, actual, wanted in checks:
        if actual != wanted:
            raise ValueError(
                f"resume manifest mismatch for {name}: existing={actual!r}, "
                f"requested={wanted!r}"
            )
    existing_stage = (existing.get("second_stage") or {}).get(
        "coordinate_stages"
    )
    expected_stage = (expected.get("second_stage") or {}).get(
        "coordinate_stages"
    )
    if existing_stage is not None and existing_stage != expected_stage:
        raise ValueError("resume manifest mismatch for second-stage coordinate grid")
    existing_stage_constraints = (existing.get("second_stage") or {}).get(
        "constraints"
    )
    expected_stage_constraints = (expected.get("second_stage") or {}).get(
        "constraints"
    )
    if (
        existing_stage_constraints is not None
        and existing_stage_constraints != expected_stage_constraints
    ):
        raise ValueError("resume manifest mismatch for second-stage constraints")


def _build_manifest(
    *,
    args: argparse.Namespace,
    strengths: list[float],
    source_domains: list[str],
    source_seeds: list[int],
    status: str,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "status": status,
        "dataset": DATASET,
        "algorithm": "DuSafe production (shared algorithms/dusafe.py)",
        "algorithm_unified": True,
        "source_domains": source_domains,
        "source_domain_definition": "HHAR user IDs 0-8",
        "source_seeds": source_seeds,
        "stream_seed": int(args.stream_seed),
        "source_splits": ["source_train", "source_test"],
        "target_transfer_flows_excluded": True,
        "target_labels_used": False,
        "target_labels_used_for_selection": False,
        "target_metrics_used": False,
        "target_data_used": False,
        "normalization_reference": "source_train",
        "target_scaler_fit_forbidden": True,
        "orientation": {
            "candidate_strengths_deg": [float(value) for value in strengths],
            "pre_registered_strengths_deg": list(PREREGISTERED_STRENGTHS),
            "sigma": ORIENTATION_SIGMA,
            "temporal_mode": ORIENTATION_TEMPORAL_MODE,
            "definition": (
                "maximum total SO(3) axis-angle rotation in degrees"
            ),
        },
        "profiles": profile_rows(strengths),
        "selection_rule": {
            "name": (
                "largest_pre_registered_strength_within_source_flip_kl_"
                "semantic_and_f1_constraints"
            ),
            "constraints": {
                "max_label_flip_rate": float(args.max_label_flip),
                "max_prediction_kl": float(args.max_kl),
                "max_semantic_distance": float(args.max_semantic_distance),
                "max_source_f1_drop": float(args.max_f1_drop),
            },
            "aggregation": (
                "worst source domain, source seed, and source split; choose "
                "largest eligible registered angle"
            ),
            "fallback": (
                "minimize normalized source constraint violation then choose "
                "smallest angle"
            ),
        },
        "second_stage": {
            "protocol": (
                "source-only paired Full versus no-SSAW optimization-profile "
                "calibration at the selected orientation"
            ),
            "coordinate_descent": True,
            "coordinate_stages": {
                "auxiliary_weight": coordinate_profile_rows(
                    "auxiliary_weight"
                ),
                "learning_rate": coordinate_profile_rows("learning_rate"),
                "steps": coordinate_profile_rows("steps"),
            },
            "conditions": list(SECOND_STAGE_CONDITIONS),
            "controlled_corruption": {
                "type": SECOND_STAGE_CORRUPTION,
                "severity": SECOND_STAGE_CORRUPTION_SEVERITY,
                "fraction": SECOND_STAGE_CORRUPTION_FRACTION,
                "seed": SECOND_STAGE_CORRUPTION_SEED,
            },
            "selection_rule": (
                "paired source clean and controlled-corruption F1 floors, "
                "then next-CE and unsafe-update-rate bounds; deterministic "
                "lower-cost tie-break"
            ),
            "constraints": {
                "max_clean_f1_drop_vs_no_ssaw": float(
                    args.max_clean_f1_drop_vs_no_ssaw
                ),
                "max_corruption_f1_drop_vs_no_ssaw": float(
                    args.max_corruption_f1_drop_vs_no_ssaw
                ),
                "max_next_ce_delta_vs_no_ssaw": float(
                    args.max_next_ce_delta_vs_no_ssaw
                ),
                "max_unsafe_update_rate_delta_vs_no_ssaw": float(
                    args.max_unsafe_update_rate_delta_vs_no_ssaw
                ),
            },
            "target_labels_used": False,
            "target_data_used": False,
            "status": "pending_orientation_selection",
            "expected_cells_by_stage": {
                stage: int(
                    len(coordinate_profile_rows(stage))
                    * len(source_domains)
                    * len(source_seeds)
                    * len(SECOND_STAGE_CONDITIONS)
                )
                for stage in SECOND_STAGE_COORDINATE_STAGES
            },
        },
        "expected_cells": int(len(source_domains) * len(source_seeds) * len(strengths)),
        "completed_cells": 0,
        "outputs": {
            "cells": "cells.csv",
            "status": "cell_status.csv",
            "profile_cells": "profile_cells.csv",
            "profile_status": "profile_cell_status.csv",
            "stage2_selected_profiles": "stage2_selected_profiles.json",
            "orientation_selected_profile": "orientation_selected_profile.json",
            "selected_profile": "selected_profile.json",
            "manifest": "manifest.json",
        },
        "checkpoint_cache": {
            "shared": True,
            "directory": str(Path(args.pretrain_cache_dir).resolve()),
            "identity": "HHAR + backbone + source user + source_seed + source hparams",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--source-seeds", default=",".join(map(str, SOURCE_SEEDS)))
    parser.add_argument("--stream-seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "hhar_orientation_source"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "calibration" / "hhar_orientation_source_v1"),
    )
    parser.add_argument(
        "--strengths",
        default=",".join(f"{value:g}" for value in PREREGISTERED_STRENGTHS),
        help="Subset of pre-registered maximum SO(3) angles in degrees.",
    )
    parser.add_argument("--sigma", type=float, default=ORIENTATION_SIGMA)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--max-label-flip", type=float, default=DEFAULT_MAX_LABEL_FLIP)
    parser.add_argument("--max-kl", type=float, default=DEFAULT_MAX_KL)
    parser.add_argument(
        "--max-semantic-distance",
        type=float,
        default=DEFAULT_MAX_SEMANTIC_DISTANCE,
    )
    parser.add_argument("--max-f1-drop", type=float, default=DEFAULT_MAX_F1_DROP)
    parser.add_argument(
        "--max-clean-f1-drop-vs-no-ssaw",
        type=float,
        default=DEFAULT_MAX_CLEAN_F1_DROP,
        help="Stage-2 source clean F1 drop allowed versus paired no-SSAW.",
    )
    parser.add_argument(
        "--max-corruption-f1-drop-vs-no-ssaw",
        type=float,
        default=DEFAULT_MAX_CORRUPTION_F1_DROP,
        help=(
            "Stage-2 controlled signal-freeze F1 drop allowed versus "
            "paired no-SSAW."
        ),
    )
    parser.add_argument(
        "--max-next-ce-delta-vs-no-ssaw",
        type=float,
        default=DEFAULT_MAX_NEXT_CE_DELTA,
    )
    parser.add_argument(
        "--max-unsafe-update-rate-delta-vs-no-ssaw",
        type=float,
        default=DEFAULT_MAX_UNSAFE_UPDATE_DELTA,
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Only run orientation calibration; retain stage-2 manifest as pending.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    strengths = validate_candidate_strengths(args.strengths)
    source_seeds = validate_source_seeds(args.source_seeds)
    if float(args.sigma) != ORIENTATION_SIGMA:
        raise ValueError("HHAR source orientation calibration requires --sigma 0")
    if int(args.eval_batch_size) < 1:
        raise ValueError("--eval-batch-size must be positive")
    for name in (
        "max_label_flip",
        "max_kl",
        "max_semantic_distance",
        "max_f1_drop",
        "max_clean_f1_drop_vs_no_ssaw",
        "max_corruption_f1_drop_vs_no_ssaw",
        "max_next_ce_delta_vs_no_ssaw",
        "max_unsafe_update_rate_delta_vs_no_ssaw",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")

    source_domains = list(SOURCE_DOMAINS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = output_dir / "cells.csv"
    status_path = output_dir / "cell_status.csv"
    manifest_path = output_dir / "manifest.json"
    selected_path = output_dir / "selected_profile.json"
    expected_keys = _expected_keys(strengths, source_domains, source_seeds)

    existing_cells = _read_csv(cells_path)
    existing_status = _read_csv(status_path)
    statuses = _status_frame(expected_keys, existing_status)
    # A status row is resumable only when its complete cell rows also survived
    # the previous atomic publication.
    cell_keys = _status_key_set(existing_cells)
    durable_completed = _completed_keys(statuses)
    durable_completed &= cell_keys
    statuses.loc[
        statuses.apply(lambda row: _key_tuple(row) not in durable_completed, axis=1)
        & statuses["status"].eq("completed"),
        "status",
    ] = "pending"
    atomic_write_csv(statuses, status_path, index=False)
    cells = (
        existing_cells.to_dict(orient="records")
        if not existing_cells.empty
        else []
    )
    manifest = _build_manifest(
        args=args,
        strengths=strengths,
        source_domains=source_domains,
        source_seeds=source_seeds,
        status="running",
    )
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read existing calibration manifest {manifest_path}"
            ) from exc
        _validate_resume_manifest(previous_manifest, manifest)
    manifest["completed_cells"] = int(len(durable_completed))
    _atomic_write_json(manifest, manifest_path)

    lock = None
    if str(args.device).lower().startswith("cuda"):
        lock = _gpu_lock(ROOT / "results" / ".current_experiment_gpu.lock")
    exit_code = 0
    try:
        for source_seed in source_seeds:
            for source_domain in source_domains:
                for strength in strengths:
                    key = (str(source_domain), int(source_seed), _float_key(strength))
                    if key in durable_completed:
                        continue
                    statuses = _mark_status(statuses, key, "running")
                    atomic_write_csv(statuses, status_path, index=False)
                    print(
                        f"[{DATASET} source orientation] source={source_domain} "
                        f"source_seed={source_seed} strength={strength:g}",
                        flush=True,
                    )
                    try:
                        cell_rows = _run_cell(
                            args=args,
                            source_domain=source_domain,
                            source_seed=int(source_seed),
                            strength=float(strength),
                        )
                    except Exception as exc:
                        oom = _is_oom_error(exc)
                        status = "oom" if oom else "failed"
                        statuses = _mark_status(
                            statuses,
                            key,
                            status,
                            rows_written=0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        atomic_write_csv(statuses, status_path, index=False)
                        if oom and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Keep the durable status and continue other cells.
                        # A later invocation can retry this key.
                        exit_code = 2
                        print(
                            f"[{DATASET} source orientation] {status} key={key}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    cells.extend(cell_rows)
                    durable_completed.add(key)
                    statuses = _mark_status(
                        statuses,
                        key,
                        "completed",
                        rows_written=len(cell_rows),
                    )
                    # Publish rows and state separately but atomically per file;
                    # startup repairs a completed status when rows are absent.
                    atomic_write_csv(pd.DataFrame(cells), cells_path, index=False)
                    atomic_write_csv(statuses, status_path, index=False)
                    manifest["completed_cells"] = int(len(durable_completed))
                    _atomic_write_json(manifest, manifest_path)

        status_values = statuses["status"].astype(str)
        all_complete = bool(status_values.eq("completed").all())
        frame = pd.DataFrame(cells)
        if all_complete and not frame.empty:
            selection = _select(
                frame,
                max_label_flip=args.max_label_flip,
                max_kl=args.max_kl,
                max_semantic_distance=args.max_semantic_distance,
                max_f1_drop=args.max_f1_drop,
            )
            selection["source_domains"] = source_domains
            selection["source_seeds"] = source_seeds
            selection["target_labels_used"] = False
            _atomic_write_json(selection, output_dir / "orientation_selected_profile.json")
            # A strength-named alias keeps compatibility with the earlier HAR
            # orientation artifact without changing the HHAR profile filename.
            _atomic_write_json(selection, output_dir / "selected_strength.json")
            manifest["orientation_selection"] = selection
            manifest["completed_cells"] = int(len(expected_keys))
            if args.skip_stage2:
                manifest["status"] = "complete_orientation_only"
                manifest["selected_profile"] = {
                    "orientation": selection,
                    "adaptation": None,
                }
                manifest["second_stage"]["status"] = "skipped_by_cli"
                final_profile = manifest["selected_profile"]
                _atomic_write_json(final_profile, selected_path)
            else:
                manifest["second_stage"]["status"] = "running"
                _atomic_write_json(manifest, manifest_path)
                stage2_code, stage2_selection = _run_stage2(
                    args=args,
                    output_dir=output_dir,
                    orientation_strength=float(selection["selected_strength_deg"]),
                    source_domains=source_domains,
                    source_seeds=source_seeds,
                )
                exit_code = max(exit_code, int(stage2_code))
                if stage2_selection is not None:
                    manifest["second_stage"]["status"] = "complete"
                    manifest["second_stage"]["selected_profile"] = stage2_selection
                    manifest["selected_profile"] = {
                        "orientation": selection,
                        "adaptation": stage2_selection,
                        "target_labels_used": False,
                        "target_data_used": False,
                    }
                    manifest["status"] = "complete"
                    _atomic_write_json(manifest["selected_profile"], selected_path)
                else:
                    manifest["second_stage"]["status"] = "incomplete"
                    manifest["status"] = "incomplete"
                    exit_code = max(exit_code, 2)
        else:
            manifest["status"] = "incomplete"
            manifest["incomplete_cells"] = int((~status_values.eq("completed")).sum())
            if exit_code == 0:
                exit_code = 2
        _atomic_write_json(manifest, manifest_path)
        # Keep a summary CSV even when a retry is needed; it is useful for
        # diagnosing an OOM/failed cell and is never used for selection unless
        # every status row is complete.
        if not frame.empty:
            summary_payload = []
            try:
                summary_payload = _select(
                    frame,
                    max_label_flip=args.max_label_flip,
                    max_kl=args.max_kl,
                    max_semantic_distance=args.max_semantic_distance,
                    max_f1_drop=args.max_f1_drop,
                )["candidate_summary"]
            except ValueError:
                summary_payload = []
            atomic_write_csv(pd.DataFrame(summary_payload), output_dir / "summary.csv", index=False)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), flush=True)
        return int(exit_code)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        _atomic_write_json(manifest, manifest_path)
        raise
    finally:
        _release_gpu_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET",
    "DEFAULT_MAX_F1_DROP",
    "GRID_STRENGTHS",
    "ORIENTATION_SIGMA",
    "ORIENTATION_STRENGTHS",
    "PREREGISTERED_STRENGTHS",
    "SOURCE_DOMAINS",
    "SOURCE_SEEDS",
    "_completed_keys",
    "_expected_keys",
    "_profile_rows",
    "_select",
    "_status_frame",
    "_run_cell",
    "main",
    "parse_args",
    "profile_rows",
    "validate_candidate_strengths",
    "validate_source_seeds",
]
