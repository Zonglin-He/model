"""First-round HAR 12->16 diagnostics for the guarded SSAW candidate method.

This runner is deliberately independent from the production DuSafe runners.  It
uses :class:`algorithms.dusafe_guarded_candidate.DuSafeGuardedCandidate` and
keeps the target labels outside the adapter call.  The labels are read only
after the adapter has returned, for the requested offline F1 diagnostics.

The unit of resume is one ``condition x source_seed`` cell.  Both ``Full`` and
``No-SSAW`` are run from a fresh copy of the same source checkpoint in that
cell, so the paired comparison cannot inherit state from the other variant.
GPU cells are serialized through the repository-wide
``results/.current_experiment_gpu.lock``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe_guarded_candidate import (  # noqa: E402
    DuSafeGuardedCandidate,
    GUARDED_CANDIDATE_VARIANTS,
)
from optim.optimizer import build_optimizer  # noqa: E402
from scripts.run_full_main_table import (  # noqa: E402
    wait_for_gpu_experiment_lock,
)
from scripts.supplementary_utils import (  # noqa: E402
    BatchTransformLoader,
    build_trainer,
    cleanup_trainer,
    ensure_dir,
    extract_primary_tensor,
    move_data_to_device,
    prepare_scenario,
)
from utils.utils import fix_randomness  # noqa: E402


PROTOCOL = "har_guarded_candidate_diagnostic_v2_frozen_bn_guard_12_to_16"
DATASET = "HAR"
SCENARIO = ("12", "16")
VARIANTS = ("Full", "No-SSAW")
DEFAULT_SOURCE_SEEDS = (1, 2, 3)
DEFAULT_STREAM_SEED = 42
DEFAULT_CORRUPTION_SEED = 1
DEFAULT_CORRUPTION_FRACTION = 0.5
HAR_DEFAULT_STEPS = 23
HAR_DEFAULT_BATCH_SIZE = 48
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "har_guarded_candidate_12to16"
DEFAULT_GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"
DEFAULT_CONDITIONS = ("clean", "signal_freeze_s3", "signal_freeze_s6")

# The physical evaluation protocol defines s3=0.20 and s6=0.60 for freeze.
# The generic corruption registry exposes only mild/moderate/severe, so this
# runner implements the two registered physical points explicitly.
SIGNAL_FREEZE_FRACTIONS = {"s3": 0.20, "s6": 0.60}
COUNTERFACTUAL_COLUMNS = (
    "dataset",
    "scenario",
    "condition",
    "source_seed",
    "stream_seed",
    "variant",
    "batch_index",
    "attempt_index",
    "learning_rate_scale",
    "candidate_finite",
    "candidate_guard_flip_count",
    "offline_counterfactual_f1",
    "rollback_model_f1",
    "counterfactual_delta_vs_rollback",
    "target_labels_used_for_decision",
)


def _csv_values(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = value.split(",")
    else:
        pieces = value
    return [str(piece).strip() for piece in pieces if str(piece).strip()]


def parse_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(item) for item in _csv_values(value))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def canonical_conditions(
    conditions: str | Sequence[str] | None = None,
    corruptions: str | Sequence[str] | None = None,
    severities: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Normalize CLI condition aliases without touching target labels."""

    if conditions is not None and _csv_values(conditions):
        requested = _csv_values(conditions)
    elif corruptions is not None and _csv_values(corruptions):
        corruption_values = _csv_values(corruptions)
        severity_values = _csv_values(severities) or ["s3", "s6"]
        requested = [
            "clean"
            if str(corruption).strip().lower() in {"clean", "none"}
            else f"{str(corruption).strip().lower()}_{str(severity).strip().lower()}"
            for corruption in corruption_values
            for severity in severity_values
        ]
    else:
        requested = list(DEFAULT_CONDITIONS)

    aliases = {
        "signal_freeze_moderate": "signal_freeze_s3",
        "signal_freeze_severe": "signal_freeze_s6",
        "signal_freeze_s3": "signal_freeze_s3",
        "signal_freeze_s6": "signal_freeze_s6",
        "clean": "clean",
    }
    normalized = []
    for item in requested:
        key = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if key != "clean" and key not in {"signal_freeze_s3", "signal_freeze_s6"}:
            raise ValueError(
                "HAR first-round runner supports only clean, "
                "signal_freeze_s3, and signal_freeze_s6; "
                f"received {item!r}"
            )
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("at least one condition is required")
    return tuple(normalized)


def _condition_spec(condition: str) -> dict[str, object]:
    condition = str(condition).strip().lower()
    if condition == "clean":
        return {
            "corruption": "none",
            "severity": "s0",
            "fraction": 0.0,
            "corruption_fraction": 0.0,
        }
    if condition not in {"signal_freeze_s3", "signal_freeze_s6"}:
        raise ValueError(f"unsupported HAR condition {condition!r}")
    severity = condition.rsplit("_", 1)[-1]
    return {
        "corruption": "signal_freeze",
        "severity": severity,
        "fraction": float(SIGNAL_FREEZE_FRACTIONS[severity]),
        "corruption_fraction": DEFAULT_CORRUPTION_FRACTION,
    }


def deterministic_corruption_mask(
    indices: torch.Tensor | Sequence[int],
    *,
    seed: int = DEFAULT_CORRUPTION_SEED,
    fraction: float = DEFAULT_CORRUPTION_FRACTION,
) -> torch.Tensor:
    """Return an index-stable, label-free mask for a corruption stream."""

    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("corruption fraction must lie in [0, 1]")
    values = torch.as_tensor(indices, dtype=torch.long).view(-1)
    # Use integer arithmetic rather than a stateful generator: the mask is
    # invariant to batch size and to a resumed cell's iteration boundaries.
    hashed = (
        values * 1_103_515_245
        + int(seed) * 12_345
        + 1_013_904_223
    ).remainder(2_147_483_647)
    cutoff = int(math.floor(float(fraction) * 2_147_483_647))
    return hashed.lt(cutoff)


def signal_freeze_at_level(inputs: torch.Tensor, severity: str) -> torch.Tensor:
    """Apply the physical protocol's exact s3/s6 freeze fraction."""

    severity = str(severity).strip().lower()
    if severity not in SIGNAL_FREEZE_FRACTIONS:
        raise ValueError(f"signal_freeze_at_level only supports s3/s6, got {severity}")
    if inputs.ndim != 3:
        raise ValueError(f"expected [B,C,T], got {tuple(inputs.shape)}")
    output = inputs.clone()
    freeze_length = max(
        1, int(round(output.size(-1) * SIGNAL_FREEZE_FRACTIONS[severity]))
    )
    pivot = max(1, output.size(-1) - freeze_length)
    output[..., pivot:] = output[..., pivot - 1 : pivot].expand_as(output[..., pivot:])
    return output


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    predictions = torch.as_tensor(logits).argmax(dim=1).detach().cpu().numpy()
    references = torch.as_tensor(labels).view(-1).detach().cpu().numpy()
    return float(
        f1_score(
            references,
            predictions,
            labels=list(range(int(num_classes))),
            average="macro",
            zero_division=0,
        )
    )


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _json_signature(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _tensor_state_sha256(model) -> str:
    """Hash the canonical source tensor state before adapter construction."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cell_directory(output_dir: Path, condition: str, source_seed: int) -> Path:
    return output_dir / "flow12_to_16" / f"source_seed_{int(source_seed)}" / condition


def _batch_indices(indices, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=device).view(-1)


def _prepare_condition_loader(trainer, condition: str, corruption_seed: int):
    spec = _condition_spec(condition)
    if condition == "clean":
        return trainer.trg_whole_dl

    def mask_fn(_data, _labels, indices, _step, _total_steps):
        # Deliberately ignore ``_labels``.  This function is part of the input
        # stream construction and cannot influence adapter decisions.
        return deterministic_corruption_mask(
            indices,
            seed=corruption_seed,
            fraction=DEFAULT_CORRUPTION_FRACTION,
        )

    return BatchTransformLoader(
        trainer.trg_whole_dl,
        signal_freeze_at_level,
        str(spec["severity"]),
        sample_mask_fn=mask_fn,
        meta={
            "condition": condition,
            "corruption": "signal_freeze",
            "severity": str(spec["severity"]),
            "corruption_seed": int(corruption_seed),
            "corruption_fraction": DEFAULT_CORRUPTION_FRACTION,
        },
        transform_seed=corruption_seed,
    )


def _make_adapter(
    trainer,
    source_seed: int,
    stream_seed: int,
    variant: str,
    steps_override: int | None,
    batch_size_override: int | None,
    *,
    scenario: tuple[str, str] = SCENARIO,
    hparam_overrides: Mapping[str, object] | None = None,
):
    overrides = {
        "candidate_guard_fraction": 0.25,
        "candidate_guard_split_seed": 271828,
        "candidate_backtracking_scale": 0.5,
        "candidate_record_gradient_diagnostics": True,
        "fixed_source_anchor_admission_mode": "joint",
    }
    if hparam_overrides:
        overrides.update(dict(hparam_overrides))
    if steps_override is not None:
        overrides["steps"] = int(steps_override)
    if batch_size_override is not None:
        overrides["batch_size"] = int(batch_size_override)
    trainer.set_runtime_hparams(overrides)
    prepare_scenario(
        trainer,
        str(scenario[0]),
        str(scenario[1]),
        run_seed=int(stream_seed),
        run_id=int(stream_seed),
    )
    # Source training/cache lookup is fixed by source seed; target-time RNG is
    # reset afterwards so Full and No-SSAW see the same stream randomness.
    fix_randomness(int(source_seed))
    source_checkpoint = trainer._pretrain_cache_path()
    _non_adapted, source_model = trainer.pre_train()
    source_model_sha256 = _tensor_state_sha256(source_model)
    fix_randomness(int(stream_seed))
    optimizer_factory = build_optimizer(trainer.hparams)
    try:
        adapter_class = GUARDED_CANDIDATE_VARIANTS[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unsupported guarded-candidate variant {variant!r}") from exc
    adapter = adapter_class(
        trainer.dataset_configs,
        trainer.hparams,
        source_model,
        optimizer_factory,
    ).to(trainer.device)
    if hasattr(adapter, "load_source_normalization_reference"):
        normalization_stats = getattr(
            trainer.src_train_dl.dataset, "normalization_stats", None
        )
        if normalization_stats is None:
            raise RuntimeError("HAR guarded candidate requires source normalization stats")
        adapter.load_source_normalization_reference(*normalization_stats)
    if getattr(adapter, "enable_confidence_gate", False):
        if trainer.source_confidence_metadata is None:
            raise RuntimeError("source confidence metadata is missing")
        adapter.load_source_confidence_reference(trainer.source_confidence_metadata)
    if getattr(adapter, "enable_source_semantic_gate", False):
        if trainer.source_semantic_metadata is None:
            raise RuntimeError("source semantic metadata is missing")
        adapter.load_source_semantic_reference(trainer.source_semantic_metadata)
    return adapter, source_checkpoint, source_model_sha256


def _clean_eval_f1(adapter, trainer, labels_num_classes: int) -> float:
    """Score the final model on the uncorrupted target stream, read-only."""

    logits = []
    labels = []
    try:
        for data, target, _indices in trainer.trg_whole_dl:
            data = move_data_to_device(data, trainer.device)
            with torch.no_grad():
                logits.append(adapter.predict_raw(data).detach().cpu())
            labels.append(torch.as_tensor(target).view(-1).long().cpu())
    except RuntimeError:
        raise
    if not logits:
        return float("nan")
    return macro_f1(torch.cat(logits), torch.cat(labels), labels_num_classes)


def _run_variant(
    *,
    args,
    condition: str,
    source_seed: int,
    variant: str,
    scenario: tuple[str, str] = SCENARIO,
    hparam_overrides: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    trainer = None
    adapter = None
    batch_rows: list[dict[str, object]] = []
    counterfactual_rows: list[dict[str, object]] = []
    labels_num_classes = 6
    try:
        trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=DATASET,
            da_method="DuSafe",
            backbone=args.backbone,
            exp_name="har_guarded_candidate_diagnostic",
            seed=args.stream_seed,
            source_seed=source_seed,
            pretrain_cache_dir=args.pretrain_cache_dir,
        )
        adapter, source_checkpoint, source_model_sha256 = _make_adapter(
            trainer,
            source_seed,
            args.stream_seed,
            variant,
            args.steps,
            args.batch_size,
            scenario=scenario,
            hparam_overrides=hparam_overrides,
        )
        scenario_name = f"{scenario[0]}->{scenario[1]}"
        labels_num_classes = int(trainer.dataset_configs.num_classes)
        loader = _prepare_condition_loader(
            trainer, condition, int(args.corruption_seed)
        )
        pre_logits_all: list[torch.Tensor] = []
        post_logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        corruption_masks_all: list[torch.Tensor] = []
        eligible_batches = 0
        first_commits = 0
        rescue_count = 0
        final_skips = 0
        guard_flips = 0
        first_guard_flips = 0
        retry_guard_flips = 0
        guard_reference_mismatches = 0
        admitted_anchors = 0
        observed_samples = 0
        selected_count = 0
        selected_positive = 0
        selected_negative = 0
        selected_flip_count = 0
        selected_kl_sum = 0.0
        raw_grad_norms: list[float] = []
        ssaw_grad_norms: list[float] = []
        weighted_ratios: list[float] = []
        unweighted_ratios: list[float] = []
        counterfactual_logits_by_attempt: dict[int, list[torch.Tensor]] = {
            1: [],
            2: [],
        }
        counterfactual_labels_by_attempt: dict[int, list[torch.Tensor]] = {
            1: [],
            2: [],
        }
        rollback_logits_by_attempt: dict[int, list[torch.Tensor]] = {
            1: [],
            2: [],
        }
        final_rejected_logits: list[torch.Tensor] = []
        final_rejected_labels: list[torch.Tensor] = []
        final_rollback_logits: list[torch.Tensor] = []
        stream_steps = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= int(args.max_batches):
                break
            data, target, indices = batch
            data = move_data_to_device(data, trainer.device)
            index_tensor = _batch_indices(indices, trainer.device)
            with torch.no_grad():
                pre_logits = adapter.predict_raw(data).detach()
            model_inputs = {
                "data": data,
                "meta": {"trg_idx": index_tensor.detach().cpu().tolist()},
            }
            adapter.forward(model_inputs, trg_idx=index_tensor)
            with torch.no_grad():
                post_logits = adapter.predict_raw(data).detach()
            # Labels are materialized only after the candidate decision has
            # returned.  They were not present in ``model_inputs`` and cannot
            # affect admission, hard-view selection, guard, or backtracking.
            labels = torch.as_tensor(target).view(-1).long().cpu()
            corruption_mask = (
                deterministic_corruption_mask(
                    indices,
                    seed=int(args.corruption_seed),
                    fraction=DEFAULT_CORRUPTION_FRACTION,
                )
                if condition != "clean"
                else torch.zeros(labels.numel(), dtype=torch.bool)
            )
            pre_logits_all.append(pre_logits.cpu())
            post_logits_all.append(post_logits.cpu())
            labels_all.append(labels)
            corruption_masks_all.append(corruption_mask.cpu())
            stream_steps += 1

            batch_log = dict(getattr(adapter, "_last_batch_log", {}) or {})
            eligible = bool(batch_log.get("candidate_eligible", 0.0))
            eligible_batches += int(eligible)
            first_commits += int(batch_log.get("first_attempt_commit", 0.0) > 0.5)
            rescue_count += int(batch_log.get("backtracking_rescue", 0.0) > 0.5)
            final_skips += int(batch_log.get("final_skip", 0.0) > 0.5)
            guard_flips += int(batch_log.get("guard_flip_count", 0.0))
            first_guard_flips += int(
                batch_log.get("first_guard_flip_count", 0.0)
            )
            retry_guard_flips += int(
                batch_log.get("retry_guard_flip_count", 0.0)
            )
            guard_reference_mismatches += int(
                batch_log.get("guard_reference_mismatch_count", 0.0)
            )
            admitted_anchors += int(
                batch_log.get("fixed_source_anchor_admission_count", 0.0)
            )
            observed_samples += int(batch_log.get("sample_count", labels.numel()))
            selected_count += int(batch_log.get("selected_view_count", 0.0))
            selected_positive += int(batch_log.get("selected_positive_count", 0.0))
            selected_negative += int(batch_log.get("selected_negative_count", 0.0))
            selected_flip_count += int(
                batch_log.get("selected_view_label_flip_count", 0.0)
            )
            selected_kl_value = _safe_float(batch_log.get("selected_kl_sum"))
            if math.isfinite(selected_kl_value):
                selected_kl_sum += selected_kl_value
            for key, sink in (
                ("raw_gradient_norm_mean", raw_grad_norms),
                ("ssaw_gradient_norm_mean", ssaw_grad_norms),
                ("weighted_ssaw_to_raw_gradient_ratio_mean", weighted_ratios),
            ):
                value = _safe_float(batch_log.get(key))
                if math.isfinite(value):
                    sink.append(value)
            raw_batch_norm = _safe_float(batch_log.get("raw_gradient_norm_mean"))
            ssaw_batch_norm = _safe_float(batch_log.get("ssaw_gradient_norm_mean"))
            unweighted_ratio = (
                ssaw_batch_norm / raw_batch_norm
                if math.isfinite(raw_batch_norm)
                and math.isfinite(ssaw_batch_norm)
                and abs(raw_batch_norm) > 1e-12
                else float("nan")
            )
            if math.isfinite(unweighted_ratio):
                unweighted_ratios.append(unweighted_ratio)

            current_labels = labels
            rollback_f1 = macro_f1(pre_logits.cpu(), current_labels, labels_num_classes)
            post_f1 = macro_f1(post_logits.cpu(), current_labels, labels_num_classes)
            for rejected in getattr(adapter, "_last_rejected_candidate_logits", []) or []:
                candidate_logits = rejected.get("candidate_logits")
                candidate_f1 = (
                    macro_f1(candidate_logits, current_labels, labels_num_classes)
                    if candidate_logits is not None
                    else float("nan")
                )
                attempt_index = int(rejected.get("attempt_index", -1))
                if candidate_logits is not None and attempt_index in (1, 2):
                    candidate_cpu = torch.as_tensor(candidate_logits).cpu()
                    counterfactual_logits_by_attempt[attempt_index].append(
                        candidate_cpu
                    )
                    counterfactual_labels_by_attempt[attempt_index].append(
                        current_labels
                    )
                    rollback_logits_by_attempt[attempt_index].append(
                        pre_logits.cpu()
                    )
                    if bool(batch_log.get("final_skip", 0.0)) and attempt_index == 2:
                        final_rejected_logits.append(candidate_cpu)
                        final_rejected_labels.append(current_labels)
                        final_rollback_logits.append(pre_logits.cpu())
                counterfactual_rows.append(
                    {
                        "dataset": DATASET,
                        "scenario": scenario_name,
                        "condition": condition,
                        "source_seed": int(source_seed),
                        "stream_seed": int(args.stream_seed),
                        "variant": variant,
                        "batch_index": int(batch_index),
                        "attempt_index": attempt_index,
                        "learning_rate_scale": _safe_float(
                            rejected.get("learning_rate_scale")
                        ),
                        "candidate_finite": bool(candidate_logits is not None),
                        "candidate_guard_flip_count": int(
                            rejected.get("guard_flip_count", -1)
                        ),
                        "offline_counterfactual_f1": candidate_f1,
                        "rollback_model_f1": rollback_f1,
                        "counterfactual_delta_vs_rollback": (
                            candidate_f1 - rollback_f1
                            if math.isfinite(candidate_f1)
                            else float("nan")
                        ),
                        "target_labels_used_for_decision": False,
                    }
                )

            batch_rows.append(
                {
                    "dataset": DATASET,
                    "scenario": scenario_name,
                    "condition": condition,
                    "source_seed": int(source_seed),
                    "stream_seed": int(args.stream_seed),
                    "variant": variant,
                    "batch_index": int(batch_index),
                    "sample_count": int(labels.numel()),
                    "pre_update_f1": rollback_f1,
                    "post_update_f1": post_f1,
                    "corruption_count": int(corruption_mask.sum().item()),
                    "ssaw_to_raw_gradient_ratio": unweighted_ratio,
                    "target_labels_used_for_decision": False,
                    **batch_log,
                }
            )

        if not labels_all:
            raise RuntimeError("target stream produced no batches")
        pre_stream_logits = torch.cat(pre_logits_all)
        post_stream_logits = torch.cat(post_logits_all)
        stream_labels = torch.cat(labels_all)
        stream_corruption_mask = torch.cat(corruption_masks_all).bool()
        pre_stream_f1 = macro_f1(
            pre_stream_logits, stream_labels, labels_num_classes
        )
        post_stream_f1 = macro_f1(
            post_stream_logits, stream_labels, labels_num_classes
        )
        corrupted_subset_f1 = (
            macro_f1(
                post_stream_logits[stream_corruption_mask],
                stream_labels[stream_corruption_mask],
                labels_num_classes,
            )
            if stream_corruption_mask.any()
            else float("nan")
        )
        uncorrupted_subset_f1 = (
            macro_f1(
                post_stream_logits[~stream_corruption_mask],
                stream_labels[~stream_corruption_mask],
                labels_num_classes,
            )
            if (~stream_corruption_mask).any()
            else float("nan")
        )
        post_clean_f1 = _clean_eval_f1(adapter, trainer, labels_num_classes)
        condition_is_clean = condition == "clean"

        def pooled_counterfactual(attempt_index: int, side: str) -> float:
            sources = (
                counterfactual_logits_by_attempt
                if side == "candidate"
                else rollback_logits_by_attempt
            )
            values = sources[int(attempt_index)]
            labels_for_attempt = counterfactual_labels_by_attempt[int(attempt_index)]
            if not values:
                return float("nan")
            return macro_f1(
                torch.cat(values),
                torch.cat(labels_for_attempt),
                labels_num_classes,
            )

        first_counterfactual_f1 = pooled_counterfactual(1, "candidate")
        first_rollback_f1 = pooled_counterfactual(1, "rollback")
        retry_counterfactual_f1 = pooled_counterfactual(2, "candidate")
        retry_rollback_f1 = pooled_counterfactual(2, "rollback")
        final_counterfactual_f1 = (
            macro_f1(
                torch.cat(final_rejected_logits),
                torch.cat(final_rejected_labels),
                labels_num_classes,
            )
            if final_rejected_logits
            else float("nan")
        )
        final_rollback_f1 = (
            macro_f1(
                torch.cat(final_rollback_logits),
                torch.cat(final_rejected_labels),
                labels_num_classes,
            )
            if final_rollback_logits
            else float("nan")
        )
        first_failure_count = int(eligible_batches - first_commits)
        counterfactual_frame = pd.DataFrame(counterfactual_rows)
        summary = {
            "status": "ok",
            "dataset": DATASET,
            "scenario": scenario_name,
            "condition": condition,
            "source_seed": int(source_seed),
            "stream_seed": int(args.stream_seed),
            "variant": variant,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_exists": bool(Path(source_checkpoint).is_file()),
            "source_model_sha256": source_model_sha256,
            "effective_learning_rate": float(trainer.hparams["learning_rate"]),
            "effective_steps": int(adapter.steps),
            "effective_batch_size": int(trainer.hparams["batch_size"]),
            "effective_ssaw_auxiliary_weight": float(
                adapter.ssaw_auxiliary_weight if adapter.enable_ssaw else 0.0
            ),
            "effective_ssaw_strength": float(adapter.ssaw.strength),
            "effective_guard_fraction": float(adapter.guard_fraction),
            "effective_backtracking_scale": float(adapter.backtracking_scale),
            "fixed_source_anchor_admission_mode": str(
                adapter.fixed_source_anchor_admission_mode
            ),
            "fixed_source_anchor_admission_count": int(admitted_anchors),
            "fixed_source_anchor_admission_rate": (
                admitted_anchors / observed_samples
                if observed_samples
                else float("nan")
            ),
            "guard_bn_mode": "guard_subset_with_frozen_source_bn_statistics",
            "stream_batches": int(stream_steps),
            "eligible_batches": int(eligible_batches),
            "first_attempt_commit_count": int(first_commits),
            "first_attempt_commit_rate": (
                first_commits / eligible_batches if eligible_batches else float("nan")
            ),
            "backtracking_rescue_count": int(rescue_count),
            "backtracking_rescue_rate": (
                rescue_count / eligible_batches if eligible_batches else float("nan")
            ),
            "first_attempt_failure_count": first_failure_count,
            "backtracking_rescue_rate_given_first_failure": (
                rescue_count / first_failure_count
                if first_failure_count
                else float("nan")
            ),
            "final_skip_count": int(final_skips),
            "final_skip_rate": (
                final_skips / eligible_batches if eligible_batches else float("nan")
            ),
            "guard_flip_count": int(guard_flips),
            "first_guard_flip_count": int(first_guard_flips),
            "retry_guard_flip_count": int(retry_guard_flips),
            "guard_reference_mismatch_count": int(
                guard_reference_mismatches
            ),
            "guard_flip_rate_per_eligible": (
                guard_flips / eligible_batches if eligible_batches else float("nan")
            ),
            "selected_view_count": int(selected_count),
            "selected_positive_count": int(selected_positive),
            "selected_negative_count": int(selected_negative),
            "selected_positive_rate": (
                selected_positive / selected_count if selected_count else float("nan")
            ),
            "selected_negative_rate": (
                selected_negative / selected_count if selected_count else float("nan")
            ),
            "selected_view_label_flip_count": int(selected_flip_count),
            "selected_view_label_flip_rate": (
                selected_flip_count / selected_count if selected_count else float("nan")
            ),
            "selected_kl_mean": (
                selected_kl_sum / selected_count if selected_count else float("nan")
            ),
            "raw_gradient_norm_mean": (
                float(np.mean(raw_grad_norms)) if raw_grad_norms else float("nan")
            ),
            "ssaw_gradient_norm_mean": (
                float(np.mean(ssaw_grad_norms)) if ssaw_grad_norms else float("nan")
            ),
            "weighted_ssaw_to_raw_gradient_ratio_mean": (
                float(np.mean(weighted_ratios)) if weighted_ratios else float("nan")
            ),
            "ssaw_to_raw_gradient_ratio_mean": (
                float(np.mean(unweighted_ratios))
                if unweighted_ratios
                else float("nan")
            ),
            "pre_stream_macro_f1": pre_stream_f1,
            "post_stream_macro_f1": post_stream_f1,
            "stream_macro_f1_delta": post_stream_f1 - pre_stream_f1,
            "clean_f1": post_stream_f1 if condition_is_clean else float("nan"),
            "corrupted_f1": post_stream_f1 if not condition_is_clean else float("nan"),
            "corrupted_subset_f1": corrupted_subset_f1,
            "uncorrupted_subset_f1": uncorrupted_subset_f1,
            "post_adaptation_clean_eval_f1": post_clean_f1,
            "offline_rejected_candidate_count": int(len(counterfactual_rows)),
            "offline_first_rejected_candidate_pooled_f1": first_counterfactual_f1,
            "offline_first_rejected_rollback_pooled_f1": first_rollback_f1,
            "offline_first_rejected_delta_pooled_f1": (
                first_counterfactual_f1 - first_rollback_f1
                if math.isfinite(first_counterfactual_f1)
                and math.isfinite(first_rollback_f1)
                else float("nan")
            ),
            "offline_retry_rejected_candidate_pooled_f1": retry_counterfactual_f1,
            "offline_retry_rejected_rollback_pooled_f1": retry_rollback_f1,
            "offline_retry_rejected_delta_pooled_f1": (
                retry_counterfactual_f1 - retry_rollback_f1
                if math.isfinite(retry_counterfactual_f1)
                and math.isfinite(retry_rollback_f1)
                else float("nan")
            ),
            "offline_final_rejected_candidate_pooled_f1": final_counterfactual_f1,
            "offline_final_rejected_rollback_pooled_f1": final_rollback_f1,
            "offline_final_rejected_delta_pooled_f1": (
                final_counterfactual_f1 - final_rollback_f1
                if math.isfinite(final_counterfactual_f1)
                and math.isfinite(final_rollback_f1)
                else float("nan")
            ),
            "offline_rejected_candidate_batch_f1_mean": (
                float(counterfactual_frame["offline_counterfactual_f1"].mean())
                if not counterfactual_frame.empty
                else float("nan")
            ),
            "offline_rejected_rollback_batch_f1_mean": (
                float(counterfactual_frame["rollback_model_f1"].mean())
                if not counterfactual_frame.empty
                else float("nan")
            ),
            "target_labels_used_for_decision": False,
        }
        return summary, batch_rows, counterfactual_rows
    finally:
        cleanup_trainer(trainer, adapter, close_summary=True)
        del adapter, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _signature(args, conditions: Sequence[str], source_seeds: Sequence[int]) -> dict:
    hparams = {
        "learning_rate": 3.325e-4,
        "steps": HAR_DEFAULT_STEPS if args.steps is None else int(args.steps),
        "batch_size": (
            HAR_DEFAULT_BATCH_SIZE
            if args.batch_size is None
            else int(args.batch_size)
        ),
        "guard_fraction": 0.25,
        "backtracking_scale": 0.5,
        "guard_split_seed": 271828,
        "guard_bn_mode": "guard_subset_with_frozen_source_bn_statistics",
        "ssaw_strength": 4.0,
        "ssaw_auxiliary_weight": 1.25,
    }
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "scenario": "12->16",
        "variants": list(VARIANTS),
        "source_seeds": [int(value) for value in source_seeds],
        "stream_seed": int(args.stream_seed),
        "conditions": list(conditions),
        "corruption_seed": int(args.corruption_seed),
        "corruption_fraction": DEFAULT_CORRUPTION_FRACTION,
        "target_labels_used_for_decision": False,
        "hparams": hparams,
        "pretrain_cache_dir": str(Path(args.pretrain_cache_dir).resolve()),
        "backbone": str(args.backbone),
        "max_batches": args.max_batches,
    }


def _completed_cell(summary_path: Path, signature_hash: str) -> bool:
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        str(payload.get("status", "")) == "ok"
        and str(payload.get("signature_hash", "")) == signature_hash
        and summary_path.with_name("batch_diagnostics.csv").is_file()
        and summary_path.with_name("rejected_counterfactuals.csv").is_file()
    )


def _flatten_summary(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for variant, values in dict(payload.get("variants", {})).items():
        row = {
            "dataset": DATASET,
            "scenario": "12->16",
            "condition": payload.get("condition"),
            "source_seed": payload.get("source_seed"),
            "stream_seed": payload.get("stream_seed"),
            "variant": variant,
            "status": payload.get("status", "ok"),
            "target_labels_used_for_decision": False,
        }
        row.update(dict(values))
        rows.append(row)
    return rows


def _write_aggregates(output_dir: Path, summary_rows: Sequence[Mapping[str, object]]) -> None:
    summary_frame = pd.DataFrame(list(summary_rows))
    if summary_frame.empty:
        _atomic_csv(summary_frame, output_dir / "summary.csv")
        _atomic_csv(summary_frame, output_dir / "aggregate.csv")
        return
    summary_frame = summary_frame.sort_values(
        ["condition", "source_seed", "variant"]
    ).reset_index(drop=True)
    _atomic_csv(summary_frame, output_dir / "summary.csv")
    numeric = [
        "post_stream_macro_f1",
        "clean_f1",
        "corrupted_f1",
        "corrupted_subset_f1",
        "uncorrupted_subset_f1",
        "post_adaptation_clean_eval_f1",
        "first_attempt_commit_rate",
        "backtracking_rescue_rate",
        "backtracking_rescue_rate_given_first_failure",
        "final_skip_rate",
        "guard_flip_count",
        "guard_reference_mismatch_count",
        "selected_positive_rate",
        "selected_negative_rate",
        "selected_view_label_flip_rate",
        "selected_kl_mean",
        "weighted_ssaw_to_raw_gradient_ratio_mean",
        "ssaw_to_raw_gradient_ratio_mean",
        "offline_first_rejected_candidate_pooled_f1",
        "offline_first_rejected_rollback_pooled_f1",
        "offline_first_rejected_delta_pooled_f1",
        "offline_retry_rejected_candidate_pooled_f1",
        "offline_retry_rejected_rollback_pooled_f1",
        "offline_retry_rejected_delta_pooled_f1",
        "offline_final_rejected_candidate_pooled_f1",
        "offline_final_rejected_rollback_pooled_f1",
        "offline_final_rejected_delta_pooled_f1",
    ]
    group_columns = ["condition", "variant"]
    present = [column for column in numeric if column in summary_frame.columns]
    if present:
        aggregate = (
            summary_frame.groupby(group_columns, dropna=False)[present]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aggregate.columns = [
            "_".join(str(part) for part in column if str(part) != "")
            if isinstance(column, tuple)
            else str(column)
            for column in aggregate.columns
        ]
    else:
        aggregate = pd.DataFrame()
    _atomic_csv(aggregate, output_dir / "aggregate.csv")


def _load_existing_summaries(
    output_dir: Path,
    *,
    signature_hash: str,
    conditions: Sequence[str],
    source_seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows = []
    for path in output_dir.glob("flow12_to_16/source_seed_*/**/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "ok"
            and payload.get("signature_hash") == signature_hash
            and str(payload.get("condition")) in set(conditions)
            and int(payload.get("source_seed", -1)) in set(source_seeds)
        ):
            rows.extend(_flatten_summary(payload))
    return rows


def run(args) -> pd.DataFrame:
    source_seeds = parse_ints(args.source_seeds)
    if any(seed not in {1, 2, 3} for seed in source_seeds):
        raise ValueError("source seeds must be drawn from 1,2,3")
    conditions = canonical_conditions(args.conditions, args.corruptions, args.severities)
    output_dir = ensure_dir(args.output_dir)
    signature = _signature(args, conditions, source_seeds)
    signature_hash = _json_signature(signature)
    _atomic_json(
        {
            "protocol": PROTOCOL,
            "signature": signature,
            "signature_hash": signature_hash,
            "expected_cells": len(source_seeds) * len(conditions),
            "cell_grain": "condition x source_seed with Full and No-SSAW variants",
            "target_labels_used_for_decision": False,
        },
        output_dir / "manifest.json",
    )
    summary_rows = _load_existing_summaries(
        output_dir,
        signature_hash=signature_hash,
        conditions=conditions,
        source_seeds=source_seeds,
    )
    summary_by_key = {
        (
            str(row.get("condition")),
            int(row.get("source_seed", -1)),
            str(row.get("variant")),
        ): row
        for row in summary_rows
    }
    device_is_gpu = str(args.device).lower().startswith("cuda")
    lock_path = Path(args.gpu_lock_path)
    lock_context = (
        lambda: wait_for_gpu_experiment_lock(lock_path)
        if device_is_gpu
        else nullcontext()
    )
    for condition in conditions:
        for source_seed in source_seeds:
            cell_dir = _cell_directory(output_dir, condition, source_seed)
            summary_path = cell_dir / "summary.json"
            if _completed_cell(summary_path, signature_hash):
                continue
            cell_dir.mkdir(parents=True, exist_ok=True)
            try:
                with lock_context():
                    variant_summaries = {}
                    all_batch_rows: list[dict[str, object]] = []
                    all_counterfactual_rows: list[dict[str, object]] = []
                    for variant in VARIANTS:
                        print(
                            f"[HAR guarded] condition={condition} source={source_seed} "
                            f"variant={variant}",
                            flush=True,
                        )
                        result, batch_rows, counterfactual_rows = _run_variant(
                            args=args,
                            condition=condition,
                            source_seed=int(source_seed),
                            variant=variant,
                        )
                        variant_summaries[variant] = result
                        all_batch_rows.extend(batch_rows)
                        all_counterfactual_rows.extend(counterfactual_rows)
                    source_hashes = {
                        str(result.get("source_model_sha256", ""))
                        for result in variant_summaries.values()
                    }
                    if len(source_hashes) != 1 or "" in source_hashes:
                        raise RuntimeError(
                            "Full and No-SSAW did not use one identical source model"
                        )
                    payload = {
                        "status": "ok",
                        "protocol": PROTOCOL,
                        "signature": signature,
                        "signature_hash": signature_hash,
                        "dataset": DATASET,
                        "scenario": "12->16",
                        "condition": condition,
                        "source_seed": int(source_seed),
                        "stream_seed": int(args.stream_seed),
                        "variants": variant_summaries,
                        "target_labels_used_for_decision": False,
                    }
                    # Publish the resume sentinel last.  A power loss between
                    # CSV writes cannot leave an apparently complete cell.
                    _atomic_csv(pd.DataFrame(all_batch_rows), cell_dir / "batch_diagnostics.csv")
                    _atomic_csv(
                        pd.DataFrame(
                            all_counterfactual_rows,
                            columns=COUNTERFACTUAL_COLUMNS,
                        ),
                        cell_dir / "rejected_counterfactuals.csv",
                    )
                    _atomic_json(payload, summary_path)
                    for variant, result in variant_summaries.items():
                        summary_by_key[(condition, int(source_seed), variant)] = {
                            **result,
                            "status": "ok",
                        }
            except Exception as exc:
                error_payload = {
                    "status": "failed",
                    "protocol": PROTOCOL,
                    "signature": signature,
                    "signature_hash": signature_hash,
                    "dataset": DATASET,
                    "scenario": "12->16",
                    "condition": condition,
                    "source_seed": int(source_seed),
                    "stream_seed": int(args.stream_seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(exc).lower(),
                    "target_labels_used_for_decision": False,
                }
                _atomic_json(error_payload, summary_path)
                print(
                    f"[HAR guarded] failed condition={condition} source={source_seed}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            _write_aggregates(output_dir, list(summary_by_key.values()))
    _write_aggregates(output_dir, list(summary_by_key.values()))
    return pd.read_csv(output_dir / "summary.csv") if (output_dir / "summary.csv").is_file() else pd.DataFrame()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrain-cache-dir", "--pretrain_cache_dir", dest="pretrain_cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--source-seeds", "--source_seeds", dest="source_seeds", default="1,2,3")
    parser.add_argument("--stream-seed", "--stream_seed", dest="stream_seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument("--corruption-seed", "--corruption_seed", dest="corruption_seed", type=int, default=DEFAULT_CORRUPTION_SEED)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--corruptions", default=None)
    parser.add_argument("--severities", default=None)
    parser.add_argument("--max-batches", "--max_batches", dest="max_batches", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--gpu-lock-path", "--gpu_lock_path", dest="gpu_lock_path", type=Path, default=DEFAULT_GPU_LOCK_PATH)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps is not None and int(args.steps) < 1:
        raise ValueError("--steps must be positive")
    if args.max_batches is not None and int(args.max_batches) < 1:
        raise ValueError("--max-batches must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
