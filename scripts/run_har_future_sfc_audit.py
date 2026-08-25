"""Future Structural False-Certainty audit for HAR 12->16.

The online trajectories are label-free and independent.  For every source
seed and registered corruption condition, the frozen source weights first
define a fixed post-hoc SFC subset.  A model updated on deployment batch
``t`` is then evaluated on the still-unseen batch ``t+1``; that future batch
is adapted only after its read-only evaluation has completed.

This file is an evidence runner.  It does not modify or register a production
algorithm variant.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.representative_causal_ablation import (  # noqa: E402
    RepresentativeConfidenceRaw,
    RepresentativeHardSSAW,
    RepresentativeRandomEligibleSpline,
)
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from dataloader import har_cross_dataset_corruptions as corruptions  # noqa: E402
from optim.optimizer import build_optimizer  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.counterfactual_horizon_common import (  # noqa: E402
    _nested_equal,
    snapshot_state,
    state_hash,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    extract_primary_tensor,
    move_data_to_device,
    replace_primary_tensor,
)
from trainers.tta_abstract_trainer import _predict_after_adaptation  # noqa: E402
from utils.utils import fix_randomness  # noqa: E402


PROTOCOL = "har_12_to_16_future_sfc_audit_v3_data_bound_full_state"
DATASET = "HAR"
SCENARIO = "12->16"
SOURCE_SEEDS = (0, 1, 2)
STREAM_SEED = 42
VARIANT_CLASSES = {
    "confidence_only": RepresentativeConfidenceRaw,
    "random_spline": RepresentativeRandomEligibleSpline,
    "dusafe": RepresentativeHardSSAW,
}
VARIANTS = tuple(VARIANT_CLASSES)
CONDITIONS = tuple(
    (corruption, severity)
    for corruption in corruptions.CORRUPTIONS
    for severity in corruptions.SEVERITIES
)
EXPECTED_BATCH_SIZES = (48, 48, 14)
FUTURE_INDICES = tuple(range(48, corruptions.TARGET_SAMPLES))
EXPECTED_FUTURE_SAMPLES = len(FUTURE_INDICES)
EXPECTED_FUTURE_CORRUPTED = 27
EXPECTED_TAU_Q = {
    0: 0.24952355027198792,
    1: 0.21749156713485718,
    2: 0.16671113669872284,
}
FULL_COMPONENT = "confidence_plus_margin_aware_hard_ssaw"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "paper_evidence_v5"
    / "har_12_to_16_future_sfc_audit_v3"
)


@dataclass(frozen=True)
class OnlineBatch:
    """The only batch fields visible to online/source-reference code."""

    data: Any
    indices: Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _membership_sha256(
    target_indices: Sequence[int], membership: Sequence[bool]
) -> str:
    """Hash a fixed subset by canonical future target-index membership."""

    indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    mask = np.asarray(membership, dtype=bool).reshape(-1)
    if len(indices) != len(mask):
        raise RuntimeError("subset membership index/mask lengths disagree")
    order = np.argsort(indices, kind="stable")
    ordered_indices = indices[order]
    ordered_mask = mask[order]
    if ordered_indices.tolist() != list(FUTURE_INDICES):
        raise RuntimeError("subset membership is not indexed by all future samples")
    selected = ordered_indices[ordered_mask].astype(int).tolist()
    return hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _production_logits(adapter: Any, data: Any) -> torch.Tensor:
    """Read logits in the deployed batch-BN mode without state/RNG drift."""

    model = adapter.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    device_data = move_data_to_device(data, device)
    rng = _capture_rng()
    try:
        return _predict_after_adaptation(adapter, {"data": device_data})
    finally:
        _restore_rng(rng)


def _top1_nll(logits: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(logits)
    if values.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    return -values.log_softmax(dim=1).amax(dim=1)


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    return float(
        f1_score(
            np.asarray(labels, dtype=np.int64),
            np.asarray(predictions, dtype=np.int64),
            labels=list(range(6)),
            average="macro",
            zero_division=0,
        )
    )


def _load_reference_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("dataset", "")).upper() == DATASET
            and str(row.get("scenario", "")) == SCENARIO
            and int(row.get("stream_seed", -1)) == STREAM_SEED
            and str(row.get("replaced_component", "")) == FULL_COMPONENT
            and str(row.get("status", "")).lower() == "ok"
            and int(row.get("source_seed", -1)) in SOURCE_SEEDS
        ]
    references: dict[int, dict[str, str]] = {}
    for row in rows:
        seed = int(row["source_seed"])
        if seed in references:
            raise RuntimeError(f"duplicate formal source identity for seed {seed}")
        references[seed] = row
    if set(references) != set(SOURCE_SEEDS):
        raise RuntimeError(
            f"formal source identities are incomplete: {sorted(references)}"
        )
    return references


def _runtime_hparams(flow_profile_json: Path) -> dict[str, Any]:
    hparams = get_hparams_class(DATASET)()
    runtime = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    profiles = load_paper_flow_profiles(flow_profile_json, datasets=[DATASET])
    runtime.update(profile_for_flow(profiles, DATASET, SCENARIO))
    runtime.update(
        {
            "spline_log_strength": 0.20,
            "enable_source_semantic_router": False,
            "dusafe_logging_mode": "evidence",
            "record_per_sample_evidence": False,
            "record_production_batch_diagnostics": True,
            "ssaw_candidate_cuda_graph": "off",
            "ssaw_production_decision_only": True,
        }
    )
    if int(runtime["batch_size"]) != 48 or int(runtime["steps"]) != 2:
        raise RuntimeError("HAR 12->16 batch/steps protocol changed")
    if not np.isclose(float(runtime["spline_log_strength"]), 0.20):
        raise RuntimeError("Future SFC audit requires alpha=0.20")
    if float(runtime["ssaw_auxiliary_weight"]) <= 0.0:
        raise RuntimeError("Future SFC audit requires a positive SSAW weight")
    return runtime


def _condition_slug(corruption: str, severity: str) -> str:
    return f"{corruption}_{severity}"


def _condition_loader(base_loader: Iterable[Any], corruption: str, severity: str):
    return corruptions.IndexStableBatchTransformLoader(
        base_loader,
        corruptions.CORRUPTION_REGISTRY[corruption],
        severity,
        sample_mask_fn=corruptions.exact_index_stable_mask_fn(
            corruptions.CORRUPTION_FRACTION,
            corruptions.CORRUPTION_SEED,
        ),
        meta={"corruption": corruption, "severity": severity},
        transform_seed=corruptions.GEOMETRY_SEED,
    )


def _instantiate_variant(
    trainer: Any,
    source_model: torch.nn.Module,
    variant_class: type,
) -> Any:
    model = copy.deepcopy(source_model)
    adapter = variant_class(
        trainer.dataset_configs,
        trainer.hparams,
        model,
        build_optimizer(trainer.hparams),
    ).to(trainer.device)
    normalization = getattr(
        trainer.src_train_dl.dataset, "normalization_stats", None
    )
    if hasattr(adapter, "load_source_normalization_reference"):
        if normalization is None:
            raise RuntimeError("SSAW requires fixed source normalization stats")
        adapter.load_source_normalization_reference(*normalization)
    if getattr(adapter, "enable_confidence_gate", False):
        adapter.load_source_confidence_reference(
            trainer.source_confidence_metadata
        )
    if getattr(adapter, "enable_source_semantic_gate", False):
        adapter.load_source_semantic_reference(trainer.source_semantic_metadata)
    return adapter


def _online_batch(batch: Any) -> OnlineBatch:
    if isinstance(batch, Mapping):
        data = batch.get("data")
        indices = batch.get("indices", batch.get("idx"))
    else:
        values = tuple(batch)
        if len(values) < 3:
            raise ValueError("online HAR batch requires data and target indices")
        # values[1] is intentionally never bound or inspected here.
        data, indices = values[0], values[2]
    if data is None or indices is None:
        raise ValueError("online HAR batch lacks data or target indices")
    return OnlineBatch(data=data, indices=indices)


def _canonical_batches(loader: Iterable[Any]) -> list[OnlineBatch]:
    batches = [_online_batch(batch) for batch in loader]
    # Batch geometry is an online protocol property.  Do not inspect target
    # labels merely to determine it.
    sizes = tuple(
        int(torch.as_tensor(batch.indices).view(-1).numel())
        for batch in batches
    )
    if sizes != EXPECTED_BATCH_SIZES:
        raise RuntimeError(
            f"registered HAR batches are {EXPECTED_BATCH_SIZES}, got {sizes}"
        )
    indices = [
        int(index)
        for batch in batches
        for index in torch.as_tensor(batch.indices).view(-1).tolist()
    ]
    if indices != list(range(corruptions.TARGET_SAMPLES)):
        raise RuntimeError("HAR target stream is not canonical 0..109")
    return batches


def _collect_source_reference(
    adapter: Any,
    raw_loader: Iterable[Any],
    corrupted_loader: Iterable[Any],
    *,
    source_seed: int,
    corruption: str,
    severity: str,
    threshold: float,
) -> pd.DataFrame:
    """Collect mixed-batch corrupted and leave-one-corruption-out source logits.

    Under TTBN, comparing an all-clean batch with a mixed-corruption batch
    confounds the focal sample's corruption with a different batch statistic.
    For each registered corrupted sample, the clean counterpart therefore
    restores only that sample while keeping every other sample and the batch
    boundary unchanged.
    """

    pre_reference_state = snapshot_state(adapter, cpu=True)
    pre_reference_state_sha256 = state_hash(pre_reference_state)
    model_hash_before = _state_sha256(adapter.model)
    traversal_rng = _capture_rng()
    rows: list[dict[str, Any]] = []
    try:
        # DataLoader iteration itself consumes torch RNG even with zero
        # workers.  Capture before materializing either stream.
        raw_batches = _canonical_batches(raw_loader)
        corr_batches = _canonical_batches(corrupted_loader)
        for batch_index, (raw_batch, corr_batch) in enumerate(
            zip(raw_batches, corr_batches, strict=True)
        ):
            raw_indices = torch.as_tensor(raw_batch.indices).view(-1).cpu().long()
            corr_indices = torch.as_tensor(corr_batch.indices).view(-1).cpu().long()
            if not torch.equal(raw_indices, corr_indices):
                raise RuntimeError("clean/corrupted source batch indices diverged")
            corr_logits = _production_logits(adapter, corr_batch.data).detach().cpu()
            corr_prediction = corr_logits.argmax(dim=1).long()
            corr_nll = _top1_nll(corr_logits).cpu()
            loco_prediction = torch.full_like(corr_prediction, -1)
            loco_nll = torch.full_like(corr_nll, float("nan"))
            raw_primary = extract_primary_tensor(raw_batch.data)
            corr_primary = extract_primary_tensor(corr_batch.data)
            for local_index, target_index in enumerate(raw_indices.tolist()):
                if int(target_index) not in corruptions.CORRUPTED_INDICES:
                    continue
                restored_primary = corr_primary.clone()
                restored_primary[local_index] = raw_primary[local_index]
                restored_data = replace_primary_tensor(
                    corr_batch.data, restored_primary
                )
                restored_logits = _production_logits(
                    adapter, restored_data
                ).detach().cpu()
                loco_prediction[local_index] = restored_logits[local_index].argmax()
                loco_nll[local_index] = _top1_nll(
                    restored_logits[local_index : local_index + 1]
                )[0]
            for local_index, target_index in enumerate(raw_indices.tolist()):
                registered = int(target_index) in corruptions.CORRUPTED_INDICES
                rows.append(
                    {
                        "protocol": PROTOCOL,
                        "dataset": DATASET,
                        "scenario": SCENARIO,
                        "source_seed": int(source_seed),
                        "stream_seed": STREAM_SEED,
                        "corruption": corruption,
                        "severity": severity,
                        "batch_index": batch_index,
                        "local_batch_index": local_index,
                        "target_index": int(target_index),
                        "registered_corrupted": bool(registered),
                        "source_corrupted_prediction": int(
                            corr_prediction[local_index]
                        ),
                        "source_corrupted_top1_nll": float(corr_nll[local_index]),
                        "source_corrupted_top1_confidence": float(
                            math.exp(-float(corr_nll[local_index]))
                        ),
                        "source_loco_clean_prediction": (
                            int(loco_prediction[local_index]) if registered else -1
                        ),
                        "source_loco_clean_top1_nll": (
                            float(loco_nll[local_index]) if registered else math.nan
                        ),
                        "confidence_nll_threshold_tau_q": float(threshold),
                        "source_reference_mode": (
                            "frozen_weights_deployment_batch_bn_reference"
                        ),
                        "clean_counterpart_mode": (
                            "leave_one_corruption_out_same_mixed_batch"
                        ),
                    }
                )
    finally:
        _restore_rng(traversal_rng)
    post_reference_state = snapshot_state(adapter, cpu=True)
    post_reference_state_sha256 = state_hash(post_reference_state)
    source_state_unchanged = (
        pre_reference_state_sha256 == post_reference_state_sha256
    )
    source_rng_unchanged = _nested_equal(
        pre_reference_state.rng_state, post_reference_state.rng_state
    )
    if not source_state_unchanged:
        raise RuntimeError("source reference changed full adapter state")
    if not source_rng_unchanged:
        raise RuntimeError("source reference changed process RNG state")
    if _state_sha256(adapter.model) != model_hash_before:
        raise RuntimeError("source reference evaluation changed model state")
    frame = pd.DataFrame(rows).sort_values("target_index").reset_index(drop=True)
    frame["source_reference_pre_state_sha256"] = pre_reference_state_sha256
    frame["source_reference_post_state_sha256"] = post_reference_state_sha256
    frame["source_reference_state_unchanged"] = source_state_unchanged
    frame["source_reference_rng_unchanged"] = source_rng_unchanged
    if len(frame) != corruptions.TARGET_SAMPLES:
        raise RuntimeError("source reference sample count is incomplete")
    if int(frame["registered_corrupted"].sum()) != 55:
        raise RuntimeError("source reference corruption mask is not 55/110")
    if frame.loc[
        frame["registered_corrupted"], "source_loco_clean_top1_nll"
    ].isna().any():
        raise RuntimeError("LOCO clean counterpart is missing on corrupted samples")
    return frame


def _collect_posthoc_labels(loader: Iterable[Any]) -> pd.DataFrame:
    """Collect immutable labels only after every online trajectory finished."""

    rows = []
    observed_sizes = []
    for batch_index, batch in enumerate(loader):
        if isinstance(batch, Mapping):
            labels_raw = batch.get("labels")
            indices_raw = batch.get("indices", batch.get("idx"))
        else:
            values = tuple(batch)
            if len(values) < 3:
                raise ValueError("post-hoc HAR batch requires labels and indices")
            labels_raw, indices_raw = values[1], values[2]
        indices = torch.as_tensor(indices_raw).view(-1).cpu().long()
        labels = torch.as_tensor(labels_raw).view(-1).cpu().long()
        observed_sizes.append(int(indices.numel()))
        if len(indices) != len(labels):
            raise RuntimeError("post-hoc label/index lengths disagree")
        for local_index, target_index in enumerate(indices.tolist()):
            rows.append(
                {
                    "target_index": int(target_index),
                    "true_label": int(labels[local_index]),
                    "batch_index": batch_index,
                    "local_batch_index": local_index,
                }
            )
    if tuple(observed_sizes) != EXPECTED_BATCH_SIZES:
        raise RuntimeError("post-hoc label batch boundaries changed")
    frame = pd.DataFrame(rows).sort_values("target_index").reset_index(drop=True)
    if frame["target_index"].tolist() != list(range(corruptions.TARGET_SAMPLES)):
        raise RuntimeError("post-hoc labels do not cover canonical target indices")
    return frame


def _run_independent_future_trajectory(
    adapter: Any,
    loader: Iterable[Any],
    *,
    variant: str,
    source_seed: int,
    corruption: str,
    severity: str,
) -> pd.DataFrame:
    batches = _canonical_batches(loader)
    rows: list[dict[str, Any]] = []
    # The final batch has no unseen successor.  Adapting it would add compute
    # and mutate state without producing a registered future endpoint.
    for update_batch_index, batch in enumerate(batches[:-1]):
        data = move_data_to_device(batch.data, next(adapter.model.parameters()).device)
        indices = torch.as_tensor(batch.indices).view(-1).cpu().long()
        # Labels are deliberately absent from both the input mapping and the
        # adapter call.  They remain only in the CPU BatchView for later
        # post-hoc metrics.
        _ = adapter(
            {
                "data": data,
                "meta": {"trg_idx": indices.tolist()},
            },
            indices,
        )
        future_batch_index = update_batch_index + 1
        if future_batch_index >= len(batches):
            continue
        future = batches[future_batch_index]
        pre_eval_state = snapshot_state(adapter, cpu=True)
        pre_eval_state_sha256 = state_hash(pre_eval_state)
        model_hash_before = _state_sha256(adapter.model)
        logits = _production_logits(adapter, future.data).detach().cpu()
        post_eval_state = snapshot_state(adapter, cpu=True)
        post_eval_state_sha256 = state_hash(post_eval_state)
        eval_state_unchanged = pre_eval_state_sha256 == post_eval_state_sha256
        eval_rng_unchanged = _nested_equal(
            pre_eval_state.rng_state, post_eval_state.rng_state
        )
        if not eval_state_unchanged:
            raise RuntimeError("future evaluation changed full adapter state")
        if not eval_rng_unchanged:
            raise RuntimeError("future evaluation changed process RNG state")
        if _state_sha256(adapter.model) != model_hash_before:
            raise RuntimeError("future read-only evaluation changed model state")
        predictions = logits.argmax(dim=1).long()
        top1_nll = _top1_nll(logits).cpu()
        future_indices = torch.as_tensor(future.indices).view(-1).cpu().long()
        for local_index, target_index in enumerate(future_indices.tolist()):
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "dataset": DATASET,
                    "scenario": SCENARIO,
                    "source_seed": int(source_seed),
                    "stream_seed": STREAM_SEED,
                    "corruption": corruption,
                    "severity": severity,
                    "variant": variant,
                    "update_batch_index": update_batch_index,
                    "future_batch_index": future_batch_index,
                    "future_local_batch_index": local_index,
                    "target_index": int(target_index),
                    "future_prediction": int(predictions[local_index]),
                    "future_top1_nll": float(top1_nll[local_index]),
                    "future_eval_pre_state_sha256": pre_eval_state_sha256,
                    "future_eval_post_state_sha256": post_eval_state_sha256,
                    "future_eval_state_unchanged": eval_state_unchanged,
                    "future_eval_rng_unchanged": eval_rng_unchanged,
                }
            )
    frame = pd.DataFrame(rows).sort_values("target_index").reset_index(drop=True)
    if len(frame) != EXPECTED_FUTURE_SAMPLES:
        raise RuntimeError(
            f"future trajectory expected {EXPECTED_FUTURE_SAMPLES} samples"
        )
    if frame["target_index"].tolist() != list(FUTURE_INDICES):
        raise RuntimeError("future samples were not evaluated exactly once")
    return frame


def _cell_output_dir(
    root: Path, source_seed: int, corruption: str, severity: str
) -> Path:
    return (
        root
        / "cells"
        / _condition_slug(corruption, severity)
        / f"source_seed_{int(source_seed)}"
    )


def _cell_complete(path: Path, protocol_sha256: str) -> bool:
    manifest_path = path / "manifest.json"
    source_path = path / "source_reference_samples.csv"
    future_path = path / "future_predictions.csv"
    labels_path = path / "posthoc_labels.csv"
    if not (
        manifest_path.is_file()
        and source_path.is_file()
        and future_path.is_file()
        and labels_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return bool(
            manifest.get("status") == "complete"
            and manifest.get("protocol_sha256") == protocol_sha256
            and manifest.get("source_reference_sha256") == _sha256_file(source_path)
            and manifest.get("future_predictions_sha256") == _sha256_file(future_path)
            and manifest.get("posthoc_labels_sha256") == _sha256_file(labels_path)
        )
    except (OSError, ValueError, TypeError):
        return False


def _protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = Path(args.reference_main_csv).resolve()
    flow_profile = Path(args.flow_profile_json).resolve()
    data_root = Path(args.data_path).resolve()
    dataset_root = data_root / DATASET
    data_artifacts = {
        name: dataset_root / name
        for name in ("train_12.pt", "test_12.pt", "train_16.pt", "test_16.pt")
    }
    missing_data = [str(path) for path in data_artifacts.values() if not path.is_file()]
    if missing_data:
        raise FileNotFoundError(
            "registered HAR data artifacts are missing: " + ", ".join(missing_data)
        )
    references = _load_reference_rows(reference_path)
    runtime = _runtime_hparams(flow_profile)
    source_artifacts = {}
    for seed, reference in references.items():
        checkpoint = Path(reference["source_checkpoint_path"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        source_artifacts[str(seed)] = {
            "source_model_sha256": reference["source_model_sha256"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_file_sha256": _sha256_file(checkpoint),
            "source_config": json.loads(reference["source_config"]),
        }
    files = {
        "runner": Path(__file__).resolve(),
        "corruption": ROOT / "dataloader" / "har_cross_dataset_corruptions.py",
        "representative_variants": (
            ROOT / "algorithms" / "representative_causal_ablation.py"
        ),
        "replacement_variant": (
            ROOT / "algorithms" / "dusafe_replacement_ablation.py"
        ),
        "production_algorithm": ROOT / "algorithms" / "dusafe.py",
        "flow_profile": flow_profile,
        "source_reference": reference_path,
    }
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "registered_before_execution",
        "registered_at_utc": _utc_now(),
        "dataset": DATASET,
        "scenario": SCENARIO,
        "backbone": str(args.backbone),
        "device": str(args.device),
        "data_root": str(data_root),
        "data_artifacts": {
            name: {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for name, path in data_artifacts.items()
        },
        "source_seeds": list(SOURCE_SEEDS),
        "expected_confidence_nll_threshold_tau_q": {
            str(seed): value for seed, value in EXPECTED_TAU_Q.items()
        },
        "stream_seed": STREAM_SEED,
        "variants": list(VARIANTS),
        "variant_classes": {
            name: variant.__name__ for name, variant in VARIANT_CLASSES.items()
        },
        "random_spline_selection": (
            RepresentativeRandomEligibleSpline.spline_selection_mode
        ),
        "corruptions": list(corruptions.CORRUPTIONS),
        "severities": list(corruptions.SEVERITIES),
        "corruption_fraction": 0.5,
        "corruption_seed": corruptions.CORRUPTION_SEED,
        "geometry_seed": corruptions.GEOMETRY_SEED,
        "corruption_mask_sha256": corruptions.corruption_mask_sha256(),
        "corrupted_samples": 55,
        "deployment_batch_sizes": list(EXPECTED_BATCH_SIZES),
        "future_sample_indices": [48, 109],
        "future_samples": EXPECTED_FUTURE_SAMPLES,
        "future_corrupted_samples_per_condition": EXPECTED_FUTURE_CORRUPTED,
        "future_evaluation": (
            "adapt B0 then evaluate untouched B1; adapt B1 then evaluate "
            "untouched B2; do not adapt terminal B2"
        ),
        "updates_per_online_trajectory": 2,
        "online_trajectories": "independent_per_variant",
        "read_only_evaluation_audit": (
            "full adapter state (model, buffers, optimizer, runtime, training "
            "flags) and process RNG must be identical before/after"
        ),
        "source_reference_mode": (
            "frozen_weights_deployment_batch_bn_reference"
        ),
        "clean_counterpart_mode": (
            "leave_one_corruption_out_same_mixed_batch"
        ),
        "sfc_definition": {
            "registered_corrupted": True,
            "source_loco_clean_prediction_correct": True,
            "source_corrupted_prediction_wrong": True,
            "source_corrupted_top1_nll_le_tau_q": True,
        },
        "reliable_definition": {
            "registered_corrupted": True,
            "source_corrupted_prediction_correct": True,
            "source_corrupted_top1_nll_le_tau_q": True,
        },
        "strict_reliable_diagnostic_adds_loco_clean_correct": True,
        "metric_definitions": {
            "sfc_correction": "adapted correct / fixed SFC subset",
            "remaining_hcw": (
                "adapted wrong and adapted top1 NLL <= frozen tau_q / fixed SFC subset"
            ),
            "remaining_wrong_low_confidence": (
                "adapted wrong and adapted top1 NLL > frozen tau_q / fixed SFC subset"
            ),
            "reliable_r_to_w": "adapted wrong / fixed reliable subset",
            "corrupted_f1": "macro-F1 on all 27 future corrupted samples",
        },
        "sfc_three_way_partition": (
            "corrected + remaining HCW + remaining wrong low-confidence = "
            "fixed SFC denominator"
        ),
        "fixed_subset_identity": "sha256 of canonical future target-index membership",
        "empty_subset_policy": (
            "rate=NaN; preserve numerator, denominator, and valid-cell count"
        ),
        "aggregation_unit": (
            "source seed; within seed sum subset numerators/denominators "
            "across eight conditions and mean corrupted F1"
        ),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_posthoc_grouping_and_metrics": True,
        "posthoc_grouping_location": "aggregate/finalizer only",
        "confirmatory": False,
        "evidence_role": "target-selected descriptive future SFC audit",
        "runtime_hparams": runtime,
        "source_artifacts": source_artifacts,
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in files.items()
        },
    }
    signature_payload = copy.deepcopy(payload)
    signature_payload.pop("registered_at_utc")
    signature_payload.pop("protocol_sha256", None)
    payload["protocol_sha256"] = hashlib.sha256(
        json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload


def run_cell(args: argparse.Namespace) -> None:
    corruption = str(args.corruption)
    severity = str(args.severity)
    source_seed = int(args.source_seed)
    if (corruption, severity) not in CONDITIONS:
        raise ValueError("unregistered HAR corruption condition")
    if source_seed not in SOURCE_SEEDS:
        raise ValueError("unregistered source seed")
    payload = _protocol_payload(args)
    if args.protocol_sha256 and args.protocol_sha256 != payload["protocol_sha256"]:
        raise RuntimeError("cell protocol signature disagrees with parent")
    references = _load_reference_rows(Path(args.reference_main_csv).resolve())
    reference = references[source_seed]
    runtime = _runtime_hparams(Path(args.flow_profile_json).resolve())
    source_config = json.loads(reference["source_config"])
    checkpoint = Path(reference["source_checkpoint_path"]).resolve()
    output = _cell_output_dir(
        Path(args.output_dir).resolve(), source_seed, corruption, severity
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {
            "protocol": PROTOCOL,
            "protocol_sha256": payload["protocol_sha256"],
            "status": "running",
            "source_seed": source_seed,
            "corruption": corruption,
            "severity": severity,
            "started_at_utc": _utc_now(),
        },
        output / "manifest.json",
    )

    source_id, target_id = SCENARIO.split("->", 1)
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=DATASET,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name=(
            f"future_sfc_s{source_seed}_{_condition_slug(corruption, severity)}"
        ),
        seed=STREAM_SEED,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        pretrained_checkpoint=str(checkpoint),
    )
    trainer.get_tta_model_class = lambda: RepresentativeHardSSAW
    adapters: dict[str, Any] = {}
    source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(runtime)
        hard_adapter, source_model = create_tta_model(
            trainer, source_id, target_id, run_seed=STREAM_SEED
        )
        adapters["dusafe"] = hard_adapter
        adapters["confidence_only"] = _instantiate_variant(
            trainer, source_model, RepresentativeConfidenceRaw
        )
        adapters["random_spline"] = _instantiate_variant(
            trainer, source_model, RepresentativeRandomEligibleSpline
        )
        source_hash = _state_sha256(source_model)
        if source_hash != str(reference["source_model_sha256"]):
            raise RuntimeError(
                f"source checkpoint mismatch: {source_hash} != "
                f"{reference['source_model_sha256']}"
            )
        model_hashes = {
            name: _state_sha256(adapter.model)
            for name, adapter in adapters.items()
        }
        if set(model_hashes.values()) != {source_hash}:
            raise RuntimeError("variants do not share the frozen source weights")
        thresholds = {
            name: float(
                adapter.confidence_nll_threshold.detach().cpu().item()
            )
            for name, adapter in adapters.items()
        }
        if max(thresholds.values()) - min(thresholds.values()) > 1e-12:
            raise RuntimeError("variants do not share source tau_q")
        threshold = thresholds["dusafe"]
        if not math.isfinite(threshold):
            raise RuntimeError("source tau_q is non-finite")
        if not math.isclose(
            threshold,
            EXPECTED_TAU_Q[source_seed],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"seed {source_seed} tau_q mismatch: {threshold} != "
                f"{EXPECTED_TAU_Q[source_seed]}"
            )

        source_frame = _collect_source_reference(
            hard_adapter,
            trainer.trg_whole_dl,
            _condition_loader(trainer.trg_whole_dl, corruption, severity),
            source_seed=source_seed,
            corruption=corruption,
            severity=severity,
            threshold=threshold,
        )
        future_frames = []
        for name in VARIANTS:
            fix_randomness(STREAM_SEED)
            future_frames.append(
                _run_independent_future_trajectory(
                    adapters[name],
                    _condition_loader(
                        trainer.trg_whole_dl, corruption, severity
                    ),
                    variant=name,
                    source_seed=source_seed,
                    corruption=corruption,
                    severity=severity,
                )
            )
        future_frame = pd.concat(future_frames, ignore_index=True)
        # This is deliberately after all three independent online
        # trajectories.  No method can observe these labels while updating.
        posthoc_labels = _collect_posthoc_labels(trainer.trg_whole_dl)
        source_path = output / "source_reference_samples.csv"
        future_path = output / "future_predictions.csv"
        labels_path = output / "posthoc_labels.csv"
        atomic_write_csv(source_frame, source_path)
        atomic_write_csv(future_frame, future_path)
        atomic_write_csv(posthoc_labels, labels_path)
        _atomic_json(
            {
                "protocol": PROTOCOL,
                "protocol_sha256": payload["protocol_sha256"],
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "dataset": DATASET,
                "scenario": SCENARIO,
                "source_seed": source_seed,
                "stream_seed": STREAM_SEED,
                "corruption": corruption,
                "severity": severity,
                "variants": list(VARIANTS),
                "source_model_sha256": source_hash,
                "source_checkpoint_path": str(checkpoint),
                "confidence_nll_threshold_tau_q": threshold,
                "source_reference_mode": (
                    "frozen_weights_deployment_batch_bn_reference"
                ),
                "clean_counterpart_mode": (
                    "leave_one_corruption_out_same_mixed_batch"
                ),
                "random_spline_selection": (
                    RepresentativeRandomEligibleSpline.spline_selection_mode
                ),
                "deployment_batch_sizes": list(EXPECTED_BATCH_SIZES),
                "registered_corrupted_samples": 55,
                "registered_future_samples": EXPECTED_FUTURE_SAMPLES,
                "registered_future_corrupted_samples": (
                    EXPECTED_FUTURE_CORRUPTED
                ),
                "corruption_mask_sha256": corruptions.corruption_mask_sha256(),
                "source_reference_rows": int(len(source_frame)),
                "future_prediction_rows": int(len(future_frame)),
                "source_reference_sha256": _sha256_file(source_path),
                "future_predictions_sha256": _sha256_file(future_path),
                "posthoc_labels_sha256": _sha256_file(labels_path),
                "target_labels_passed_to_online_adapter": False,
                "target_labels_used_for_online_decision": False,
                "target_labels_persisted_for_posthoc_finalizer": True,
                "posthoc_subset_flags_computed_in_cell": False,
                "posthoc_labels_collected_after_all_trajectories": True,
                "independent_online_trajectories": True,
                "future_samples_evaluated_once": True,
                "future_evaluation_before_own_update": True,
                "updates_per_trajectory": 2,
                "source_reference_state_preserved": True,
                "source_reference_full_adapter_state_preserved": bool(
                    source_frame["source_reference_state_unchanged"].all()
                ),
                "source_reference_rng_preserved": bool(
                    source_frame["source_reference_rng_unchanged"].all()
                ),
                "future_evaluation_full_adapter_state_preserved": bool(
                    future_frame["future_eval_state_unchanged"].all()
                ),
                "future_evaluation_rng_preserved": bool(
                    future_frame["future_eval_rng_unchanged"].all()
                ),
            },
            output / "manifest.json",
        )
    finally:
        cleanup_trainer(
            trainer,
            *adapters.values(),
            source_model,
            close_summary=True,
        )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _boolean_values(values: pd.Series, *, field: str) -> np.ndarray:
    """Parse a persisted boolean column without treating ``"False"`` as true."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized).difference(allowed))
    if unexpected:
        raise RuntimeError(f"invalid boolean values in {field}: {unexpected}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def _cell_metrics(
    source: pd.DataFrame,
    future: pd.DataFrame,
    posthoc_labels: pd.DataFrame,
) -> pd.DataFrame:
    required_source = {
        "target_index",
        "registered_corrupted",
        "source_corrupted_prediction",
        "source_corrupted_top1_nll",
        "source_loco_clean_prediction",
        "confidence_nll_threshold_tau_q",
    }
    required_future = {
        "target_index",
        "variant",
        "future_prediction",
        "future_top1_nll",
    }
    if not required_source.issubset(source) or not required_future.issubset(future):
        raise RuntimeError("cell sample schema is incomplete")
    if not {"target_index", "true_label"}.issubset(posthoc_labels):
        raise RuntimeError("post-hoc label schema is incomplete")
    if posthoc_labels.duplicated("target_index").any():
        raise RuntimeError("post-hoc labels contain duplicate target indices")
    source_with_labels = source.merge(
        posthoc_labels[["target_index", "true_label"]],
        on="target_index",
        how="left",
        validate="one_to_one",
    )
    merged = future.merge(
        source_with_labels[[*required_source, "true_label"]],
        on="target_index",
        how="left",
        validate="many_to_one",
    )
    if merged[list(required_source - {"target_index"})].isna().all(axis=None):
        raise RuntimeError("future/source sample join failed")
    rows = []
    for variant, group in merged.groupby("variant", sort=False):
        group = group.sort_values("target_index").reset_index(drop=True)
        target_indices = group["target_index"].to_numpy(dtype=np.int64)
        if target_indices.tolist() != list(FUTURE_INDICES):
            raise RuntimeError("metric input does not cover canonical future indices")
        registered = _boolean_values(
            group["registered_corrupted"], field="registered_corrupted"
        )
        if int(registered.sum()) != EXPECTED_FUTURE_CORRUPTED:
            raise RuntimeError("future corruption mask is not 27/62")
        labels = group["true_label"].to_numpy(dtype=np.int64)
        source_corr = group["source_corrupted_prediction"].to_numpy(
            dtype=np.int64
        )
        source_clean = group["source_loco_clean_prediction"].to_numpy(
            dtype=np.int64
        )
        source_nll = group["source_corrupted_top1_nll"].to_numpy(dtype=float)
        threshold = group["confidence_nll_threshold_tau_q"].to_numpy(dtype=float)
        adapted = group["future_prediction"].to_numpy(dtype=np.int64)
        adapted_nll = group["future_top1_nll"].to_numpy(dtype=float)
        if not np.isfinite(adapted_nll).all():
            raise RuntimeError("future top1 NLL contains a non-finite value")
        sfc = (
            registered
            & (source_clean == labels)
            & (source_corr != labels)
            & (source_nll <= threshold)
        )
        reliable = (
            registered
            & (source_corr == labels)
            & (source_nll <= threshold)
        )
        strict_reliable = reliable & (source_clean == labels)
        sfc_den = int(sfc.sum())
        sfc_corrected = int((sfc & (adapted == labels)).sum())
        remaining_hcw = int(
            (sfc & (adapted != labels) & (adapted_nll <= threshold)).sum()
        )
        remaining_wrong_low_confidence = int(
            (sfc & (adapted != labels) & (adapted_nll > threshold)).sum()
        )
        if (
            sfc_corrected
            + remaining_hcw
            + remaining_wrong_low_confidence
            != sfc_den
        ):
            raise RuntimeError("SFC three-way outcome partition is incomplete")
        reliable_den = int(reliable.sum())
        reliable_r_to_w = int((reliable & (adapted != labels)).sum())
        strict_den = int(strict_reliable.sum())
        strict_r_to_w = int((strict_reliable & (adapted != labels)).sum())
        corrupted_f1 = _macro_f1(labels[registered], adapted[registered])
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "scenario": str(group["scenario"].iloc[0]),
                "source_seed": int(group["source_seed"].iloc[0]),
                "stream_seed": int(group["stream_seed"].iloc[0]),
                "corruption": str(group["corruption"].iloc[0]),
                "severity": str(group["severity"].iloc[0]),
                "variant": str(variant),
                "sfc_correction_numerator": sfc_corrected,
                "sfc_denominator": sfc_den,
                "sfc_correction": _safe_rate(sfc_corrected, sfc_den),
                "sfc_status": "ok" if sfc_den else "empty_subset",
                "sfc_subset_sha256": _membership_sha256(
                    target_indices, sfc
                ),
                "remaining_hcw_numerator": remaining_hcw,
                "remaining_hcw_denominator": sfc_den,
                "remaining_hcw": _safe_rate(remaining_hcw, sfc_den),
                "remaining_wrong_low_confidence_numerator": (
                    remaining_wrong_low_confidence
                ),
                "remaining_wrong_low_confidence_denominator": sfc_den,
                "remaining_wrong_low_confidence": _safe_rate(
                    remaining_wrong_low_confidence, sfc_den
                ),
                "sfc_three_way_partition_passed": True,
                "reliable_r_to_w_numerator": reliable_r_to_w,
                "reliable_denominator": reliable_den,
                "reliable_r_to_w": _safe_rate(reliable_r_to_w, reliable_den),
                "reliable_status": (
                    "ok" if reliable_den else "empty_subset"
                ),
                "reliable_subset_sha256": _membership_sha256(
                    target_indices, reliable
                ),
                "strict_reliable_r_to_w_numerator": strict_r_to_w,
                "strict_reliable_denominator": strict_den,
                "strict_reliable_r_to_w": _safe_rate(strict_r_to_w, strict_den),
                "strict_reliable_status": (
                    "ok" if strict_den else "empty_subset"
                ),
                "strict_reliable_subset_sha256": _membership_sha256(
                    target_indices, strict_reliable
                ),
                "corrupted_f1": corrupted_f1,
                "future_corrupted_samples": int(registered.sum()),
            }
        )
    return pd.DataFrame(rows)


def _source_sanity_metrics(
    source: pd.DataFrame, posthoc_labels: pd.DataFrame
) -> dict[str, Any]:
    """Verify the source-defined subset identities before method comparison."""

    merged = source.merge(
        posthoc_labels[["target_index", "true_label"]],
        on="target_index",
        how="left",
        validate="one_to_one",
    )
    merged = merged[merged["target_index"].isin(FUTURE_INDICES)].sort_values(
        "target_index"
    )
    target_indices = merged["target_index"].to_numpy(dtype=np.int64)
    if target_indices.tolist() != list(FUTURE_INDICES):
        raise RuntimeError("source sanity input lacks canonical future samples")
    registered = _boolean_values(
        merged["registered_corrupted"], field="registered_corrupted"
    )
    if int(registered.sum()) != EXPECTED_FUTURE_CORRUPTED:
        raise RuntimeError("source sanity corruption mask is not 27/62")
    labels = merged["true_label"].to_numpy(dtype=np.int64)
    source_corr = merged["source_corrupted_prediction"].to_numpy(dtype=np.int64)
    source_clean = merged["source_loco_clean_prediction"].to_numpy(dtype=np.int64)
    source_nll = merged["source_corrupted_top1_nll"].to_numpy(dtype=float)
    threshold = merged["confidence_nll_threshold_tau_q"].to_numpy(dtype=float)
    sfc = (
        registered
        & (source_clean == labels)
        & (source_corr != labels)
        & (source_nll <= threshold)
    )
    reliable = (
        registered
        & (source_corr == labels)
        & (source_nll <= threshold)
    )
    strict_reliable = reliable & (source_clean == labels)
    sfc_den = int(sfc.sum())
    reliable_den = int(reliable.sum())
    strict_den = int(strict_reliable.sum())
    source_sfc_corrected = int((sfc & (source_corr == labels)).sum())
    source_remaining_hcw = int(
        (sfc & (source_corr != labels) & (source_nll <= threshold)).sum()
    )
    source_reliable_r_to_w = int(
        (reliable & (source_corr != labels)).sum()
    )
    source_strict_reliable_r_to_w = int(
        (strict_reliable & (source_corr != labels)).sum()
    )
    passed = bool(
        source_sfc_corrected == 0
        and source_remaining_hcw == sfc_den
        and source_reliable_r_to_w == 0
        and source_strict_reliable_r_to_w == 0
    )
    if not passed:
        raise RuntimeError("source subset sanity identity failed")
    return {
        "dataset": str(merged["dataset"].iloc[0]),
        "scenario": str(merged["scenario"].iloc[0]),
        "source_seed": int(merged["source_seed"].iloc[0]),
        "stream_seed": int(merged["stream_seed"].iloc[0]),
        "corruption": str(merged["corruption"].iloc[0]),
        "severity": str(merged["severity"].iloc[0]),
        "sfc_denominator": sfc_den,
        "sfc_status": "ok" if sfc_den else "empty_subset",
        "sfc_subset_sha256": _membership_sha256(target_indices, sfc),
        "source_sfc_correction_numerator": source_sfc_corrected,
        "source_sfc_correction": _safe_rate(source_sfc_corrected, sfc_den),
        "source_remaining_hcw_numerator": source_remaining_hcw,
        "source_remaining_hcw": _safe_rate(source_remaining_hcw, sfc_den),
        "reliable_denominator": reliable_den,
        "reliable_status": "ok" if reliable_den else "empty_subset",
        "reliable_subset_sha256": _membership_sha256(
            target_indices, reliable
        ),
        "source_reliable_r_to_w_numerator": source_reliable_r_to_w,
        "source_reliable_r_to_w": _safe_rate(
            source_reliable_r_to_w, reliable_den
        ),
        "strict_reliable_denominator": strict_den,
        "strict_reliable_status": "ok" if strict_den else "empty_subset",
        "strict_reliable_subset_sha256": _membership_sha256(
            target_indices, strict_reliable
        ),
        "source_strict_reliable_r_to_w_numerator": (
            source_strict_reliable_r_to_w
        ),
        "source_strict_reliable_r_to_w": _safe_rate(
            source_strict_reliable_r_to_w, strict_den
        ),
        "source_sanity_passed": passed,
    }


def _ratio_summary(
    frame: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    output: str,
) -> dict[str, Any]:
    num = int(frame[numerator].sum())
    den = int(frame[denominator].sum())
    valid = int((frame[denominator] > 0).sum())
    return {
        f"{output}_numerator": num,
        f"{output}_denominator": den,
        f"{output}_valid_cells": valid,
        output: _safe_rate(num, den),
    }


def _aggregate_final(
    condition_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (variant, source_seed), group in condition_metrics.groupby(
        ["variant", "source_seed"], sort=False
    ):
        if len(group) != len(CONDITIONS):
            raise RuntimeError("method/seed does not contain eight conditions")
        seed_rows.append(
            {
                "variant": variant,
                "source_seed": int(source_seed),
                **_ratio_summary(
                    group,
                    numerator="sfc_correction_numerator",
                    denominator="sfc_denominator",
                    output="sfc_correction",
                ),
                **_ratio_summary(
                    group,
                    numerator="remaining_hcw_numerator",
                    denominator="remaining_hcw_denominator",
                    output="remaining_hcw",
                ),
                **_ratio_summary(
                    group,
                    numerator="remaining_wrong_low_confidence_numerator",
                    denominator="remaining_wrong_low_confidence_denominator",
                    output="remaining_wrong_low_confidence",
                ),
                **_ratio_summary(
                    group,
                    numerator="reliable_r_to_w_numerator",
                    denominator="reliable_denominator",
                    output="reliable_r_to_w",
                ),
                **_ratio_summary(
                    group,
                    numerator="strict_reliable_r_to_w_numerator",
                    denominator="strict_reliable_denominator",
                    output="strict_reliable_r_to_w",
                ),
                "corrupted_f1": float(group["corrupted_f1"].mean()),
                "corrupted_f1_condition_count": int(len(group)),
            }
        )
    seed_summary = pd.DataFrame(seed_rows).sort_values(
        ["variant", "source_seed"]
    )

    paper_rows = []
    for variant, group in seed_summary.groupby("variant", sort=False):
        record: dict[str, Any] = {
            "variant": variant,
            "source_seed_count": int(group["source_seed"].nunique()),
            "sfc_denominators_by_source_seed": json.dumps(
                {
                    str(int(row.source_seed)): int(
                        row.sfc_correction_denominator
                    )
                    for row in group.itertuples()
                },
                sort_keys=True,
            ),
            "reliable_denominators_by_source_seed": json.dumps(
                {
                    str(int(row.source_seed)): int(
                        row.reliable_r_to_w_denominator
                    )
                    for row in group.itertuples()
                },
                sort_keys=True,
            ),
        }
        for metric in (
            "sfc_correction",
            "remaining_hcw",
            "remaining_wrong_low_confidence",
            "reliable_r_to_w",
            "strict_reliable_r_to_w",
            "corrupted_f1",
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.dropna()
            record[f"{metric}_mean"] = (
                float(valid.mean()) if len(valid) else math.nan
            )
            record[f"{metric}_std"] = (
                float(valid.std(ddof=1)) if len(valid) > 1 else math.nan
            )
            record[f"{metric}_valid_source_seeds"] = int(len(valid))
        paper_rows.append(record)
    paper_summary = pd.DataFrame(paper_rows)

    condition_rows = []
    for (variant, corruption, severity), group in condition_metrics.groupby(
        ["variant", "corruption", "severity"], sort=False
    ):
        record = {
            "variant": variant,
            "corruption": corruption,
            "severity": severity,
        }
        for metric in (
            "sfc_correction",
            "remaining_hcw",
            "remaining_wrong_low_confidence",
            "reliable_r_to_w",
            "strict_reliable_r_to_w",
            "corrupted_f1",
        ):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else math.nan
            )
            record[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
            record[f"{metric}_valid_source_seeds"] = int(len(values))
        condition_rows.append(record)
    condition_summary = pd.DataFrame(condition_rows)

    pooled_rows = []
    for variant, group in condition_metrics.groupby("variant", sort=False):
        pooled_rows.append(
            {
                "variant": variant,
                **_ratio_summary(
                    group,
                    numerator="sfc_correction_numerator",
                    denominator="sfc_denominator",
                    output="sfc_correction",
                ),
                **_ratio_summary(
                    group,
                    numerator="remaining_hcw_numerator",
                    denominator="remaining_hcw_denominator",
                    output="remaining_hcw",
                ),
                **_ratio_summary(
                    group,
                    numerator="remaining_wrong_low_confidence_numerator",
                    denominator="remaining_wrong_low_confidence_denominator",
                    output="remaining_wrong_low_confidence",
                ),
                **_ratio_summary(
                    group,
                    numerator="reliable_r_to_w_numerator",
                    denominator="reliable_denominator",
                    output="reliable_r_to_w",
                ),
                **_ratio_summary(
                    group,
                    numerator="strict_reliable_r_to_w_numerator",
                    denominator="strict_reliable_denominator",
                    output="strict_reliable_r_to_w",
                ),
                "corrupted_f1_mean_across_24_condition_seed_cells": float(
                    group["corrupted_f1"].mean()
                ),
                "condition_seed_cells": int(len(group)),
            }
        )
    pooled_summary = pd.DataFrame(pooled_rows)
    return seed_summary, paper_summary, condition_summary, pooled_summary


def _paper_markdown(paper: pd.DataFrame) -> str:
    labels = {
        "confidence_only": "Confidence-only",
        "random_spline": "Random Spline",
        "dusafe": "DuSafe",
    }
    lines = [
        "| Method | SFC correction ↑ | Remaining HCW ↓ | Reliable R→W ↓ | Corrupted F1 ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = paper[paper["variant"].eq(variant)].iloc[0]
        cells = []
        for metric in (
            "sfc_correction",
            "remaining_hcw",
            "reliable_r_to_w",
            "corrupted_f1",
        ):
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            cells.append(
                "NA" if not math.isfinite(mean) else f"{100*mean:.2f} ± {100*std:.2f}"
            )
        lines.append(f"| {labels[variant]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def finalize(output_dir: Path, payload: Mapping[str, Any]) -> pd.DataFrame:
    source_frames = []
    future_frames = []
    label_frames = []
    manifests = []
    for corruption, severity in CONDITIONS:
        for source_seed in SOURCE_SEEDS:
            cell = _cell_output_dir(output_dir, source_seed, corruption, severity)
            if not _cell_complete(cell, str(payload["protocol_sha256"])):
                raise RuntimeError(f"incomplete Future SFC cell: {cell}")
            manifest = json.loads((cell / "manifest.json").read_text("utf-8"))
            manifests.append(manifest)
            source_cell = pd.read_csv(cell / "source_reference_samples.csv")
            future_cell = pd.read_csv(cell / "future_predictions.csv")
            labels_cell = pd.read_csv(cell / "posthoc_labels.csv")
            labels_cell["corruption"] = corruption
            labels_cell["severity"] = severity
            labels_cell["source_seed"] = source_seed
            source_frames.append(source_cell)
            future_frames.append(future_cell)
            label_frames.append(labels_cell)
    source = pd.concat(source_frames, ignore_index=True)
    future = pd.concat(future_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    if len(source) != len(CONDITIONS) * len(SOURCE_SEEDS) * 110:
        raise RuntimeError("source-reference panel row count is incomplete")
    if len(future) != (
        len(CONDITIONS)
        * len(SOURCE_SEEDS)
        * len(VARIANTS)
        * EXPECTED_FUTURE_SAMPLES
    ):
        raise RuntimeError("future-prediction panel row count is incomplete")
    if len(labels) != len(CONDITIONS) * len(SOURCE_SEEDS) * 110:
        raise RuntimeError("post-hoc label panel row count is incomplete")
    required_source_audit = {
        "source_reference_pre_state_sha256",
        "source_reference_post_state_sha256",
        "source_reference_state_unchanged",
        "source_reference_rng_unchanged",
    }
    required_future_audit = {
        "future_eval_pre_state_sha256",
        "future_eval_post_state_sha256",
        "future_eval_state_unchanged",
        "future_eval_rng_unchanged",
    }
    if not required_source_audit.issubset(source):
        raise RuntimeError("source full-state audit schema is incomplete")
    if not required_future_audit.issubset(future):
        raise RuntimeError("future full-state audit schema is incomplete")
    if not _boolean_values(
        source["source_reference_state_unchanged"],
        field="source_reference_state_unchanged",
    ).all() or not _boolean_values(
        source["source_reference_rng_unchanged"],
        field="source_reference_rng_unchanged",
    ).all():
        raise RuntimeError("source reference state/RNG audit failed")
    if not (
        source["source_reference_pre_state_sha256"].astype(str)
        == source["source_reference_post_state_sha256"].astype(str)
    ).all():
        raise RuntimeError("source reference state hashes differ")
    if not _boolean_values(
        future["future_eval_state_unchanged"],
        field="future_eval_state_unchanged",
    ).all() or not _boolean_values(
        future["future_eval_rng_unchanged"],
        field="future_eval_rng_unchanged",
    ).all():
        raise RuntimeError("future evaluation state/RNG audit failed")
    if not (
        future["future_eval_pre_state_sha256"].astype(str)
        == future["future_eval_post_state_sha256"].astype(str)
    ).all():
        raise RuntimeError("future evaluation state hashes differ")
    for manifest in manifests:
        for field in (
            "source_reference_full_adapter_state_preserved",
            "source_reference_rng_preserved",
            "future_evaluation_full_adapter_state_preserved",
            "future_evaluation_rng_preserved",
        ):
            if manifest.get(field) is not True:
                raise RuntimeError(f"cell manifest state audit failed: {field}")
    source_key = ["corruption", "severity", "source_seed", "target_index"]
    future_key = [
        "corruption",
        "severity",
        "source_seed",
        "variant",
        "target_index",
    ]
    if source.duplicated(source_key).any() or future.duplicated(future_key).any():
        raise RuntimeError("duplicate sample key in Future SFC panel")
    label_key = ["corruption", "severity", "source_seed", "target_index"]
    if labels.duplicated(label_key).any():
        raise RuntimeError("duplicate post-hoc label key in Future SFC panel")
    canonical_labels = None
    for _, group in labels.groupby(
        ["corruption", "severity", "source_seed"], sort=False
    ):
        ordered = group.sort_values("target_index")
        if ordered["target_index"].tolist() != list(range(110)):
            raise RuntimeError("post-hoc labels do not cover 0..109")
        identity = ordered["true_label"].astype(int).tolist()
        if canonical_labels is None:
            canonical_labels = identity
        elif identity != canonical_labels:
            raise RuntimeError("post-hoc target labels differ across cells")
    source_artifacts = payload.get("source_artifacts", {})
    for source_seed in SOURCE_SEEDS:
        seed_source = source[source["source_seed"].eq(source_seed)]
        thresholds = pd.to_numeric(
            seed_source["confidence_nll_threshold_tau_q"], errors="raise"
        ).unique()
        if len(thresholds) != 1 or not math.isclose(
            float(thresholds[0]),
            EXPECTED_TAU_Q[source_seed],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"seed {source_seed} tau_q identity changed")
        manifest_hashes = {
            str(item["source_model_sha256"])
            for item in manifests
            if int(item["source_seed"]) == source_seed
        }
        if len(manifest_hashes) != 1:
            raise RuntimeError(
                f"seed {source_seed} source hash differs across conditions"
            )
        expected_artifact = source_artifacts.get(str(source_seed))
        if expected_artifact is not None and manifest_hashes != {
            str(expected_artifact["source_model_sha256"])
        }:
            raise RuntimeError(
                f"seed {source_seed} source hash differs from preregistration"
            )
    for (corruption, severity, source_seed), group in source.groupby(
        ["corruption", "severity", "source_seed"], sort=False
    ):
        ordered = group.sort_values("target_index")
        if ordered["target_index"].tolist() != list(range(110)):
            raise RuntimeError("source reference index coverage is incomplete")
        mask = _boolean_values(
            ordered["registered_corrupted"], field="registered_corrupted"
        ).astype(np.uint8)
        if int(mask.sum()) != 55:
            raise RuntimeError("registered corruption mask is not exact 55/110")
        if hashlib.sha256(mask.tobytes()).hexdigest() != (
            corruptions.corruption_mask_sha256()
        ):
            raise RuntimeError("registered corruption mask hash changed")
        future_mask = _boolean_values(
            ordered[ordered["target_index"].isin(FUTURE_INDICES)][
                "registered_corrupted"
            ],
            field="registered_corrupted",
        )
        if int(future_mask.sum()) != EXPECTED_FUTURE_CORRUPTED:
            raise RuntimeError("future corruption mask is not exact 27/62")

    # Frozen clean/corrupted outputs and tau_q are method-independent by
    # construction.  They are joined only after every online trajectory has
    # finished, so true labels cannot influence an update or subset exposure.
    metric_frames = []
    source_sanity_rows = []
    for corruption, severity in CONDITIONS:
        for source_seed in SOURCE_SEEDS:
            source_cell = source[
                source["corruption"].eq(corruption)
                & source["severity"].eq(severity)
                & source["source_seed"].eq(source_seed)
            ]
            future_cell = future[
                future["corruption"].eq(corruption)
                & future["severity"].eq(severity)
                & future["source_seed"].eq(source_seed)
            ]
            labels_cell = labels[
                labels["corruption"].eq(corruption)
                & labels["severity"].eq(severity)
                & labels["source_seed"].eq(source_seed)
            ]
            metric_frames.append(
                _cell_metrics(source_cell, future_cell, labels_cell)
            )
            source_sanity_rows.append(
                _source_sanity_metrics(source_cell, labels_cell)
            )
    condition_metrics = pd.concat(metric_frames, ignore_index=True)
    source_sanity = pd.DataFrame(source_sanity_rows)
    if len(condition_metrics) != len(CONDITIONS) * len(SOURCE_SEEDS) * len(VARIANTS):
        raise RuntimeError("condition/seed/method metric grid is incomplete")
    if len(source_sanity) != len(CONDITIONS) * len(SOURCE_SEEDS):
        raise RuntimeError("source sanity grid is incomplete")
    if not source_sanity["source_sanity_passed"].astype(bool).all():
        raise RuntimeError("source sanity grid contains a failed identity")
    for group_key, group in condition_metrics.groupby(
        ["corruption", "severity", "source_seed"], sort=False
    ):
        if group["sfc_denominator"].nunique() != 1:
            raise RuntimeError(f"fixed SFC subset differs by method: {group_key}")
        if group["reliable_denominator"].nunique() != 1:
            raise RuntimeError(f"fixed reliable subset differs by method: {group_key}")
        if group["strict_reliable_denominator"].nunique() != 1:
            raise RuntimeError(
                f"fixed strict reliable subset differs by method: {group_key}"
            )
        for field in (
            "sfc_subset_sha256",
            "reliable_subset_sha256",
            "strict_reliable_subset_sha256",
        ):
            if group[field].nunique(dropna=False) != 1:
                raise RuntimeError(
                    f"fixed subset membership differs by method: {group_key}/{field}"
                )
        if not group["sfc_three_way_partition_passed"].astype(bool).all():
            raise RuntimeError(f"SFC partition failed: {group_key}")
        if not (
            group["sfc_correction_numerator"]
            + group["remaining_hcw_numerator"]
            + group["remaining_wrong_low_confidence_numerator"]
            == group["sfc_denominator"]
        ).all():
            raise RuntimeError(f"SFC count identity failed: {group_key}")
        sanity = source_sanity[
            source_sanity["corruption"].eq(group_key[0])
            & source_sanity["severity"].eq(group_key[1])
            & source_sanity["source_seed"].eq(group_key[2])
        ]
        if len(sanity) != 1:
            raise RuntimeError(f"source sanity key missing: {group_key}")
        sanity_row = sanity.iloc[0]
        for field in (
            "sfc_subset_sha256",
            "reliable_subset_sha256",
            "strict_reliable_subset_sha256",
        ):
            if str(sanity_row[field]) != str(group[field].iloc[0]):
                raise RuntimeError(
                    f"source/method membership hash differs: {group_key}/{field}"
                )

    seed_summary, paper, condition_summary, pooled = _aggregate_final(
        condition_metrics
    )
    atomic_write_csv(source, output_dir / "source_reference_samples_all.csv")
    atomic_write_csv(future, output_dir / "future_sample_predictions_all.csv")
    atomic_write_csv(labels, output_dir / "posthoc_labels_all.csv")
    atomic_write_csv(
        condition_metrics, output_dir / "condition_seed_metrics.csv"
    )
    atomic_write_csv(source_sanity, output_dir / "source_sanity.csv")
    atomic_write_csv(seed_summary, output_dir / "method_seed_summary.csv")
    atomic_write_csv(paper, output_dir / "paper_summary.csv")
    atomic_write_csv(condition_summary, output_dir / "condition_summary.csv")
    atomic_write_csv(pooled, output_dir / "pooled_diagnostic_summary.csv")
    (output_dir / "paper_summary.md").write_text(
        _paper_markdown(paper), encoding="utf-8"
    )
    final_manifest = {
        **dict(payload),
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "cell_count": int(len(manifests)),
        "source_reference_rows": int(len(source)),
        "future_sample_rows": int(len(future)),
        "posthoc_label_rows": int(len(labels)),
        "condition_seed_method_rows": int(len(condition_metrics)),
        "independent_source_seed_units": len(SOURCE_SEEDS),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_posthoc_grouping_and_metrics": True,
        "sfc_and_reliable_subsets_fixed_before_method_comparison": True,
        "all_methods_share_source_defined_subsets": True,
        "empty_subset_cells_are_nan_not_zero": True,
        "source_sanity_passed": True,
        "fixed_subset_membership_hashes_verified": True,
        "sfc_three_way_partition_verified": True,
        "source_reference_full_adapter_state_and_rng_preserved": True,
        "future_evaluation_full_adapter_state_and_rng_preserved": True,
        "outputs": [
            "source_reference_samples_all.csv",
            "future_sample_predictions_all.csv",
            "posthoc_labels_all.csv",
            "condition_seed_metrics.csv",
            "source_sanity.csv",
            "method_seed_summary.csv",
            "paper_summary.csv",
            "paper_summary.md",
            "condition_summary.csv",
            "pooled_diagnostic_summary.csv",
            "final_manifest.json",
        ],
    }
    _atomic_json(final_manifest, output_dir / "final_manifest.json")
    return paper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(ROOT / "configs" / "paper_flow_profiles_v1.json"),
    )
    parser.add_argument(
        "--reference-main-csv",
        default=str(
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "final_claim_preserving"
            / "main_raw_normalized.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gpu-lock-path",
        type=Path,
        default=ROOT / "results" / ".current_experiment_gpu.lock",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--corruption", help=argparse.SUPPRESS)
    parser.add_argument("--severity", help=argparse.SUPPRESS)
    parser.add_argument("--protocol-sha256", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _protocol_payload(args)
    if args.cell:
        if args.source_seed is None or not args.corruption or not args.severity:
            raise ValueError("cell requires source seed, corruption, and severity")
        run_cell(args)
        return 0

    preregistered = output_dir / "preregistered_protocol.json"
    if preregistered.is_file():
        existing = json.loads(preregistered.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != payload["protocol_sha256"]:
            raise RuntimeError("existing preregistered protocol does not match")
        payload = existing
    else:
        _atomic_json(payload, preregistered)
    if args.finalize_only:
        paper = finalize(output_dir, payload)
        print(_paper_markdown(paper))
        return 0

    cells = []
    for corruption, severity in CONDITIONS:
        for source_seed in SOURCE_SEEDS:
            cell_output = _cell_output_dir(
                output_dir, source_seed, corruption, severity
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cell",
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                str(args.device),
                "--backbone",
                str(args.backbone),
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
                "--flow-profile-json",
                str(Path(args.flow_profile_json).resolve()),
                "--reference-main-csv",
                str(Path(args.reference_main_csv).resolve()),
                "--output-dir",
                str(output_dir),
                "--source-seed",
                str(source_seed),
                "--corruption",
                corruption,
                "--severity",
                severity,
                "--protocol-sha256",
                str(payload["protocol_sha256"]),
            ]
            cells.append(
                {
                    "source_seed": source_seed,
                    "corruption": corruption,
                    "severity": severity,
                    "output_dir": str(cell_output),
                    "command": command,
                }
            )
    plan = {
        "protocol": PROTOCOL,
        "protocol_sha256": payload["protocol_sha256"],
        "status": "planned" if not args.execute else "running",
        "cell_count": len(cells),
        "cells": cells,
    }
    _atomic_json(plan, output_dir / "plan.json")
    if not args.execute:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    for cell in cells:
        cell_path = Path(cell["output_dir"])
        if _cell_complete(cell_path, str(payload["protocol_sha256"])):
            continue
        lock = (
            wait_for_gpu_experiment_lock(args.gpu_lock_path)
            if str(args.device).lower().startswith("cuda")
            else contextlib.nullcontext()
        )
        with lock:
            completed = subprocess.run(cell["command"], cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Future SFC cell failed: "
                f"{cell['corruption']} {cell['severity']} seed{cell['source_seed']}"
            )
    paper = finalize(output_dir, payload)
    print(_paper_markdown(paper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "EXPECTED_BATCH_SIZES",
    "EXPECTED_FUTURE_CORRUPTED",
    "EXPECTED_FUTURE_SAMPLES",
    "FUTURE_INDICES",
    "PROTOCOL",
    "SOURCE_SEEDS",
    "VARIANTS",
    "_aggregate_final",
    "_cell_metrics",
    "_protocol_payload",
    "finalize",
]
