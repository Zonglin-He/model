"""HAR 12->16 source-quality audit and target-stream stability figure.

The protocol keeps the paper deployment batch size (48), so the 110-sample
target stream has three real updates (48/48/14). Target coverage and
batch-start prequential predictions are assigned to ten sample deciles;
source-calibration F1 is measured only at the four real model states and is
carried forward causally to decile endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.representative_causal_ablation import (  # noqa: E402
    RepresentativeAcceptAllRaw,
    RepresentativeConfidenceRaw,
    RepresentativeHardSSAW,
)
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from trainers.tta_abstract_trainer import _predict_after_adaptation  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    move_data_to_device,
)


PROTOCOL = "har_source_quality_stream_stability_v1_alpha020"
DATASET = "HAR"
SCENARIO = "12->16"
STREAM_SEED = 42
SOURCE_SEEDS = (0, 1, 2)
NUM_DECILES = 10
FULL_COMPONENT = "confidence_plus_margin_aware_hard_ssaw"
VARIANTS = {
    "Raw TTA": RepresentativeAcceptAllRaw,
    "Confidence-only": RepresentativeConfidenceRaw,
    "Full DuSafe": RepresentativeHardSSAW,
}


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True), path)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_reference_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("dataset", "")).upper() == DATASET
            and str(row.get("scenario", "")) == SCENARIO
            and int(row.get("stream_seed", -1)) == STREAM_SEED
            and str(row.get("replaced_component", "")) == FULL_COMPONENT
            and str(row.get("status", "")).lower() == "ok"
            and int(row.get("source_seed", -1)) in SOURCE_SEEDS
        ]
    result: dict[int, dict[str, str]] = {}
    for row in selected:
        seed = int(row["source_seed"])
        if seed in result:
            raise RuntimeError(f"duplicate formal source identity for seed {seed}")
        result[seed] = row
    if set(result) != set(SOURCE_SEEDS):
        raise RuntimeError(
            f"formal source identities are incomplete: {sorted(result)}"
        )
    return result


def _macro_f1(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    num_classes: int,
) -> float:
    return float(
        f1_score(
            np.asarray(labels, dtype=np.int64),
            np.asarray(predictions, dtype=np.int64),
            labels=list(range(int(num_classes))),
            average="macro",
            zero_division=0,
        )
    )


def _decile_indices(sample_count: int, num_deciles: int = NUM_DECILES) -> np.ndarray:
    if sample_count < num_deciles or sample_count % num_deciles != 0:
        raise ValueError(
            "this registered HAR audit requires equal non-empty deciles"
        )
    positions = np.arange(sample_count, dtype=np.int64)
    return np.minimum(
        num_deciles,
        (positions * num_deciles // sample_count) + 1,
    )


def _carry_source_states_to_deciles(
    *,
    sample_count: int,
    update_end_positions: Sequence[int],
    source_f1_values: Sequence[float],
    num_deciles: int = NUM_DECILES,
) -> list[float]:
    positions = np.asarray(update_end_positions, dtype=np.int64)
    values = np.asarray(source_f1_values, dtype=np.float64)
    if positions.ndim != 1 or values.ndim != 1 or positions.size != values.size:
        raise ValueError("source-state positions and values must align")
    if positions.size == 0 or positions[0] != 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("source-state positions must start at zero and increase")
    if positions[-1] != sample_count:
        raise ValueError("the final source state must be measured at stream end")
    endpoints = np.arange(1, num_deciles + 1) * sample_count // num_deciles
    result = []
    for endpoint in endpoints:
        available = np.flatnonzero(positions <= endpoint)
        result.append(float(values[int(available[-1])]))
    return result


def _evaluate_loader(adapter, loader, *, num_classes: int) -> tuple[float, int]:
    # Constructing/iterating a DataLoader consumes the process-wide torch RNG
    # even with ``num_workers=0``.  Source-retention measurement is offline
    # instrumentation and must not perturb the subsequent target update (or
    # the SSAW candidate sequence), so preserve all global RNG streams around
    # the complete loader traversal rather than only around each model call.
    torch_rng = torch.random.get_rng_state()
    model_values = list(adapter.model.parameters()) + list(
        adapter.model.buffers()
    )
    model_uses_cuda = any(value.is_cuda for value in model_values)
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if model_uses_cuda and torch.cuda.is_available()
        else None
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    labels: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    model_hash_before = _state_sha256(adapter.model)
    try:
        for data, target, _indices in loader:
            logits = _production_logits(adapter, data)
            labels.append(torch.as_tensor(target).view(-1).cpu().long())
            predictions.append(logits.argmax(dim=1).detach().cpu().long())
        model_hash_after = _state_sha256(adapter.model)
        if model_hash_before != model_hash_after:
            raise RuntimeError("read-only source evaluation changed model state")
    finally:
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if not labels:
        raise RuntimeError("source evaluation loader is empty")
    truth = torch.cat(labels).numpy()
    pred = torch.cat(predictions).numpy()
    return _macro_f1(truth, pred, num_classes), int(truth.size)


def _production_logits(adapter, data) -> torch.Tensor:
    """Read logits in the exact production module modes without state drift.

    DuSafe intentionally keeps batch-normalization modules in batch-statistics
    mode with ``track_running_stats=False``.  Calling ``model.eval()`` changes
    that numerical path on modules whose running buffers still exist, so this
    probe delegates to the trainer's audited post-update predictor instead.
    Global RNG is restored to keep the probe observational.
    """

    model = adapter.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    device_data = move_data_to_device(data, device)
    torch_rng = torch.random.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        return _predict_after_adaptation(adapter, {"data": device_data})
    finally:
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


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
            "dusafe_logging_mode": "production",
            "record_per_sample_evidence": False,
            "record_production_batch_diagnostics": True,
            "ssaw_candidate_cuda_graph": "off",
            "ssaw_production_decision_only": True,
        }
    )
    if int(runtime["batch_size"]) != 48 or int(runtime["steps"]) != 2:
        raise RuntimeError("HAR 12->16 formal batch/steps protocol changed")
    if not np.isclose(float(runtime["spline_log_strength"]), 0.20):
        raise RuntimeError("paper audit requires alpha=0.20")
    return runtime


def _run_cell(
    *,
    args: argparse.Namespace,
    reference: Mapping[str, str],
    source_seed: int,
    method: str,
    variant_class,
    runtime_hparams: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_id, target_id = SCENARIO.split("->", 1)
    source_config = json.loads(reference["source_config"])
    checkpoint = Path(reference["source_checkpoint_path"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=DATASET,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name=(
            "har_source_quality_stream_stability_v1_"
            f"s{source_seed}_{method.lower().replace(' ', '_')}"
        ),
        seed=STREAM_SEED,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        pretrained_checkpoint=str(checkpoint),
    )
    trainer.get_tta_model_class = lambda: variant_class
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(dict(runtime_hparams))
        adapter, source_model = create_tta_model(
            trainer, source_id, target_id, run_seed=STREAM_SEED
        )
        actual_source_hash = _state_sha256(source_model)
        expected_source_hash = str(reference["source_model_sha256"])
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                f"seed {source_seed} source hash mismatch: "
                f"{actual_source_hash} != {expected_source_hash}"
            )
        source_evaluation_batch_size = int(
            getattr(
                args,
                "source_evaluation_batch_size",
                runtime_hparams["batch_size"],
            )
        )
        source_evaluation_loader = DataLoader(
            trainer.src_test_dl.dataset,
            batch_size=source_evaluation_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        source_f1_before, source_samples = _evaluate_loader(
            adapter, source_evaluation_loader, num_classes=trainer.num_classes
        )
        threshold = float(adapter.confidence_nll_threshold.detach().cpu().item())
        if not np.isfinite(threshold):
            raise RuntimeError("source confidence threshold is non-finite")

        sample_rows: list[dict[str, Any]] = []
        source_rows = [
            {
                "protocol": PROTOCOL,
                "dataset": DATASET,
                "scenario": SCENARIO,
                "source_seed": source_seed,
                "stream_seed": STREAM_SEED,
                "method": method,
                "stream_samples_completed": 0,
                "source_calibration_f1": source_f1_before,
                "source_calibration_samples": source_samples,
            }
        ]
        stream_position = 0
        target_index_order: list[int] = []
        for batch_index, (data, labels, target_indices) in enumerate(
            trainer.trg_whole_dl
        ):
            labels_cpu = torch.as_tensor(labels).view(-1).cpu().long()
            target_indices_cpu = torch.as_tensor(target_indices).view(-1).cpu().long()
            batch_start_logits = _production_logits(adapter, data)
            batch_start_predictions = (
                batch_start_logits.argmax(dim=1).detach().cpu().long()
            )
            device_data = move_data_to_device(data, trainer.device)
            online_logits = adapter(
                {
                    "data": device_data,
                    "meta": {"trg_idx": target_indices_cpu.tolist()},
                }
            )
            post_update_logits = _production_logits(adapter, data)
            online_predictions = online_logits.argmax(dim=1).detach().cpu().long()
            post_update_predictions = (
                post_update_logits.argmax(dim=1).detach().cpu().long()
            )
            gate_log = adapter._last_gate_log
            admission = torch.as_tensor(
                gate_log["admission_mask"], dtype=torch.bool
            ).detach().cpu().view(-1)
            pseudo_labels = torch.as_tensor(
                gate_log["pseudo_labels"], dtype=torch.long
            ).detach().cpu().view(-1)
            batch_size = int(labels_cpu.numel())
            vectors = (
                target_indices_cpu,
                batch_start_predictions,
                online_predictions,
                post_update_predictions,
                admission,
                pseudo_labels,
            )
            if any(int(vector.numel()) != batch_size for vector in vectors):
                raise RuntimeError("target batch vectors have inconsistent lengths")
            for local_index in range(batch_size):
                true_label = int(labels_cpu[local_index].item())
                admitted = bool(admission[local_index].item())
                pseudo_label = int(pseudo_labels[local_index].item())
                target_index = int(target_indices_cpu[local_index].item())
                target_index_order.append(target_index)
                sample_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "dataset": DATASET,
                        "scenario": SCENARIO,
                        "source_seed": source_seed,
                        "stream_seed": STREAM_SEED,
                        "method": method,
                        "batch_index": batch_index,
                        "local_batch_index": local_index,
                        "stream_position": stream_position + local_index,
                        "target_index": target_index,
                        "true_label": true_label,
                        "batch_start_prediction": int(
                            batch_start_predictions[local_index].item()
                        ),
                        "online_output_prediction": int(
                            online_predictions[local_index].item()
                        ),
                        "post_update_prediction": int(
                            post_update_predictions[local_index].item()
                        ),
                        "admitted": admitted,
                        "admission_pseudo_label": pseudo_label,
                        "admitted_correct": bool(admitted and pseudo_label == true_label),
                    }
                )
            stream_position += batch_size
            source_f1_after, after_samples = _evaluate_loader(
                adapter,
                source_evaluation_loader,
                num_classes=trainer.num_classes,
            )
            if after_samples != source_samples:
                raise RuntimeError("source evaluation sample count changed")
            source_rows.append(
                {
                    "protocol": PROTOCOL,
                    "dataset": DATASET,
                    "scenario": SCENARIO,
                    "source_seed": source_seed,
                    "stream_seed": STREAM_SEED,
                    "method": method,
                    "stream_samples_completed": stream_position,
                    "source_calibration_f1": source_f1_after,
                    "source_calibration_samples": source_samples,
                }
            )

        sample_frame = pd.DataFrame(sample_rows)
        source_frame = pd.DataFrame(source_rows)
        expected_target_samples = int(
            getattr(args, "expected_target_samples", 110)
        )
        if stream_position != expected_target_samples:
            raise RuntimeError(
                f"{DATASET} {SCENARIO} expected {expected_target_samples} "
                f"target samples, got {stream_position}"
            )
        if sample_frame["stream_position"].duplicated().any():
            raise RuntimeError("duplicate stream positions within a cell")
        if target_index_order != list(range(expected_target_samples)):
            raise RuntimeError(
                "target stream index order is not the registered sequential order"
            )
        labels_np = sample_frame["true_label"].to_numpy(dtype=np.int64)
        prequential_f1 = _macro_f1(
            labels_np,
            sample_frame["batch_start_prediction"].to_numpy(dtype=np.int64),
            trainer.num_classes,
        )
        online_f1 = _macro_f1(
            labels_np,
            sample_frame["online_output_prediction"].to_numpy(dtype=np.int64),
            trainer.num_classes,
        )
        post_update_f1 = _macro_f1(
            labels_np,
            sample_frame["post_update_prediction"].to_numpy(dtype=np.int64),
            trainer.num_classes,
        )
        admitted_count = int(sample_frame["admitted"].sum())
        admitted_accuracy = (
            float("nan")
            if admitted_count == 0
            else float(sample_frame["admitted_correct"].sum() / admitted_count)
        )
        summary = {
            "protocol": PROTOCOL,
            "dataset": DATASET,
            "scenario": SCENARIO,
            "source_seed": source_seed,
            "stream_seed": STREAM_SEED,
            "method": method,
            "execution_device": str(args.device),
            "source_model_sha256": actual_source_hash,
            "source_checkpoint_path": str(checkpoint),
            "source_config": source_config,
            "runtime_hparams": dict(runtime_hparams),
            "source_calibration_f1_before": source_f1_before,
            "source_calibration_f1_after": float(
                source_frame.iloc[-1]["source_calibration_f1"]
            ),
            "source_retention_delta": float(
                source_frame.iloc[-1]["source_calibration_f1"] - source_f1_before
            ),
            "source_calibration_samples": source_samples,
            "source_evaluation_batch_size": source_evaluation_batch_size,
            "confidence_nll_threshold_tau_q": threshold,
            "target_samples": stream_position,
            "target_admission_coverage": float(admitted_count / stream_position),
            "target_admitted_accuracy": admitted_accuracy,
            "target_batch_start_prequential_f1": prequential_f1,
            "target_online_output_f1": online_f1,
            "target_post_update_f1": post_update_f1,
            "target_labels_used_for_online_decision": False,
            "target_labels_used_for_offline_metrics": True,
        }
        return sample_frame, source_frame, summary
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def _build_deciles(
    sample_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    *,
    num_classes: int,
) -> pd.DataFrame:
    rows = []
    for (source_seed, method), group in sample_frame.groupby(
        ["source_seed", "method"], sort=False
    ):
        group = group.sort_values("stream_position").reset_index(drop=True)
        count = len(group)
        deciles = _decile_indices(count)
        group = group.assign(decile=deciles)
        source_group = source_frame[
            (source_frame["source_seed"] == source_seed)
            & (source_frame["method"] == method)
        ].sort_values("stream_samples_completed")
        carried_source_f1 = _carry_source_states_to_deciles(
            sample_count=count,
            update_end_positions=source_group[
                "stream_samples_completed"
            ].to_numpy(dtype=np.int64),
            source_f1_values=source_group[
                "source_calibration_f1"
            ].to_numpy(dtype=np.float64),
        )
        for decile in range(1, NUM_DECILES + 1):
            local = group[group["decile"] == decile]
            cumulative = group[group["decile"] <= decile]
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "dataset": DATASET,
                    "scenario": SCENARIO,
                    "source_seed": int(source_seed),
                    "stream_seed": STREAM_SEED,
                    "method": str(method),
                    "decile": decile,
                    "decile_start_position": int(local["stream_position"].min()),
                    "decile_end_position": int(local["stream_position"].max() + 1),
                    "decile_samples": int(len(local)),
                    "admission_coverage": float(local["admitted"].mean()),
                    "local_prequential_macro_f1": _macro_f1(
                        local["true_label"],
                        local["batch_start_prediction"],
                        num_classes,
                    ),
                    "cumulative_prequential_macro_f1": _macro_f1(
                        cumulative["true_label"],
                        cumulative["batch_start_prediction"],
                        num_classes,
                    ),
                    "source_calibration_f1": carried_source_f1[decile - 1],
                }
            )
    return pd.DataFrame(rows)


def _aggregate_deciles(deciles: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "admission_coverage",
        "local_prequential_macro_f1",
        "cumulative_prequential_macro_f1",
        "source_calibration_f1",
    )
    key_columns = ["method", "decile", "source_seed"]
    if deciles.duplicated(key_columns).any():
        duplicates = deciles.loc[
            deciles.duplicated(key_columns, keep=False), key_columns
        ].drop_duplicates()
        raise RuntimeError(
            "duplicate method-decile-source_seed rows: "
            f"{duplicates.to_dict(orient='records')}"
        )
    expected_methods = set(VARIANTS)
    observed_methods = set(deciles["method"].astype(str))
    if observed_methods != expected_methods:
        raise RuntimeError(
            "decile rows have unexpected methods: "
            f"{sorted(observed_methods)} != {sorted(expected_methods)}"
        )
    rows = []
    for (method, decile), group in deciles.groupby(["method", "decile"], sort=False):
        if len(group) != len(SOURCE_SEEDS) or set(
            group["source_seed"].astype(int)
        ) != set(SOURCE_SEEDS):
            raise RuntimeError(
                f"{method} decile {decile} lacks exactly one row per source seed"
            )
        row: dict[str, Any] = {
            "protocol": PROTOCOL,
            "dataset": DATASET,
            "scenario": SCENARIO,
            "method": str(method),
            "decile": int(decile),
            "source_seed_count": int(group["source_seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{metric} contains non-finite values")
            if ((values < 0.0) | (values > 1.0)).any():
                raise RuntimeError(f"{metric} lies outside [0, 1]")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(VARIANTS) * NUM_DECILES:
        raise RuntimeError("decile aggregate has an unexpected row count")
    if not (result["source_seed_count"] == len(SOURCE_SEEDS)).all():
        raise RuntimeError("a decile aggregate is missing source seeds")
    return result


def _source_quality_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    full = summary_frame[summary_frame["method"] == "Full DuSafe"].copy()
    full = full.sort_values("source_seed")
    if list(full["source_seed"].astype(int)) != list(SOURCE_SEEDS):
        raise RuntimeError("Full source-quality table lacks registered seeds")
    return full[
        [
            "source_seed",
            "source_model_sha256",
            "source_calibration_f1_before",
            "confidence_nll_threshold_tau_q",
            "target_admission_coverage",
            "target_admitted_accuracy",
            "target_post_update_f1",
            "target_batch_start_prequential_f1",
            "source_calibration_f1_after",
            "source_retention_delta",
        ]
    ].reset_index(drop=True)


def _table_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Source seed | Source F1 | $\\tau_q$ | Coverage | Admitted Acc. | Target F1 | Source retention $\\Delta$ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            f"{int(row.source_seed)} | "
            f"{row.source_calibration_f1_before * 100:.2f} | "
            f"{row.confidence_nll_threshold_tau_q:.4f} | "
            f"{row.target_admission_coverage * 100:.2f} | "
            f"{row.target_admitted_accuracy * 100:.2f} | "
            f"{row.target_post_update_f1 * 100:.2f} | "
            f"{row.source_retention_delta * 100:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def _plot(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    styles = {
        "Raw TTA": {"color": "#666666", "ls": ":", "marker": "o"},
        "Confidence-only": {"color": "#24537A", "ls": "--", "marker": "s"},
        "Full DuSafe": {"color": "#E58606", "ls": "-", "marker": "^"},
    }
    figure, axes = plt.subplots(3, 1, figsize=(6.9, 7.7), sharex=True, constrained_layout=True)
    panels = (
        ("admission_coverage", "(a) Admission coverage", "Coverage"),
        (
            "cumulative_prequential_macro_f1",
            "(b) Cumulative batch-start prequential Macro-F1",
            "Macro-F1",
        ),
        (
            "source_calibration_f1",
            "(c) Source-calibration F1 at causally available states",
            "Macro-F1",
        ),
    )
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for method in VARIANTS:
            group = summary[summary["method"] == method].sort_values("decile")
            x = group["decile"].to_numpy(dtype=float)
            mean = group[f"{metric}_mean"].to_numpy(dtype=float)
            std = group[f"{metric}_std"].to_numpy(dtype=float)
            style = styles[method]
            axis.plot(
                x,
                mean,
                label=method,
                color=style["color"],
                ls=style["ls"],
                marker=style["marker"],
                lw=1.8,
                ms=4.5,
            )
            axis.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )
        axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D8DDE3", lw=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.0, 1.03)
    axes[1].set_ylim(0.0, 1.03)
    source_low = float(summary["source_calibration_f1_mean"].min())
    source_high = float(summary["source_calibration_f1_mean"].max())
    padding = max(0.01, 0.18 * (source_high - source_low + 1e-8))
    axes[2].set_ylim(max(0.0, source_low - padding), min(1.0, source_high + padding))
    axes[2].text(
        0.01,
        0.04,
        "Focused y-axis; source state changes only after completed deployment batches",
        transform=axes[2].transAxes,
        fontsize=7.5,
        color="#4B5563",
    )
    axes[2].text(
        0.99,
        0.92,
        "All methods/seeds overlap at 1.000",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#4B5563",
    )
    for boundary, label in ((48 / 11.0, "48"), (96 / 11.0, "96")):
        axes[2].axvline(boundary, color="#9AA3AD", lw=0.8, ls=":")
        axes[2].text(
            boundary,
            axes[2].get_ylim()[1],
            f" update @{label}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="#6E7781",
        )
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    axes[1].text(
        0.99,
        0.04,
        "Lines: seed mean; bands: ±1 SD (descriptive)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )
    axes[2].set_xlabel("Target-stream decile")
    axes[2].set_xticks(range(1, NUM_DECILES + 1))
    png = output_dir / "har_12_to_16_stream_stability.png"
    pdf = output_dir / "har_12_to_16_stream_stability.pdf"
    figure.savefig(png, dpi=600, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    return png, pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
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
    parser.add_argument(
        "--gpu-lock-path",
        default=str(ROOT / "results" / ".current_experiment_gpu.lock"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "har_source_quality_stream_stability"
        ),
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference_main_csv).resolve()
    references = _load_reference_rows(reference_path)
    runtime = _runtime_hparams(Path(args.flow_profile_json).resolve())
    # Publish an incomplete state before touching the GPU.  A failed rerun
    # must not leave a stale previous ``complete`` manifest that downstream
    # paper finalizers could accidentally accept.
    _atomic_json(
        {
            "protocol": PROTOCOL,
            "status": "running",
            "dataset": DATASET,
            "scenario": SCENARIO,
            "source_seeds": list(SOURCE_SEEDS),
            "stream_seed": STREAM_SEED,
            "methods": list(VARIANTS),
            "confirmatory": False,
        },
        output_dir / "manifest.json",
    )

    sample_frames = []
    source_frames = []
    summaries = []
    with wait_for_gpu_experiment_lock(args.gpu_lock_path):
        for source_seed in SOURCE_SEEDS:
            per_seed_hashes = set()
            per_seed_orders = []
            for method, variant_class in VARIANTS.items():
                samples, source_states, summary = _run_cell(
                    args=args,
                    reference=references[source_seed],
                    source_seed=source_seed,
                    method=method,
                    variant_class=variant_class,
                    runtime_hparams=runtime,
                )
                sample_frames.append(samples)
                source_frames.append(source_states)
                summaries.append(summary)
                per_seed_hashes.add(summary["source_model_sha256"])
                per_seed_orders.append(samples["target_index"].tolist())
            if len(per_seed_hashes) != 1:
                raise RuntimeError(f"seed {source_seed} methods used different checkpoints")
            if any(order != per_seed_orders[0] for order in per_seed_orders[1:]):
                raise RuntimeError(f"seed {source_seed} methods used different streams")

    sample_frame = pd.concat(sample_frames, ignore_index=True)
    source_frame = pd.concat(source_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    if len(sample_frame) != len(SOURCE_SEEDS) * len(VARIANTS) * 110:
        raise RuntimeError("sample record count is incomplete")
    if summary_frame.duplicated(["source_seed", "method"]).any():
        raise RuntimeError("duplicate seed-method summary")
    # All variants must start from the same source operating point per seed.
    initial_spread = summary_frame.groupby("source_seed")[
        "source_calibration_f1_before"
    ].agg(lambda values: float(max(values) - min(values)))
    tau_spread = summary_frame.groupby("source_seed")[
        "confidence_nll_threshold_tau_q"
    ].agg(lambda values: float(max(values) - min(values)))
    if (initial_spread > 1e-12).any() or (tau_spread > 1e-12).any():
        raise RuntimeError("paired variants have mismatched source operating points")

    deciles = _build_deciles(sample_frame, source_frame, num_classes=6)
    decile_summary = _aggregate_deciles(deciles)
    source_quality = _source_quality_table(summary_frame)
    atomic_write_csv(sample_frame, output_dir / "stream_sample_records.csv")
    atomic_write_csv(source_frame, output_dir / "source_retention_checkpoints.csv")
    atomic_write_csv(summary_frame, output_dir / "method_seed_summary.csv")
    atomic_write_csv(deciles, output_dir / "stream_deciles_by_seed.csv")
    atomic_write_csv(decile_summary, output_dir / "stream_deciles_summary.csv")
    atomic_write_csv(source_quality, output_dir / "source_quality_audit.csv")
    _atomic_text(_table_markdown(source_quality), output_dir / "source_quality_audit.md")
    png, pdf = _plot(decile_summary, output_dir)
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "dataset": DATASET,
        "scenario": SCENARIO,
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "methods": list(VARIANTS),
        "execution_device": str(args.device),
        "cuda_allocator_environment": {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get(
                "PYTORCH_NO_CUDA_MEMORY_CACHING"
            ),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
        },
        "target_samples_per_cell": 110,
        "source_calibration_samples": 96,
        "deployment_batch_sizes": [48, 48, 14],
        "target_deciles": 10,
        "spline_log_strength": 0.20,
        "source_split_role": (
            "held out from source weight training but reused label-free for "
            "confidence-threshold calibration; not an independent holdout"
        ),
        "prequential_definition": "batch-start raw prediction before any update on that batch",
        "coverage_definition": "actual final-inner-step admission mask",
        "source_retention_definition": "source-calibration F1 after minus before the target stream",
        "target_f1_table_definition": "post-update prediction on each deployment batch",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_offline_metrics": True,
        "confirmatory": False,
        "reference_main_csv_for_source_identity_only": str(reference_path),
        "flow_profile_json": str(Path(args.flow_profile_json).resolve()),
        "runtime_hparams": runtime,
        "outputs": [
            "source_quality_audit.csv",
            "source_quality_audit.md",
            "stream_sample_records.csv",
            "source_retention_checkpoints.csv",
            "method_seed_summary.csv",
            "stream_deciles_by_seed.csv",
            "stream_deciles_summary.csv",
            png.name,
            pdf.name,
            "manifest.json",
        ],
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(_table_markdown(source_quality))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
