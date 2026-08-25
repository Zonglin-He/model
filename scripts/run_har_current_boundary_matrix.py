"""Run the final source-feasible current-boundary SSAW mechanism screen.

Execution is deliberately staged:

1. a no-training candidate audit on HAR 6->23, 9->18, and 12->16;
2. the two-step N2 raw baseline;
3. an exact residual-KL raw duplicate invariant;
4. Fixed-KL and CurrentBoundary-KL only if the invariant passes.

All threshold calibration uses labelled source data only.  Target labels are
read by the trainer after prediction solely for final offline metrics.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe import (  # noqa: E402
    SSAWPhysicalView,
    _extract_features,
    _extract_primary_tensor,
    evaluate_candidate_pool_sequential,
)
from algorithms.dusafe_adaptive_frontier import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    AdaptiveFrontierRunner,
)
from algorithms.dusafe_current_boundary import (  # noqa: E402
    CURRENT_BOUNDARY_CALIBRATION_QUANTILE,
    CURRENT_BOUNDARY_CALIBRATION_SOBOL_SEED,
    CURRENT_BOUNDARY_RUNNERS,
    CurrentBoundaryRunner,
    _pseudo_class_probability_gap,
    get_current_boundary_runner,
)
from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView  # noqa: E402
from scripts.dusafe_factorial_runner_common import (  # noqa: E402
    current_profiles,
    tensor_state_sha256,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_har_spline_router_ablation import _LimitedLoader  # noqa: E402
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


PROTOCOL = "har_current_boundary_v1_source_only_q25_steps2"
FLOWS = (("6", "23"), ("9", "18"), ("12", "16"))
SOURCE_SEED = 1
STREAM_SEED = 42
INNER_STEPS = 2
AUDIT_RUNNER = "CurrentBoundary_KL"
MATRIX_RUNNERS = (
    "N2_confidence_raw",
    "Fixed_KL_current_B4",
    "CurrentBoundary_KL",
    "CurrentBoundary_Dup",
)
EXECUTION_ORDER = (
    "N2_confidence_raw",
    "CurrentBoundary_Dup",
    "Fixed_KL_current_B4",
    "CurrentBoundary_KL",
)
CURRENT_BOUNDARY_PROFILE = {
    "steps": INNER_STEPS,
    "spline_control_points": 10,
    "spline_num_directions": 4,
    "spline_log_strength": 0.20,
    "spline_radius_levels": (1.0, 0.5, 0.25),
    "spline_search_steps": 2,
    "spline_search_step_size": 0.5,
    "spline_search_log_strength": 0.20,
    "frontier_alpha_grid": DEFAULT_ALPHA_GRID,
    "frontier_hard_quantile": 0.90,
    "frontier_restore_quantile": 0.75,
    "frontier_gradient_budget": 0.50,
    "frontier_source_preservation": 0.99,
    "current_boundary_calibration_quantile": (
        CURRENT_BOUNDARY_CALIBRATION_QUANTILE
    ),
    "record_optimizer_diagnostics": True,
}
AUDIT_PROFILE = {
    **CURRENT_BOUNDARY_PROFILE,
    "steps": 1,
    "enable_adaptation": False,
    "record_current_boundary_candidates": True,
}
HELDOUT_SOBOL_SEED = 161_803
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "har_current_boundary_seed1_steps2_v1"
)
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _cell_dir(
    output_dir: Path,
    phase: str,
    flow: Sequence[str],
    runner: str,
) -> Path:
    return (
        output_dir
        / "cells"
        / phase
        / f"flow_{flow[0]}_to_{flow[1]}"
        / runner
    )


def _signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "phase": str(spec["phase"]),
        "flow": list(spec["flow"]),
        "runner": str(spec["runner"]),
        "source_seed": int(spec["source_seed"]),
        "stream_seed": int(spec["stream_seed"]),
        "source_config": spec["source_config"],
        "tta_config": spec["tta_config"],
        "max_batches": spec.get("max_batches"),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_boundary_selection": False,
        "target_metrics_used_for_boundary_selection": False,
    }


def _required_cell_files(phase: str) -> tuple[str, ...]:
    if phase == "audit":
        return (
            "summary.json",
            "candidate_records.csv",
            "audit_sample_records.csv",
            "batch_diagnostics.csv",
        )
    return (
        "summary.json",
        "sample_records.csv",
        "batch_diagnostics.csv",
        "heldout_stable_radius.csv",
    )


def _complete(cell_dir: Path, phase: str, signature_hash: str) -> bool:
    summary_path = cell_dir / "summary.json"
    if not all((cell_dir / name).is_file() for name in _required_cell_files(phase)):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "ok"
        and summary.get("protocol") == PROTOCOL
        and summary.get("phase") == phase
        and summary.get("signature_hash") == signature_hash
    )


def _reference_cache_signature(
    spec: Mapping[str, object], source_hash: str
) -> dict[str, object]:
    tta = dict(spec["tta_config"])
    return {
        "protocol": PROTOCOL,
        "source_domain": str(spec["flow"][0]),
        "source_seed": int(spec["source_seed"]),
        "source_model_sha256": source_hash,
        "frontier_alpha_grid": list(tta["frontier_alpha_grid"]),
        "frontier_restore_quantile": float(tta["frontier_restore_quantile"]),
        "frontier_source_preservation": float(
            tta["frontier_source_preservation"]
        ),
        "frontier_calibration_sobol_seed": 271_828,
        "current_boundary_calibration_quantile": float(
            tta["current_boundary_calibration_quantile"]
        ),
        "current_boundary_calibration_sobol_seed": (
            CURRENT_BOUNDARY_CALIBRATION_SOBOL_SEED
        ),
        "spline_control_points": int(tta["spline_control_points"]),
        "spline_num_directions": int(tta["spline_num_directions"]),
        "spline_search_steps": int(tta["spline_search_steps"]),
        "spline_search_step_size": float(tta["spline_search_step_size"]),
        "target_labels_used": False,
        "target_metrics_used": False,
    }


def _reference_cache_path(spec: Mapping[str, object]) -> Path:
    cache_dir = Path(spec["reference_cache_dir"])
    return cache_dir / (
        f"HAR_source_{spec['flow'][0]}_seed_{spec['source_seed']}.pt"
    )


def _load_reference_payload(
    spec: Mapping[str, object], source_hash: str
) -> tuple[dict[str, object] | None, str, bool]:
    cache_path = _reference_cache_path(spec)
    signature_hash = _hash_json(_reference_cache_signature(spec, source_hash))
    if not cache_path.is_file():
        return None, str(cache_path), False
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None, str(cache_path), False
    if not isinstance(payload, Mapping) or payload.get("signature_hash") != signature_hash:
        return None, str(cache_path), False
    if not isinstance(payload.get("source_frontier"), Mapping) or not isinstance(
        payload.get("current_boundary"), Mapping
    ):
        return None, str(cache_path), False
    return dict(payload), str(cache_path), True


def _load_or_fit_references(
    adapter,
    trainer,
    spec: Mapping[str, object],
    source_hash: str,
) -> tuple[dict[str, object], str, bool]:
    payload, cache_path_string, hit = _load_reference_payload(spec, source_hash)
    if payload is not None:
        if isinstance(adapter, CurrentBoundaryRunner):
            adapter.load_source_frontier_reference(payload["source_frontier"])
            adapter.load_current_boundary_reference(payload["current_boundary"])
        return payload, cache_path_string, hit
    if not isinstance(adapter, CurrentBoundaryRunner):
        raise RuntimeError(
            "current-boundary source cache is missing; audit phase must run first"
        )
    source_frontier = adapter.fit_source_frontier_reference(
        trainer.src_test_dl,
        reference_samples=4096,
        calibration_sobol_seed=271_828,
    )
    current_boundary = adapter.fit_current_boundary_reference(
        trainer.src_test_dl,
        reference_samples=4096,
        calibration_sobol_seed=CURRENT_BOUNDARY_CALIBRATION_SOBOL_SEED,
    )
    signature = _reference_cache_signature(spec, source_hash)
    payload = {
        "signature": signature,
        "signature_hash": _hash_json(signature),
        "source_frontier": source_frontier,
        "current_boundary": current_boundary,
    }
    cache_path = Path(cache_path_string)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return payload, str(cache_path), False


@torch.no_grad()
def _frozen_source_forward(
    adapter,
    frozen_classifier,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapter._configure_frozen_semantic_extractor()
    frozen_classifier.eval()
    features = adapter.source_semantic_feature_extractor(inputs)
    if isinstance(features, (tuple, list)):
        features = features[0]
    logits = frozen_classifier(features)
    normalized = torch.nn.functional.normalize(features.flatten(1), dim=1)
    semantic = (normalized @ adapter.source_semantic_prototypes.t()).argmax(dim=1)
    return logits, semantic


@torch.no_grad()
def _frozen_source_candidate_forward(
    adapter,
    frozen_classifier,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, semantics = [], []
    for candidate in candidates.unbind(dim=0):
        current_logits, current_semantic = _frozen_source_forward(
            adapter, frozen_classifier, candidate
        )
        logits.append(current_logits)
        semantics.append(current_semantic)
    return torch.stack(logits), torch.stack(semantics)


def _heldout_stable_radius_audit(
    adapter,
    frozen_classifier,
    target_loader,
    *,
    source_safe_alpha_cap: float,
    max_batches: int | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Measure raywise stable radius on directions never used for training."""
    device = next(adapter.model.parameters()).device
    maximum_alpha = max(DEFAULT_ALPHA_GRID)
    heldout_view = UnifiedSplineHardView(
        num_control_points=int(adapter.spline_control_points),
        num_directions=int(adapter.spline_num_directions),
        log_strength=maximum_alpha,
        radius_levels=tuple(value / maximum_alpha for value in DEFAULT_ALPHA_GRID),
        sobol_seed=HELDOUT_SOBOL_SEED,
    )
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(target_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = _extract_primary_tensor(batch)
        if not torch.is_tensor(inputs) or inputs.dim() != 3:
            raise ValueError("held-out radius audit expects [B,C,T]")
        inputs = inputs.float().to(device)
        with SSAWPhysicalView._preserved_bn_buffers(adapter.model):
            raw_features = _extract_features(adapter.model, inputs)
            raw_logits = adapter.model.classifier(raw_features)
        raw_labels = raw_logits.argmax(dim=1)
        raw_nll = -raw_logits.log_softmax(dim=1).gather(
            1, raw_labels[:, None]
        ).squeeze(1)
        confidence = raw_nll.le(adapter.confidence_nll_threshold)
        raw_source_logits, raw_semantic = _frozen_source_forward(
            adapter, frozen_classifier, inputs
        )
        raw_source_supported = (
            raw_source_logits.argmax(dim=1).eq(raw_labels)
            & raw_semantic.eq(raw_labels)
        )
        prepared = heldout_view.prepare_view_inputs(
            inputs,
            normalization_mean=adapter.source_normalization_mean,
            normalization_std=adapter.source_normalization_std,
            reuse_cached_view=False,
        )
        candidates = torch.as_tensor(prepared["view_inputs"])
        _, candidate_logits = evaluate_candidate_pool_sequential(
            adapter.model, candidates, require_grad=False
        )
        source_logits, source_semantic = _frozen_source_candidate_forward(
            adapter, frozen_classifier, candidates
        )
        expanded_labels = raw_labels[None].expand(candidates.size(0), -1)
        candidate_alpha = (
            maximum_alpha * heldout_view._cached_radius_values
        ).to(device)
        within_cap = candidate_alpha.le(source_safe_alpha_cap + 1e-8)
        stable = (
            within_cap[:, None]
            & raw_source_supported[None]
            & candidate_logits.argmax(dim=2).eq(expanded_labels)
            & source_logits.argmax(dim=2).eq(expanded_labels)
            & source_semantic.eq(expanded_labels)
        )
        ray_stable = stable.reshape(
            heldout_view.ray_count, len(DEFAULT_ALPHA_GRID), inputs.size(0)
        )
        ray_alpha = candidate_alpha.reshape(
            heldout_view.ray_count, len(DEFAULT_ALPHA_GRID), 1
        )
        stable_radius_by_ray = ray_alpha.masked_fill(~ray_stable, 0.0).amax(dim=1)
        stable_radius = stable_radius_by_ray.mean(dim=0)
        cap_rows = torch.isclose(
            candidate_alpha,
            torch.tensor(
                source_safe_alpha_cap,
                device=device,
                dtype=candidate_alpha.dtype,
            ),
            rtol=0.0,
            atol=1e-6,
        )
        cap_stable_fraction = (
            stable[cap_rows].float().mean(dim=0)
            if cap_rows.any()
            else torch.zeros(inputs.size(0), device=device)
        )
        for sample_index in range(inputs.size(0)):
            rows.append(
                {
                    "batch_index": batch_index,
                    "sample_in_batch": sample_index,
                    "confidence_admitted": bool(confidence[sample_index]),
                    "raw_source_supported": bool(
                        raw_source_supported[sample_index]
                    ),
                    "eligible": bool(
                        confidence[sample_index] & raw_source_supported[sample_index]
                    ),
                    "stable_radius": float(stable_radius[sample_index]),
                    "stable_radius_over_cap": float(
                        stable_radius[sample_index]
                        / max(source_safe_alpha_cap, 1e-12)
                    ),
                    "cap_stable_ray_fraction": float(
                        cap_stable_fraction[sample_index]
                    ),
                }
            )
        heldout_view.clear_cached_view()
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["eligible"]] if not frame.empty else frame
    if eligible.empty:
        summary = {
            "heldout_stable_radius_mean": math.nan,
            "heldout_stable_radius_over_cap_mean": math.nan,
            "heldout_cap_stable_ray_fraction_mean": math.nan,
            "heldout_eligible_samples": 0.0,
        }
    else:
        summary = {
            "heldout_stable_radius_mean": float(eligible["stable_radius"].mean()),
            "heldout_stable_radius_over_cap_mean": float(
                eligible["stable_radius_over_cap"].mean()
            ),
            "heldout_cap_stable_ray_fraction_mean": float(
                eligible["cap_stable_ray_fraction"].mean()
            ),
            "heldout_eligible_samples": float(len(eligible)),
        }
        for quantile in (0.1, 0.5, 0.9):
            summary[f"heldout_stable_radius_q{int(100 * quantile)}"] = float(
                eligible["stable_radius"].quantile(quantile)
            )
    return summary, frame


def _run_cell(spec: Mapping[str, object]):
    phase = str(spec["phase"])
    flow = tuple(str(value) for value in spec["flow"])
    runner_name = str(spec["runner"])
    trainer = build_trainer(
        data_path=str(spec["data_path"]),
        device=str(spec["device"]),
        dataset="HAR",
        da_method="DuSafe",
        backbone=str(spec["backbone"]),
        exp_name=f"current_boundary_{phase}_{runner_name}",
        seed=int(spec["stream_seed"]),
        source_seed=int(spec["source_seed"]),
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    adapted = source_model = frozen_classifier = None
    try:
        runner_class = get_current_boundary_runner(runner_name)
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(spec["source_config"]))
        trainer.set_runtime_hparams(dict(spec["tta_config"]))
        adapted, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=int(spec["stream_seed"])
        )
        source_hash = tensor_state_sha256(source_model)
        source_checkpoint = str(trainer._pretrain_cache_path() or "")
        frozen_classifier = copy.deepcopy(adapted.model.classifier).eval()
        for parameter in frozen_classifier.parameters():
            parameter.requires_grad_(False)
        reference_payload, reference_cache_path, reference_cache_hit = (
            _load_or_fit_references(
                adapted, trainer, spec, source_hash
            )
        )
        if spec.get("max_batches") is not None:
            trainer.trg_whole_dl = _LimitedLoader(
                trainer.trg_whole_dl, int(spec["max_batches"])
            )
        metrics = trainer.calculate_metrics(adapted)
        common = {
            "status": "ok",
            "protocol": PROTOCOL,
            "phase": phase,
            "dataset": "HAR",
            "scenario": _flow_label(flow),
            "source_seed": int(spec["source_seed"]),
            "stream_seed": int(spec["stream_seed"]),
            "inner_steps": int(spec["tta_config"]["steps"]),
            "runner": runner_name,
            "runner_class": runner_class.__name__,
            "source_model_sha256": source_hash,
            "source_checkpoint_path": source_checkpoint,
            "reference_cache_path": reference_cache_path,
            "reference_cache_hit": bool(reference_cache_hit),
            "source_safe_alpha_cap": float(
                reference_payload["source_frontier"]["safe_alpha_cap"]
            ),
            "current_boundary_rho_star": float(
                reference_payload["current_boundary"]["rho_star"]
            ),
            "current_boundary_tau_g": float(
                reference_payload["current_boundary"]["tau_g"]
            ),
            "source_boundary_joint_reach_rate": float(
                reference_payload["current_boundary"]["joint_reach_rate"]
            ),
            "target_labels_used_for_online_decision": False,
            "target_labels_used_for_boundary_selection": False,
            "target_metrics_used_for_boundary_selection": False,
            "evaluation_partition": "target_selected_descriptive",
        }
        batches = getattr(trainer, "last_batch_log_records", pd.DataFrame()).copy()
        if not batches.empty:
            batches.insert(0, "batch_index", range(len(batches)))
        for name, value in reversed(
            (
                ("dataset", "HAR"),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("runner", runner_name),
                ("phase", phase),
            )
        ):
            batches.insert(0, name, value)

        if phase == "audit":
            if not isinstance(adapted, CurrentBoundaryRunner):
                raise RuntimeError("audit requires CurrentBoundaryRunner")
            candidate_records = pd.DataFrame(
                adapted.current_boundary_candidate_records
            )
            audit_samples = pd.DataFrame(adapted.current_boundary_sample_records)
            for frame in (candidate_records, audit_samples):
                for name, value in reversed(
                    (
                        ("dataset", "HAR"),
                        ("scenario", _flow_label(flow)),
                        ("source_seed", int(spec["source_seed"])),
                        ("stream_seed", int(spec["stream_seed"])),
                    )
                ):
                    frame.insert(0, name, value)
            result = {
                **common,
                "accuracy": float(metrics[0]),
                "f1": float(metrics[1]),
                "candidate_record_count": int(len(candidate_records)),
                "audit_sample_count": int(len(audit_samples)),
                "target_training_performed": False,
            }
            return result, candidate_records, audit_samples, batches, pd.DataFrame()

        samples = trainer.last_safety_records.copy()
        for name, value in reversed(
            (
                ("dataset", "HAR"),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("runner", runner_name),
            )
        ):
            samples.insert(0, name, value)
        heldout_summary, heldout_records = _heldout_stable_radius_audit(
            adapted,
            frozen_classifier,
            trainer.trg_whole_dl,
            source_safe_alpha_cap=float(
                reference_payload["source_frontier"]["safe_alpha_cap"]
            ),
            max_batches=spec.get("max_batches"),
        )
        for name, value in reversed(
            (
                ("dataset", "HAR"),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("runner", runner_name),
            )
        ):
            heldout_records.insert(0, name, value)
        result = {
            **common,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            "sample_count": int(len(samples)),
            "batch_count": int(samples["batch_index"].nunique()),
            **heldout_summary,
            **dict(getattr(trainer, "last_prediction_metric_summary", {}) or {}),
            **dict(getattr(trainer, "last_safety_summary", {}) or {}),
        }
        result.update(
            {
                f"diag_{name}": float(value)
                for name, value in (
                    getattr(trainer, "last_batch_log_summary", {}) or {}
                ).items()
            }
        )
        return result, samples, pd.DataFrame(), batches, heldout_records
    finally:
        cleanup_trainer(
            trainer,
            adapted,
            source_model,
            frozen_classifier,
            close_summary=True,
        )
        adapted = source_model = frozen_classifier = None
        release_cuda()
        gc.collect()


def _worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    phase = str(spec["phase"])
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    signature_hash = _hash_json(_signature(spec))
    if _complete(cell_dir, phase, signature_hash):
        return 0
    try:
        lock = (
            wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
            if str(spec["device"]).lower().startswith("cuda")
            else None
        )
        if lock is None:
            outputs = _run_cell(spec)
        else:
            with lock:
                outputs = _run_cell(spec)
        result, samples, audit_samples, batches, heldout = outputs
        if phase == "audit":
            atomic_write_csv(samples, cell_dir / "candidate_records.csv", index=False)
            atomic_write_csv(
                audit_samples, cell_dir / "audit_sample_records.csv", index=False
            )
        else:
            atomic_write_csv(samples, cell_dir / "sample_records.csv", index=False)
            atomic_write_csv(
                heldout, cell_dir / "heldout_stable_radius.csv", index=False
            )
        atomic_write_csv(batches, cell_dir / "batch_diagnostics.csv", index=False)
        result["signature_hash"] = signature_hash
        atomic_write_json(result, cell_dir / "summary.json")
        return 0
    except BaseException as exc:
        atomic_write_json(
            {
                "status": "failed",
                "protocol": PROTOCOL,
                "phase": phase,
                "scenario": _flow_label(spec["flow"]),
                "runner": spec["runner"],
                "signature_hash": signature_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exc).lower(),
            },
            cell_dir / "summary.json",
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        release_cuda()
        gc.collect()


def _read_phase(output_dir: Path, phase: str, runners: Sequence[str]):
    summaries, samples, audit_samples, batches, heldout = [], [], [], [], []
    for flow in FLOWS:
        for runner in runners:
            cell_dir = _cell_dir(output_dir, phase, flow, runner)
            summary_path = cell_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("status") != "ok"
                or summary.get("protocol") != PROTOCOL
                or summary.get("phase") != phase
            ):
                continue
            summaries.append(summary)
            if phase == "audit":
                samples.append(pd.read_csv(cell_dir / "candidate_records.csv"))
                audit_samples.append(
                    pd.read_csv(cell_dir / "audit_sample_records.csv")
                )
            else:
                samples.append(pd.read_csv(cell_dir / "sample_records.csv"))
                heldout.append(
                    pd.read_csv(cell_dir / "heldout_stable_radius.csv")
                )
            batches.append(pd.read_csv(cell_dir / "batch_diagnostics.csv"))
    return (
        pd.DataFrame(summaries),
        pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(),
        pd.concat(audit_samples, ignore_index=True)
        if audit_samples
        else pd.DataFrame(),
        pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(),
        pd.concat(heldout, ignore_index=True) if heldout else pd.DataFrame(),
    )


def _quantiles(series: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {f"{prefix}_p{value}": math.nan for value in (10, 50, 90)}
    return {
        f"{prefix}_p{int(100 * quantile)}": float(numeric.quantile(quantile))
        for quantile in (0.1, 0.5, 0.9)
    }


def _audit_summary(
    candidates: pd.DataFrame, samples: pd.DataFrame
) -> pd.DataFrame:
    if candidates.empty or samples.empty:
        return pd.DataFrame()
    rows = []
    for scenario in sorted(samples["scenario"].unique()):
        candidate_group = candidates.loc[candidates["scenario"] == scenario].copy()
        sample_group = samples.loc[samples["scenario"] == scenario].copy()
        feasible = candidate_group.loc[
            candidate_group["source_valid"].astype(bool)
            & candidate_group["current_label_preserving"].astype(bool)
        ]
        supported = sample_group.loc[
            sample_group["raw_source_supported"].astype(bool)
        ]
        old_reached = supported.loc[
            supported["source_frontier_reach"].astype(bool)
        ]
        spearman = (
            math.nan
            if len(feasible) < 2
            else float(
                feasible[["source_percentile", "probability_gap_reduction"]]
                .corr(method="spearman")
                .iloc[0, 1]
            )
        )
        reached = supported.loc[
            supported["current_boundary_reach"].astype(bool)
        ]
        row = {
            "scenario": scenario,
            "candidate_rows": int(len(candidate_group)),
            "source_feasible_candidate_rows": int(len(feasible)),
            "raw_supported_samples": int(len(supported)),
            "source_percentile_gap_reduction_spearman": spearman,
            "raw_source_percentile_ge_90_rate": float(
                supported["raw_source_percentile"].ge(0.90).mean()
            )
            if len(supported)
            else math.nan,
            "source_frontier_reach_rate": float(
                supported["source_frontier_reach"].astype(bool).mean()
            )
            if len(supported)
            else math.nan,
            "current_boundary_reach_rate": float(
                supported["current_boundary_reach"].astype(bool).mean()
            )
            if len(supported)
            else math.nan,
            "source_frontier_reached_current_boundary_fraction": float(
                old_reached["current_boundary_reach"].astype(bool).mean()
            )
            if len(old_reached)
            else math.nan,
            "selected_cap_hit_rate": float(reached["cap_hit"].astype(bool).mean())
            if len(reached)
            else math.nan,
            "no_reach_rate": float(supported["no_reach"].astype(bool).mean())
            if len(supported)
            else math.nan,
            **_quantiles(
                supported["raw_source_percentile"], "raw_source_percentile"
            ),
            **_quantiles(
                feasible["source_percentile_delta"],
                "view_source_percentile_delta",
            ),
            **_quantiles(reached["selected_alpha"], "selected_alpha"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _publish_audit(output_dir: Path) -> pd.DataFrame:
    raw, candidates, samples, batches, _ = _read_phase(
        output_dir, "audit", (AUDIT_RUNNER,)
    )
    atomic_write_csv(raw, output_dir / "audit_raw.csv", index=False)
    atomic_write_csv(
        candidates, output_dir / "audit_candidate_records.csv", index=False
    )
    atomic_write_csv(samples, output_dir / "audit_sample_records.csv", index=False)
    atomic_write_csv(
        batches, output_dir / "audit_batch_diagnostics.csv", index=False
    )
    summary = _audit_summary(candidates, samples)
    atomic_write_csv(summary, output_dir / "no_training_audit_summary.csv", index=False)
    return summary


def _paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    pivot = raw.pivot(index="scenario", columns="runner", values="f1")
    baseline = "N2_confidence_raw"
    if baseline not in pivot.columns:
        return pd.DataFrame()
    rows = []
    for runner in MATRIX_RUNNERS:
        if runner == baseline or runner not in pivot.columns:
            continue
        delta = pivot[runner] - pivot[baseline]
        for scenario, value in delta.items():
            rows.append(
                {
                    "scenario": scenario,
                    "contrast": f"{runner}-N2",
                    "delta_f1": float(value),
                }
            )
        rows.append(
            {
                "scenario": "mean_three_flows",
                "contrast": f"{runner}-N2",
                "delta_f1": float(delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def _duplicate_invariant(
    raw: pd.DataFrame, samples: pd.DataFrame, batches: pd.DataFrame
) -> dict[str, object]:
    baseline = "N2_confidence_raw"
    duplicate = "CurrentBoundary_Dup"
    if raw.empty or not {baseline, duplicate}.issubset(set(raw["runner"])):
        return {"status": "incomplete"}
    problems = []
    compared_samples = 0
    for scenario in (_flow_label(flow) for flow in FLOWS):
        pair = raw.loc[
            (raw["scenario"] == scenario)
            & raw["runner"].isin((baseline, duplicate))
        ]
        if len(pair) != 2:
            problems.append({"scenario": scenario, "reason": "missing summary"})
            continue
        f1_by_runner = pair.set_index("runner")["f1"]
        if float(f1_by_runner[baseline]) != float(f1_by_runner[duplicate]):
            problems.append(
                {
                    "scenario": scenario,
                    "reason": "f1_mismatch",
                    "n2": float(f1_by_runner[baseline]),
                    "dup": float(f1_by_runner[duplicate]),
                }
            )
        columns = [
            "batch_index",
            "sample_index",
            "pseudo_label",
            "prediction",
            "post_update_prediction",
            "pre_final_update_prediction",
            "admitted",
            "selected",
        ]
        left = samples.loc[
            (samples["scenario"] == scenario) & (samples["runner"] == baseline),
            columns,
        ].sort_values(["batch_index", "sample_index"]).reset_index(drop=True)
        right = samples.loc[
            (samples["scenario"] == scenario) & (samples["runner"] == duplicate),
            columns,
        ].sort_values(["batch_index", "sample_index"]).reset_index(drop=True)
        compared_samples += len(left)
        if not left.equals(right):
            problems.append(
                {
                    "scenario": scenario,
                    "reason": "sample_trajectory_mismatch",
                    "n2_rows": int(len(left)),
                    "dup_rows": int(len(right)),
                }
            )
        duplicate_batches = batches.loc[
            (batches["scenario"] == scenario)
            & (batches["runner"] == duplicate)
        ]
        for loss_column in (
            "ssaw_consistency_loss",
            "ssaw_weighted_consistency_loss",
        ):
            if loss_column in duplicate_batches and not duplicate_batches[
                loss_column
            ].fillna(0.0).eq(0.0).all():
                problems.append(
                    {
                        "scenario": scenario,
                        "reason": f"nonzero_{loss_column}",
                        "maximum": float(
                            duplicate_batches[loss_column].abs().max()
                        ),
                    }
                )
    return {
        "status": "passed" if not problems else "failed",
        "compared_samples": int(compared_samples),
        "problems": problems,
        "contract": "same current-boundary mask and denominator; exact raw residual KL is zero",
    }


def _promotion_decision(
    raw: pd.DataFrame,
    batches: pd.DataFrame,
    duplicate_invariant: Mapping[str, object],
) -> dict[str, object]:
    primary = "CurrentBoundary_KL"
    baseline = "N2_confidence_raw"
    if raw.empty or not {primary, baseline}.issubset(set(raw["runner"])):
        return {"status": "incomplete"}
    pivot = raw.pivot(index="scenario", columns="runner", values="f1")
    if len(pivot) != len(FLOWS) or pivot[[primary, baseline]].isna().any().any():
        return {"status": "incomplete"}
    delta = pivot[primary] - pivot[baseline]
    mean_delta = float(delta.mean())
    positive = int(delta.gt(0.0).sum())
    worst = float(delta.min())
    primary_batches = batches.loc[batches["runner"] == primary]
    violation_count = (
        math.inf
        if "current_boundary_gathered_violation_count" not in primary_batches
        else float(
            primary_batches["current_boundary_gathered_violation_count"]
            .fillna(0.0)
            .sum()
        )
    )
    coverage_by_flow = (
        primary_batches.groupby("scenario")["current_boundary_final_coverage"]
        .mean()
        .to_dict()
        if "current_boundary_final_coverage" in primary_batches
        else {}
    )
    stable = raw.pivot(
        index="scenario", columns="runner", values="heldout_stable_radius_mean"
    )
    stable_delta = (
        float((stable[primary] - stable[baseline]).mean())
        if {primary, baseline}.issubset(stable.columns)
        else math.nan
    )
    gathered_pass = bool(
        violation_count == 0.0
        and len(coverage_by_flow) == len(FLOWS)
        and all(float(value) > 0.0 for value in coverage_by_flow.values())
    )
    passed = bool(
        duplicate_invariant.get("status") == "passed"
        and mean_delta >= 0.003
        and positive >= 2
        and worst >= -0.01
        and gathered_pass
        and math.isfinite(stable_delta)
        and stable_delta > 0.0
    )
    return {
        "status": "passed" if passed else "failed",
        "primary": primary,
        "baseline": baseline,
        "mean_delta_f1": mean_delta,
        "positive_flows": positive,
        "worst_flow_delta_f1": worst,
        "gathered_boundary_violation_count": violation_count,
        "final_coverage_by_flow": {
            key: float(value) for key, value in coverage_by_flow.items()
        },
        "heldout_stable_radius_mean_delta": stable_delta,
        "duplicate_invariant": duplicate_invariant.get("status"),
        "thresholds": {
            "mean_delta_f1": 0.003,
            "positive_flows": 2,
            "worst_flow_delta_f1": -0.01,
            "gathered_boundary_violations": 0,
            "positive_coverage_each_flow": True,
            "heldout_stable_radius_mean_delta": ">0",
        },
    }


def _publish_matrix(output_dir: Path):
    raw, samples, _, batches, heldout = _read_phase(
        output_dir, "matrix", MATRIX_RUNNERS
    )
    if not raw.empty:
        raw = raw.sort_values(["scenario", "runner"]).reset_index(drop=True)
    atomic_write_csv(raw, output_dir / "matrix_raw.csv", index=False)
    atomic_write_csv(samples, output_dir / "matrix_sample_records.csv", index=False)
    atomic_write_csv(batches, output_dir / "matrix_batch_diagnostics.csv", index=False)
    atomic_write_csv(
        heldout, output_dir / "heldout_stable_radius_records.csv", index=False
    )
    if not raw.empty:
        table = raw.pivot(index="scenario", columns="runner", values="f1")
        atomic_write_csv(
            table.reset_index(), output_dir / "flow_f1_table.csv", index=False
        )
    atomic_write_csv(
        _paired_effects(raw), output_dir / "paired_effects.csv", index=False
    )
    invariant = _duplicate_invariant(raw, samples, batches)
    atomic_write_json(invariant, output_dir / "duplicate_invariant.json")
    promotion = _promotion_decision(raw, batches, invariant)
    atomic_write_json(promotion, output_dir / "promotion_decision.json")
    return raw, samples, batches, invariant, promotion


def _build_spec(
    args,
    *,
    phase: str,
    flow: Sequence[str],
    runner: str,
) -> dict[str, object]:
    source_config, tta_config = current_profiles("HAR")
    profile = AUDIT_PROFILE if phase == "audit" else CURRENT_BOUNDARY_PROFILE
    tta_config = {**tta_config, **profile}
    return {
        "cell_dir": str(
            _cell_dir(args.output_dir, phase, flow, runner).resolve()
        ),
        "reference_cache_dir": str(
            (args.output_dir / "source_reference_cache").resolve()
        ),
        "phase": phase,
        "flow": list(flow),
        "runner": runner,
        "source_seed": SOURCE_SEED,
        "stream_seed": STREAM_SEED,
        "source_config": source_config,
        "tta_config": tta_config,
        "data_path": str(args.data_path.resolve()),
        "device": args.device,
        "backbone": args.backbone,
        "pretrain_cache_dir": str(args.pretrain_cache_dir.resolve()),
        "gpu_lock_path": str(args.gpu_lock_path.resolve()),
        "max_batches": args.max_batches,
    }


def _run_spec(spec: Mapping[str, object]) -> tuple[bool, dict[str, object] | None]:
    cell_dir = Path(spec["cell_dir"])
    phase = str(spec["phase"])
    signature_hash = _hash_json(_signature(spec))
    if _complete(cell_dir, phase, signature_hash):
        return True, None
    cell_dir.mkdir(parents=True, exist_ok=True)
    spec_path = cell_dir / "worker_spec.json"
    atomic_write_json(dict(spec), spec_path)
    log_path = cell_dir / "worker.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode == 0:
        return True, None
    return False, {
        "phase": phase,
        "scenario": _flow_label(spec["flow"]),
        "runner": spec["runner"],
        "returncode": int(process.returncode),
        "log": str(log_path),
    }


def _validate_source_identity(raw: pd.DataFrame) -> dict[str, object]:
    if raw.empty:
        return {"status": "incomplete"}
    problems = []
    for scenario, group in raw.groupby("scenario"):
        hashes = sorted(set(group["source_model_sha256"].dropna().astype(str)))
        if len(hashes) != 1:
            problems.append({"scenario": scenario, "hashes": hashes})
    return {"status": "passed" if not problems else "failed", "problems": problems}


def _run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_specs = [
        _build_spec(args, phase="audit", flow=flow, runner=AUDIT_RUNNER)
        for flow in FLOWS
    ]
    matrix_specs = [
        _build_spec(args, phase="matrix", flow=flow, runner=runner)
        for runner in EXECUTION_ORDER
        for flow in FLOWS
    ]
    manifest = {
        "protocol": PROTOCOL,
        "status": "running_audit",
        "dataset": "HAR",
        "flows": [_flow_label(flow) for flow in FLOWS],
        "source_seed": SOURCE_SEED,
        "stream_seed": STREAM_SEED,
        "audit_cells": len(audit_specs),
        "matrix_cells": len(matrix_specs),
        "matrix_runners": list(MATRIX_RUNNERS),
        "execution_order": list(EXECUTION_ORDER),
        "inner_steps": INNER_STEPS,
        "profile": CURRENT_BOUNDARY_PROFILE,
        "source_threshold_rule": "q25 of per-anchor hardest source-feasible current probability gap and ratio",
        "source_feasible_rule": "per-candidate frozen source classifier and semantic agreement within source-safe cap",
        "candidate_evaluation": "one [B,C,T] forward per view; gathered mixed batch re-forwarded",
        "easy_view_fallback_trained": False,
        "duplicate_contract": "same masks/weights/denominator; exact raw residual KL equals zero",
        "heldout_direction_seed": HELDOUT_SOBOL_SEED,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_boundary_selection": False,
        "target_metrics_used_for_boundary_selection": False,
        "max_batches": args.max_batches,
        "fail_closed_after_duplicate_invariant": True,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    failures = []
    completed = 0

    for spec in audit_specs:
        ok, failure = _run_spec(spec)
        if not ok:
            failures.append(failure)
            break
        completed += 1
        _publish_audit(args.output_dir)
        atomic_write_json(
            {
                **manifest,
                "status": "running_audit",
                "completed_cells": completed,
                "failures": failures,
            },
            args.output_dir / "status.json",
        )
    audit_summary = _publish_audit(args.output_dir)
    audit_complete = bool(
        not failures
        and len(audit_summary) == len(FLOWS)
        and audit_summary["source_feasible_candidate_rows"].gt(0).all()
    )
    if not audit_complete:
        final = {
            **manifest,
            "status": "failed_audit",
            "completed_cells": completed,
            "failures": failures,
        }
        atomic_write_json(final, args.output_dir / "manifest.json")
        atomic_write_json(final, args.output_dir / "status.json")
        return 1

    invariant_checked = False
    invariant = {"status": "incomplete"}
    for spec in matrix_specs:
        ok, failure = _run_spec(spec)
        if not ok:
            failures.append(failure)
            break
        completed += 1
        raw, _, batches, invariant, _ = _publish_matrix(args.output_dir)
        # The first six matrix cells are N2 and Dup for all three flows.
        if str(spec["runner"]) == "CurrentBoundary_Dup" and tuple(spec["flow"]) == FLOWS[-1]:
            invariant_checked = True
            if invariant.get("status") != "passed":
                failures.append(
                    {
                        "phase": "matrix",
                        "reason": "duplicate_invariant_failed",
                        "details": invariant,
                    }
                )
                break
        atomic_write_json(
            {
                **manifest,
                "status": "running_matrix",
                "completed_cells": completed,
                "duplicate_invariant": invariant,
                "failures": failures,
            },
            args.output_dir / "status.json",
        )

    raw, _, batches, invariant, promotion = _publish_matrix(args.output_dir)
    source_identity = _validate_source_identity(raw)
    expected_total = len(audit_specs) + len(matrix_specs)
    status = (
        "complete"
        if completed == expected_total
        and not failures
        and invariant_checked
        and invariant.get("status") == "passed"
        and source_identity["status"] == "passed"
        else "failed"
    )
    final = {
        **manifest,
        "status": status,
        "completed_cells": completed,
        "expected_cells": expected_total,
        "audit_complete": audit_complete,
        "duplicate_invariant": invariant,
        "source_identity": source_identity,
        "promotion_decision": promotion,
        "failures": failures,
    }
    atomic_write_json(final, args.output_dir / "manifest.json")
    atomic_write_json(final, args.output_dir / "status.json")
    return 0 if status == "complete" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "Dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--max-batches", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        return _worker(args.worker_spec)
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
