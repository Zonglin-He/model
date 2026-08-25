"""Audit diagnostics against the current production DuSafe protocol.

This runner deliberately does not import archived SSAW candidate-search or
HCW/SFC code.  It evaluates the production ``SSAWPhysicalView`` (a fixed
antithetic sensor-calibration pair), and it reuses the trainer's independent
synthetic corruption-mask accounting for safety metrics.

The runner has two phases:

* ``plausibility`` records per-view residual spectrum, total variation, and
  distance in the frozen source semantic feature space for every five-domain
  transfer scenario.
* ``safety`` evaluates the known deterministic corruption mask and writes
  corruption-rejection recall, clean-correct false rejection, accepted
  pseudo-label accuracy, unsafe-update rate, and risk-coverage curves.

Target labels are never passed into DuSafe.  They are used only after a run
to score pseudo-labels and predictions.  A synthetic mask is an annotation
for this audit; it is not a real structural label and is not called HCW.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.get_tta_class import get_algorithm_class  # noqa: E402
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from dataloader.corruption_transforms import CORRUPTION_REGISTRY  # noqa: E402
from scripts.run_controlled_safety_benchmark import (  # noqa: E402
    run_job as run_controlled_job,
)
from scripts.run_optuna_stepwise import parse_csv, scenario_label, scenario_pairs  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
    move_data_to_device,
)


SUPPORTED_METHODS = ("NoAdap", "DuSafe")
REQUESTED_BASELINES = ("NoAdap", "Tent", "EATA", "SAR", "ACCUPOfficial")
REQUESTED_METHODS = (*SUPPORTED_METHODS, "Tent", "EATA", "SAR", "ACCUPOfficial")
PLAUSIBILITY_METRICS = (
    "raw_low_energy_ratio",
    "raw_high_energy_ratio",
    "view_low_energy_ratio",
    "view_high_energy_ratio",
    "delta_low_energy_ratio",
    "delta_high_energy_ratio",
    "delta_total_variation",
    "delta_relative_tv",
    "physical_curve_tv",
    "physical_curve_amplitude",
    "relative_rms",
    "source_semantic_distance",
    "ssaw_label_flip",
    "ssaw_selected_kl",
    "ssaw_entropy_rise",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def _band_energy_ratio(signal: torch.Tensor, low: bool) -> torch.Tensor:
    """Return a per-sample FFT energy ratio for [0,.1] or [.4,.5] cycles/sample."""
    if signal.size(-1) < 2:
        return torch.full(
            signal.shape[:-2], float("nan"), device=signal.device, dtype=signal.dtype
        )
    spectrum = torch.fft.rfft(signal, dim=-1)
    power = spectrum.abs().square().mean(dim=-2)
    frequencies = torch.fft.rfftfreq(
        signal.size(-1), d=1.0, device=signal.device
    )
    mask = frequencies <= 0.1 if low else frequencies >= 0.4
    band = power[..., mask].sum(dim=-1)
    return _safe_ratio(band, power.sum(dim=-1))


def _total_variation(signal: torch.Tensor) -> torch.Tensor:
    """Mean absolute first difference over channels and time."""
    if signal.size(-1) < 2:
        return torch.zeros(signal.shape[:-2], device=signal.device, dtype=signal.dtype)
    return signal.diff(dim=-1).abs().mean(dim=(-2, -1))


def _feature_vectors(extractor, inputs: torch.Tensor) -> torch.Tensor:
    features = extractor(inputs)
    if isinstance(features, (tuple, list)):
        features = features[0]
    return F.normalize(features.flatten(1), dim=1)


def _view_matrix(
    value,
    *,
    view_count: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    fill_value: float,
) -> torch.Tensor:
    """Normalize a metadata vector to [views,batch] without inventing views."""
    if value is None:
        return torch.full(
            (view_count, batch_size),
            fill_value,
            device=device,
            dtype=dtype,
        )
    matrix = torch.as_tensor(value, device=device, dtype=dtype)
    if matrix.dim() == 1:
        if matrix.numel() != batch_size:
            raise RuntimeError("SSAW per-sample metadata has the wrong batch size")
        matrix = matrix.unsqueeze(0).expand(view_count, -1)
    if tuple(matrix.shape) != (view_count, batch_size):
        raise RuntimeError(
            "SSAW per-view metadata has shape "
            f"{tuple(matrix.shape)}, expected {(view_count, batch_size)}"
        )
    return matrix


def current_configs(dataset: str) -> tuple[dict, dict]:
    """Return source and target settings from the current production config."""
    hparams = get_hparams_class(dataset)()
    source = {
        **dict(hparams.alg_hparams["NoAdap"]),
        **dict(hparams.source_train_params),
    }
    tta = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    return source, tta


def _scenario_filter(dataset: str, requested: str | None) -> list[tuple[str, str]]:
    scenarios = scenario_pairs(dataset)
    if not requested:
        return scenarios
    value = requested.strip().replace("->", ",")
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 2:
        raise ValueError("--scenario must look like 0->11")
    pair = (pieces[0], pieces[1])
    if pair not in scenarios:
        raise ValueError(f"Unknown {dataset} scenario: {requested}")
    return [pair]


def _primary(data):
    return extract_primary_tensor(data)


def _physical_rows(
    *,
    raw: torch.Tensor,
    views: torch.Tensor,
    adapter,
    labels: torch.Tensor,
    indices: torch.Tensor,
    metadata: Mapping[str, object],
) -> list[dict]:
    """Compute current-view diagnostics without retaining model tensors."""
    if views.dim() == 3:
        views = views.unsqueeze(0)
    if views.dim() != 4 or views.size(1) != raw.size(0):
        raise RuntimeError(
            "Current SSAW view tensor must have shape [views,batch,channels,time]"
        )
    view_count, batch_size = int(views.size(0)), int(views.size(1))
    residual = views - raw.unsqueeze(0)
    raw_broadcast = raw.unsqueeze(0).expand_as(views)
    # Keep the raw reference one value per sample; the view-specific arrays
    # below retain the leading view dimension.
    raw_low = _band_energy_ratio(raw, low=True)
    raw_high = _band_energy_ratio(raw, low=False)
    view_low = _band_energy_ratio(views, low=True)
    view_high = _band_energy_ratio(views, low=False)
    delta_low = _band_energy_ratio(residual, low=True)
    delta_high = _band_energy_ratio(residual, low=False)
    delta_tv = _total_variation(residual)
    raw_tv = _total_variation(raw_broadcast)
    relative_tv = _safe_ratio(delta_tv, raw_tv)
    relative_rms = residual.square().mean(dim=(-2, -1)).sqrt() / (
        raw_broadcast.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-8)
    )

    # ``last_warp_curve`` contains only the gain curve.  For the antithetic
    # reflected view, the corresponding gain curve is 2-gain.  HAR rotation
    # therefore reports zero here; its actual input residual remains measured
    # above and is not mislabeled as a gain diagnostic.
    curve = getattr(adapter.ssaw, "last_warp_curve", None)
    if curve is None:
        curve_tv = torch.full_like(delta_tv, float("nan"))
        curve_amplitude = torch.full_like(delta_tv, float("nan"))
    else:
        curve = torch.as_tensor(curve, device=raw.device, dtype=raw.dtype)
        if curve.dim() == 3:
            curve = curve.unsqueeze(0)
        if curve.dim() != 4 or curve.size(1) != batch_size:
            raise RuntimeError("Current SSAW gain curve has an unexpected shape")
        curves = torch.cat(
            (curve, 2.0 - curve), dim=0
        ) if bool(getattr(adapter.ssaw, "antithetic", False)) else curve
        curves = curves[:view_count]
        curve_tv = _total_variation(curves - 1.0)
        curve_amplitude = (curves - 1.0).abs().mean(dim=(-2, -1))

    # The production semantic gate uses this frozen source feature extractor.
    # This is a representation-distance diagnostic, not a structural label.
    extractor = adapter.source_semantic_feature_extractor
    flat_views = views.reshape(view_count * batch_size, *views.shape[2:])
    with torch.inference_mode():
        raw_features = _feature_vectors(extractor, raw)
        view_features = _feature_vectors(extractor, flat_views)
    semantic_distance = 1.0 - (
        view_features
        * raw_features.repeat(view_count, 1)
    ).sum(dim=1).reshape(view_count, batch_size)

    ssaw_metadata = getattr(adapter.ssaw, "last_metadata", {})
    label_flip_by_view = _view_matrix(
        ssaw_metadata.get("ssaw_label_flip_by_view"),
        view_count=view_count,
        batch_size=batch_size,
        device=raw.device,
        dtype=torch.float32,
        fill_value=float("nan"),
    )
    selected_kl_by_view = _view_matrix(
        ssaw_metadata.get("selected_kl_by_view"),
        view_count=view_count,
        batch_size=batch_size,
        device=raw.device,
        dtype=raw.dtype,
        fill_value=float("nan"),
    )
    entropy_rise_by_view = _view_matrix(
        ssaw_metadata.get("entropy_rise_by_view", ssaw_metadata.get("entropy_rise")),
        view_count=view_count,
        batch_size=batch_size,
        device=raw.device,
        dtype=raw.dtype,
        fill_value=float("nan"),
    )

    pseudo_labels = torch.as_tensor(
        getattr(adapter, "_last_gate_log", {}).get(
            "pseudo_labels", torch.full((batch_size,), -1)
        ),
        device=raw.device,
        dtype=torch.long,
    ).view(-1)
    if pseudo_labels.numel() != batch_size:
        pseudo_labels = torch.full((batch_size,), -1, device=raw.device, dtype=torch.long)
    labels = labels.view(-1).to(device=raw.device, dtype=torch.long)
    indices = indices.view(-1).to(device=raw.device, dtype=torch.long)

    rows: list[dict] = []
    for view_index in range(view_count):
        role = (
            "antithetic_positive"
            if bool(getattr(adapter.ssaw, "antithetic", False)) and view_index < view_count // 2
            else "antithetic_reflection"
            if bool(getattr(adapter.ssaw, "antithetic", False))
            else "single_physical_view"
        )
        for sample_index in range(batch_size):
            row = {
                **dict(metadata),
                "view_index": int(view_index),
                "view_role": role,
                "sample_index": int(indices[sample_index].item()),
                "pseudo_label": int(pseudo_labels[sample_index].item()),
                # Labels are scored only after the online decision has run.
                "raw_correct_posthoc": bool(
                    pseudo_labels[sample_index].item() == labels[sample_index].item()
                ) if pseudo_labels[sample_index].item() >= 0 else None,
                "raw_low_energy_ratio": float(raw_low[sample_index].item()),
                "raw_high_energy_ratio": float(raw_high[sample_index].item()),
                "view_low_energy_ratio": float(view_low[view_index, sample_index].item()),
                "view_high_energy_ratio": float(view_high[view_index, sample_index].item()),
                "delta_low_energy_ratio": float(delta_low[view_index, sample_index].item()),
                "delta_high_energy_ratio": float(delta_high[view_index, sample_index].item()),
                "delta_total_variation": float(delta_tv[view_index, sample_index].item()),
                "delta_relative_tv": float(relative_tv[view_index, sample_index].item()),
                "physical_curve_tv": float(curve_tv[view_index, sample_index].item()),
                "physical_curve_amplitude": float(curve_amplitude[view_index, sample_index].item()),
                "relative_rms": float(relative_rms[view_index, sample_index].item()),
                "source_semantic_distance": float(
                    semantic_distance[view_index, sample_index].item()
                ),
                "ssaw_label_flip": bool(
                    label_flip_by_view[view_index, sample_index].item()
                ) if torch.isfinite(label_flip_by_view[view_index, sample_index]) else None,
                "ssaw_selected_kl": float(
                    selected_kl_by_view[view_index, sample_index].item()
                ),
                "ssaw_entropy_rise": float(
                    entropy_rise_by_view[view_index, sample_index].item()
                ),
            }
            rows.append(row)
    return rows


def run_plausibility_job(
    *,
    dataset: str,
    scenario: tuple[str, str],
    source_seed: int,
    test_time_seed: int,
    data_path: str,
    device: str,
    backbone: str,
    pretrain_cache_dir: str,
) -> list[dict]:
    source_config, tta_config = current_configs(dataset)
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=backbone,
        exp_name="current_v2_plausibility",
        seed=test_time_seed,
        source_seed=source_seed,
        pretrain_cache_dir=pretrain_cache_dir,
    )
    adapter = source_model = None
    rows: list[dict] = []
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model = create_tta_model(
            trainer,
            scenario[0],
            scenario[1],
            run_seed=test_time_seed,
        )
        if not getattr(adapter, "enable_ssaw", False):
            raise RuntimeError("Current plausibility audit requires SSAW enabled")
        for batch_index, (data, labels, target_indices) in enumerate(
            trainer.trg_whole_dl
        ):
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            target_indices = torch.as_tensor(target_indices).view(-1)
            raw = _primary(data).detach()
            model_inputs = {
                "data": data,
                "labels": labels,
                "meta": {
                    "trg_idx": target_indices.detach().cpu().tolist(),
                },
            }
            adapter(model_inputs)
            views = adapter.ssaw.last_view_inputs
            if views is None:
                raise RuntimeError("Current DuSafe did not retain its physical view")
            job_meta = {
                "dataset": dataset,
                "scenario": scenario_label(scenario),
                "source_seed": int(source_seed),
                "test_time_seed": int(test_time_seed),
                "batch_index": int(batch_index),
                "transform_family": str(adapter.ssaw.last_metadata.get("transform_family")),
                "temporal_mode": str(adapter.ssaw.last_metadata.get("temporal_mode")),
                "antithetic": bool(adapter.ssaw.last_metadata.get("antithetic", False)),
                "view_count": int(adapter.ssaw.last_metadata.get("view_count", 0)),
                "ssaw_sigma": float(adapter.ssaw.sigma),
                "ssaw_strength": float(adapter.ssaw.strength),
                "ssaw_control_points": int(adapter.ssaw.num_control_points),
            }
            rows.extend(
                _physical_rows(
                    raw=raw,
                    views=torch.as_tensor(views, device=trainer.device),
                    adapter=adapter,
                    labels=labels,
                    indices=target_indices,
                    metadata=job_meta,
                )
            )
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)
    return rows


def _supported_method_status() -> list[dict]:
    statuses = [
        {
            "method": "DuSafe",
            "status": "runnable",
            "source": "algorithms/dusafe.py current production path",
            "comparison_note": "fixed antithetic physical sensor-calibration pair",
        }
    ]
    for method in REQUESTED_BASELINES:
        if method == "NoAdap":
            statuses.append(
                {
                    "method": method,
                    "status": "runnable",
                    "source": "trainers/tta_abstract_trainer.py special-case",
                    "comparison_note": "source-only; selected/admitted update coverage is zero by design",
                }
            )
            continue
        try:
            get_algorithm_class(method)
        except (KeyError, NotImplementedError):
            statuses.append(
                {
                    "method": method,
                    "status": "unavailable_current_registry",
                    "source": "historical/third_party code only",
                    "comparison_note": "not run; no current trainer adapter or protocol validation",
                }
            )
        else:
            statuses.append(
                {
                    "method": method,
                    "status": "runnable_registry_unverified",
                    "source": "algorithms.get_tta_class",
                    "comparison_note": "runner will require explicit protocol validation",
                }
            )
    return statuses


def _safety_args(args, scenario_map: Mapping[str, tuple[str, str]]):
    return SimpleNamespace(
        data_path=args.data_path,
        device=args.device,
        backbone=args.backbone,
        pretrain_cache_dir=args.pretrain_cache_dir,
        scenario_map=dict(scenario_map),
        corruption_seed=args.corruption_seed,
        corruption_fraction=args.corruption_fraction,
    )


def run_safety_job(
    *,
    args,
    dataset: str,
    scenario: tuple[str, str],
    method: str,
    corruption: str,
    severity: str,
    test_time_seed: int,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    if corruption not in CORRUPTION_REGISTRY:
        raise ValueError(f"Unknown corruption: {corruption}")
    seed = int(args.source_seed if args.corruption_seed is None else args.corruption_seed)
    safety_args = _safety_args(args, {dataset: scenario})
    safety_args.corruption_seed = seed
    summary, records, curves = run_controlled_job(
        safety_args,
        dataset,
        method,
        "full",
        corruption,
        severity,
        int(args.source_seed),
        int(test_time_seed),
    )
    summary.update(
        {
            "audit_version": "current-v2",
            "known_corruption_annotation": "deterministic synthetic sample mask; not a structural/HCW label",
            "target_labels_used_for_updates": False,
        }
    )
    for row in curves:
        row["audit_version"] = "current-v2"
        row["known_corruption_annotation"] = (
            "deterministic synthetic sample mask; not a structural/HCW label"
        )
    return summary, records, curves


def _write_manifest(
    output_dir: Path,
    args,
    datasets: Iterable[str],
    phases: list[str],
    *,
    run_record: dict | None = None,
) -> None:
    manifest_path = output_dir / "manifest.json"
    previous = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    previous_datasets = [str(item).upper() for item in previous.get("datasets", [])]
    previous_phases = [str(item) for item in previous.get("phases", [])]
    previous_seeds = [int(item) for item in previous.get("test_time_seeds", [])]
    run_history = list(previous.get("run_history", []))
    if run_record is not None:
        run_id = str(run_record.get("run_id", ""))
        run_history = [
            item for item in run_history if str(item.get("run_id", "")) != run_id
        ]
        run_history.append(dict(run_record))
    datasets = sorted(set(previous_datasets).union(str(item).upper() for item in datasets))
    phases = sorted(set(previous_phases).union(str(item) for item in phases))
    test_time_seeds = sorted(set(previous_seeds).union(int(seed) for seed in args.test_time_seeds))
    source_configs = {dataset: current_configs(dataset)[0] for dataset in datasets}
    tta_configs = {dataset: current_configs(dataset)[1] for dataset in datasets}
    payload = {
        "audit_version": "current-v2",
        "git_commit": _git_commit(),
        "datasets": list(datasets),
        "phases": phases,
        "source_seed": int(args.source_seed),
        "test_time_seeds": test_time_seeds,
        "production_method": "DuSafe",
        "production_description": (
            "fixed antithetic physical sensor-calibration pair; source-calibrated "
            "confidence and semantic admission; raw CE anchor plus SSAW KL auxiliary"
        ),
        "source_configs": source_configs,
        "tta_configs": tta_configs,
        "target_labels_used_for_updates": False,
        "plausibility_metric_semantics": {
            "low_high_energy": "FFT residual/view energy bands; 0-.1 and .4-.5 cycles/sample",
            "total_variation": "mean absolute first difference; physical_curve_* is gain-only",
            "source_semantic_distance": "1-cosine in frozen source semantic feature space; not a structural label",
            "ssaw_label_flip": "per-view categorical label change relative to the raw view; descriptive model response",
            "ssaw_selected_kl": "raw predictive distribution to each physical view; continuous SSAW auxiliary risk",
            "ssaw_entropy_rise": "per-view entropy minus raw entropy; not the production selection objective",
            "har_rotation_caveat": "HAR rotation is measured through input residual metrics; gain-curve metrics are zero because sigma=0",
        },
        "safety_annotation": (
            "corruption_mask is deterministic and known only for evaluation; it is not HCW/SFC ground truth"
        ),
        "baseline_status": _supported_method_status(),
        "requested_baselines": list(REQUESTED_BASELINES),
        "run_history": run_history,
        "artifact_origin": {
            "plausibility": {
                "status": "completed; command reconstructed from recorded run output",
                "command": (
                    "python scripts/run_current_v2_audit.py --phase plausibility "
                    "--datasets EEG,HAR,FD --source-seed 1 "
                    "--test-time-seeds 1,2,3 --device cuda "
                    "--output-dir results/diagnostics/current_v2_audit"
                ),
                "jobs": 45,
                "summary_rows": 90,
                "result_files": [
                    "plausibility_summary.csv",
                    "plausibility_aggregate.csv",
                    "plausibility_scenario_aggregate.csv",
                ],
                "provenance_note": (
                    "The later run_history entries with jobs_published_this_run=0 "
                    "are resume/manifest-repair invocations, not the original job execution."
                ),
            },
            "safety": {
                "status": "completed; command reconstructed from recorded run output",
                "command": (
                    "python scripts/run_current_v2_audit.py --phase safety "
                    "--datasets HAR --scenario 2->11 --methods NoAdap,DuSafe "
                    "--test-time-seeds 1 --corruptions signal_freeze "
                    "--severities moderate --device cuda "
                    "--output-dir results/diagnostics/current_v2_audit"
                ),
                "jobs": 2,
                "summary_rows": 2,
                "result_files": [
                    "safety_summary.csv",
                    "safety_aggregate.csv",
                    "risk_coverage.csv",
                ],
                "provenance_note": (
                    "The later run_history entry with jobs_published_this_run=0 "
                    "is a resume/manifest-repair invocation, not the original job execution."
                ),
            },
        },
        "legacy_artifact_policy": {
            "scripts/diagnose_ssaw_pipeline.py": "do_not_reuse: archived multi-candidate/rescue/veto admission state",
            "scripts/diagnose_ssaw_update_counterfactual.py": "do_not_reuse: depends on archived admission-state protocol",
            "scripts/run_simplified_ssaw_validation.py": "do_not_reuse_as_current: reads historical tuning states and reports legacy variants",
            "scripts/run_controlled_safety_benchmark.py": "reuse_core: deterministic corruption mask and trainer safety metrics; current-v2 wrapper supplies current protocol",
            "main_acmmm.tex:tab:spectral": "rerun_or_remove: hard-coded old perturbation comparisons do not describe the antithetic pair",
            "main_acmmm.tex:tab:entropy_shift/fig:entropy_pdfs/fig:ssaw_module": "rerun_or_remove: old random-vs-ranked entropy candidate-selection protocol",
            "main_acmmm.tex:fig:sensitivity": "remove_or_redesign: semantic-threshold and candidate-count axes are removed",
            "main_acmmm.tex:tab:corruption": "rerun: F1-vs-NoAdap alone omits known-mask safety and coverage metrics",
            "Table 4 SFC/HCW diagnostics": "not_found_in_current_source: no current artifact may be relabeled as that table",
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", choices=("all", "plausibility", "safety"), default="all")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--scenario", default=None, help="Optional one scenario, e.g. 0->11")
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--test-time-seeds", default="1,2,3")
    parser.add_argument("--methods", default=",".join(REQUESTED_METHODS))
    parser.add_argument("--corruptions", default="signal_freeze")
    parser.add_argument("--severities", default="moderate")
    parser.add_argument("--corruption-fraction", type=float, default=0.5)
    parser.add_argument("--corruption-seed", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "diagnostics" / "current_v2_audit"),
    )
    args = parser.parse_args(argv)
    args.datasets = [str(name).upper() for name in parse_csv(args.datasets)]
    args.test_time_seeds = parse_csv(args.test_time_seeds, int)
    args.methods = parse_csv(args.methods)
    args.corruptions = parse_csv(args.corruptions)
    args.severities = parse_csv(args.severities)
    if not args.datasets:
        parser.error("--datasets must not be empty")
    if not args.test_time_seeds:
        parser.error("--test-time-seeds must not be empty")
    if not 0.0 <= args.corruption_fraction <= 1.0:
        parser.error("--corruption-fraction must lie in [0,1]")
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    unknown = sorted(set(args.methods) - set(REQUESTED_METHODS))
    if unknown:
        parser.error(f"Unknown methods: {unknown}")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    output_dir = ensure_dir(args.output_dir)
    plausibility_dir = ensure_dir(output_dir / "plausibility_samples")
    safety_dir = ensure_dir(output_dir / "safety_records")
    datasets = args.datasets
    phases = ["plausibility", "safety"] if args.phase == "all" else [args.phase]
    run_record = {
        "run_id": f"{_utc_now()}_{Path(sys.argv[0]).stem}",
        "started_at": _utc_now(),
        "status": "running",
        "phase": args.phase,
        "datasets": list(datasets),
        "scenario": args.scenario,
        "source_seed": int(args.source_seed),
        "test_time_seeds": [int(seed) for seed in args.test_time_seeds],
        "methods": list(args.methods),
        "corruptions": list(args.corruptions),
        "severities": list(args.severities),
        "device": str(args.device),
        "output_dir": str(output_dir),
        "target_labels_used_for_updates": False,
        "known_corruption_annotation": (
            "deterministic synthetic sample mask; not a structural/HCW label"
        ),
    }
    _write_manifest(output_dir, args, datasets, phases, run_record=run_record)

    plausibility_summary_path = output_dir / "plausibility_summary.csv"
    safety_summary_path = output_dir / "safety_summary.csv"
    risk_curve_path = output_dir / "risk_coverage.csv"
    plausibility_summary_rows = (
        pd.read_csv(plausibility_summary_path).to_dict("records")
        if plausibility_summary_path.exists()
        else []
    )
    safety_summary_rows = (
        pd.read_csv(safety_summary_path).to_dict("records")
        if safety_summary_path.exists()
        else []
    )
    risk_curve_rows = (
        pd.read_csv(risk_curve_path).to_dict("records")
        if risk_curve_path.exists()
        else []
    )
    jobs_done = 0

    if "plausibility" in phases:
        for dataset in datasets:
            scenarios = _scenario_filter(dataset, args.scenario)
            for scenario in scenarios:
                for test_time_seed in args.test_time_seeds:
                    stem = (
                        f"{dataset}_{scenario[0]}_to_{scenario[1]}_"
                        f"source{args.source_seed}_seed{test_time_seed}"
                    )
                    sample_path = plausibility_dir / f"{stem}.csv"
                    key = (dataset, scenario_label(scenario), int(args.source_seed), int(test_time_seed))
                    already = {
                        (
                            str(row.get("dataset")),
                            str(row.get("scenario")),
                            int(row.get("source_seed", -1)),
                            int(row.get("test_time_seed", -1)),
                        )
                        for row in plausibility_summary_rows
                    }
                    required_sample_columns = {
                        "ssaw_label_flip",
                        "ssaw_selected_kl",
                        "ssaw_entropy_rise",
                    }
                    sample_has_current_metrics = False
                    if sample_path.exists():
                        sample_columns = set(
                            pd.read_csv(sample_path, nrows=0).columns
                        )
                        sample_has_current_metrics = required_sample_columns.issubset(
                            sample_columns
                        )
                    if key in already and sample_has_current_metrics:
                        continue
                    if args.max_jobs is not None and jobs_done >= args.max_jobs:
                        break
                    print(
                        f"[Current-v2 plausibility] {dataset} {scenario_label(scenario)} seed={test_time_seed}",
                        flush=True,
                    )
                    rows = run_plausibility_job(
                        dataset=dataset,
                        scenario=scenario,
                        source_seed=args.source_seed,
                        test_time_seed=int(test_time_seed),
                        data_path=args.data_path,
                        device=args.device,
                        backbone=args.backbone,
                        pretrain_cache_dir=args.pretrain_cache_dir,
                    )
                    frame = pd.DataFrame(rows)
                    atomic_write_csv(frame, sample_path, index=False)
                    if key in already:
                        plausibility_summary_rows = [
                            row
                            for row in plausibility_summary_rows
                            if (
                                str(row.get("dataset")),
                                str(row.get("scenario")),
                                int(row.get("source_seed", -1)),
                                int(row.get("test_time_seed", -1)),
                            )
                            != key
                        ]
                    if not frame.empty:
                        grouped = frame.groupby(["dataset", "scenario", "source_seed", "test_time_seed", "view_role"], as_index=False)
                        summary = grouped[list(PLAUSIBILITY_METRICS)].mean(numeric_only=True)
                        summary["sample_count"] = grouped.size()["size"]
                        plausibility_summary_rows.extend(summary.to_dict("records"))
                    atomic_write_csv(pd.DataFrame(plausibility_summary_rows), plausibility_summary_path, index=False)
                    jobs_done += 1
                if args.max_jobs is not None and jobs_done >= args.max_jobs:
                    break
            if args.max_jobs is not None and jobs_done >= args.max_jobs:
                break

    if "safety" in phases and (args.max_jobs is None or jobs_done < args.max_jobs):
        runnable_methods = [method for method in args.methods if method in SUPPORTED_METHODS]
        unavailable = [method for method in args.methods if method not in SUPPORTED_METHODS]
        if unavailable:
            print(
                "[Current-v2 safety] skipping unavailable current adapters: "
                + ", ".join(unavailable),
                flush=True,
            )
        for dataset in datasets:
            scenarios = _scenario_filter(dataset, args.scenario)
            for scenario in scenarios:
                for method in runnable_methods:
                    for corruption in args.corruptions:
                        for severity in args.severities:
                            for test_time_seed in args.test_time_seeds:
                                seed = int(
                                    args.source_seed
                                    if args.corruption_seed is None
                                    else args.corruption_seed
                                )
                                key = (
                                    dataset,
                                    scenario_label(scenario),
                                    method,
                                    corruption,
                                    severity,
                                    int(args.source_seed),
                                    int(test_time_seed),
                                    seed,
                                )
                                existing = {
                                    (
                                        str(row.get("dataset")),
                                        str(row.get("scenario")),
                                        str(row.get("method")),
                                        str(row.get("corruption")),
                                        str(row.get("severity")),
                                        int(row.get("source_seed", -1)),
                                        int(row.get("stream_seed", -1)),
                                        int(row.get("corruption_seed", -1)),
                                    )
                                    for row in safety_summary_rows
                                }
                                if key in existing:
                                    continue
                                if args.max_jobs is not None and jobs_done >= args.max_jobs:
                                    break
                                print(
                                    f"[Current-v2 safety] {key}", flush=True
                                )
                                summary, records, curves = run_safety_job(
                                    args=args,
                                    dataset=dataset,
                                    scenario=scenario,
                                    method=method,
                                    corruption=corruption,
                                    severity=severity,
                                    test_time_seed=int(test_time_seed),
                                )
                                safety_summary_rows.append(summary)
                                risk_curve_rows.extend(curves)
                                record_name = (
                                    f"{dataset}_{scenario[0]}_to_{scenario[1]}_"
                                    f"{method}_{corruption}_{severity}_source{args.source_seed}_"
                                    f"seed{test_time_seed}_corruption{seed}.csv"
                                )
                                atomic_write_csv(records, safety_dir / record_name, index=False)
                                atomic_write_csv(pd.DataFrame(safety_summary_rows), safety_summary_path, index=False)
                                atomic_write_csv(pd.DataFrame(risk_curve_rows), risk_curve_path, index=False)
                                jobs_done += 1
                            if args.max_jobs is not None and jobs_done >= args.max_jobs:
                                break
                        if args.max_jobs is not None and jobs_done >= args.max_jobs:
                            break
                    if args.max_jobs is not None and jobs_done >= args.max_jobs:
                        break
                if args.max_jobs is not None and jobs_done >= args.max_jobs:
                    break
            if args.max_jobs is not None and jobs_done >= args.max_jobs:
                break

    # Aggregate risk-coverage is already written per job.  Add a compact
    # method-level summary only for cells that actually ran; never synthesize
    # rows for unavailable baselines.
    if safety_summary_rows:
        safety_frame = pd.DataFrame(safety_summary_rows)
        aggregate = (
            safety_frame.groupby(
                ["dataset", "scenario", "method", "corruption", "severity"],
                as_index=False,
            )
            .agg(
                f1_mean=("f1", "mean"),
                coverage_mean=("coverage", "mean"),
                accepted_pseudo_label_accuracy_mean=(
                    "accepted_pseudo_label_accuracy", "mean"
                ),
                unsafe_update_rate_mean=("unsafe_update_rate", "mean"),
                corruption_rejection_recall_mean=(
                    "corruption_rejection_recall", "mean"
                ),
                clean_correct_false_rejection_rate_mean=(
                    "clean_correct_false_rejection_rate", "mean"
                ),
            )
        )
        atomic_write_csv(aggregate, output_dir / "safety_aggregate.csv", index=False)
    if plausibility_summary_rows:
        plausibility_frame = pd.DataFrame(plausibility_summary_rows)
        group_columns = ["dataset", "view_role"]
        aggregate = (
            plausibility_frame.groupby(group_columns, as_index=False)[
                list(PLAUSIBILITY_METRICS)
            ]
            .agg(["mean", "std", "count"])
        )
        aggregate.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in aggregate.columns
        ]
        atomic_write_csv(
            aggregate,
            output_dir / "plausibility_aggregate.csv",
            index=False,
        )
        scenario_aggregate = (
            plausibility_frame.groupby(
                ["dataset", "scenario", "view_role"], as_index=False
            )[list(PLAUSIBILITY_METRICS)]
            .mean()
        )
        atomic_write_csv(
            scenario_aggregate,
            output_dir / "plausibility_scenario_aggregate.csv",
            index=False,
        )
    run_record["status"] = "completed"
    run_record["finished_at"] = _utc_now()
    run_record["jobs_published_this_run"] = int(jobs_done)
    _write_manifest(output_dir, args, datasets, phases, run_record=run_record)
    print(f"Current-v2 audit output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
