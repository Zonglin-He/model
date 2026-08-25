"""Static source/target spline radius audit for search/radius/manifold diagnosis."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe import (  # noqa: E402
    SSAWPhysicalView,
    _extract_features,
    evaluate_candidate_pool_sequential,
)
from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView  # noqa: E402
from algorithms.dusafe_spline_mechanism_matrix import (  # noqa: E402
    BoundarySeekingSplineHardView,
    _pseudo_class_margin,
    get_mechanism_runner,
)
from scripts.dusafe_factorial_runner_common import current_profiles  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_har_spline_mechanism_matrix import (  # noqa: E402
    FLOWS,
    SOURCE_SEED,
    STREAM_SEED,
)
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


PROTOCOL = "har_spline_radius_audit_v2_per_view_bn_gathered_recheck"
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80)
VIEW_MODES = ("random", "boundary")
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "har_spline_radius_audit_seed1_v2_corrected"
)
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _quantiles(values: torch.Tensor) -> tuple[float, float, float]:
    values = values.detach().float().cpu()
    return tuple(float(torch.quantile(values, q)) for q in (0.1, 0.5, 0.9))


def _evaluate_loader(adapter, loader, *, view_mode: str, alpha: float, split: str):
    view_class = (
        UnifiedSplineHardView if view_mode == "random" else BoundarySeekingSplineHardView
    )
    kwargs = {
        "num_control_points": 10,
        "num_directions": 4,
        "log_strength": float(alpha),
        "radius_levels": (1.0,),
        "sobol_seed": adapter.ssaw_effective_sobol_seed,
    }
    if view_mode == "boundary":
        kwargs.update({"search_steps": 2, "search_step_size": 0.5})
    view = view_class(**kwargs)
    if view_mode == "boundary":
        view.search_model = adapter.model
    raw_margins, search_margins, gathered_margins = [], [], []
    flips, raw_correct, selected_correct = [], [], []
    search_gather_prediction_disagreement = []
    selected_true_nll = []
    sample_count = 0
    for batch_index, (data, labels, _) in enumerate(loader):
        model_device = next(adapter.model.parameters()).device
        data = data.float().to(model_device)
        labels = labels.view(-1).long().to(data.device)
        with SSAWPhysicalView._preserved_bn_buffers(adapter.model), torch.no_grad():
            raw_logits = adapter.model.classifier(_extract_features(adapter.model, data))
        pseudo = raw_logits.argmax(dim=1)
        raw_margin = _pseudo_class_margin(raw_logits, pseudo)
        prepared = view.prepare_view_inputs(
            data,
            normalization_mean=adapter.source_normalization_mean,
            normalization_std=adapter.source_normalization_std,
            reuse_cached_view=False,
        )
        candidates = torch.as_tensor(prepared["view_inputs"])
        _, logits = evaluate_candidate_pool_sequential(
            adapter.model, candidates, require_grad=False
        )
        expanded_pseudo = pseudo[None].expand(candidates.size(0), -1)
        margins = _pseudo_class_margin(logits, expanded_pseudo)
        selected_index = margins.argmin(dim=0)
        batch_indices = torch.arange(data.size(0), device=data.device)
        search_selected_logits = logits[selected_index, batch_indices]
        search_selected_margin = margins[selected_index, batch_indices]
        gathered_inputs = candidates[selected_index, batch_indices]
        with SSAWPhysicalView._preserved_bn_buffers(adapter.model), torch.no_grad():
            gathered_logits = adapter.model.classifier(
                _extract_features(adapter.model, gathered_inputs)
            )
        gathered_margin = _pseudo_class_margin(gathered_logits, pseudo)
        selected_prediction = gathered_logits.argmax(dim=1)
        raw_margins.append(raw_margin.cpu())
        search_margins.append(search_selected_margin.cpu())
        gathered_margins.append(gathered_margin.cpu())
        flips.append(selected_prediction.ne(pseudo).cpu())
        search_gather_prediction_disagreement.append(
            search_selected_logits.argmax(dim=1).ne(selected_prediction).cpu()
        )
        raw_correct.append(pseudo.eq(labels).cpu())
        selected_correct.append(selected_prediction.eq(labels).cpu())
        selected_true_nll.append(
            F.cross_entropy(gathered_logits, labels, reduction="none").cpu()
        )
        sample_count += int(labels.numel())
        view.clear_cached_view()
        if hasattr(view, "_spline_call_index"):
            view._spline_call_index = batch_index + 1

    raw_margin = torch.cat(raw_margins)
    search_margin = torch.cat(search_margins)
    gathered_margin = torch.cat(gathered_margins)
    ratio = gathered_margin / raw_margin.clamp_min(1e-8)
    flip = torch.cat(flips)
    raw_ok = torch.cat(raw_correct)
    selected_ok = torch.cat(selected_correct)
    nll = torch.cat(selected_true_nll)
    raw_q = _quantiles(raw_margin)
    search_q = _quantiles(search_margin)
    gathered_q = _quantiles(gathered_margin)
    ratio_q = _quantiles(ratio)
    preservation = (
        float(selected_ok[raw_ok].float().mean()) if raw_ok.any() else math.nan
    )
    return {
        "split": split,
        "view_mode": view_mode,
        "alpha": float(alpha),
        "samples": sample_count,
        "raw_accuracy": float(raw_ok.float().mean()),
        "selected_accuracy": float(selected_ok.float().mean()),
        "raw_correct_preservation": preservation,
        "raw_prediction_flip_rate": float(flip.float().mean()),
        "search_gather_prediction_disagreement_rate": float(
            torch.cat(search_gather_prediction_disagreement).float().mean()
        ),
        "selected_true_nll": float(nll.mean()),
        "raw_margin_p10": raw_q[0],
        "raw_margin_p50": raw_q[1],
        "raw_margin_p90": raw_q[2],
        "search_selected_margin_p10": search_q[0],
        "search_selected_margin_p50": search_q[1],
        "search_selected_margin_p90": search_q[2],
        "gathered_actual_margin_p10": gathered_q[0],
        "gathered_actual_margin_p50": gathered_q[1],
        "gathered_actual_margin_p90": gathered_q[2],
        # Backward-compatible alias now explicitly refers to the actual mixed
        # training/evaluation batch, never the search-time candidate batch.
        "selected_margin_p10": gathered_q[0],
        "selected_margin_p50": gathered_q[1],
        "selected_margin_p90": gathered_q[2],
        "normalized_margin_ratio_p10": ratio_q[0],
        "normalized_margin_ratio_p50": ratio_q[1],
        "normalized_margin_ratio_p90": ratio_q[2],
    }


def _run_flow(spec):
    flow = tuple(str(value) for value in spec["flow"])
    trainer = build_trainer(
        data_path=spec["data_path"],
        device=spec["device"],
        dataset="HAR",
        da_method="DuSafe",
        backbone=spec["backbone"],
        exp_name="spline_radius_audit",
        seed=STREAM_SEED,
        source_seed=SOURCE_SEED,
        pretrain_cache_dir=spec["pretrain_cache_dir"],
        ablation_mode=None,
    )
    adapter = source_model = None
    try:
        source_config, tta_config = current_profiles("HAR")
        trainer.get_tta_model_class = lambda: get_mechanism_runner("B0_raw_only")
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=STREAM_SEED
        )
        rows = []
        for view_mode in VIEW_MODES:
            for alpha in ALPHAS:
                for split, loader in (
                    ("source_test", trainer.src_test_dl),
                    ("target_stream", trainer.trg_whole_dl),
                ):
                    rows.append(
                        {
                            "protocol": PROTOCOL,
                            "dataset": "HAR",
                            "scenario": _flow_label(flow),
                            "source_seed": SOURCE_SEED,
                            "stream_seed": STREAM_SEED,
                            **_evaluate_loader(
                                adapter,
                                loader,
                                view_mode=view_mode,
                                alpha=alpha,
                                split=split,
                            ),
                        }
                    )
        return pd.DataFrame(rows)
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)
        adapter = source_model = None
        release_cuda()
        gc.collect()


def _worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock = wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
        with lock:
            rows = _run_flow(spec)
        atomic_write_csv(rows, cell_dir / "radius_rows.csv", index=False)
        atomic_write_json(
            {
                "status": "ok",
                "protocol": PROTOCOL,
                "scenario": _flow_label(spec["flow"]),
                "rows": int(len(rows)),
            },
            cell_dir / "summary.json",
        )
        return 0
    except BaseException as exc:
        atomic_write_json(
            {
                "status": "failed",
                "protocol": PROTOCOL,
                "scenario": _flow_label(spec["flow"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exc).lower(),
            },
            cell_dir / "summary.json",
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _aggregate(output_dir: Path):
    frames = []
    for flow in FLOWS:
        path = output_dir / "cells" / f"flow_{flow[0]}_to_{flow[1]}" / "radius_rows.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    atomic_write_csv(rows, output_dir / "radius_rows.csv", index=False)
    if rows.empty:
        return
    source = rows[rows["split"].eq("source_test")].copy()
    source["source_valid_99pct"] = source["raw_correct_preservation"].ge(0.99)
    selected = []
    for (scenario, mode), group in source.groupby(["scenario", "view_mode"]):
        valid = group[group["source_valid_99pct"]].sort_values("alpha")
        if valid.empty:
            continue
        row = valid.iloc[-1]
        target = rows[
            rows["scenario"].eq(scenario)
            & rows["view_mode"].eq(mode)
            & rows["split"].eq("target_stream")
            & rows["alpha"].eq(row["alpha"])
        ].iloc[0]
        selected.append(
            {
                "scenario": scenario,
                "view_mode": mode,
                "max_source_valid_alpha": float(row["alpha"]),
                "source_raw_correct_preservation": float(
                    row["raw_correct_preservation"]
                ),
                "source_flip_rate": float(row["raw_prediction_flip_rate"]),
                "target_flip_rate": float(target["raw_prediction_flip_rate"]),
                "target_margin_ratio_p10": float(
                    target["normalized_margin_ratio_p10"]
                ),
                "target_margin_ratio_p50": float(
                    target["normalized_margin_ratio_p50"]
                ),
                "target_margin_ratio_p90": float(
                    target["normalized_margin_ratio_p90"]
                ),
            }
        )
    atomic_write_csv(
        pd.DataFrame(selected), output_dir / "max_source_valid_radius.csv", index=False
    )


def _run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed, failures = 0, []
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "flows": [_flow_label(flow) for flow in FLOWS],
        "alphas": list(ALPHAS),
        "view_modes": list(VIEW_MODES),
        "candidate_evaluation": "one [B,C,T] forward per view",
        "gathered_batch_rechecked": True,
        "source_validity_rule": "raw-correct source samples remain correct >= 99%",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_offline_analysis": True,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    for flow in FLOWS:
        cell_dir = args.output_dir / "cells" / f"flow_{flow[0]}_to_{flow[1]}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "flow": list(flow),
            "cell_dir": str(cell_dir.resolve()),
            "data_path": str(args.data_path.resolve()),
            "device": args.device,
            "backbone": args.backbone,
            "pretrain_cache_dir": str(args.pretrain_cache_dir.resolve()),
            "gpu_lock_path": str(args.gpu_lock_path.resolve()),
        }
        spec_path = cell_dir / "worker_spec.json"
        atomic_write_json(spec, spec_path)
        log_path = cell_dir / "worker.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            failures.append(
                {
                    "scenario": _flow_label(flow),
                    "returncode": int(process.returncode),
                    "log": str(log_path),
                }
            )
            if args.fail_fast:
                break
        else:
            completed += 1
        _aggregate(args.output_dir)
    _aggregate(args.output_dir)
    status = "complete" if completed == len(FLOWS) and not failures else "failed"
    final = {
        **manifest,
        "status": status,
        "completed_flows": completed,
        "failures": failures,
    }
    atomic_write_json(final, args.output_dir / "manifest.json")
    atomic_write_json(final, args.output_dir / "status.json")
    return 0 if status == "complete" else 1


def _parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "Dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        return _worker(args.worker_spec)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
