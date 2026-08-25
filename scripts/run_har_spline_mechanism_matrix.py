"""Run the controlled HAR B0--B4 spline mechanism matrix.

Scope is fixed to source seed 1, stream seed 42, and flows 6->23, 9->18,
12->16.  The checked-in HAR TTA profile is target-selected/descriptive; the
new spline search parameters are fixed without target-F1 tuning.
"""

from __future__ import annotations

import argparse
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

from algorithms.dusafe_spline_mechanism_matrix import (  # noqa: E402
    MECHANISM_RUNNERS,
    get_mechanism_runner,
)
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


PROTOCOL = "har_spline_mechanism_matrix_v2_per_view_bn_gathered_recheck"
FLOWS = (("6", "23"), ("9", "18"), ("12", "16"))
RUNNERS = tuple(MECHANISM_RUNNERS)
SOURCE_SEED = 1
STREAM_SEED = 42
B0_EXPECTED_F1 = {
    "6->23": 0.9918505549430847,
    "9->18": 0.8277944326400757,
    "12->16": 0.8598269820213318,
}
B0_REPRODUCTION_ATOL = 1e-6
SPLINE_PROFILE = {
    "spline_control_points": 10,
    "spline_num_directions": 4,
    "spline_log_strength": 0.2,
    "spline_radius_levels": (1.0, 0.5, 0.25),
    "spline_search_steps": 2,
    "spline_search_step_size": 0.5,
    "record_optimizer_diagnostics": True,
}
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "ablation"
    / "har_spline_mechanism_matrix_seed1_v2_corrected"
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


def _cell_dir(output_dir: Path, flow: Sequence[str], runner: str) -> Path:
    return output_dir / "cells" / f"flow_{flow[0]}_to_{flow[1]}" / runner


def _signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "flow": list(spec["flow"]),
        "runner": str(spec["runner"]),
        "source_seed": int(spec["source_seed"]),
        "stream_seed": int(spec["stream_seed"]),
        "source_config": spec["source_config"],
        "tta_config": spec["tta_config"],
        "max_batches": spec.get("max_batches"),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_new_spline_parameter_selection": False,
    }


def _complete(cell_dir: Path, signature_hash: str) -> bool:
    summary_path = cell_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "ok"
        and summary.get("signature_hash") == signature_hash
        and (cell_dir / "sample_records.csv").is_file()
        and (cell_dir / "batch_diagnostics.csv").is_file()
    )


def _run_cell(spec: Mapping[str, object]):
    flow = tuple(str(value) for value in spec["flow"])
    runner_name = str(spec["runner"])
    dataset = str(spec.get("dataset", "HAR")).upper()
    trainer = build_trainer(
        data_path=str(spec["data_path"]),
        device=str(spec["device"]),
        dataset=dataset,
        da_method="DuSafe",
        backbone=str(spec["backbone"]),
        exp_name=f"spline_mechanism_{dataset}_{runner_name}",
        seed=int(spec["stream_seed"]),
        source_seed=int(spec["source_seed"]),
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        runner_class = get_mechanism_runner(runner_name)
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(spec["source_config"]))
        trainer.set_runtime_hparams(dict(spec["tta_config"]))
        adapted, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=int(spec["stream_seed"])
        )
        source_hash = tensor_state_sha256(source_model)
        source_checkpoint = str(trainer._pretrain_cache_path() or "")
        if spec.get("max_batches") is not None:
            trainer.trg_whole_dl = _LimitedLoader(
                trainer.trg_whole_dl, int(spec["max_batches"])
            )
        metrics = trainer.calculate_metrics(adapted)

        samples = trainer.last_safety_records.copy()
        for name, value in reversed(
            (
                ("dataset", dataset),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("runner", runner_name),
            )
        ):
            samples.insert(0, name, value)
        batches = getattr(trainer, "last_batch_log_records", pd.DataFrame()).copy()
        if not batches.empty:
            batches.insert(0, "batch_index", range(len(batches)))
            candidate_hashes = getattr(
                trainer, "last_candidate_hash_records", pd.DataFrame()
            ).copy()
            if not candidate_hashes.empty:
                batches = batches.merge(
                    candidate_hashes,
                    on="batch_index",
                    how="left",
                    validate="one_to_one",
                )
        for name, value in reversed(
            (
                ("dataset", dataset),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("runner", runner_name),
            )
        ):
            batches.insert(0, name, value)

        result = {
            "status": "ok",
            "protocol": PROTOCOL,
            "dataset": dataset,
            "scenario": _flow_label(flow),
            "source_seed": int(spec["source_seed"]),
            "stream_seed": int(spec["stream_seed"]),
            "runner": runner_name,
            "runner_class": runner_class.__name__,
            "source_model_sha256": source_hash,
            "source_checkpoint_path": source_checkpoint,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            "sample_count": int(len(samples)),
            "batch_count": int(samples["batch_index"].nunique()),
            "target_labels_used_for_online_decision": False,
            "target_labels_used_for_parameter_selection": True,
            "target_labels_used_for_new_spline_parameter_selection": False,
            "evaluation_partition": "target_selected_descriptive",
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
        return result, samples, batches
    finally:
        cleanup_trainer(trainer, adapted, source_model, close_summary=True)
        adapted = source_model = None
        release_cuda()
        gc.collect()


def _worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    signature_hash = _hash_json(_signature(spec))
    if _complete(cell_dir, signature_hash):
        return 0
    try:
        lock = (
            wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
            if str(spec["device"]).lower().startswith("cuda")
            else None
        )
        if lock is None:
            result, samples, batches = _run_cell(spec)
        else:
            with lock:
                result, samples, batches = _run_cell(spec)
        atomic_write_csv(samples, cell_dir / "sample_records.csv", index=False)
        atomic_write_csv(batches, cell_dir / "batch_diagnostics.csv", index=False)
        result["signature_hash"] = signature_hash
        atomic_write_json(result, cell_dir / "summary.json")
        return 0
    except BaseException as exc:
        atomic_write_json(
            {
                "status": "failed",
                "protocol": PROTOCOL,
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


def _read_completed(output_dir: Path):
    summaries, samples, batches = [], [], []
    for flow in FLOWS:
        for runner in RUNNERS:
            cell_dir = _cell_dir(output_dir, flow, runner)
            summary_path = cell_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "ok" or summary.get("protocol") != PROTOCOL:
                continue
            summaries.append(summary)
            samples.append(pd.read_csv(cell_dir / "sample_records.csv"))
            batches.append(pd.read_csv(cell_dir / "batch_diagnostics.csv"))
    return (
        pd.DataFrame(summaries),
        pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(),
        pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(),
    )


def _variant_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    diagnostic_columns = [
        column
        for column in raw.columns
        if column.startswith("diag_")
        and not column.startswith("diag_parameter_delta_norm__")
    ]
    metrics = ["f1", "accuracy", "auroc", "risk", *diagnostic_columns]
    return raw.groupby("runner", as_index=False)[metrics].mean(numeric_only=True)


def _paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    pivot = raw.pivot(index="scenario", columns="runner", values="f1")
    baseline = "B0_raw_only"
    if baseline not in pivot.columns:
        return pd.DataFrame()
    rows = []
    for runner in RUNNERS:
        if runner == baseline or runner not in pivot.columns:
            continue
        delta = pivot[runner] - pivot[baseline]
        for scenario, value in delta.items():
            rows.append(
                {
                    "scenario": scenario,
                    "contrast": f"{runner}-B0",
                    "delta_f1": float(value),
                }
            )
        rows.append(
            {
                "scenario": "mean_three_flows",
                "contrast": f"{runner}-B0",
                "delta_f1": float(delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "p10": float(numeric.quantile(0.1)),
        "p50": float(numeric.quantile(0.5)),
        "p90": float(numeric.quantile(0.9)),
    }


def _margin_quantiles(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = []
    for (scenario, runner), group in samples.groupby(["scenario", "runner"]):
        row = {"scenario": scenario, "runner": runner, "samples": int(len(group))}
        for name in (
            "ssaw_raw_pseudo_margin",
            "ssaw_selected_margin",
            "ssaw_selected_margin_drop",
            "ssaw_selected_normalized_margin_ratio",
            "ssaw_gathered_actual_margin",
            "ssaw_gathered_actual_margin_drop",
            "ssaw_gathered_actual_normalized_margin_ratio",
        ):
            quantiles = _quantiles(group[name])
            row.update({f"{name}_{key}": value for key, value in quantiles.items()})
        row["positive_margin_reduction_fraction"] = float(
            pd.to_numeric(
                group["ssaw_selected_margin_drop"], errors="coerce"
            ).gt(0.0).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _layer_update_norms(batches: pd.DataFrame) -> pd.DataFrame:
    if batches.empty:
        return pd.DataFrame()
    layer_columns = [
        column for column in batches if column.startswith("parameter_delta_norm__")
    ]
    rows = []
    for (scenario, runner), group in batches.groupby(["scenario", "runner"]):
        for column in layer_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "runner": runner,
                    "layer": column.removeprefix("parameter_delta_norm__"),
                    "mean_update_norm": float(values.mean()),
                    "max_update_norm": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _publish(output_dir: Path) -> None:
    raw, samples, batches = _read_completed(output_dir)
    if not raw.empty:
        raw = raw.sort_values(["scenario", "runner"]).reset_index(drop=True)
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    atomic_write_csv(_variant_summary(raw), output_dir / "variant_summary.csv", index=False)
    atomic_write_csv(_paired_effects(raw), output_dir / "paired_effects.csv", index=False)
    if not raw.empty:
        table = raw.pivot(index="scenario", columns="runner", values="f1")
        atomic_write_csv(table.reset_index(), output_dir / "flow_f1_table.csv", index=False)
    atomic_write_csv(_margin_quantiles(samples), output_dir / "margin_quantiles.csv", index=False)
    atomic_write_csv(batches, output_dir / "batch_diagnostics.csv", index=False)
    atomic_write_csv(
        _layer_update_norms(batches), output_dir / "layer_update_norms.csv", index=False
    )


def _build_specs(args) -> list[dict[str, object]]:
    source_config, tta_config = current_profiles("HAR")
    tta_config = {**tta_config, **SPLINE_PROFILE}
    specs = [
        {
            "cell_dir": str(_cell_dir(args.output_dir, flow, runner).resolve()),
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
        for flow in FLOWS
        for runner in RUNNERS
    ]
    ordered = sorted(
        specs,
        key=lambda spec: (
            0 if spec["runner"] == "B0_raw_only" else 1,
            FLOWS.index(tuple(spec["flow"])),
            RUNNERS.index(spec["runner"]),
        ),
    )
    if args.b0_only:
        return [spec for spec in ordered if spec["runner"] == "B0_raw_only"]
    return ordered


def _validate_b0_reproduction(output_dir: Path) -> dict[str, object]:
    observed = {}
    for flow in FLOWS:
        scenario = _flow_label(flow)
        summary_path = _cell_dir(
            output_dir, flow, "B0_raw_only"
        ) / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"B0 reproduction cell is missing: {scenario}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        observed[scenario] = float(summary["f1"])
    failures = {
        scenario: {
            "expected": expected,
            "observed": observed[scenario],
            "absolute_error": abs(observed[scenario] - expected),
        }
        for scenario, expected in B0_EXPECTED_F1.items()
        if abs(observed[scenario] - expected) > B0_REPRODUCTION_ATOL
    }
    if failures:
        raise RuntimeError(f"B0 reproduction failed: {failures}")
    return {
        "status": "passed",
        "expected": B0_EXPECTED_F1,
        "observed": observed,
        "absolute_tolerance": B0_REPRODUCTION_ATOL,
    }


def _run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(args)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "dataset": "HAR",
        "flows": [_flow_label(flow) for flow in FLOWS],
        "runners": list(RUNNERS),
        "source_seed": SOURCE_SEED,
        "stream_seed": STREAM_SEED,
        "expected_cells": len(specs),
        "auxiliary_denominator": "all confidence-admitted anchors |C|",
        "candidate_evaluation": "one [B,C,T] forward per view",
        "gathered_training_batch_rechecked": True,
        "bdup_contract": "B1 candidates/masks/weights/denominator; raw tensor only",
        "effective_ssaw_seed": 42_001_855,
        "semantic_role": "fixed C_intersect_S auxiliary router",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_new_spline_parameter_selection": False,
        "effective_source_config": specs[0]["source_config"],
        "effective_tta_config": specs[0]["tta_config"],
        "max_batches": args.max_batches,
        "b0_only": bool(args.b0_only),
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    completed = 0
    failures = []
    b0_reproduction = None
    for cell_index, spec in enumerate(specs, start=1):
        cell_dir = Path(spec["cell_dir"])
        signature_hash = _hash_json(_signature(spec))
        if _complete(cell_dir, signature_hash):
            completed += 1
        else:
            cell_dir.mkdir(parents=True, exist_ok=True)
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
                        "scenario": _flow_label(spec["flow"]),
                        "runner": spec["runner"],
                        "returncode": int(process.returncode),
                        "log": str(log_path),
                    }
                )
                if args.fail_fast:
                    break
            else:
                completed += 1
        _publish(args.output_dir)
        if (
            cell_index == len(FLOWS)
            and args.max_batches is None
            and not failures
        ):
            try:
                b0_reproduction = _validate_b0_reproduction(args.output_dir)
            except RuntimeError as exc:
                failures.append(
                    {
                        "phase": "B0_reproduction_gate",
                        "error": str(exc),
                    }
                )
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_cells": completed,
                "current_cell": cell_index,
                "b0_reproduction": b0_reproduction,
                "failures": failures,
            },
            args.output_dir / "status.json",
        )
        if failures and (
            args.fail_fast or cell_index == len(FLOWS)
        ):
            break
    _publish(args.output_dir)
    status = "complete" if completed == len(specs) and not failures else "failed"
    final = {
        **manifest,
        "status": status,
        "completed_cells": completed,
        "b0_reproduction": b0_reproduction,
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
    parser.add_argument("--b0-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
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
