"""Run the untuned HAR R1--R4 unified-spline router experiment.

Every cell uses the checked-in HAR source/TTA profile, source seed 1, stream
seed 42, and one of the five registered HAR flows.  Target labels are kept out
of the adapter and are read only after online inference for descriptive F1 and
diagnostics.  The checked-in HAR TTA profile was previously target-selected on
these flows; only the new spline profile is untuned.  Each GPU cell is isolated
in a child process for crash/OOM-safe resume.
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
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe_spline_hard_view import (  # noqa: E402
    SPLINE_ROUTER_RUNNERS,
    get_spline_router_runner,
)
from configs.data_model_configs import HAR  # noqa: E402
from scripts.dusafe_factorial_runner_common import (  # noqa: E402
    current_profiles,
    tensor_state_sha256,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
)


PROTOCOL = "har_spline_router_r1_r4_v2_target_selected_base_profile"
RUNNERS = tuple(SPLINE_ROUTER_RUNNERS)
FLOWS = tuple((str(source), str(target)) for source, target in HAR.scenarios)
SOURCE_SEED = 1
STREAM_SEED = 42
SPLINE_PROFILE = {
    "spline_control_points": 10,
    "spline_num_directions": 4,
    "spline_log_strength": 0.2,
    "spline_radius_levels": (1.0, 0.5, 0.25),
}
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ablation" / "har_spline_router_seed1_v1"
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"


class _LimitedLoader:
    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = int(max_batches)
        self.meta = getattr(loader, "meta", None)

    def __iter__(self):
        for batch_index, batch in enumerate(self.loader):
            if batch_index >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


def _hash_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _cell_dir(output_dir: Path, flow: Sequence[str], runner: str) -> Path:
    return output_dir / "cells" / f"flow_{flow[0]}_to_{flow[1]}" / runner


def _signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "dataset": "HAR",
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
        "evaluation_partition": "target_selected_descriptive",
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


def _run_cell(spec: Mapping[str, object]) -> dict[str, object]:
    flow = tuple(str(value) for value in spec["flow"])
    runner_name = str(spec["runner"])
    trainer = build_trainer(
        data_path=str(spec["data_path"]),
        device=str(spec["device"]),
        dataset="HAR",
        da_method="DuSafe",
        backbone=str(spec["backbone"]),
        exp_name=f"spline_router_{runner_name}",
        seed=int(spec["stream_seed"]),
        source_seed=int(spec["source_seed"]),
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        runner_class = get_spline_router_runner(runner_name)
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(spec["source_config"]))
        trainer.set_runtime_hparams(dict(spec["tta_config"]))
        adapted, source_model = create_tta_model(
            trainer,
            flow[0],
            flow[1],
            run_seed=int(spec["stream_seed"]),
        )
        source_hash = tensor_state_sha256(source_model)
        source_checkpoint = str(trainer._pretrain_cache_path() or "")
        if spec.get("max_batches") is not None:
            trainer.trg_whole_dl = _LimitedLoader(
                trainer.trg_whole_dl, int(spec["max_batches"])
            )
        metrics = trainer.calculate_metrics(adapted)

        samples = trainer.last_safety_records.copy()
        samples.insert(0, "runner", runner_name)
        samples.insert(0, "stream_seed", int(spec["stream_seed"]))
        samples.insert(0, "source_seed", int(spec["source_seed"]))
        samples.insert(0, "scenario", _flow_label(flow))
        samples.insert(0, "dataset", "HAR")

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
            )
        ):
            batches.insert(0, name, value)

        result = {
            "status": "ok",
            "protocol": PROTOCOL,
            "dataset": "HAR",
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
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
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
        failure = {
            "status": "failed",
            "protocol": PROTOCOL,
            "scenario": _flow_label(spec["flow"]),
            "runner": spec["runner"],
            "source_seed": int(spec["source_seed"]),
            "stream_seed": int(spec["stream_seed"]),
            "signature_hash": signature_hash,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower(),
        }
        atomic_write_json(failure, cell_dir / "summary.json")
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 1
    finally:
        release_cuda()
        gc.collect()


def _read_completed(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    sample_frames = []
    batch_frames = []
    for flow in FLOWS:
        for runner in RUNNERS:
            cell_dir = _cell_dir(output_dir, flow, runner)
            summary_path = cell_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "ok":
                continue
            summaries.append(summary)
            sample_frames.append(pd.read_csv(cell_dir / "sample_records.csv"))
            batch_frames.append(pd.read_csv(cell_dir / "batch_diagnostics.csv"))
    return (
        pd.DataFrame(summaries),
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(),
        pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame(),
    )


def _variant_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    diagnostic_columns = [
        column
        for column in raw.columns
        if column.startswith("diag_")
        and any(
            token in column
            for token in (
                "gradient",
                "router",
                "selected_radius",
                "endpoint_flip",
                "backtracking",
                "final_skip",
                "positive_selection",
            )
        )
    ]
    metrics = ["f1", "accuracy", "auroc", "risk", *diagnostic_columns]
    return raw.groupby("runner", as_index=False)[metrics].mean(numeric_only=True)


def _paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    pivot = raw.pivot(index="scenario", columns="runner", values="f1")
    rows = []
    baseline = "r1_confidence_raw_only"
    if baseline not in pivot.columns:
        return pd.DataFrame()
    for runner in RUNNERS[1:]:
        if runner not in pivot.columns:
            continue
        delta = pivot[runner] - pivot[baseline]
        for scenario, value in delta.items():
            rows.append(
                {
                    "scenario": scenario,
                    "contrast": f"{runner}-R1",
                    "delta_f1": float(value),
                }
            )
        rows.append(
            {
                "scenario": "mean_five_flows",
                "contrast": f"{runner}-R1",
                "delta_f1": float(delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def _router_subset_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = []
    for (scenario, runner), group in samples.groupby(["scenario", "runner"]):
        confidence = group["admitted"].astype(bool)
        semantic = group["source_semantic_router_agree"].astype(bool)
        subsets = {
            "C": confidence,
            "C_intersect_S": confidence & semantic,
            "C_minus_S": confidence & (~semantic),
        }
        pseudo_correct = group["pseudo_label"].eq(group["label"])
        for subset_name, mask in subsets.items():
            count = int(mask.sum())
            rows.append(
                {
                    "scenario": scenario,
                    "runner": runner,
                    "subset": subset_name,
                    "count": count,
                    "coverage": float(mask.mean()),
                    "pseudo_label_accuracy": (
                        float(pseudo_correct[mask].mean()) if count else math.nan
                    ),
                    "post_update_accuracy": (
                        float(group.loc[mask, "correct"].mean()) if count else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _class_coverage(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in samples.groupby(["scenario", "runner", "label"]):
        confidence = group["admitted"].astype(bool)
        semantic = group["source_semantic_router_agree"].astype(bool)
        rows.append(
            {
                "scenario": keys[0],
                "runner": keys[1],
                "label": int(keys[2]),
                "samples": int(len(group)),
                "confidence_coverage": float(confidence.mean()),
                "semantic_agree_within_confidence": (
                    float(semantic[confidence].mean()) if confidence.any() else math.nan
                ),
                "ssaw_router_coverage": float(
                    group["ssaw_router_selected"].astype(bool).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _batch_performance(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in samples.groupby(["scenario", "runner", "batch_index"]):
        labels = group["label"].astype(int)
        predictions = group["prediction"].astype(int)
        rows.append(
            {
                "scenario": keys[0],
                "runner": keys[1],
                "batch_index": int(keys[2]),
                "samples": int(len(group)),
                "macro_f1": float(
                    f1_score(labels, predictions, average="macro", zero_division=0)
                ),
                "true_label_nll": float(
                    pd.to_numeric(
                        group["post_update_true_label_nll"], errors="coerce"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _batch_trajectory_descriptive(
    batch_performance: pd.DataFrame, batch_diagnostics: pd.DataFrame
) -> pd.DataFrame:
    if batch_performance.empty:
        return pd.DataFrame()
    baseline = batch_performance[
        batch_performance["runner"].eq("r1_confidence_raw_only")
    ][["scenario", "batch_index", "macro_f1", "true_label_nll"]].rename(
        columns={"macro_f1": "r1_f1", "true_label_nll": "r1_nll"}
    )
    rows = []
    for runner in RUNNERS[1:]:
        variant = batch_performance[batch_performance["runner"].eq(runner)].copy()
        if variant.empty:
            continue
        variant = variant.merge(baseline, on=["scenario", "batch_index"], how="inner")
        variant["same_batch_trajectory_delta_f1"] = (
            variant["macro_f1"] - variant["r1_f1"]
        )
        variant["same_batch_trajectory_delta_nll"] = (
            variant["true_label_nll"] - variant["r1_nll"]
        )
        if not batch_diagnostics.empty:
            diagnostics = batch_diagnostics[
                batch_diagnostics["runner"].eq(runner)
            ].copy()
            keep = [
                column
                for column in diagnostics.columns
                if column in ("scenario", "batch_index")
                or any(
                    token in column
                    for token in (
                        "gradient",
                        "router_selected",
                        "selected_radius",
                        "endpoint_flip",
                        "backtracking",
                        "final_skip",
                    )
                )
            ]
            variant = variant.merge(
                diagnostics[keep],
                on=["scenario", "batch_index"],
                how="left",
            )
        variant.insert(1, "contrast", f"{runner}-R1")
        rows.append(variant)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _publish(output_dir: Path) -> None:
    raw, samples, batch_diagnostics = _read_completed(output_dir)
    if not raw.empty:
        raw = raw.sort_values(["scenario", "runner"]).reset_index(drop=True)
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    atomic_write_csv(_variant_summary(raw), output_dir / "variant_summary.csv", index=False)
    atomic_write_csv(_paired_effects(raw), output_dir / "paired_effects.csv", index=False)
    if not raw.empty:
        flow_table = raw.pivot(index="scenario", columns="runner", values="f1")
        atomic_write_csv(flow_table.reset_index(), output_dir / "flow_f1_table.csv", index=False)
    atomic_write_csv(
        _router_subset_metrics(samples), output_dir / "router_subset_metrics.csv", index=False
    )
    atomic_write_csv(
        _class_coverage(samples), output_dir / "class_coverage.csv", index=False
    )
    batch_performance = _batch_performance(samples)
    atomic_write_csv(batch_performance, output_dir / "batch_performance.csv", index=False)
    atomic_write_csv(
        _batch_trajectory_descriptive(batch_performance, batch_diagnostics),
        output_dir / "batch_trajectory_descriptive.csv",
        index=False,
    )


def _build_specs(args) -> list[dict[str, object]]:
    source_config, tta_config = current_profiles("HAR")
    tta_config = {**tta_config, **SPLINE_PROFILE}
    specs = []
    for flow in FLOWS:
        for runner in RUNNERS:
            cell_dir = _cell_dir(args.output_dir, flow, runner)
            specs.append(
                {
                    "protocol": PROTOCOL,
                    "cell_dir": str(cell_dir.resolve()),
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
            )
    return specs


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
        "spline_profile": SPLINE_PROFILE,
        "selection_rule": "minimum pseudo-class logit margin among first label-preserving radius per antithetic spline ray",
        "raw_admission": "frozen-source confidence only",
        "semantic_role": "raw-prediction SSAW router only; never raw admission",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_new_spline_parameter_selection": False,
        "untuned": True,
        "max_batches": args.max_batches,
        "effective_source_config": specs[0]["source_config"],
        "effective_tta_config": specs[0]["tta_config"],
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    completed = 0
    failures = []
    for cell_index, spec in enumerate(specs, start=1):
        cell_dir = Path(spec["cell_dir"])
        signature_hash = _hash_json(_signature(spec))
        if _complete(cell_dir, signature_hash):
            completed += 1
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        spec_path = cell_dir / "worker_spec.json"
        atomic_write_json(spec, spec_path)
        log_path = cell_dir / "worker.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)]
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
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
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_cells": completed,
                "failures": failures,
                "current_cell": cell_index,
            },
            args.output_dir / "status.json",
        )
    _publish(args.output_dir)
    final_status = "complete" if completed == len(specs) and not failures else "failed"
    atomic_write_json(
        {
            **manifest,
            "status": final_status,
            "completed_cells": completed,
            "failures": failures,
        },
        args.output_dir / "status.json",
    )
    manifest["status"] = final_status
    manifest["completed_cells"] = completed
    manifest["failures"] = failures
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    return 0 if final_status == "complete" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "Dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--max-batches", type=int, default=None)
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
