"""Run the preregistered HAR adaptive-frontier SSAW mechanism screen.

The scope is deliberately small: source seed 1, stream seed 42, HAR flows
6->23, 9->18, and 12->16, with exactly two online inner steps.  Numeric
frontier settings are fixed before target evaluation and source-safe caps are
derived only from each source checkpoint's labelled source split.
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

from algorithms.dusafe_adaptive_frontier import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    FRONTIER_RUNNERS,
    AdaptiveFrontierRunner,
    get_frontier_runner,
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


PROTOCOL = "har_adaptive_frontier_matrix_v2_steps2_strict_frontier_membership"
FLOWS = (("6", "23"), ("9", "18"), ("12", "16"))
RUNNERS = tuple(FRONTIER_RUNNERS)
SOURCE_SEED = 1
STREAM_SEED = 42
INNER_STEPS = 2
FRONTIER_PROFILE = {
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
    "record_optimizer_diagnostics": True,
}
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "har_adaptive_frontier_seed1_steps2_v2"
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
        "target_labels_used_for_new_frontier_parameter_selection": False,
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


def _frontier_cache_signature(
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
        "target_labels_used": False,
    }


def _load_or_fit_frontier_reference(
    adapter: AdaptiveFrontierRunner,
    trainer,
    spec: Mapping[str, object],
    source_hash: str,
) -> tuple[dict[str, object], str, bool]:
    cache_dir = Path(spec["frontier_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"HAR_source_{spec['flow'][0]}_seed_{spec['source_seed']}.pt"
    )
    signature = _frontier_cache_signature(spec, source_hash)
    signature_hash = _hash_json(signature)
    if cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError, TypeError):
            payload = None
        if (
            isinstance(payload, Mapping)
            and payload.get("signature_hash") == signature_hash
            and isinstance(payload.get("metadata"), Mapping)
        ):
            metadata = dict(payload["metadata"])
            adapter.load_source_frontier_reference(metadata)
            return metadata, str(cache_path), True

    metadata = adapter.fit_source_frontier_reference(
        trainer.src_test_dl,
        reference_samples=4096,
        calibration_sobol_seed=271_828,
    )
    payload = {
        "signature": signature,
        "signature_hash": signature_hash,
        "metadata": metadata,
    }
    temporary = cache_path.with_suffix(
        cache_path.suffix + f".{os.getpid()}.tmp"
    )
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return metadata, str(cache_path), False


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
        exp_name=f"adaptive_frontier_{dataset}_{runner_name}",
        seed=int(spec["stream_seed"]),
        source_seed=int(spec["source_seed"]),
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        runner_class = get_frontier_runner(runner_name)
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(spec["source_config"]))
        trainer.set_runtime_hparams(dict(spec["tta_config"]))
        adapted, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=int(spec["stream_seed"])
        )
        source_hash = tensor_state_sha256(source_model)
        source_checkpoint = str(trainer._pretrain_cache_path() or "")
        frontier_metadata = None
        frontier_cache_path = ""
        frontier_cache_hit = False
        if isinstance(adapted, AdaptiveFrontierRunner):
            (
                frontier_metadata,
                frontier_cache_path,
                frontier_cache_hit,
            ) = _load_or_fit_frontier_reference(
                adapted, trainer, spec, source_hash
            )
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
            "inner_steps": int(dict(spec["tta_config"])["steps"]),
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
            "target_labels_used_for_new_frontier_parameter_selection": False,
            "evaluation_partition": "target_selected_descriptive",
            "frontier_cache_path": frontier_cache_path,
            "frontier_cache_hit": bool(frontier_cache_hit),
            "source_safe_alpha_cap": (
                math.nan
                if frontier_metadata is None
                else float(frontier_metadata["safe_alpha_cap"])
            ),
            "source_frontier_raw_supported_samples": (
                0
                if frontier_metadata is None
                else int(frontier_metadata["raw_supported_samples"])
            ),
            "source_preservation_by_alpha": (
                {}
                if frontier_metadata is None
                else frontier_metadata["preservation_by_alpha"]
            ),
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


def _paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    pivot = raw.pivot(index="scenario", columns="runner", values="f1")
    baseline = "N2_confidence_raw"
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


def _promotion_decision(raw: pd.DataFrame) -> dict[str, object]:
    primary = "Adaptive_Restore_Budget"
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
    comparisons = {}
    for comparator in ("Adaptive_KL", "Matched_Dup"):
        comparisons[comparator] = (
            math.nan
            if comparator not in pivot
            else float((pivot[primary] - pivot[comparator]).mean())
        )
    passed = bool(
        mean_delta >= 0.003
        and positive >= 2
        and worst >= -0.01
        and comparisons["Adaptive_KL"] > 0.0
        and comparisons["Matched_Dup"] > 0.0
    )
    return {
        "status": "passed" if passed else "failed",
        "primary": primary,
        "baseline": baseline,
        "mean_delta_f1": mean_delta,
        "positive_flows": positive,
        "worst_flow_delta_f1": worst,
        "mean_delta_vs_comparators": comparisons,
        "thresholds": {
            "mean_delta_f1": 0.003,
            "positive_flows": 2,
            "worst_flow_delta_f1": -0.01,
            "must_exceed": ["Adaptive_KL", "Matched_Dup"],
        },
    }


def _publish(output_dir: Path) -> None:
    raw, samples, batches = _read_completed(output_dir)
    if not raw.empty:
        raw = raw.sort_values(["scenario", "runner"]).reset_index(drop=True)
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    atomic_write_csv(samples, output_dir / "sample_records.csv", index=False)
    atomic_write_csv(batches, output_dir / "batch_diagnostics.csv", index=False)
    if not raw.empty:
        numeric = [
            column
            for column in raw.columns
            if column in {"f1", "accuracy", "auroc", "risk"}
            or column.startswith("diag_")
        ]
        summary = raw.groupby("runner", as_index=False)[numeric].mean(
            numeric_only=True
        )
        atomic_write_csv(summary, output_dir / "variant_summary.csv", index=False)
        table = raw.pivot(index="scenario", columns="runner", values="f1")
        atomic_write_csv(
            table.reset_index(), output_dir / "flow_f1_table.csv", index=False
        )
    atomic_write_csv(
        _paired_effects(raw), output_dir / "paired_effects.csv", index=False
    )
    atomic_write_json(_promotion_decision(raw), output_dir / "promotion_decision.json")


def _build_specs(args) -> list[dict[str, object]]:
    source_config, tta_config = current_profiles("HAR")
    tta_config = {**tta_config, **FRONTIER_PROFILE}
    specs = [
        {
            "cell_dir": str(_cell_dir(args.output_dir, flow, runner).resolve()),
            "frontier_cache_dir": str(
                (args.output_dir / "source_frontier_cache").resolve()
            ),
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
    # Establish the two-step raw baseline first, then compare every mechanism
    # from the same source checkpoints.  Each cell runs in an isolated process.
    return sorted(
        specs,
        key=lambda spec: (
            RUNNERS.index(str(spec["runner"])),
            FLOWS.index(tuple(spec["flow"])),
        ),
    )


def _validate_source_identity(raw: pd.DataFrame) -> dict[str, object]:
    if raw.empty:
        return {"status": "incomplete"}
    problems = []
    for scenario, group in raw.groupby("scenario"):
        hashes = sorted(set(group["source_model_sha256"].dropna().astype(str)))
        if len(hashes) != 1:
            problems.append({"scenario": scenario, "hashes": hashes})
    return {
        "status": "passed" if not problems else "failed",
        "problems": problems,
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
        "inner_steps": INNER_STEPS,
        "expected_cells": len(specs),
        "frontier_profile": FRONTIER_PROFILE,
        "source_safe_cap_rule": "largest grid alpha with >=99% frozen-source classifier-and-prototype label preservation",
        "source_uncertainty": "class-conditional frozen-source pseudo-class NLL percentile",
        "auxiliary_denominator": "all confidence-admitted anchors",
        "candidate_evaluation": "one [B,C,T] forward per view",
        "gathered_training_batch_rechecked": True,
        "matched_duplicate_contract": "same adaptive eligibility rule; exact same-batch-start equality requires causal replay",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_new_frontier_parameter_selection": False,
        "effective_source_config": specs[0]["source_config"],
        "effective_tta_config": specs[0]["tta_config"],
        "max_batches": args.max_batches,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    completed = 0
    failures = []
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
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-spec",
                        str(spec_path),
                    ],
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
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_cells": completed,
                "current_cell": cell_index,
                "failures": failures,
            },
            args.output_dir / "status.json",
        )
    _publish(args.output_dir)
    raw, _, _ = _read_completed(args.output_dir)
    source_identity = _validate_source_identity(raw)
    status = (
        "complete"
        if completed == len(specs)
        and not failures
        and source_identity["status"] == "passed"
        else "failed"
    )
    final = {
        **manifest,
        "status": status,
        "completed_cells": completed,
        "source_identity": source_identity,
        "promotion_decision": _promotion_decision(raw),
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
