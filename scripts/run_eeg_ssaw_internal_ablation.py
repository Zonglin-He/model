"""Run the seed-1 one-factor-at-a-time SSAW internal ablation on EEG."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    SSAW_INTERNAL_ABLATION_RUNNERS,
)
from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from scripts.dusafe_factorial_runner_common import current_profiles  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_har_spline_mechanism_matrix import (  # noqa: E402
    SPLINE_PROFILE,
    _run_cell as _run_mechanism_cell,
)
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402


PROTOCOL = "eeg_ssaw_internal_ablation_seed1_v1_steps2"
DATASET = "EEG"
FLOWS = tuple(formal_scenario_pairs(DATASET))
RUNNERS = tuple(SSAW_INTERNAL_ABLATION_RUNNERS)
SOURCE_SEED = 1
STREAM_SEED = 42
INNER_STEPS = 2
FULL_RUNNER = "B4_boundary_spline_residual_kl"
BASELINE_RUNNER = "B0_raw_only"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "eeg_ssaw_internal_ablation_seed1_v1"
)
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"

COMPONENT_REMOVALS = {
    "A2_no_semantic_router": "source_semantic_router",
    "A3_no_coefficient_search": "coefficient_boundary_search",
    "A4_no_margin_filter": "hardness_margin_filter",
    "A5_no_radius_backtracking": "radius_backtracking",
    "A6_no_gathered_recheck": "gathered_batch_recheck",
    BASELINE_RUNNER: "entire_ssaw_branch",
}


def _hash(payload: Mapping[str, object]) -> str:
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
        "dataset": DATASET,
        "flow": list(spec["flow"]),
        "runner": str(spec["runner"]),
        "source_seed": int(spec["source_seed"]),
        "stream_seed": int(spec["stream_seed"]),
        "source_config": spec["source_config"],
        "tta_config": spec["tta_config"],
        "max_batches": spec.get("max_batches"),
        "target_labels_used_for_online_decision": False,
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
        and summary.get("protocol") == PROTOCOL
        and summary.get("signature_hash") == signature_hash
        and (cell_dir / "sample_records.csv").is_file()
        and (cell_dir / "batch_diagnostics.csv").is_file()
    )


def _build_specs(args) -> list[dict[str, object]]:
    source_config, tta_config = current_profiles(DATASET)
    tta_config = {
        **tta_config,
        **SPLINE_PROFILE,
        "dusafe_variant": "fixed_kl_b4",
        "steps": INNER_STEPS,
    }
    return [
        {
            "dataset": DATASET,
            "cell_dir": str(_cell_dir(args.output_dir, flow, runner).resolve()),
            "flow": list(flow),
            "runner": runner,
            "source_seed": SOURCE_SEED,
            "stream_seed": STREAM_SEED,
            "source_config": dict(source_config),
            "tta_config": dict(tta_config),
            "data_path": str(args.data_path.resolve()),
            "device": str(args.device),
            "backbone": str(args.backbone),
            "pretrain_cache_dir": str(args.pretrain_cache_dir.resolve()),
            "gpu_lock_path": str(args.gpu_lock_path.resolve()),
            "max_batches": args.max_batches,
        }
        for runner in RUNNERS
        for flow in FLOWS
    ]


def _worker(spec_path: Path) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    signature_hash = _hash(_signature(spec))
    if _complete(cell_dir, signature_hash):
        return 0
    result = samples = batches = None
    try:
        lock = (
            wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
            if str(spec["device"]).lower().startswith("cuda")
            else None
        )
        if lock is None:
            result, samples, batches = _run_mechanism_cell(spec)
        else:
            with lock:
                result, samples, batches = _run_mechanism_cell(spec)
        result.update(
            {
                "protocol": PROTOCOL,
                "signature_hash": signature_hash,
                "component_removed": COMPONENT_REMOVALS.get(
                    str(spec["runner"]), "none_full"
                ),
                "evaluation_partition": "target_selected_evaluation",
                "confirmatory": False,
                "target_labels_used_for_online_decision": False,
                "target_labels_used_for_parameter_selection": True,
            }
        )
        atomic_write_csv(samples, cell_dir / "sample_records.csv", index=False)
        atomic_write_csv(batches, cell_dir / "batch_diagnostics.csv", index=False)
        atomic_write_json(result, cell_dir / "summary.json")
        return 0
    except BaseException as exc:
        atomic_write_json(
            {
                "status": "failed",
                "protocol": PROTOCOL,
                "signature_hash": signature_hash,
                "dataset": DATASET,
                "scenario": _flow_label(spec["flow"]),
                "source_seed": int(spec["source_seed"]),
                "stream_seed": int(spec["stream_seed"]),
                "runner": str(spec["runner"]),
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
        del result, samples, batches
        release_cuda()
        gc.collect()


def _collect(specs: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        path = Path(spec["cell_dir"]) / "summary.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _validate(raw: pd.DataFrame) -> dict[str, object]:
    expected = {
        (_flow_label(flow), runner) for flow in FLOWS for runner in RUNNERS
    }
    if raw.empty:
        return {"status": "failed", "reason": "no result rows"}
    ok = raw.loc[raw["status"].eq("ok")].copy()
    keys = list(zip(ok["scenario"].astype(str), ok["runner"].astype(str)))
    observed = set(keys)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    source_mismatches = []
    for scenario, group in ok.groupby("scenario", dropna=False):
        hashes = sorted(set(group["source_model_sha256"].astype(str)))
        if len(hashes) != 1:
            source_mismatches.append({"scenario": str(scenario), "hashes": hashes})
    missing = sorted(expected - observed)
    foreign = sorted(observed - expected)
    passed = not (missing or foreign or duplicates or source_mismatches)
    return {
        "status": "passed" if passed else "failed",
        "expected_cells": len(expected),
        "completed_cells": len(observed & expected),
        "missing": missing,
        "foreign": foreign,
        "duplicate_keys": duplicates,
        "source_checkpoint_mismatches": source_mismatches,
    }


def _effects(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw.loc[raw["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    pivot = ok.pivot(index="scenario", columns="runner", values="f1")
    rows = []
    for runner in RUNNERS:
        if runner == FULL_RUNNER or runner not in pivot:
            continue
        for scenario in pivot.index:
            rows.append(
                {
                    "scenario": scenario,
                    "runner": runner,
                    "component_removed": COMPONENT_REMOVALS[runner],
                    "ablation_f1": float(pivot.loc[scenario, runner]),
                    "full_f1": float(pivot.loc[scenario, FULL_RUNNER]),
                    "ablation_minus_full": float(
                        pivot.loc[scenario, runner]
                        - pivot.loc[scenario, FULL_RUNNER]
                    ),
                    "ablation_minus_no_ssaw": float(
                        pivot.loc[scenario, runner]
                        - pivot.loc[scenario, BASELINE_RUNNER]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _publish(raw: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    ok = raw.loc[raw["status"].eq("ok")].copy()
    if ok.empty:
        summary = {"status": "incomplete", "completed_cells": 0}
        atomic_write_json(summary, output_dir / "analysis.json")
        return summary
    table = ok.pivot(index="scenario", columns="runner", values="f1")
    atomic_write_csv(table.reset_index(), output_dir / "flow_f1_table.csv", index=False)
    effects = _effects(raw)
    atomic_write_csv(effects, output_dir / "component_effects_by_flow.csv", index=False)
    summary = (
        effects.groupby(["runner", "component_removed"], as_index=False)
        .agg(
            flows=("scenario", "size"),
            f1_mean=("ablation_f1", "mean"),
            removal_effect_mean=("ablation_minus_full", "mean"),
            vs_no_ssaw_mean=("ablation_minus_no_ssaw", "mean"),
            positive_flows=("ablation_minus_full", lambda x: int((x > 0).sum())),
            zero_flows=("ablation_minus_full", lambda x: int((x == 0).sum())),
            negative_flows=("ablation_minus_full", lambda x: int((x < 0).sum())),
        )
    )
    full_row = pd.DataFrame(
        [
            {
                "runner": FULL_RUNNER,
                "component_removed": "none_full",
                "flows": len(FLOWS),
                "f1_mean": float(table[FULL_RUNNER].mean()),
                "removal_effect_mean": 0.0,
                "vs_no_ssaw_mean": float(
                    (table[FULL_RUNNER] - table[BASELINE_RUNNER]).mean()
                ),
                "positive_flows": 0,
                "zero_flows": len(FLOWS),
                "negative_flows": 0,
            }
        ]
    )
    summary = pd.concat([summary, full_row], ignore_index=True).sort_values(
        "f1_mean", ascending=False
    )
    atomic_write_csv(summary, output_dir / "component_summary.csv", index=False)
    best_f1 = float(summary["f1_mean"].max())
    best_runners = sorted(
        summary.loc[
            summary["f1_mean"].sub(best_f1).abs().le(1e-12), "runner"
        ].astype(str)
    )
    analysis = {
        "status": (
            "complete"
            if len(ok) == len(FLOWS) * len(RUNNERS)
            else "incomplete"
        ),
        "full_f1_mean": float(table[FULL_RUNNER].mean()),
        "no_ssaw_f1_mean": float(table[BASELINE_RUNNER].mean()),
        "best_runner": best_runners[0] if len(best_runners) == 1 else None,
        "best_runners": best_runners,
        "best_f1_mean": best_f1,
    }
    atomic_write_json(analysis, output_dir / "analysis.json")
    return analysis


def _run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(args)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "dataset": DATASET,
        "flows": [_flow_label(flow) for flow in FLOWS],
        "source_seed": SOURCE_SEED,
        "stream_seed": STREAM_SEED,
        "runners": list(RUNNERS),
        "component_removals": COMPONENT_REMOVALS,
        "expected_cells": len(specs),
        "one_factor_at_a_time": True,
        "shared_raw_admission": "source-calibrated confidence admission",
        "shared_auxiliary_objective": "residual KL normalized by admitted raw anchors",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "effective_source_config": specs[0]["source_config"],
        "effective_tta_config": specs[0]["tta_config"],
        "max_batches": args.max_batches,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    completed = 0
    failures = []
    for index, spec in enumerate(specs, start=1):
        cell_dir = Path(spec["cell_dir"])
        signature_hash = _hash(_signature(spec))
        if not _complete(cell_dir, signature_hash):
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
                        "runner": str(spec["runner"]),
                        "returncode": int(process.returncode),
                        "log": str(log_path),
                    }
                )
                if args.fail_fast:
                    break
        if _complete(cell_dir, signature_hash):
            completed += 1
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_cells": completed,
                "current_cell": index,
                "current_runner": str(spec["runner"]),
                "current_scenario": _flow_label(spec["flow"]),
                "failures": failures,
            },
            args.output_dir / "status.json",
        )
    raw = _collect(specs)
    validation = _validate(raw)
    analysis = _publish(raw, args.output_dir)
    status = (
        "complete"
        if validation["status"] == "passed" and analysis["status"] == "complete"
        else "failed"
    )
    final = {
        **manifest,
        "status": status,
        "completed_cells": int(validation.get("completed_cells", 0)),
        "validation": validation,
        "analysis": analysis,
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
