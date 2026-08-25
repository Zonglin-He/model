"""Run the selected HAR Fixed-KL/B4 profile against its no-SSAW control.

The experiment uses all five registered HAR source-to-target flows, source
seeds 1/2/3, and paired stream seed 42.  Each cell is isolated in a child
process and acquires the shared GPU lock, so a native failure or OOM cannot
invalidate already completed cells.
"""

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

from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from scripts.dusafe_factorial_runner_common import current_profiles  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_har_adaptive_frontier_matrix import (  # noqa: E402
    FRONTIER_PROFILE,
    _run_cell as _run_fixed_kl_cell,
)
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402


PROTOCOL = "har_fixed_kl_b4_five_flow_v1_steps2"
WORKER_PROTOCOL = "har_adaptive_frontier_matrix_v2_steps2_strict_frontier_membership"
FLOWS = tuple(formal_scenario_pairs("HAR"))
RUNNERS = ("N2_confidence_raw", "Fixed_KL_current_B4")
DEFAULT_SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
INNER_STEPS = 2
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "har_fixed_kl_b4_five_flow_steps2_v1"
)
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"


def _hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_source_seeds(value: str | Sequence[int]) -> tuple[int, ...]:
    pieces = value.split(",") if isinstance(value, str) else value
    seeds = tuple(int(piece) for piece in pieces if str(piece).strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("source seeds must be a non-empty unique list")
    if any(seed not in DEFAULT_SOURCE_SEEDS for seed in seeds):
        raise ValueError("formal HAR source seeds must be drawn from 1,2,3")
    return seeds


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _cell_dir(
    output_dir: Path,
    flow: Sequence[str],
    source_seed: int,
    runner: str,
) -> Path:
    return (
        output_dir
        / "cells"
        / f"flow_{flow[0]}_to_{flow[1]}"
        / f"source_seed_{int(source_seed)}"
        / runner
    )


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


def _build_specs(args, source_seeds: Sequence[int]) -> list[dict[str, object]]:
    source_config, base_tta_config = current_profiles("HAR")
    base_tta_config = {
        **base_tta_config,
        **FRONTIER_PROFILE,
        "steps": INNER_STEPS,
    }
    if args.ssaw_auxiliary_weight is not None:
        base_tta_config["ssaw_auxiliary_weight"] = float(
            args.ssaw_auxiliary_weight
        )
    if args.spline_log_strength is not None:
        base_tta_config["spline_log_strength"] = float(args.spline_log_strength)
    if args.learning_rate is not None:
        base_tta_config["learning_rate"] = float(args.learning_rate)
    specs: list[dict[str, object]] = []
    for runner in RUNNERS:
        for flow in FLOWS:
            for source_seed in source_seeds:
                tta_config = dict(base_tta_config)
                tta_config["dusafe_variant"] = (
                    "fixed_kl_b4"
                    if runner == "Fixed_KL_current_B4"
                    else "confidence_raw_n2"
                )
                specs.append(
                    {
                        "cell_dir": str(
                            _cell_dir(
                                Path(args.output_dir), flow, source_seed, runner
                            ).resolve()
                        ),
                        "frontier_cache_dir": str(
                            (Path(args.output_dir) / "source_frontier_cache").resolve()
                        ),
                        "flow": list(flow),
                        "runner": runner,
                        "source_seed": int(source_seed),
                        "stream_seed": int(args.stream_seed),
                        "source_config": dict(source_config),
                        "tta_config": tta_config,
                        "data_path": str(Path(args.data_path).resolve()),
                        "device": str(args.device),
                        "backbone": str(args.backbone),
                        "pretrain_cache_dir": str(
                            Path(args.pretrain_cache_dir).resolve()
                        ),
                        "gpu_lock_path": str(Path(args.gpu_lock_path).resolve()),
                        "max_batches": args.max_batches,
                    }
                )
    return specs


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
            result, samples, batches = _run_fixed_kl_cell(spec)
        else:
            with lock:
                result, samples, batches = _run_fixed_kl_cell(spec)
        result.update(
            {
                "protocol": PROTOCOL,
                "worker_protocol": WORKER_PROTOCOL,
                "signature_hash": signature_hash,
                "evaluation_partition": "target_selected_evaluation",
                "confirmatory": False,
                "target_labels_used_for_parameter_selection": True,
                "target_labels_used_for_online_decision": False,
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
                "scenario": _flow_label(spec["flow"]),
                "source_seed": int(spec["source_seed"]),
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
        summary_path = Path(spec["cell_dir"]) / "summary.json"
        if summary_path.is_file():
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _validate_complete(
    raw: pd.DataFrame,
    source_seeds: Sequence[int],
) -> dict[str, object]:
    expected = {
        (_flow_label(flow), int(seed), runner)
        for flow in FLOWS
        for seed in source_seeds
        for runner in RUNNERS
    }
    if raw.empty:
        return {"status": "failed", "reason": "no result rows"}
    ok = raw.loc[raw["status"].eq("ok")].copy()
    keys = list(
        zip(
            ok["scenario"].astype(str),
            ok["source_seed"].astype(int),
            ok["runner"].astype(str),
        )
    )
    observed = set(keys)
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    source_mismatches = []
    for (scenario, source_seed), group in ok.groupby(
        ["scenario", "source_seed"], dropna=False
    ):
        hashes = sorted(set(group["source_model_sha256"].astype(str)))
        if len(hashes) != 1:
            source_mismatches.append(
                {
                    "scenario": str(scenario),
                    "source_seed": int(source_seed),
                    "hashes": hashes,
                }
            )
    missing = sorted(expected - observed)
    foreign = sorted(observed - expected)
    passed = not (missing or foreign or duplicate_keys or source_mismatches)
    return {
        "status": "passed" if passed else "failed",
        "expected_cells": len(expected),
        "completed_cells": len(observed & expected),
        "missing": missing,
        "foreign": foreign,
        "duplicate_keys": duplicate_keys,
        "source_checkpoint_mismatches": source_mismatches,
    }


def paired_results(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw.loc[raw["status"].eq("ok")].copy()
    pivot = ok.pivot(
        index=["scenario", "source_seed", "stream_seed"],
        columns="runner",
        values="f1",
    ).reset_index()
    required = set(RUNNERS)
    if not required.issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=list(required)).copy()
    pivot = pivot.rename(
        columns={
            "Fixed_KL_current_B4": "full_f1",
            "N2_confidence_raw": "no_ssaw_f1",
        }
    )
    pivot["full_minus_no_ssaw"] = pivot["full_f1"] - pivot["no_ssaw_f1"]
    return pivot.sort_values(["scenario", "source_seed"]).reset_index(drop=True)


def _publish(raw: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    paired = paired_results(raw)
    atomic_write_csv(paired, output_dir / "paired_results.csv", index=False)
    ok = raw.loc[raw["status"].eq("ok")].copy()
    flow_summary = (
        ok.groupby(["scenario", "runner"], dropna=False)
        .agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
        .reset_index()
        .sort_values(["scenario", "runner"])
    )
    atomic_write_csv(flow_summary, output_dir / "flow_summary.csv", index=False)
    method_summary = (
        ok.groupby("runner", dropna=False)
        .agg(units=("f1", "size"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )
    atomic_write_csv(method_summary, output_dir / "method_summary.csv", index=False)
    if paired.empty:
        summary = {"status": "incomplete"}
    else:
        summary = {
            "status": "complete",
            "paired_units": int(len(paired)),
            "full_f1_mean": float(paired["full_f1"].mean()),
            "no_ssaw_f1_mean": float(paired["no_ssaw_f1"].mean()),
            "full_minus_no_ssaw_mean": float(
                paired["full_minus_no_ssaw"].mean()
            ),
            "positive_units": int(paired["full_minus_no_ssaw"].gt(0.0).sum()),
            "zero_units": int(paired["full_minus_no_ssaw"].eq(0.0).sum()),
            "negative_units": int(paired["full_minus_no_ssaw"].lt(0.0).sum()),
        }
    atomic_write_json(summary, output_dir / "paired_summary.json")
    return summary


def _run_parent(args) -> int:
    source_seeds = _parse_source_seeds(args.source_seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(args, source_seeds)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "dataset": "HAR",
        "flows": [_flow_label(flow) for flow in FLOWS],
        "runners": list(RUNNERS),
        "source_seeds": list(source_seeds),
        "stream_seed": int(args.stream_seed),
        "inner_steps": INNER_STEPS,
        "expected_cells": len(specs),
        "profile_shared_across_all_flows": True,
        "full_variant": "Fixed-KL/B4 boundary spline residual KL",
        "no_ssaw_variant": "N2 confidence-admitted raw TTA",
        "candidate_evaluation": "one [B,C,T] forward per view",
        "gathered_training_batch_rechecked": True,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "evaluation_partition": "target_selected_evaluation",
        "confirmatory": False,
        "effective_source_config": specs[0]["source_config"],
        "effective_tta_configs": {
            runner: next(
                spec["tta_config"]
                for spec in specs
                if spec["runner"] == runner
            )
            for runner in RUNNERS
        },
        "max_batches": args.max_batches,
        "runtime_profile_overrides": {
            key: value
            for key, value in {
                "ssaw_auxiliary_weight": args.ssaw_auxiliary_weight,
                "spline_log_strength": args.spline_log_strength,
                "learning_rate": args.learning_rate,
            }.items()
            if value is not None
        },
    }
    atomic_write_json(manifest, output_dir / "manifest.json")

    failures = []
    completed = 0
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
                    [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                    cwd=str(ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode != 0:
                failures.append(
                    {
                        "scenario": _flow_label(spec["flow"]),
                        "source_seed": int(spec["source_seed"]),
                        "runner": spec["runner"],
                        "returncode": int(process.returncode),
                        "cell_dir": str(cell_dir),
                    }
                )
                if args.fail_fast:
                    break
        if _complete(cell_dir, signature_hash):
            completed += 1
        status = {
            **manifest,
            "status": "running",
            "completed_cells": completed,
            "current_cell": index,
            "current_scenario": _flow_label(spec["flow"]),
            "current_source_seed": int(spec["source_seed"]),
            "current_runner": spec["runner"],
            "failures": failures,
        }
        atomic_write_json(status, output_dir / "status.json")

    raw = _collect(specs)
    validation = _validate_complete(raw, source_seeds)
    paired_summary = _publish(raw, output_dir)
    final_status = (
        "complete"
        if validation["status"] == "passed" and paired_summary["status"] == "complete"
        else "failed"
    )
    final = {
        **manifest,
        "status": final_status,
        "completed_cells": int(validation.get("completed_cells", 0)),
        "validation": validation,
        "paired_summary": paired_summary,
        "failures": failures,
    }
    atomic_write_json(final, output_dir / "manifest.json")
    atomic_write_json(final, output_dir / "status.json")
    return 0 if final_status == "complete" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "Dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--ssaw-auxiliary-weight", type=float)
    parser.add_argument("--spline-log-strength", type=float)
    parser.add_argument("--learning-rate", type=float)
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
