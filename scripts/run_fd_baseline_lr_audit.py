"""Audit FD TENT/SAR learning-rate sensitivity and collapse trajectories.

The best learning rate in this script is an oracle diagnostic selected with
target labels.  It is never substituted into the fixed-protocol main table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_main_table import GPUExperimentLock, tensor_state_sha256
from scripts.supplementary_utils import (
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
)


DEFAULT_LRS = (
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    2.5e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
)
JOB_KEYS = (
    "method",
    "learning_rate_key",
    "scenario",
    "source_seed",
    "stream_seed",
)


def parse_csv(raw: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def lr_key(value: float) -> str:
    return format(float(value), ".12g")


def _job_key(row: dict) -> tuple:
    raw_learning_rate = row.get(
        "learning_rate_key", row.get("learning_rate")
    )
    return (
        str(row["method"]),
        lr_key(float(raw_learning_rate)),
        str(row["scenario"]),
        int(row["source_seed"]),
        int(row["stream_seed"]),
    )


def _trajectory_key(row: dict) -> tuple:
    return (*_job_key(row), int(row["batch_index"]))


def _deduplicate_rows(rows: list[dict], key_fn) -> tuple[list[dict], list[dict]]:
    """Keep the latest atomic record per normalized key and archive prior rows."""

    latest = {}
    discarded = []
    for row in rows:
        key = key_fn(row)
        if key in latest:
            discarded.append(latest[key])
        latest[key] = row
    return list(latest.values()), discarded


def _atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _trajectory(records: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    required = {"batch_index", "label", "prediction"}
    if records.empty or not required.issubset(records.columns):
        return pd.DataFrame()
    rows = []
    cumulative = []
    cumulative_labels = []
    cumulative_pre = []
    for batch_index, batch in records.groupby("batch_index", sort=True):
        labels = batch["label"].to_numpy(dtype=np.int64)
        predictions = batch["prediction"].to_numpy(dtype=np.int64)
        pre_predictions = batch.get(
            "pre_final_update_prediction", batch["prediction"]
        ).to_numpy(dtype=np.int64)
        cumulative.extend(predictions.tolist())
        cumulative_pre.extend(pre_predictions.tolist())
        cumulative_labels.extend(labels.tolist())
        selected = batch.get(
            "selected", pd.Series(False, index=batch.index)
        ).astype(bool)
        rows.append(
            {
                **metadata,
                "batch_index": int(batch_index),
                "batch_samples": int(len(batch)),
                "batch_accuracy": float(np.mean(predictions == labels)),
                "batch_macro_f1": float(
                    f1_score(
                        labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "cumulative_accuracy": float(
                    np.mean(
                        np.asarray(cumulative)
                        == np.asarray(cumulative_labels)
                    )
                ),
                "cumulative_macro_f1": float(
                    f1_score(
                        cumulative_labels,
                        cumulative,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "cumulative_pre_update_macro_f1": float(
                    f1_score(
                        cumulative_labels,
                        cumulative_pre,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "selected_fraction": float(selected.mean()),
            }
        )
    return pd.DataFrame(rows)


def run_job(args, method: str, learning_rate: float, scenario, source_seed: int):
    src_id, trg_id = (str(scenario[0]), str(scenario[1]))
    scenario_label = f"{src_id}->{trg_id}"
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset="FD",
        da_method=method,
        backbone=args.backbone,
        exp_name=f"fd_lr_audit_{method}_{lr_key(learning_rate)}",
        seed=args.stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        algorithm_registry="benchmark",
    )
    default_lr = float(trainer.hparams.get("learning_rate", float("nan")))
    trainer.set_runtime_hparams({"learning_rate": float(learning_rate)})
    tta_model = source_model = None
    source_hash = ""
    try:
        def pre_tta_hook(_trainer, model):
            nonlocal source_hash
            source_hash = tensor_state_sha256(model)

        tta_model, source_model = create_tta_model(
            trainer,
            src_id,
            trg_id,
            run_seed=args.stream_seed,
            pre_tta_hook=pre_tta_hook,
        )
        accuracy, macro_f1, auroc, risk = trainer.calculate_metrics(tta_model)
        metadata = {
            "method": method,
            "learning_rate": float(learning_rate),
            "learning_rate_key": lr_key(learning_rate),
            "scenario": scenario_label,
            "source_seed": int(source_seed),
            "stream_seed": int(args.stream_seed),
        }
        trajectory = _trajectory(trainer.last_safety_records.copy(), metadata)
        row = {
            "status": "ok",
            **metadata,
            "default_learning_rate": default_lr,
            "accuracy": float(accuracy),
            "f1": float(macro_f1),
            "auroc": float(auroc),
            "risk": float(risk),
            "source_model_sha256": source_hash,
            "source_checkpoint_path": str(
                trainer._pretrain_cache_path() or ""
            ),
            "runtime_hparams": json.dumps(
                trainer.hparams, sort_keys=True, default=str
            ),
            "error_type": "",
            "error": "",
            "is_oom": False,
        }
        return row, trajectory
    except Exception as error:
        row = {
            "status": "failed",
            "method": method,
            "learning_rate": float(learning_rate),
            "learning_rate_key": lr_key(learning_rate),
            "scenario": scenario_label,
            "source_seed": int(source_seed),
            "stream_seed": int(args.stream_seed),
            "default_learning_rate": default_lr,
            "source_model_sha256": source_hash,
            "source_checkpoint_path": str(
                trainer._pretrain_cache_path() or ""
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=20),
            "is_oom": bool(
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
            ),
        }
        return row, pd.DataFrame()
    finally:
        cleanup_trainer(
            trainer, tta_model, source_model, close_summary=True
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate(summary: pd.DataFrame, trajectory: pd.DataFrame, output_dir: Path) -> None:
    ok = summary[summary["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        raise RuntimeError("FD LR audit has no successful cells")
    per_seed = (
        ok.groupby(
            ["method", "learning_rate", "learning_rate_key", "source_seed"],
            as_index=False,
        )[["accuracy", "f1", "auroc"]]
        .mean()
    )
    grid = (
        per_seed.groupby(
            ["method", "learning_rate", "learning_rate_key"], as_index=False
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auroc_mean=("auroc", "mean"),
            source_seed_count=("source_seed", "nunique"),
        )
    )
    selection_rows = []
    chosen = {}
    for method, group in grid.groupby("method", sort=True):
        ordered = group.sort_values(
            ["f1_mean", "learning_rate"], ascending=[False, True]
        )
        best = ordered.iloc[0]
        defaults = sorted(
            set(
                float(value)
                for value in ok.loc[
                    ok["method"].eq(method), "default_learning_rate"
                ].dropna()
            )
        )
        default_lr = defaults[0] if len(defaults) == 1 else float("nan")
        chosen[method] = {lr_key(best["learning_rate"]), lr_key(default_lr)}
        selection_rows.append(
            {
                "method": method,
                "default_learning_rate": default_lr,
                "oracle_learning_rate": float(best["learning_rate"]),
                "oracle_f1_mean": float(best["f1_mean"]),
                "target_labels_used_for_oracle_selection": True,
                "main_table_uses_oracle_learning_rate": False,
            }
        )
    selection = pd.DataFrame(selection_rows)
    grid = grid.merge(
        selection[["method", "default_learning_rate", "oracle_learning_rate"]],
        on="method",
        how="left",
    )
    grid["is_default"] = np.isclose(
        grid["learning_rate"], grid["default_learning_rate"]
    )
    grid["is_oracle_best"] = np.isclose(
        grid["learning_rate"], grid["oracle_learning_rate"]
    )
    atomic_write_csv(per_seed, output_dir / "per_source_seed.csv", index=False)
    atomic_write_csv(grid, output_dir / "lr_grid_summary.csv", index=False)
    atomic_write_csv(selection, output_dir / "oracle_selection.csv", index=False)

    if not trajectory.empty:
        trajectory_aggregate = (
            trajectory.groupby(
                ["method", "learning_rate", "learning_rate_key", "batch_index"],
                as_index=False,
            )
            .agg(
                cumulative_macro_f1_mean=("cumulative_macro_f1", "mean"),
                cumulative_macro_f1_std=("cumulative_macro_f1", "std"),
                cumulative_accuracy_mean=("cumulative_accuracy", "mean"),
                batch_macro_f1_mean=("batch_macro_f1", "mean"),
                selected_fraction_mean=("selected_fraction", "mean"),
                independent_source_seeds=("source_seed", "nunique"),
                scenario_seed_cells=("scenario", "count"),
            )
        )
        selected_mask = trajectory_aggregate.apply(
            lambda row: row["learning_rate_key"] in chosen.get(row["method"], set()),
            axis=1,
        )
        atomic_write_csv(
            trajectory_aggregate,
            output_dir / "trajectory_aggregate_all_lrs.csv",
            index=False,
        )
        atomic_write_csv(
            trajectory_aggregate[selected_mask],
            output_dir / "trajectory_default_vs_oracle.csv",
            index=False,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--methods", default="Tent,SAR")
    parser.add_argument(
        "--learning-rates",
        default=",".join(format(value, ".12g") for value in DEFAULT_LRS),
    )
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--limit-jobs", type=int, default=None)
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "diagnostics" / "fd_tent_sar_lr_audit_v1"),
    )
    args = parser.parse_args(argv)
    methods = parse_csv(args.methods)
    if not methods or set(methods) - {"Tent", "SAR"}:
        parser.error("--methods must be a non-empty subset of Tent,SAR")
    learning_rates = parse_csv(args.learning_rates, float)
    if not learning_rates or any(value <= 0 for value in learning_rates):
        parser.error("--learning-rates must contain positive values")
    source_seeds = parse_csv(args.source_seeds, int)
    if not source_seeds or len(source_seeds) != len(set(source_seeds)):
        parser.error("--source-seeds must contain unique values")
    output_dir = Path(ensure_dir(args.output_dir))
    summary_path = output_dir / "cell_results.csv"
    trajectory_path = output_dir / "batch_trajectories.csv"
    loaded_summary_rows = (
        pd.read_csv(summary_path).to_dict("records")
        if summary_path.exists()
        else []
    )
    loaded_trajectory_rows = (
        pd.read_csv(trajectory_path).to_dict("records")
        if trajectory_path.exists()
        else []
    )
    summary_rows, duplicate_summary_rows = _deduplicate_rows(
        loaded_summary_rows, _job_key
    )
    trajectory_rows, duplicate_trajectory_rows = _deduplicate_rows(
        loaded_trajectory_rows, _trajectory_key
    )
    if duplicate_summary_rows:
        atomic_write_csv(
            pd.DataFrame(duplicate_summary_rows),
            output_dir / "duplicate_cell_rows_archive.csv",
            index=False,
        )
    if duplicate_trajectory_rows:
        atomic_write_csv(
            pd.DataFrame(duplicate_trajectory_rows),
            output_dir / "duplicate_trajectory_rows_archive.csv",
            index=False,
        )
    # Publish the normalized active state before launching another GPU cell.
    atomic_write_csv(pd.DataFrame(summary_rows), summary_path, index=False)
    atomic_write_csv(
        pd.DataFrame(trajectory_rows), trajectory_path, index=False
    )
    completed = {
        _job_key(row)
        for row in summary_rows
        if str(row.get("status")) == "ok"
    }

    probe = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset="FD",
        da_method=methods[0],
        backbone=args.backbone,
        exp_name="fd_lr_audit_probe",
        seed=args.stream_seed,
        source_seed=source_seeds[0],
        pretrain_cache_dir=args.pretrain_cache_dir,
        algorithm_registry="benchmark",
    )
    try:
        scenarios = dataset_scenarios(probe)
    finally:
        cleanup_trainer(probe, close_summary=True)
    requested_scenarios = set(parse_csv(args.scenarios))
    if requested_scenarios:
        scenarios = [
            scenario
            for scenario in scenarios
            if f"{scenario[0]}->{scenario[1]}" in requested_scenarios
        ]
    jobs_run = 0
    lock_path = ROOT / "results" / ".current_experiment_gpu.lock"
    with GPUExperimentLock(lock_path):
        for method in methods:
            for learning_rate in learning_rates:
                for scenario in scenarios:
                    for source_seed in source_seeds:
                        key = (
                            method,
                            lr_key(learning_rate),
                            f"{scenario[0]}->{scenario[1]}",
                            int(source_seed),
                            int(args.stream_seed),
                        )
                        if key in completed:
                            continue
                        if args.limit_jobs is not None and jobs_run >= args.limit_jobs:
                            break
                        print(f"[FD LR audit] {key}", flush=True)
                        row, trajectory = run_job(
                            args, method, learning_rate, scenario, source_seed
                        )
                        summary_rows = [
                            existing
                            for existing in summary_rows
                            if _job_key(existing) != key
                        ]
                        trajectory_rows = [
                            existing
                            for existing in trajectory_rows
                            if _job_key(existing) != key
                        ]
                        summary_rows.append(row)
                        trajectory_rows.extend(trajectory.to_dict("records"))
                        if str(row.get("status")) == "ok":
                            completed.add(key)
                        atomic_write_csv(
                            pd.DataFrame(summary_rows), summary_path, index=False
                        )
                        atomic_write_csv(
                            pd.DataFrame(trajectory_rows), trajectory_path, index=False
                        )
                        jobs_run += 1
                    if args.limit_jobs is not None and jobs_run >= args.limit_jobs:
                        break
                if args.limit_jobs is not None and jobs_run >= args.limit_jobs:
                    break
            if args.limit_jobs is not None and jobs_run >= args.limit_jobs:
                break

    summary = pd.DataFrame(summary_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    expected = len(methods) * len(learning_rates) * len(scenarios) * len(source_seeds)
    if args.limit_jobs is None and len(summary) != expected:
        raise RuntimeError(f"Expected {expected} LR cells, found {len(summary)}")
    aggregate(summary, trajectory, output_dir)
    manifest = {
        "protocol": "FD TENT/SAR LR and collapse trajectory audit v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "FD",
        "methods": methods,
        "learning_rates": learning_rates,
        "scenarios": [f"{src}->{trg}" for src, trg in scenarios],
        "source_seeds": source_seeds,
        "source_seed_is_independent_unit": True,
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "target_labels_used_for_oracle_selection": True,
        "main_table_uses_oracle_learning_rate": False,
        "selection_scope": "diagnose sensitivity/collapse only",
        "expected_cells": expected,
        "recorded_cells": int(len(summary)),
        "failed_cells": int(summary["status"].astype(str).eq("failed").sum()),
    }
    manifest["discarded_duplicate_cell_rows"] = int(
        len(duplicate_summary_rows)
    )
    manifest["discarded_duplicate_trajectory_rows"] = int(
        len(duplicate_trajectory_rows)
    )
    _atomic_write_json(manifest, output_dir / "manifest.json")
    return 1 if manifest["failed_cells"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
