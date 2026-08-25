"""Run and summarize the current full benchmark main table.

The runner is intentionally separate from the production CLI and from the
controlled-corruption audit.  It evaluates the benchmark-only registry under
one fixed source-training recipe:

* every source checkpoint is keyed by ``dataset/source/source_seed`` (the
  trainer cache intentionally reuses it across all target flows from that
  source) and is reused by every method;
* source seeds are the independent replication units;
* one fixed target stream seed is paired across all methods;
* EATA receives a diagonal Fisher computed from the *same* source checkpoint;
* every completed or failed job is atomically recorded, so an interrupted
  sweep can be resumed without silently dropping failures.

The script does not register methods in ``algorithms.get_tta_class`` and does
not modify DuSafe.  Use ``--device cpu --limit-jobs 1`` for a protocol smoke
run.  GPU sweeps acquire ``results/.current_experiment_gpu.lock`` and wait with
bounded backoff when another live experiment owns it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_baselines.fisher import ensure_source_fisher, sha256_file
from scripts.run_significance_test import (
    exact_paired_sign_flip,
    hierarchical_mean_ci,
    hierarchical_paired_ci,
    holm_adjust,
    paired_effect_dz,
    safe_wilcoxon,
)
from scripts.supplementary_utils import (
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
)


# The tuple is part of the protocol and is deliberately not inferred from the
# registry order.  NoAdap is handled by TTATrainer itself; the remaining names
# are resolved only through ``benchmark_baselines.registry``.
DEFAULT_METHODS = (
    "NoAdap",
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
    "DuSafe",
)
DEFAULT_DATASETS = ("EEG", "HAR", "FD")
DATASET_CHOICES = (*DEFAULT_DATASETS, "HHAR")
DEFAULT_SOURCE_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_STREAM_SEED = 42
METRICS = ("accuracy", "f1", "auroc")
JOB_KEY_COLUMNS = ("dataset", "scenario", "method", "source_seed", "stream_seed")
RAW_COLUMNS = (
    "status",
    "dataset",
    "scenario",
    "src_id",
    "trg_id",
    "method",
    "source_seed",
    "stream_seed",
    "run_signature",
    "accuracy",
    "f1",
    "auroc",
    "risk",
    "pre_final_accuracy",
    "pre_final_f1",
    "pre_final_auroc",
    "pre_final_risk",
    "source_model_sha256",
    "source_checkpoint_path",
    "source_checkpoint_file_sha256",
    "source_checkpoint_protocol",
    "fisher_enabled",
    "fisher_cache_path",
    "fisher_cache_hash",
    "fisher_samples",
    "fisher_batches",
    "fisher_source_checkpoint_sha256",
    "runtime_hparams",
    "source_hparams",
    "error_type",
    "is_oom",
    "error",
    "traceback",
)


def parse_csv_list(raw: str | Iterable[Any] | None, cast=str) -> list[Any]:
    """Parse comma-separated CLI values without accepting an empty item."""

    if raw is None:
        return []
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    result = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(cast(text))
    return result


def parse_override_entries(entries: Iterable[str] | None) -> dict[str, Any]:
    """Parse trainer runtime overrides using the same literal rules as CLI."""

    result: dict[str, Any] = {}
    for entry in entries or ():
        if "=" not in str(entry):
            raise ValueError(f"Invalid override '{entry}'; expected key=value")
        key, value = str(entry).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{entry}'; empty key")
        text = value.strip()
        lowered = text.lower()
        if lowered == "none":
            parsed: Any = None
        elif lowered == "true":
            parsed = True
        elif lowered == "false":
            parsed = False
        else:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text
        result[key] = parsed
    return result


def tensor_state_sha256(model: Any) -> str:
    """Hash model tensors in stable name order, excluding mutable metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def is_cuda_oom(error: BaseException) -> bool:
    """Return true for both native and wrapped CUDA OOM exceptions."""

    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(error).lower()


def _process_is_alive(pid: int) -> bool:
    """Return whether a lock-owner PID still exists without mutating it."""

    pid = int(pid)
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            try:
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                ):
                    # A terminated Windows process object can remain openable
                    # while another process still owns a handle to it.  Only
                    # STILL_ACTIVE identifies a live lock owner.
                    return exit_code.value == 259
                return True
            finally:
                kernel32.CloseHandle(handle)
        # Access denied means the process exists but is protected.
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GPUExperimentLockBusy(RuntimeError):
    """The shared GPU lock is currently owned by another live process."""


class GPUExperimentLockInvalid(RuntimeError):
    """The shared GPU lock exists but has no auditable live/dead owner."""


class GPUExperimentLock:
    """Exclusive, recoverable lock for long-running GPU experiments."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": str(Path.cwd()),
        }
        def open_exclusive():
            return os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )

        try:
            descriptor = open_exclusive()
        except FileExistsError as first_error:
            try:
                owner_text = self.path.read_text(encoding="utf-8")
                owner_payload = json.loads(owner_text)
                owner_pid = int(owner_payload.get("pid", -1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                owner_text = "unreadable or invalid lock owner"
                owner_pid = -1
            if owner_pid > 0 and not _process_is_alive(owner_pid):
                # Only a parseable, positively identified dead owner may be
                # reclaimed.  The second O_EXCL open protects against a race
                # with another process claiming the path after unlink.
                self.path.unlink(missing_ok=True)
                try:
                    descriptor = open_exclusive()
                except FileExistsError as second_error:
                    try:
                        owner_text = self.path.read_text(encoding="utf-8")
                    except OSError:
                        owner_text = "unreadable lock owner"
                    raise GPUExperimentLockBusy(
                        "GPU experiment lock was reclaimed concurrently at "
                        f"{self.path}: {owner_text}"
                    ) from second_error
            else:
                error_type = (
                    GPUExperimentLockBusy
                    if owner_pid > 0
                    else GPUExperimentLockInvalid
                )
                raise error_type(
                    "GPU experiment lock already exists at "
                    f"{self.path}: {owner_text}"
                ) from first_error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self._owned = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False
        return False


@contextmanager
def wait_for_gpu_experiment_lock(
    path: str | os.PathLike[str],
    *,
    initial_poll_seconds: float = 0.25,
    max_poll_seconds: float = 5.0,
    timeout_seconds: float = 0.0,
    on_wait: Any | None = None,
    _sleep: Any | None = None,
):
    """Acquire the shared GPU lock without charging contention as a run.

    ``GPUExperimentLock`` remains a fail-fast primitive.  Formal queues use
    this wrapper so another live queue causes bounded exponential backoff;
    only a launched experiment consumes a cell/group attempt.
    ``timeout_seconds=0`` means wait indefinitely.
    """

    initial = max(0.01, float(initial_poll_seconds))
    maximum = max(initial, float(max_poll_seconds))
    timeout = max(0.0, float(timeout_seconds))
    sleep = time.sleep if _sleep is None else _sleep
    started = time.monotonic()
    waits = 0
    while True:
        lock = GPUExperimentLock(path)
        try:
            lock.__enter__()
        except GPUExperimentLockBusy as exc:
            elapsed = time.monotonic() - started
            if timeout > 0.0 and elapsed >= timeout:
                raise TimeoutError(
                    f"timed out waiting for GPU experiment lock at {Path(path)}"
                ) from exc
            waits += 1
            delay = min(maximum, initial * (2 ** min(waits - 1, 8)))
            if on_wait is not None:
                on_wait(
                    {
                        "path": str(Path(path)),
                        "wait_count": waits,
                        "elapsed_seconds": elapsed,
                        "next_poll_seconds": delay,
                    }
                )
            sleep(delay)
            continue
        break
    try:
        yield lock
    finally:
        lock.__exit__(None, None, None)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _job_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["dataset"]),
        str(row["scenario"]),
        str(row["method"]),
        int(row["source_seed"]),
        int(row["stream_seed"]),
    )


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.to_dict("records")


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    for column in RAW_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    ordered = [column for column in RAW_COLUMNS if column in frame.columns]
    extra = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + extra]
    if not frame.empty:
        sort_columns = [column for column in JOB_KEY_COLUMNS if column in frame]
        frame = frame.sort_values(sort_columns, kind="stable")
    atomic_write_csv(frame, path)


def _error_row(
    *,
    dataset: str,
    scenario: str,
    src_id: str,
    trg_id: str,
    method: str,
    source_seed: int,
    stream_seed: int,
    error: BaseException,
    source_model_sha256: str = "",
    source_checkpoint_path: str = "",
    fisher_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "status": "failed",
        "dataset": dataset,
        "scenario": scenario,
        "src_id": str(src_id),
        "trg_id": str(trg_id),
        "method": method,
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "source_model_sha256": source_model_sha256,
        "source_checkpoint_path": source_checkpoint_path,
        "error_type": type(error).__name__,
        "is_oom": bool(is_cuda_oom(error)),
        "error": str(error),
        "traceback": traceback.format_exc(limit=20),
    }
    if fisher_info:
        row.update(fisher_info)
    return row


def run_job(
    args: argparse.Namespace,
    *,
    dataset: str,
    src_id: str,
    trg_id: str,
    method: str,
    source_seed: int,
    stream_seed: int,
) -> dict[str, Any]:
    """Run one paired source/method cell and always return an auditable row."""

    scenario = f"{src_id}->{trg_id}"
    trainer = None
    tta_model = None
    pre_trained_model = None
    source_model_sha256 = ""
    fisher_info: dict[str, Any] = {}
    source_checkpoint_path = ""
    try:
        trainer = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method=method,
            backbone=args.backbone,
            exp_name=f"full_main_table_{method}_src{source_seed}",
            seed=stream_seed,
            source_seed=source_seed,
            pretrain_cache_dir=args.pretrain_cache_dir,
            algorithm_registry="benchmark",
        )
        # The main table consumes stream predictions and aggregate metrics, not
        # target-label-dependent per-sample safety evidence. Safety runners
        # explicitly enable the full evidence contract for every method.
        trainer.set_runtime_hparams({"record_per_sample_evidence": False})
        if getattr(args, "batch_policy", "method_default") == "common":
            common_batch = int(args.common_batch_sizes[dataset])
            trainer.set_runtime_hparams({"batch_size": common_batch})
        if getattr(args, "overrides", None):
            trainer.set_runtime_hparams(dict(args.overrides))

        def pre_tta_hook(hook_trainer, hook_model):
            nonlocal source_model_sha256, source_checkpoint_path, fisher_info
            source_model_sha256 = tensor_state_sha256(hook_model)
            source_checkpoint_path = str(
                hook_trainer._pretrain_cache_path() or ""
            )
            if method != "EATA":
                return
            fisher_info = ensure_source_fisher(
                model=hook_model,
                source_loader=hook_trainer.src_train_dl,
                cache_dir=getattr(
                    args,
                    "eata_fisher_cache_dir",
                    ROOT / "results" / "eata_fisher_cache" / "full_main_table",
                ),
                dataset=dataset,
                source_seed=source_seed,
                source_checkpoint_sha256=source_model_sha256,
                samples=int(
                    hook_trainer.hparams.get(
                        "fisher_samples", getattr(args, "eata_fisher_samples", 2000)
                    )
                ),
                adapt_keywords=hook_trainer.hparams.get(
                    "adapt_keywords", ("classifier", "adapter")
                ),
            )
            # The adapter constructor requires an explicit source diagonal;
            # this mutation occurs before construction and is local to this
            # trainer instance, never to the production registry/config.
            hook_trainer.hparams["fisher_enabled"] = True
            hook_trainer.hparams["fisher_path"] = fisher_info["fisher_cache_path"]

        tta_model, pre_trained_model = create_tta_model(
            trainer,
            src_id,
            trg_id,
            run_seed=stream_seed,
            pre_tta_hook=pre_tta_hook,
        )
        source_checkpoint = trainer._pretrain_cache_path()
        source_checkpoint_path = str(source_checkpoint or "")
        source_file_hash = ""
        if source_checkpoint and Path(source_checkpoint).exists():
            source_file_hash = sha256_file(source_checkpoint)
        accuracy, macro_f1, auroc, risk = trainer.calculate_metrics(tta_model)
        prediction_summary = dict(
            getattr(trainer, "last_prediction_metric_summary", {}) or {}
        )
        row = {
            "status": "ok",
            "dataset": dataset,
            "scenario": scenario,
            "src_id": str(src_id),
            "trg_id": str(trg_id),
            "method": method,
            "source_seed": int(source_seed),
            "stream_seed": int(stream_seed),
            "run_signature": str(getattr(args, "run_signature", "")),
            "accuracy": float(accuracy),
            "f1": float(macro_f1),
            "auroc": float(auroc),
            "risk": float(risk),
            "pre_final_accuracy": float(
                prediction_summary.get("pre_final_update_accuracy", np.nan)
            ),
            "pre_final_f1": float(
                prediction_summary.get("pre_final_update_macro_f1", np.nan)
            ),
            "pre_final_auroc": float(
                prediction_summary.get("pre_final_update_auroc", np.nan)
            ),
            "pre_final_risk": float(
                prediction_summary.get("pre_final_update_risk", np.nan)
            ),
            "source_model_sha256": source_model_sha256,
            "source_checkpoint_path": source_checkpoint_path,
            "source_checkpoint_file_sha256": source_file_hash,
            "source_checkpoint_protocol": "pretrain_protocol_version=2",
            "runtime_hparams": _json_dumps(trainer.hparams),
            "source_hparams": _json_dumps(trainer.source_hparams),
        }
        if method == "EATA":
            row.update(fisher_info)
        else:
            row.update(
                {
                    "fisher_enabled": False,
                    "fisher_cache_path": "",
                    "fisher_cache_hash": "",
                    "fisher_samples": 0,
                    "fisher_batches": 0,
                    "fisher_source_checkpoint_sha256": "",
                }
            )
        safety = dict(getattr(trainer, "last_safety_summary", {}) or {})
        row.update({f"safety_{key}": value for key, value in safety.items()})
        return row
    except Exception as error:
        row = _error_row(
            dataset=dataset,
            scenario=scenario,
            src_id=src_id,
            trg_id=trg_id,
            method=method,
            source_seed=source_seed,
            stream_seed=stream_seed,
            error=error,
            source_model_sha256=source_model_sha256,
            source_checkpoint_path=source_checkpoint_path,
            fisher_info=fisher_info,
        )
        row["run_signature"] = str(getattr(args, "run_signature", ""))
        return row
    finally:
        if trainer is not None:
            cleanup_trainer(
                trainer,
                tta_model,
                pre_trained_model,
                close_summary=True,
            )


def _scenario_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = set()
    for item in parse_csv_list(raw):
        text = str(item).replace("_to_", "->").replace(" ", "")
        if "->" not in text:
            raise ValueError(f"Invalid scenario '{item}'; expected src->trg")
        src, trg = text.split("->", 1)
        if not src or not trg:
            raise ValueError(f"Invalid scenario '{item}'; expected src->trg")
        values.add(f"{src}->{trg}")
    return values


def collect(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """Execute missing jobs and atomically publish the raw per-cell table."""

    raw_path = output_dir / "per_source_seed_results.csv"
    rows = _load_rows(raw_path)
    completed = {}
    for row in rows:
        try:
            completed[_job_key(row)] = row
        except (KeyError, TypeError, ValueError):
            continue

    methods = list(args.methods)
    datasets = list(args.datasets)
    source_seeds = [int(value) for value in args.source_seeds]
    scenario_filter = _scenario_filter(args.scenarios)
    run_signature = str(getattr(args, "run_signature", ""))
    jobs_run = 0
    for dataset in datasets:
        probe = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method=methods[0],
            backbone=args.backbone,
            exp_name="full_main_table_probe",
            seed=args.stream_seed,
            source_seed=source_seeds[0],
            pretrain_cache_dir=args.pretrain_cache_dir,
            algorithm_registry="benchmark",
        )
        try:
            scenarios = dataset_scenarios(probe)
        finally:
            cleanup_trainer(probe, close_summary=True)
        for src_id, trg_id in scenarios:
            scenario = f"{src_id}->{trg_id}"
            if scenario_filter is not None and scenario not in scenario_filter:
                continue
            for source_seed in source_seeds:
                for method in methods:
                    key = (
                        dataset,
                        scenario,
                        method,
                        int(source_seed),
                        int(args.stream_seed),
                    )
                    old = completed.get(key)
                    if old is not None and (
                        str(old.get("run_signature", ""))
                        == run_signature
                        and (
                            str(old.get("status", "ok")) == "ok"
                            or not args.retry_failures
                        )
                    ):
                        continue
                    if args.limit_jobs is not None and jobs_run >= args.limit_jobs:
                        _write_rows(rows, raw_path)
                        return pd.DataFrame(rows)
                    print(
                        f"[MainTable] {dataset} {scenario} source={source_seed} {method}",
                        flush=True,
                    )
                    row = run_job(
                        args,
                        dataset=dataset,
                        src_id=src_id,
                        trg_id=trg_id,
                        method=method,
                        source_seed=int(source_seed),
                        stream_seed=int(args.stream_seed),
                    )
                    rows = [
                        existing
                        for existing in rows
                        if _job_key(existing) != key
                    ]
                    rows.append(row)
                    completed[key] = row
                    _write_rows(rows, raw_path)
                    jobs_run += 1
    _write_rows(rows, raw_path)
    return pd.DataFrame(rows)


def _metric_ci(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    try:
        return hierarchical_mean_ci(frame, value=metric)
    except (KeyError, ValueError):
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            return float("nan"), float("nan")
        return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def analyze(raw: pd.DataFrame, output_dir: Path, reference_method: str = "DuSafe") -> None:
    """Write aggregate main tables and source-seed paired comparisons."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if "status" in raw.columns:
        status = raw["status"].astype(str)
    else:
        status = pd.Series("ok", index=raw.index, dtype="object")
    ok = raw[status.eq("ok")].copy()
    for metric in METRICS:
        if metric in ok:
            ok[metric] = pd.to_numeric(ok[metric], errors="coerce")
    summary_rows: list[dict[str, Any]] = []
    if not ok.empty:
        for (dataset, method), group in ok.groupby(["dataset", "method"], sort=True):
            row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "n_successful_cells": int(len(group)),
                "n_source_seeds": int(group["source_seed"].nunique()),
                "n_scenarios": int(group["scenario"].nunique()),
            }
            per_seed = group.groupby("source_seed", as_index=False)[list(METRICS)].mean()
            for metric in METRICS:
                values = per_seed[metric].to_numpy(dtype=float)
                ci_low, ci_high = _metric_ci(group, metric)
                row[f"{metric}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
                row[f"{metric}_std"] = (
                    float(np.nanstd(values, ddof=1)) if values.size > 1 else float("nan")
                )
                row[f"{metric}_ci_low"] = ci_low
                row[f"{metric}_ci_high"] = ci_high
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "main_table.csv", index=False)

    # Preserve the scenario/source means used for the paired tests.  This is
    # useful when a partial/resumed run has one failed cell.
    if not ok.empty:
        scenario_summary = (
            ok.groupby(["dataset", "scenario", "method", "source_seed"], as_index=False)[
                list(METRICS)
            ]
            .mean()
        )
    else:
        scenario_summary = pd.DataFrame()
    scenario_summary.to_csv(output_dir / "scenario_source_means.csv", index=False)

    if not ok.empty:
        per_source_summary = (
            ok.groupby(["dataset", "method", "source_seed"], as_index=False)[
                list(METRICS)
            ]
            .mean()
        )
    else:
        per_source_summary = pd.DataFrame()
    per_source_summary.to_csv(
        output_dir / "per_source_seed_summary.csv", index=False
    )

    consistency_rows: list[dict[str, Any]] = []
    if not ok.empty and "source_model_sha256" in ok.columns:
        for (dataset, src_id, source_seed), group in ok.groupby(
            ["dataset", "src_id", "source_seed"], sort=True
        ):
            hashes = sorted(
                {
                    str(value)
                    for value in group["source_model_sha256"].dropna()
                    if str(value)
                }
            )
            paths = sorted(
                {
                    str(value)
                    for value in group.get("source_checkpoint_path", pd.Series(dtype=str)).dropna()
                    if str(value)
                }
            )
            consistency_rows.append(
                {
                    "dataset": dataset,
                    "src_id": str(src_id),
                    "source_seed": int(source_seed),
                    "n_rows": int(len(group)),
                    "n_source_model_hashes": int(len(hashes)),
                    "source_model_sha256": ";".join(hashes),
                    "n_checkpoint_paths": int(len(paths)),
                    "source_checkpoint_paths": ";".join(paths),
                    "consistent": bool(len(hashes) <= 1 and len(paths) <= 1),
                }
            )
    pd.DataFrame(consistency_rows).to_csv(
        output_dir / "checkpoint_consistency.csv", index=False
    )

    comparison_rows: list[dict[str, Any]] = []
    if not scenario_summary.empty and reference_method in set(scenario_summary["method"]):
        for dataset, dataset_frame in scenario_summary.groupby("dataset", sort=True):
            reference = dataset_frame[dataset_frame["method"] == reference_method]
            for method in sorted(set(dataset_frame["method"]) - {reference_method}):
                baseline = dataset_frame[dataset_frame["method"] == method]
                for metric in METRICS:
                    paired = reference.merge(
                        baseline,
                        on=["dataset", "scenario", "source_seed"],
                        suffixes=("_reference", "_baseline"),
                        validate="one_to_one",
                    )
                    if paired.empty:
                        continue
                    difference_column = f"{metric}_reference"
                    baseline_column = f"{metric}_baseline"
                    paired = paired.assign(
                        _difference=paired[difference_column] - paired[baseline_column]
                    )
                    seed_level = paired.groupby("source_seed", as_index=False)["_difference"].mean()
                    differences = seed_level["_difference"].to_numpy(dtype=float)
                    ci_low, ci_high = hierarchical_paired_ci(
                        paired,
                        difference_column,
                        baseline_column,
                    )
                    comparison_rows.append(
                        {
                            "dataset": dataset,
                            "reference": reference_method,
                            "baseline": method,
                            "metric": metric,
                            "mean_paired_difference": float(np.nanmean(differences)),
                            "hierarchical_ci_low": ci_low,
                            "hierarchical_ci_high": ci_high,
                            "exact_sign_flip_p": exact_paired_sign_flip(differences),
                            "wilcoxon_p": safe_wilcoxon(differences),
                            "paired_effect_dz": paired_effect_dz(differences),
                            "n_independent_source_seeds": int(len(differences)),
                            "n_paired_scenario_seed_cells": int(len(paired)),
                        }
                    )
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["holm_exact_p"] = np.nan
        comparisons["holm_wilcoxon_p"] = np.nan
        for (_, metric), indices in comparisons.groupby(["dataset", "metric"]).groups.items():
            index_list = list(indices)
            comparisons.loc[index_list, "holm_exact_p"] = holm_adjust(
                comparisons.loc[index_list, "exact_sign_flip_p"].to_numpy()
            )
            comparisons.loc[index_list, "holm_wilcoxon_p"] = holm_adjust(
                comparisons.loc[index_list, "wilcoxon_p"].to_numpy()
            )
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)


def write_manifest(args: argparse.Namespace, output_dir: Path, raw: pd.DataFrame) -> None:
    manifest = {
        "protocol_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "registry": "benchmark",
        "datasets": list(args.datasets),
        "methods": list(args.methods),
        "source_seeds": [int(value) for value in args.source_seeds],
        "stream_seed": int(args.stream_seed),
        "source_seed_is_independent_unit": True,
        "stream_seed_is_paired_control": True,
        "source_training": "one trainer pre-training cache per dataset/source/source_seed, shared across all target flows and methods",
        "source_checkpoint_hash": "sha256 over sorted model state tensors",
        "eata_fisher": "source-only diagonal Fisher from the same source checkpoint; no L2 fallback",
        "batch_policy": args.batch_policy,
        "common_batch_sizes": dict(args.common_batch_sizes),
        "runtime_overrides": dict(args.overrides),
        "run_signature": str(getattr(args, "run_signature", "")),
        "retry_failures": bool(args.retry_failures),
        "analysis_only": bool(getattr(args, "analyze_only", False)),
        "limit_jobs": args.limit_jobs,
        "raw_rows": int(len(raw)),
        "successful_rows": int((raw.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum()) if not raw.empty else 0,
        "failed_rows": int((raw.get("status", pd.Series(dtype=str)).astype(str) == "failed").sum()) if not raw.empty else 0,
        "gpu_lock": str(ROOT / "results" / ".current_experiment_gpu.lock"),
        "gpu_lock_scope": "runner_invocation",
        "gpu_lock_wait_policy": "bounded_exponential_backoff",
        "gpu_lock_busy_consumes_attempt": False,
        "outputs": {
            "raw": "per_source_seed_results.csv",
            "main_table": "main_table.csv",
            "scenario_source_means": "scenario_source_means.csv",
            "per_source_seed_summary": "per_source_seed_summary.csv",
            "checkpoint_consistency": "checkpoint_consistency.csv",
            "paired_comparisons": "paired_comparisons.csv",
        },
    }
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")


def _git_head() -> str:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--source-seeds",
        "--source_seeds",
        dest="source_seeds",
        default=",".join(str(value) for value in DEFAULT_SOURCE_SEEDS),
    )
    parser.add_argument("--stream-seed", "--stream_seed", dest="stream_seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument("--scenarios", default=None, help="Optional comma-separated src->trg filters.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(ROOT / "results" / "tta_experiments_logs" / "full_main_table"))
    parser.add_argument("--pretrain-cache-dir", "--pretrain_cache_dir", dest="pretrain_cache_dir", default=str(ROOT / "results" / "pretrain_cache" / "full_main_table"))
    parser.add_argument("--eata-fisher-cache-dir", "--eata_fisher_cache_dir", dest="eata_fisher_cache_dir", default=str(ROOT / "results" / "eata_fisher_cache" / "full_main_table"))
    parser.add_argument("--eata-fisher-samples", "--eata_fisher_samples", dest="eata_fisher_samples", type=int, default=2000)
    parser.add_argument("--batch-policy", choices=("method_default", "common"), default="method_default")
    parser.add_argument(
        "--common-batch-sizes",
        default="EEG=96,HAR=16,FD=128,HHAR=48",
    )
    parser.add_argument("--override", action="append", default=None, help="Runtime key=value override applied to every method.")
    parser.add_argument(
        "--run-signature",
        default="",
        help=(
            "Optional protocol/profile signature. Existing cells with a "
            "different signature are recomputed instead of resumed."
        ),
    )
    parser.add_argument("--limit-jobs", "--limit_jobs", dest="limit_jobs", type=int, default=None)
    parser.add_argument("--retry-failures", "--retry_failures", dest="retry_failures", action="store_true")
    parser.add_argument("--reference-method", default="DuSafe")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help=(
            "Rebuild aggregate/statistical artifacts from the existing "
            "per_source_seed_results.csv without launching GPU jobs."
        ),
    )
    return parser


def _parse_common_batch_sizes(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in parse_csv_list(raw):
        if "=" not in item:
            raise ValueError(f"Invalid common batch entry '{item}'; expected DATASET=INTEGER")
        dataset, value = item.split("=", 1)
        result[dataset.strip()] = int(value)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.datasets = parse_csv_list(args.datasets)
    args.methods = parse_csv_list(args.methods)
    args.source_seeds = parse_csv_list(args.source_seeds, int)
    if not args.datasets or not args.methods or not args.source_seeds:
        parser.error("datasets, methods, and source_seeds must be non-empty")
    unknown_datasets = set(args.datasets) - set(DATASET_CHOICES)
    if unknown_datasets:
        parser.error(f"Unknown datasets: {sorted(unknown_datasets)}")
    unknown_methods = set(args.methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        parser.error(f"Unknown methods: {sorted(unknown_methods)}")
    args.overrides = parse_override_entries(args.override)
    args.common_batch_sizes = _parse_common_batch_sizes(args.common_batch_sizes)
    for dataset in args.datasets:
        if args.batch_policy == "common" and dataset not in args.common_batch_sizes:
            parser.error(f"Missing common batch size for {dataset}")
    args.pretrain_cache_dir = str(ensure_dir(args.pretrain_cache_dir))
    args.eata_fisher_cache_dir = str(ensure_dir(args.eata_fisher_cache_dir))
    output_dir = ensure_dir(args.output_dir)
    if args.analyze_only:
        raw = pd.DataFrame(
            _load_rows(output_dir / "per_source_seed_results.csv")
        )
        analyze(raw, output_dir, reference_method=args.reference_method)
        write_manifest(args, output_dir, raw)
        print(f"Results: {output_dir}")
        return 0
    lock_path = ROOT / "results" / ".current_experiment_gpu.lock"
    lock_context = (
        wait_for_gpu_experiment_lock(lock_path)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )
    with lock_context:
        raw = collect(args, output_dir)
        analyze(raw, output_dir, reference_method=args.reference_method)
        write_manifest(args, output_dir, raw)
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
