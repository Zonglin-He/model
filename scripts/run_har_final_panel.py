"""Run the frozen HAR Full-vs-no-SSAW transfer safety panel.

The unit of resumability is one ``flow x condition x source_seed`` cell.  A
cell is evaluated by one child process of
``run_controlled_safety_benchmark.py``; that child receives both DuSafe
variants so the variants share the same source checkpoint, stream seed, and
corruption mask.  This runner contains no tuning logic and never reads target
labels itself.
"""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.har_frozen_profile import (  # noqa: E402
    FROZEN_HAR_TTA_PARAMS,
    PROFILE_ID as FROZEN_HAR_PROFILE_ID,
    validate_frozen_har_profile,
)
from scripts.run_controlled_safety_benchmark import REQUIRED_SAFETY_METRICS
from scripts.run_optuna_stepwise import scenario_pairs
from scripts.supplementary_utils import atomic_write_csv, ensure_dir


PROTOCOL = "HAR final transfer safety Full-vs-no-SSAW v2 frozen profile"
DATASET = "HAR"
VARIANTS = ("full", "no_ssaw")
CONDITIONS = ("clean", "signal_freeze_moderate")
CONDITION_FRACTIONS = {
    "clean": 0.0,
    "signal_freeze_moderate": 0.5,
}
CONDITION_CORRUPTION = {
    "clean": "signal_freeze",
    "signal_freeze_moderate": "signal_freeze",
}
CONDITION_SEVERITY = {
    "clean": "moderate",
    "signal_freeze_moderate": "moderate",
}
DEFAULT_SOURCE_SEEDS = (1, 2, 3)
DEFAULT_STREAM_SEED = 42
DEFAULT_CORRUPTION_SEED = 1
FLOWS = tuple(f"{source}->{target}" for source, target in scenario_pairs(DATASET))
SUMMARY_COLUMNS = (
    "dataset",
    "scenario",
    "method",
    "variant",
    "corruption",
    "severity",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    "f1",
    *REQUIRED_SAFETY_METRICS,
)
STATUS_COLUMNS = (
    "flow",
    "condition",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    "resume_key",
    "output_dir",
    "status",
    "returncode",
    "resumed",
    "error",
    "recorded_at_unix",
)
RESUME_KEY_FIELDS = (
    "flow",
    "condition",
    "source_seed",
    "stream_seed",
    "corruption_seed",
)


def acquire_gpu_lock(path: Path):
    """Hold the repository-wide GPU lock for the runner lifetime."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"GPU lock is busy: {path}") from exc
    return handle


def parse_source_seeds(text: str) -> tuple[int, ...]:
    """Parse and validate the independent source checkpoint seed list."""

    values = tuple(
        int(value.strip()) for value in str(text).split(",") if value.strip()
    )
    if not values:
        raise ValueError("--source_seeds must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("--source_seeds must contain unique seeds")
    return values


def cell_key(
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
) -> tuple[str, str, int, int, int]:
    """Return the immutable key used for status and resume decisions."""

    return (
        str(flow),
        str(condition),
        int(source_seed),
        int(stream_seed),
        int(corruption_seed),
    )


def resume_key(
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
) -> str:
    """Return a human-readable, stable serialization of :func:`cell_key`."""

    key = cell_key(flow, condition, source_seed, stream_seed, corruption_seed)
    return "|".join(
        f"{field}={value}" for field, value in zip(RESUME_KEY_FIELDS, key)
    )


def cell_dir(output_dir: Path, flow: str, condition: str, source_seed: int) -> Path:
    source, target = str(flow).split("->", 1)
    return (
        Path(output_dir)
        / f"flow_{source}_to_{target}"
        / str(condition)
        / f"source_seed_{int(source_seed)}"
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically publish JSON so an interrupted run is never resumable by it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_int(value, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {field}: {value!r}") from exc


def _metadata_matches(
    metadata: dict,
    *,
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
    runtime_hparam_overrides: dict | None = None,
) -> bool:
    expected = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "flow": str(flow),
        "condition": str(condition),
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
        "corruption": CONDITION_CORRUPTION[condition],
        "severity": CONDITION_SEVERITY[condition],
        "corruption_fraction": float(CONDITION_FRACTIONS[condition]),
        "variants": list(VARIANTS),
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_tuning": True,
        "target_labels_used_for_selection": True,
        "frozen_har_profile_id": FROZEN_HAR_PROFILE_ID,
        "frozen_har_tta_hparams": dict(FROZEN_HAR_TTA_PARAMS),
    }
    for key, expected_value in expected.items():
        observed = metadata.get(key)
        if isinstance(expected_value, float):
            try:
                if abs(float(observed) - expected_value) > 1e-12:
                    return False
            except (TypeError, ValueError):
                return False
        elif observed != expected_value:
            return False
    if metadata.get("runtime_hparam_overrides", {}) != dict(
        runtime_hparam_overrides or {}
    ):
        return False
    return metadata.get("resume_key") == resume_key(
        flow, condition, source_seed, stream_seed, corruption_seed
    )


def summary_matches(
    cell_output_dir: Path,
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
    runtime_hparam_overrides: dict | None = None,
) -> bool:
    """Return true only for a complete, key-matching two-variant cell."""

    cell_output_dir = Path(cell_output_dir)
    metadata_path = cell_output_dir / "cell_metadata.json"
    summary_path = cell_output_dir / "summary_raw.csv"
    if not metadata_path.exists() or not summary_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _metadata_matches(
            metadata,
            flow=flow,
            condition=condition,
            source_seed=source_seed,
            stream_seed=stream_seed,
            corruption_seed=corruption_seed,
            runtime_hparam_overrides=runtime_hparam_overrides,
        ):
            return False
        frame = pd.read_csv(summary_path)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    required = set(SUMMARY_COLUMNS)
    if len(frame) != len(VARIANTS) or not required.issubset(frame.columns):
        return False
    if set(frame["variant"].astype(str)) != set(VARIANTS):
        return False
    if frame.duplicated(
        ["dataset", "scenario", "method", "variant", "corruption", "severity",
         "source_seed", "stream_seed", "corruption_seed"]
    ).any():
        return False
    expected = {
        "dataset": DATASET,
        "scenario": str(flow),
        "method": "DuSafe",
        "corruption": CONDITION_CORRUPTION[condition],
        "severity": CONDITION_SEVERITY[condition],
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
    }
    for column, expected_value in expected.items():
        if column not in frame:
            return False
        if column in {"source_seed", "stream_seed", "corruption_seed"}:
            try:
                if not frame[column].map(lambda value: int(value) == expected_value).all():
                    return False
            except (TypeError, ValueError):
                return False
        elif not frame[column].astype(str).eq(str(expected_value)).all():
            return False
    for metric in ("f1", *REQUIRED_SAFETY_METRICS):
        values = pd.to_numeric(frame[metric], errors="coerce")
        if metric == "corruption_rejection_recall" and condition == "clean":
            # There are no corrupted samples in the clean condition, so the
            # recall denominator is zero and NaN is the only valid value.
            if not values.isna().all():
                return False
        elif values.isna().any():
            return False
    return True


def load_status(status_path: Path) -> dict[tuple, dict]:
    """Load status rows keyed by the immutable cell key.

    Malformed status files are treated as empty by the runner.  The strict
    aggregator does not use this leniency and rejects malformed or incomplete
    status files.
    """

    path = Path(status_path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    if not set(RESUME_KEY_FIELDS).issubset(frame.columns):
        return {}
    rows = {}
    for row in frame.to_dict("records"):
        try:
            key = cell_key(
                row["flow"],
                row["condition"],
                _as_int(row["source_seed"], "source_seed"),
                _as_int(row["stream_seed"], "stream_seed"),
                _as_int(row["corruption_seed"], "corruption_seed"),
            )
        except (KeyError, TypeError, ValueError):
            continue
        rows[key] = row
    return rows


def write_status(status_path: Path, status_by_key: dict[tuple, dict]) -> None:
    rows = []
    for key, row in status_by_key.items():
        normalized = dict(row)
        normalized.setdefault("flow", key[0])
        normalized.setdefault("condition", key[1])
        normalized.setdefault("source_seed", key[2])
        normalized.setdefault("stream_seed", key[3])
        normalized.setdefault("corruption_seed", key[4])
        normalized.setdefault("resume_key", resume_key(*key))
        rows.append(normalized)
    frame = pd.DataFrame(rows, columns=STATUS_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["flow", "condition", "source_seed"]
        ).reset_index(drop=True)
    atomic_write_csv(frame, status_path, index=False)


def _expected_keys(source_seeds, stream_seed, corruption_seed):
    return [
        cell_key(flow, condition, source_seed, stream_seed, corruption_seed)
        for flow in FLOWS
        for condition in CONDITIONS
        for source_seed in source_seeds
    ]


def build_command(args, *, flow: str, condition: str, source_seed: int, output_dir: Path) -> list[str]:
    """Build exactly one controlled-benchmark child invocation per cell."""

    runner = Path(args.controlled_script)
    command = [
        sys.executable,
        str(runner),
        "--data_path",
        str(args.data_path),
        "--device",
        str(args.device),
        "--backbone",
        str(args.backbone),
        "--registry",
        str(args.registry),
        "--datasets",
        DATASET,
        "--methods",
        "DuSafe",
        "--variants",
        ",".join(VARIANTS),
        "--scenarios",
        f"{DATASET}:{flow}",
        "--corruptions",
        CONDITION_CORRUPTION[condition],
        "--severities",
        CONDITION_SEVERITY[condition],
        "--source_seeds",
        str(int(source_seed)),
        "--stream_seeds",
        str(int(args.stream_seed)),
        "--corruption_fraction",
        str(float(CONDITION_FRACTIONS[condition])),
        "--corruption_seed",
        str(int(args.corruption_seed)),
        "--output_dir",
        str(output_dir),
        "--pretrain_cache_dir",
        str(args.pretrain_cache_dir),
    ]
    auxiliary_weight = getattr(args, "ssaw_auxiliary_weight", None)
    if auxiliary_weight is not None:
        if auxiliary_weight != FROZEN_HAR_TTA_PARAMS["ssaw_auxiliary_weight"]:
            raise ValueError(
                "The formal HAR panel only accepts the frozen auxiliary "
                f"weight {FROZEN_HAR_TTA_PARAMS['ssaw_auxiliary_weight']}"
            )
        command.extend(
            ["--override", f"ssaw_auxiliary_weight={float(auxiliary_weight):g}"]
        )
    return command


def _runtime_hparam_overrides(args) -> dict:
    auxiliary_weight = getattr(args, "ssaw_auxiliary_weight", None)
    if auxiliary_weight is None:
        return {}
    return {"ssaw_auxiliary_weight": float(auxiliary_weight)}


def _base_manifest(args, output_dir: Path, source_seeds: tuple[int, ...]) -> dict:
    expected_jobs = len(FLOWS) * len(CONDITIONS) * len(source_seeds)
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "flows": list(FLOWS),
        "variants": list(VARIANTS),
        "conditions": list(CONDITIONS),
        "condition_fractions": dict(CONDITION_FRACTIONS),
        "condition_corruption": dict(CONDITION_CORRUPTION),
        "condition_severity": dict(CONDITION_SEVERITY),
        "source_seeds": [int(value) for value in source_seeds],
        "source_seed_is_independent_unit": True,
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "corruption_seed": int(args.corruption_seed),
        "expected_job_count": expected_jobs,
        "job_count_expected": expected_jobs,
        "expected_row_count": expected_jobs * len(VARIANTS),
        "row_count_expected": expected_jobs * len(VARIANTS),
        "cell_count_expected": expected_jobs,
        "cell_grain": (
            "one subprocess and one summary_raw.csv per flow, condition, "
            "and independent source_seed; both variants are in that summary"
        ),
        "resume_key_fields": list(RESUME_KEY_FIELDS),
        "status_path": str(Path(output_dir) / "cell_status.csv"),
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_tuning": True,
        "target_labels_used_for_selection": True,
        "target_labels_used_online": False,
        "controlled_benchmark_script": str(Path(args.controlled_script)),
        "required_safety_metrics": list(REQUIRED_SAFETY_METRICS),
        "runtime_hparam_overrides": _runtime_hparam_overrides(args),
        "frozen_har_profile_id": FROZEN_HAR_PROFILE_ID,
        "frozen_har_tta_hparams": dict(FROZEN_HAR_TTA_PARAMS),
    }


def _metadata(
    *,
    flow: str,
    condition: str,
    source_seed: int,
    stream_seed: int,
    corruption_seed: int,
    runtime_hparam_overrides: dict | None = None,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "flow": str(flow),
        "condition": str(condition),
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
        "corruption": CONDITION_CORRUPTION[condition],
        "severity": CONDITION_SEVERITY[condition],
        "corruption_fraction": float(CONDITION_FRACTIONS[condition]),
        "variants": list(VARIANTS),
        "resume_key": resume_key(
            flow, condition, source_seed, stream_seed, corruption_seed
        ),
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_tuning": True,
        "target_labels_used_for_selection": True,
        "target_labels_used_online": False,
        "runtime_hparam_overrides": dict(runtime_hparam_overrides or {}),
        "frozen_har_profile_id": FROZEN_HAR_PROFILE_ID,
        "frozen_har_tta_hparams": dict(FROZEN_HAR_TTA_PARAMS),
    }


def _write_manifest(
    manifest: dict,
    path: Path,
    status_by_key: dict[tuple, dict],
    *,
    status: str,
    error_count: int = 0,
) -> None:
    completed = sum(row.get("status") == "completed" for row in status_by_key.values())
    failed = sum(row.get("status") == "failed" for row in status_by_key.values())
    running = sum(row.get("status") == "running" for row in status_by_key.values())
    pending = sum(row.get("status") == "pending" for row in status_by_key.values())
    updated = {
        **manifest,
        "status": status,
        "completed_jobs": int(completed),
        "failed_jobs": int(failed),
        "running_jobs": int(running),
        "pending_jobs": int(pending),
        "error_count": int(error_count),
        "updated_at_unix": time.time(),
    }
    atomic_write_json(path, updated)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--data_path",
        "--data-path",
        dest="data_path",
        default=str(ROOT / "data" / "Dataset"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--registry", choices=("production", "benchmark"), default="production")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default=str(ROOT / "results" / "diagnostics" / "har_final_panel_v1"),
    )
    parser.add_argument(
        "--pretrain_cache_dir",
        "--pretrain-cache-dir",
        dest="pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument(
        "--controlled_script",
        "--controlled-script",
        dest="controlled_script",
        default=str(ROOT / "scripts" / "run_controlled_safety_benchmark.py"),
    )
    parser.add_argument("--source_seeds", "--source-seeds", dest="source_seeds", default="1,2,3")
    parser.add_argument("--stream_seed", "--stream-seed", dest="stream_seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument(
        "--corruption_seed",
        "--corruption-seed",
        dest="corruption_seed",
        type=int,
        default=DEFAULT_CORRUPTION_SEED,
    )
    parser.add_argument(
        "--max_jobs",
        "--max-jobs",
        dest="max_jobs",
        type=int,
        default=None,
        help="Run at most this many newly executed cells, preserving resume state.",
    )
    parser.add_argument(
        "--ssaw_auxiliary_weight",
        "--ssaw-auxiliary-weight",
        dest="ssaw_auxiliary_weight",
        type=float,
        default=None,
        help=(
            "Optional recorded runtime override for an exploratory paired "
            "SSAW-weight panel."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_frozen_har_profile()
    source_seeds = parse_source_seeds(args.source_seeds)
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("--max_jobs must be positive")
    if (
        args.ssaw_auxiliary_weight is not None
        and args.ssaw_auxiliary_weight < 0.0
    ):
        raise ValueError("--ssaw_auxiliary_weight must be non-negative")
    if (
        args.ssaw_auxiliary_weight is not None
        and args.ssaw_auxiliary_weight
        != FROZEN_HAR_TTA_PARAMS["ssaw_auxiliary_weight"]
    ):
        raise ValueError(
            "The formal HAR panel is frozen at "
            f"ssaw_auxiliary_weight={FROZEN_HAR_TTA_PARAMS['ssaw_auxiliary_weight']}; "
            "use a separate exploratory runner for other values"
        )
    if len(FLOWS) != 5:
        raise RuntimeError(f"HAR protocol requires five scenario_pairs, found {FLOWS}")

    # Keep the handle alive until main returns. Windows releases the byte lock
    # when the handle is closed during normal return or exception unwinding.
    gpu_lock_handle = acquire_gpu_lock(
        ROOT / "results" / ".current_experiment_gpu.lock"
    )

    output_dir = ensure_dir(args.output_dir)
    output_dir = Path(output_dir)
    status_path = output_dir / "cell_status.csv"
    manifest_path = output_dir / "manifest.json"
    manifest = _base_manifest(args, output_dir, source_seeds)
    runtime_hparam_overrides = _runtime_hparam_overrides(args)
    expected_keys = _expected_keys(source_seeds, args.stream_seed, args.corruption_seed)
    loaded = load_status(status_path)
    status_by_key = {}
    for key in expected_keys:
        previous = dict(loaded.get(key, {}))
        status_by_key[key] = {
            "flow": key[0],
            "condition": key[1],
            "source_seed": key[2],
            "stream_seed": key[3],
            "corruption_seed": key[4],
            "resume_key": resume_key(*key),
            "output_dir": str(cell_dir(output_dir, key[0], key[1], key[2])),
            "status": previous.get("status", "pending"),
            "returncode": previous.get("returncode", None),
            "resumed": previous.get("resumed", False),
            "error": previous.get("error", ""),
            "recorded_at_unix": previous.get("recorded_at_unix", None),
        }
    write_status(status_path, status_by_key)
    _write_manifest(manifest, manifest_path, status_by_key, status="running")

    executed = 0
    errors = 0
    stopped_early = False
    for key in expected_keys:
        flow, condition, source_seed, stream_seed, corruption_seed = key
        output_cell_dir = cell_dir(output_dir, flow, condition, source_seed)
        row = status_by_key[key]
        if summary_matches(
            output_cell_dir,
            flow,
            condition,
            source_seed,
            stream_seed,
            corruption_seed,
            runtime_hparam_overrides=runtime_hparam_overrides,
        ):
            row.update(
                {
                    "status": "completed",
                    "returncode": 0,
                    "resumed": True,
                    "error": "",
                    "recorded_at_unix": time.time(),
                }
            )
            write_status(status_path, status_by_key)
            _write_manifest(manifest, manifest_path, status_by_key, status="running", error_count=errors)
            continue
        if args.max_jobs is not None and executed >= args.max_jobs:
            stopped_early = True
            break

        output_cell_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            args,
            flow=flow,
            condition=condition,
            source_seed=source_seed,
            output_dir=output_cell_dir,
        )
        row.update(
            {
                "status": "running",
                "returncode": None,
                "resumed": False,
                "error": "",
                "recorded_at_unix": time.time(),
            }
        )
        write_status(status_path, status_by_key)
        _write_manifest(manifest, manifest_path, status_by_key, status="running", error_count=errors)
        print(
            f"[HAR final panel] flow={flow} condition={condition} "
            f"source_seed={source_seed} stream_seed={stream_seed}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        executed += 1
        returncode = int(completed.returncode)
        error = ""
        if returncode == 0:
            atomic_write_json(
                output_cell_dir / "cell_metadata.json",
                _metadata(
                    flow=flow,
                    condition=condition,
                    source_seed=source_seed,
                    stream_seed=stream_seed,
                    corruption_seed=corruption_seed,
                    runtime_hparam_overrides=runtime_hparam_overrides,
                ),
            )
            valid = summary_matches(
                output_cell_dir,
                flow,
                condition,
                source_seed,
                stream_seed,
                corruption_seed,
                runtime_hparam_overrides=runtime_hparam_overrides,
            )
            if not valid:
                error = "child exited zero but did not publish a valid two-variant summary"
        else:
            error = f"controlled benchmark child exited with returncode {returncode}"
        if error:
            errors += 1
        row.update(
            {
                "status": "completed" if not error else "failed",
                "returncode": returncode,
                "resumed": False,
                "error": error,
                "recorded_at_unix": time.time(),
            }
        )
        write_status(status_path, status_by_key)
        _write_manifest(manifest, manifest_path, status_by_key, status="running", error_count=errors)

    all_completed = all(row.get("status") == "completed" for row in status_by_key.values())
    final_status = "complete" if all_completed else ("partial" if stopped_early else "failed")
    _write_manifest(manifest, manifest_path, status_by_key, status=final_status, error_count=errors)
    print(f"HAR final panel {final_status}: {output_dir}", flush=True)
    gpu_lock_handle.close()
    return 0 if final_status in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
