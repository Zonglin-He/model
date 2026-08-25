"""Run the preregistered FD source-only confidence-gate calibration grid.

The calibration unit is one source checkpoint and one fixed stream.  Every
cell evaluates a held-out same-domain source test stream, so the transfer
flows are not part of the selection data.  The runner delegates the actual
benchmark execution to :mod:`run_controlled_safety_benchmark` and only
publishes a cell after both its summary and its metadata are complete.

The protocol is intentionally fixed in code:

* source domains ``0, 1, 2, 3``;
* independent source checkpoint seeds ``1, 2, 3``;
* paired stream seed ``42`` and corruption seed ``1``;
* clean and deterministic 50% moderate ``signal_freeze`` conditions; and
* preregistered keep fractions ``.95, .975, .99, 1.0``.

The default device is CPU.  This module is safe to import in tests without
constructing a trainer or touching CUDA.
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


PROTOCOL = "FD source-only confidence keep-fraction calibration v2"
SOURCE_DOMAINS = (0, 1, 2, 3)
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
CORRUPTION_SEED = 1
CORRUPTION_FRACTION = 0.5
CONDITIONS = ("clean", "signal_freeze_moderate")
CONDITION_FRACTIONS = {
    "clean": 0.0,
    "signal_freeze_moderate": 0.5,
}

# Keep the labels human-readable and filesystem-safe.  Do not derive these
# from floating-point formatting: q=.975 must remain q0975, not q0.975.
CANDIDATE_KEEP_FRACTIONS = (0.95, 0.975, 0.99, 1.0)
CANDIDATE_LABELS = ("095", "0975", "099", "100")
CANDIDATE_LABEL_TO_VALUE = dict(zip(CANDIDATE_LABELS, CANDIDATE_KEEP_FRACTIONS))
CANDIDATE_VALUE_TO_LABEL = {
    value: label for label, value in CANDIDATE_LABEL_TO_VALUE.items()
}


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

SUMMARY_REQUIRED_COLUMNS = (
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
    "clean_correct_false_rejection_rate",
)


def _float_equal(left, right, tolerance=1e-12):
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def candidate_label(value) -> str:
    """Return the preregistered filesystem label for a keep fraction."""
    if isinstance(value, str) and value in CANDIDATE_LABEL_TO_VALUE:
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown confidence keep fraction: {value!r}") from exc
    for registered, label in CANDIDATE_VALUE_TO_LABEL.items():
        if _float_equal(numeric, registered):
            return label
    raise ValueError(
        "Candidate confidence keep fractions are preregistered as "
        f"{CANDIDATE_KEEP_FRACTIONS}; got {value!r}"
    )


def parse_candidates(text: str) -> tuple[float, ...]:
    """Parse and validate the complete preregistered candidate list."""
    values = tuple(
        float(item.strip())
        for item in str(text).split(",")
        if item.strip()
    )
    if values != CANDIDATE_KEEP_FRACTIONS:
        raise ValueError(
            "--candidates must exactly match the preregistered ordered list "
            f"{CANDIDATE_KEEP_FRACTIONS}; got {values}"
        )
    return values


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")
    return values


def _validate_fixed_protocol(
    source_domains, source_seeds, stream_seed, corruption_seed, corruption_fraction
):
    if tuple(source_domains) != SOURCE_DOMAINS:
        raise ValueError(
            f"source domains are fixed at {SOURCE_DOMAINS}; got {tuple(source_domains)}"
        )
    if tuple(source_seeds) != SOURCE_SEEDS:
        raise ValueError(
            f"source seeds are fixed at {SOURCE_SEEDS}; got {tuple(source_seeds)}"
        )
    if int(stream_seed) != STREAM_SEED:
        raise ValueError(f"stream seed is fixed at {STREAM_SEED}")
    if int(corruption_seed) != CORRUPTION_SEED:
        raise ValueError(f"corruption seed is fixed at {CORRUPTION_SEED}")
    if not _float_equal(corruption_fraction, CORRUPTION_FRACTION):
        raise ValueError(
            f"corruption fraction is fixed at {CORRUPTION_FRACTION}"
        )


def cell_key(
    candidate,
    source_domain,
    condition,
    source_seed,
    stream_seed=STREAM_SEED,
    corruption_seed=CORRUPTION_SEED,
):
    """Return the immutable identity of one calibration cell."""
    label = candidate_label(candidate)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown calibration condition: {condition!r}")
    return (
        label,
        int(source_domain),
        str(condition),
        int(source_seed),
        int(stream_seed),
        int(corruption_seed),
    )


def expected_cell_keys(
    candidates=CANDIDATE_KEEP_FRACTIONS,
    source_domains=SOURCE_DOMAINS,
    source_seeds=SOURCE_SEEDS,
    stream_seed=STREAM_SEED,
    corruption_seed=CORRUPTION_SEED,
):
    return tuple(
        cell_key(
            candidate,
            source_domain,
            condition,
            source_seed,
            stream_seed,
            corruption_seed,
        )
        for candidate in candidates
        for source_domain in source_domains
        for source_seed in source_seeds
        for condition in CONDITIONS
    )


def cell_dir(
    root: Path,
    candidate,
    source_domain: int,
    condition: str,
    source_seed: int,
) -> Path:
    """Return the canonical directory for one cell."""
    return (
        Path(root)
        / f"q{candidate_label(candidate)}"
        / f"source_domain_{int(source_domain)}"
        / str(condition)
        / f"source_seed_{int(source_seed)}"
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically publish JSON so an interrupted run is never resumable."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically publish a CSV status artifact."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_metadata(
    candidate,
    source_domain,
    condition,
    source_seed,
    stream_seed,
    corruption_seed,
):
    label = candidate_label(candidate)
    return {
        "protocol": PROTOCOL,
        "candidate_label": label,
        "confidence_keep_fraction": CANDIDATE_LABEL_TO_VALUE[label],
        "source_domain": int(source_domain),
        "calibration_flow": f"{int(source_domain)}->{int(source_domain)}",
        "condition": str(condition),
        "corruption_type": "signal_freeze",
        "severity": "moderate",
        "corruption_fraction": CONDITION_FRACTIONS[condition],
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
        "source_checkpoint_is_independent_unit": True,
        "stream_seed_is_paired_control": True,
        "held_out_source_labels_used_for_scoring": True,
        "target_labels_used_for_selection": False,
        "target_transfer_flows_excluded": True,
    }


def summary_matches(
    cell_path: Path,
    candidate,
    source_domain: int,
    condition: str,
    source_seed: int,
    stream_seed: int = STREAM_SEED,
    corruption_seed: int = CORRUPTION_SEED,
) -> bool:
    """Check that a cell has a complete summary and matching metadata."""
    if condition not in CONDITIONS:
        return False
    cell_path = Path(cell_path)
    metadata_path = cell_path / "cell_metadata.json"
    summary_path = cell_path / "summary_raw.csv"
    if not metadata_path.exists() or not summary_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(summary_path)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    if len(frame) != 1:
        return False
    expected_metadata = _expected_metadata(
        candidate,
        source_domain,
        condition,
        source_seed,
        stream_seed,
        corruption_seed,
    )
    for key, expected in expected_metadata.items():
        observed = metadata.get(key)
        if isinstance(expected, float):
            if not _float_equal(observed, expected):
                return False
        elif observed != expected:
            return False
    row = frame.iloc[0]
    if any(column not in row.index for column in SUMMARY_REQUIRED_COLUMNS):
        return False
    expected_row = {
        "dataset": "FD",
        "scenario": f"{int(source_domain)}->{int(source_domain)}",
        "method": "DuSafe",
        "variant": "full",
        "corruption": "signal_freeze",
        "severity": "moderate",
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
    }
    for column, expected in expected_row.items():
        observed = row[column]
        if column in {"source_seed", "stream_seed", "corruption_seed"}:
            try:
                if int(observed) != int(expected):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(observed) != str(expected):
            return False
    for metric in ("f1", "clean_correct_false_rejection_rate"):
        try:
            if not pd.notna(float(row[metric])):
                return False
        except (TypeError, ValueError):
            return False
    # A controlled benchmark summary may carry this optional provenance flag;
    # a true value would make the cell invalid for source-only selection.
    if "target_labels_used_for_selection" in row.index:
        if bool(row["target_labels_used_for_selection"]):
            return False
    if "target_data_used" in row.index:
        if bool(row["target_data_used"]):
            return False
    return True


STATUS_COLUMNS = (
    "candidate_label",
    "confidence_keep_fraction",
    "source_domain",
    "calibration_flow",
    "condition",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    "output_dir",
    "status",
    "returncode",
    "resumed",
    "recorded_at_unix",
)


def load_status(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    if not set(STATUS_COLUMNS).issubset(frame.columns):
        return {}
    values = {}
    for row in frame.to_dict(orient="records"):
        try:
            key = cell_key(
                row["candidate_label"],
                row["source_domain"],
                row["condition"],
                row["source_seed"],
                row["stream_seed"],
                row["corruption_seed"],
            )
        except (TypeError, ValueError):
            continue
        values[key] = row
    return values


def write_status(path: Path, status_by_key: dict) -> None:
    rows = []
    for row in status_by_key.values():
        rows.append({column: row.get(column) for column in STATUS_COLUMNS})
    frame = pd.DataFrame(rows, columns=STATUS_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["candidate_label", "source_domain", "source_seed", "condition"]
        )
    atomic_write_csv(frame, path)


def _cell_metadata(
    candidate, source_domain, condition, source_seed, stream_seed, corruption_seed
):
    return _expected_metadata(
        candidate,
        source_domain,
        condition,
        source_seed,
        stream_seed,
        corruption_seed,
    )


def _override_text(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def build_cell_command(args, candidate, source_domain, condition, source_seed, output):
    fraction = CONDITION_FRACTIONS[condition]
    benchmark = ROOT / "scripts" / "run_controlled_safety_benchmark.py"
    return [
        sys.executable,
        str(benchmark),
        "--data_path",
        str(args.data_path),
        "--device",
        str(args.device),
        "--datasets",
        "FD",
        "--methods",
        "DuSafe",
        "--variants",
        "full",
        "--scenarios",
        f"FD:{int(source_domain)}->{int(source_domain)}",
        "--corruptions",
        "signal_freeze",
        "--severities",
        "moderate",
        "--source_seeds",
        str(int(source_seed)),
        "--stream_seeds",
        str(int(args.stream_seed)),
        "--corruption_fraction",
        str(float(fraction)),
        "--corruption_seed",
        str(int(args.corruption_seed)),
        "--override",
        f"confidence_keep_fraction={_override_text(candidate)}",
        "--output_dir",
        str(output),
        "--pretrain_cache_dir",
        str(args.pretrain_cache_dir),
        "--registry",
        str(args.registry),
    ]


def _manifest_payload(args, source_domains, source_seeds, candidates, status, completed):
    return {
        "protocol": PROTOCOL,
        "status": status,
        "source_domains": [int(value) for value in source_domains],
        "calibration_flows": [
            f"{int(value)}->{int(value)}" for value in source_domains
        ],
        "source_seeds": [int(value) for value in source_seeds],
        "source_seed_is_independent_unit": True,
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "corruption_seed": int(args.corruption_seed),
        "conditions": {
            "clean": "signal_freeze moderate with fraction=0.0",
            "signal_freeze_moderate": {
                "severity": "moderate",
                "fraction": float(args.corruption_fraction),
                "mask": "deterministic_index_hash_fraction_0.5_seed1",
            },
        },
        "candidates": [
            {
                "candidate_label": candidate_label(value),
                "confidence_keep_fraction": float(value),
                "baseline": _float_equal(value, 0.95),
            }
            for value in candidates
        ],
        "f1_tolerance_absolute": 0.002,
        "selection_rule": {
            "eligibility": (
                "clean and signal_freeze_moderate F1 must each be at least "
                "the q=.95 mean minus 0.002"
            ),
            "priority": [
                "minimize clean-condition clean-correct false rejection rate",
                "minimize signal_freeze_moderate-condition clean-correct false rejection rate",
                "maximize signal_freeze_moderate F1",
            ],
        },
        "target_labels_used_for_selection": False,
        "target_transfer_flows_excluded": True,
        "held_out_same_domain_source_labels_only": True,
        "expected_cells": int(
            len(source_domains)
            * len(source_seeds)
            * len(candidates)
            * len(CONDITIONS)
        ),
        "completed_cells": int(completed),
        "cell_grain": (
            "one controlled benchmark summary per candidate, source domain, "
            "condition, and independent source checkpoint seed"
        ),
        "status_path": "cell_status.csv",
        "aggregate_command": (
            "scripts/aggregate_fd_gate_calibration_extended.py "
            "--input-dir <this-output-dir>"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", "--data_path", dest="data_path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--source-domains", "--source_domains", dest="source_domains", default="0,1,2,3")
    parser.add_argument("--source-seeds", "--source_seeds", dest="source_seeds", default="1,2,3")
    parser.add_argument("--stream-seed", "--stream_seed", dest="stream_seed", type=int, default=STREAM_SEED)
    parser.add_argument("--corruption-seed", "--corruption_seed", dest="corruption_seed", type=int, default=CORRUPTION_SEED)
    parser.add_argument("--corruption-fraction", "--corruption_fraction", dest="corruption_fraction", type=float, default=CORRUPTION_FRACTION)
    parser.add_argument("--candidates", default=",".join(str(value) for value in CANDIDATE_KEEP_FRACTIONS))
    parser.add_argument(
        "--pretrain-cache-dir",
        "--pretrain_cache_dir",
        dest="pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "fd_source_calibration_extended"),
    )
    parser.add_argument("--registry", choices=("production", "benchmark"), default="production")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=str(ROOT / "results" / "calibration" / "fd_source_gate_q95_q975_q99_q100_v2"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_domains = parse_int_list(args.source_domains, "--source-domains")
    source_seeds = parse_int_list(args.source_seeds, "--source-seeds")
    candidates = parse_candidates(args.candidates)
    _validate_fixed_protocol(
        source_domains,
        source_seeds,
        args.stream_seed,
        args.corruption_seed,
        args.corruption_fraction,
    )
    gpu_lock_handle = acquire_gpu_lock(
        ROOT / "results" / ".current_experiment_gpu.lock"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "cell_status.csv"
    manifest_path = output_dir / "manifest.json"
    expected_keys = expected_cell_keys(
        candidates,
        source_domains,
        source_seeds,
        args.stream_seed,
        args.corruption_seed,
    )
    status_by_key = load_status(status_path)
    # Never carry a key from a different protocol or malformed status file
    # into a new run.
    status_by_key = {
        key: value for key, value in status_by_key.items() if key in set(expected_keys)
    }
    manifest = _manifest_payload(
        args, source_domains, source_seeds, candidates, "running", 0
    )
    if manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid existing manifest: {manifest_path}") from exc
        if old_manifest.get("protocol") != PROTOCOL:
            raise ValueError("Existing output directory belongs to another protocol")
        if old_manifest.get("target_labels_used_for_selection") is not False:
            raise ValueError("Existing manifest is not target-label excluded")
        old_keys = {
            (
                str(item.get("candidate_label")),
                float(item.get("confidence_keep_fraction")),
            )
            for item in old_manifest.get("candidates", [])
        }
        new_keys = {
            (candidate_label(value), float(value)) for value in candidates
        }
        if old_keys and old_keys != new_keys:
            raise ValueError("Existing manifest candidate grid does not match")
    atomic_write_json(manifest_path, manifest)
    write_status(status_path, status_by_key)

    for candidate in candidates:
        for source_domain in source_domains:
            for source_seed in source_seeds:
                for condition in CONDITIONS:
                    key = cell_key(
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                        args.stream_seed,
                        args.corruption_seed,
                    )
                    output = cell_dir(
                        output_dir,
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                    )
                    if summary_matches(
                        output,
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                        args.stream_seed,
                        args.corruption_seed,
                    ):
                        status_by_key[key] = {
                            "candidate_label": candidate_label(candidate),
                            "confidence_keep_fraction": float(candidate),
                            "source_domain": int(source_domain),
                            "calibration_flow": f"{int(source_domain)}->{int(source_domain)}",
                            "condition": condition,
                            "source_seed": int(source_seed),
                            "stream_seed": int(args.stream_seed),
                            "corruption_seed": int(args.corruption_seed),
                            "output_dir": str(output),
                            "status": "completed",
                            "returncode": 0,
                            "resumed": True,
                            "recorded_at_unix": time.time(),
                        }
                        write_status(status_path, status_by_key)
                        completed_count = sum(
                            value.get("status") == "completed"
                            for value in status_by_key.values()
                        )
                        atomic_write_json(
                            manifest_path,
                            _manifest_payload(
                                args,
                                source_domains,
                                source_seeds,
                                candidates,
                                "running",
                                completed_count,
                            ),
                        )
                        continue

                    output.mkdir(parents=True, exist_ok=True)
                    command = build_cell_command(
                        args,
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                        output,
                    )
                    print(
                        f"[FD source calibration] q={candidate_label(candidate)} "
                        f"source={source_domain} source_seed={source_seed} "
                        f"condition={condition}",
                        flush=True,
                    )
                    completed = subprocess.run(command, cwd=ROOT, check=False)
                    returncode = int(completed.returncode)
                    if returncode == 0:
                        atomic_write_json(
                            output / "cell_metadata.json",
                            _cell_metadata(
                                candidate,
                                source_domain,
                                condition,
                                source_seed,
                                args.stream_seed,
                                args.corruption_seed,
                            ),
                        )
                    valid = returncode == 0 and summary_matches(
                        output,
                        candidate,
                        source_domain,
                        condition,
                        source_seed,
                        args.stream_seed,
                        args.corruption_seed,
                    )
                    status_by_key[key] = {
                        "candidate_label": candidate_label(candidate),
                        "confidence_keep_fraction": float(candidate),
                        "source_domain": int(source_domain),
                        "calibration_flow": f"{int(source_domain)}->{int(source_domain)}",
                        "condition": condition,
                        "source_seed": int(source_seed),
                        "stream_seed": int(args.stream_seed),
                        "corruption_seed": int(args.corruption_seed),
                        "output_dir": str(output),
                        "status": "completed" if valid else "failed",
                        "returncode": returncode,
                        "resumed": False,
                        "recorded_at_unix": time.time(),
                    }
                    write_status(status_path, status_by_key)
                    completed_count = sum(
                        value.get("status") == "completed"
                        for value in status_by_key.values()
                    )
                    atomic_write_json(
                        manifest_path,
                        _manifest_payload(
                            args,
                            source_domains,
                            source_seeds,
                            candidates,
                            "running",
                            completed_count,
                        ),
                    )
                    if not valid:
                        raise RuntimeError(
                            "FD source calibration cell failed or published an "
                            f"invalid summary: {output} (returncode={returncode})"
                        )

    completed_count = len(expected_keys)
    atomic_write_json(
        manifest_path,
        _manifest_payload(
            args,
            source_domains,
            source_seeds,
            candidates,
            "complete",
            completed_count,
        ),
    )
    print(f"Results: {output_dir}", flush=True)
    gpu_lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
