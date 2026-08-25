"""One-time FD transfer safety panel with frozen source-calibrated q=.95.

This runner is intentionally separate from the source-only selector.  It
does not inspect target labels and has no parameter-selection logic.  The
five paper flows are evaluated once for independent source checkpoint seeds
1--3 and one paired stream seed, under clean and the fixed moderate 50%
signal-freeze condition.
"""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FLOWS = ("0->1", "1->2", "3->1", "1->0", "2->3")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import atomic_write_csv


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"GPU lock is busy: {path}") from exc
    try:
        yield
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def parse_source_seeds(text):
    seeds = [
        int(value.strip())
        for value in str(text).split(",")
        if value.strip()
    ]
    if not seeds:
        raise ValueError("--source_seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--source_seeds must contain unique seeds")
    return seeds


def cell_key(flow, condition, source_seed, stream_seed, corruption_seed):
    """Return the immutable identity for one final-evaluation cell."""
    return (
        str(flow),
        str(condition),
        int(source_seed),
        int(stream_seed),
        int(corruption_seed),
    )


def summary_matches(cell_dir, flow, condition, source_seed, stream_seed,
                    corruption_seed, corruption_fraction):
    """Check that a per-cell benchmark completed for the requested key."""
    metadata_path = Path(cell_dir) / "cell_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected_metadata = {
        "flow": str(flow),
        "condition": str(condition),
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
        "corruption_fraction": float(corruption_fraction),
        "confidence_keep_fraction": 0.95,
    }
    for key, value in expected_metadata.items():
        observed = metadata.get(key)
        if isinstance(value, float):
            try:
                if abs(float(observed) - value) > 1e-12:
                    return False
            except (TypeError, ValueError):
                return False
        elif observed != value:
            return False
    summary_path = Path(cell_dir) / "summary_raw.csv"
    if not summary_path.exists():
        return False
    try:
        frame = pd.read_csv(summary_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if len(frame) != 1:
        return False
    row = frame.iloc[0]
    expected = {
        "dataset": "FD",
        "scenario": str(flow),
        "method": "DuSafe",
        "variant": "full",
        "corruption": "signal_freeze",
        "severity": "moderate",
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
        "corruption_seed": int(corruption_seed),
    }
    for column, value in expected.items():
        if column not in row.index:
            return False
        observed = row[column]
        if column in {"source_seed", "stream_seed", "corruption_seed"}:
            try:
                if int(observed) != int(value):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(observed) != str(value):
            return False
    return True


def atomic_write_json(path, payload):
    """Publish cell metadata atomically so resume cannot trust a partial file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_status(status_path):
    if not Path(status_path).exists():
        return {}
    try:
        frame = pd.read_csv(status_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    status_by_key = {}
    required = {
        "flow", "condition", "source_seed", "stream_seed",
        "corruption_seed",
    }
    if not required.issubset(frame.columns):
        return {}
    for row in frame.to_dict("records"):
        try:
            key = cell_key(
                row["flow"], row["condition"], row["source_seed"],
                row["stream_seed"], row["corruption_seed"],
            )
        except (TypeError, ValueError):
            continue
        status_by_key[key] = row
    return status_by_key


def write_status(status_path, status_by_key):
    columns = [
        "flow", "condition", "source_seed", "stream_seed",
        "corruption_seed", "output_dir", "status", "returncode",
        "resumed", "recorded_at_unix",
    ]
    rows = list(status_by_key.values())
    if rows:
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[columns].sort_values(
            ["flow", "condition", "source_seed"]
        )
    else:
        frame = pd.DataFrame(columns=columns)
    atomic_write_csv(frame, status_path, index=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "results" / "diagnostics" / "fd_gate_q095_final_v2"),
    )
    parser.add_argument(
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument("--source_seeds", default="1,2,3")
    parser.add_argument("--stream_seed", type=int, default=42)
    parser.add_argument("--corruption_seed", type=int, default=1)
    parser.add_argument("--corruption_fraction", type=float, default=0.5)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_seeds = parse_source_seeds(args.source_seeds)
    conditions = (
        ("clean", "0.0"),
        ("signal_freeze_moderate", str(float(args.corruption_fraction))),
    )
    runner = ROOT / "scripts" / "run_controlled_safety_benchmark.py"
    status_path = output_dir / "cell_status.csv"
    status_by_key = load_status(status_path)
    commands = []
    with gpu_lock(ROOT / "results" / ".current_experiment_gpu.lock"):
        for flow in FLOWS:
            source, target = flow.split("->", 1)
            for condition, fraction in conditions:
                for source_seed in source_seeds:
                    key = cell_key(
                        flow, condition, source_seed, args.stream_seed,
                        args.corruption_seed,
                    )
                    cell_dir = (
                        output_dir / f"flow_{source}_to_{target}" / condition
                        / f"source_seed_{source_seed}"
                    )
                    if summary_matches(
                        cell_dir, flow, condition, source_seed,
                        args.stream_seed, args.corruption_seed, float(fraction),
                    ):
                        status_by_key[key] = {
                            "flow": flow,
                            "condition": condition,
                            "source_seed": int(source_seed),
                            "stream_seed": int(args.stream_seed),
                            "corruption_seed": int(args.corruption_seed),
                            "output_dir": str(cell_dir),
                            "status": "completed",
                            "returncode": 0,
                            "resumed": True,
                            "recorded_at_unix": time.time(),
                        }
                        write_status(status_path, status_by_key)
                        commands.append({
                            "flow": flow,
                            "condition": condition,
                            "source_seed": int(source_seed),
                            "stream_seed": int(args.stream_seed),
                            "output_dir": str(cell_dir),
                            "status": "skipped_existing",
                        })
                        print(
                            f"[FD final safety] resume flow={flow} "
                            f"condition={condition} source_seed={source_seed} "
                            f"stream_seed={args.stream_seed}",
                            flush=True,
                        )
                        continue
                    command = [
                        sys.executable,
                        str(runner),
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
                        f"FD:{flow}",
                        "--corruptions",
                        "signal_freeze",
                        "--severities",
                        "moderate",
                        "--source_seeds",
                        str(int(source_seed)),
                        "--stream_seeds",
                        str(int(args.stream_seed)),
                        "--corruption_fraction",
                        fraction,
                        "--corruption_seed",
                        str(args.corruption_seed),
                        "--override",
                        "confidence_keep_fraction=0.95",
                        "--output_dir",
                        str(cell_dir),
                        "--pretrain_cache_dir",
                        str(args.pretrain_cache_dir),
                    ]
                    print(
                        f"[FD final safety] flow={flow} condition={condition} "
                        f"source_seed={source_seed} "
                        f"stream_seed={args.stream_seed}",
                        flush=True,
                    )
                    completed = subprocess.run(command, cwd=ROOT, check=False)
                    if completed.returncode == 0:
                        atomic_write_json(
                            cell_dir / "cell_metadata.json",
                            {
                                "flow": flow,
                                "condition": condition,
                                "source_seed": int(source_seed),
                                "stream_seed": int(args.stream_seed),
                                "corruption_seed": int(args.corruption_seed),
                                "corruption_fraction": float(fraction),
                                "confidence_keep_fraction": 0.95,
                                "selection_completed_before_evaluation": True,
                                "target_labels_used_for_selection": False,
                            },
                        )
                    cell_complete = (
                        completed.returncode == 0
                        and summary_matches(
                            cell_dir, flow, condition, source_seed,
                            args.stream_seed, args.corruption_seed,
                            float(fraction),
                        )
                    )
                    if completed.returncode == 0 and not cell_complete:
                        raise RuntimeError(
                            f"Final safety job exited successfully but did not "
                            f"publish a valid summary for {flow} {condition} "
                            f"source_seed={source_seed}"
                        )
                    cell_status = "completed" if cell_complete else "failed"
                    status_by_key[key] = {
                        "flow": flow,
                        "condition": condition,
                        "source_seed": int(source_seed),
                        "stream_seed": int(args.stream_seed),
                        "corruption_seed": int(args.corruption_seed),
                        "output_dir": str(cell_dir),
                        "status": cell_status,
                        "returncode": int(completed.returncode),
                        "resumed": False,
                        "recorded_at_unix": time.time(),
                    }
                    write_status(status_path, status_by_key)
                    commands.append(
                        {
                            "flow": flow,
                            "condition": condition,
                            "source_seed": int(source_seed),
                            "stream_seed": int(args.stream_seed),
                            "returncode": int(completed.returncode),
                            "output_dir": str(cell_dir),
                        }
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"Final safety job failed for {flow} {condition} "
                            f"source_seed={source_seed} with code "
                            f"{completed.returncode}"
                        )
    manifest = {
        "protocol": "FD final transfer safety q=.95 v2",
        "confidence_keep_fraction": 0.95,
        "source_seeds": source_seeds,
        "source_seed_is_independent_unit": True,
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "flows": list(FLOWS),
        "conditions": {
            "clean": "corruption_fraction=0.0",
            "signal_freeze_moderate": {
                "severity": "moderate",
                "fraction": float(args.corruption_fraction),
                "corruption_seed": int(args.corruption_seed),
            },
        },
        "selection_completed_before_evaluation": True,
        "target_labels_used_for_selection": False,
        "cell_count_expected": len(FLOWS) * len(source_seeds) * len(conditions),
        "cell_grain": (
            "one subprocess and one summary_raw.csv per flow, condition, "
            "source_seed, and fixed stream_seed"
        ),
        "completed_key_fields": [
            "flow", "condition", "source_seed", "stream_seed",
            "corruption_seed",
        ],
        "status_path": str(status_path),
        "invalid_previous_runs": [
            "results/diagnostics/fd_gate_q095_final_v1/"
            "INVALID_PROTOCOL.json"
        ],
        "commands": commands,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
