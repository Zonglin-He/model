"""Retune EEG TTA profiles with paired three-source-seed means.

This is a target-selected descriptive protocol.  Target labels are never
passed to the online adapter, but final target Macro-F1 is used to select each
coordinate.  Source-training hyperparameters are frozen; every candidate
reuses the same source checkpoint for a given flow/source seed.  Full and
No-SSAW are always evaluated as a paired unit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from scripts.dusafe_factorial_runner_common import current_profiles  # noqa: E402
from scripts.paper_flow_profiles import load_paper_flow_profiles  # noqa: E402
from scripts.run_flowwise_optuna_full_no_ssaw import (  # noqa: E402
    WorkerFailure,
    evaluate_pair,
)
from scripts.run_final_ssaw_full_no_ssaw_five_flow import (  # noqa: E402
    DEFAULT_GPU_LOCK,
    production_code_sha256,
)
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402
from scripts.run_optuna_stepwise import atomic_write_json  # noqa: E402


PROTOCOL = "eeg_flowwise_three_seed_tta_tuning_v1"
DATASET = "EEG"
DEFAULT_SEEDS = (0, 1, 2)
STREAM_SEED = 42

# Coordinates are intentionally compact.  Earlier sweeps already showed that
# tiny EEG deployment batches are both slow and poor, so they are excluded
# rather than re-spending the same target labels and GPU time.
STAGES: tuple[tuple[str, tuple[float | int, ...]], ...] = (
    ("batch_size", (48, 96, 192)),
    ("learning_rate", (1e-4, 3e-4, 7.5e-4, 1.5e-3, 3e-3, 5e-3)),
    ("steps", (1, 2, 4)),
    ("confidence_keep_fraction", (0.95, 0.975, 0.995, 1.0)),
    ("grad_clip", (0.01, 0.03, 0.1)),
    ("ssaw_auxiliary_weight", (0.1, 0.3, 1.0, 2.0, 4.0)),
    ("spline_log_strength", (0.1, 0.2, 0.3)),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_flowwise_optuna_full_no_ssaw.py",
        ROOT / "scripts" / "run_final_ssaw_full_no_ssaw_five_flow.py",
    ):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _flow_slug(flow: Sequence[str]) -> str:
    return f"{flow[0]}_to_{flow[1]}"


def _value_slug(value: float | int) -> str:
    rendered = f"{float(value):.12g}" if isinstance(value, float) else str(value)
    return rendered.replace("-", "m").replace(".", "p").replace("+", "")


def _stable_values(values: Sequence[float | int], current: float | int) -> list:
    result: list[float | int] = []
    for value in (*values, current):
        if value not in result:
            result.append(value)
    return result


def select_winner(
    records: Sequence[Mapping[str, Any]], *, f1_tolerance_pp: float
) -> dict[str, Any]:
    """Select by three-seed Full mean, then paired delta near the maximum."""

    completed = [
        dict(record)
        for record in records
        if record.get("status") == "complete"
        and math.isfinite(float(record["full_f1_mean"]))
        and math.isfinite(float(record["full_minus_no_ssaw_mean"]))
    ]
    if not completed:
        raise RuntimeError("stage has no complete three-seed candidate")
    maximum = max(float(record["full_f1_mean"]) for record in completed)
    tolerance = float(f1_tolerance_pp) / 100.0
    eligible = [
        record
        for record in completed
        if float(record["full_f1_mean"]) >= maximum - tolerance - 1e-12
    ]
    winner = max(
        eligible,
        key=lambda record: (
            float(record["full_minus_no_ssaw_mean"]),
            float(record["full_f1_mean"]),
            -float(record["candidate_value"]),
        ),
    )
    winner["stage_max_full_f1_mean"] = maximum
    winner["eligible_candidate_count"] = len(eligible)
    return winner


def _candidate_signature(
    *,
    flow: Sequence[str],
    seeds: Sequence[int],
    parameter: str,
    candidate: float | int,
    source_config: Mapping[str, Any],
    tta_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "code_sha256": _code_sha256(),
        "production_code_sha256": production_code_sha256(),
        "dataset": DATASET,
        "flow": list(flow),
        "source_seeds": list(map(int, seeds)),
        "stream_seed": STREAM_SEED,
        "parameter": parameter,
        "candidate": candidate,
        "source_config": dict(source_config),
        "tta_config": dict(tta_config),
    }


def _complete_cached_candidate(path: Path, signature_sha256: str) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") == "complete"
        and payload.get("candidate_signature_sha256") == signature_sha256
    ):
        return payload
    return None


def evaluate_candidate(
    *,
    args: argparse.Namespace,
    candidate_dir: Path,
    flow: Sequence[str],
    seeds: Sequence[int],
    parameter: str,
    candidate: float | int,
    source_config: Mapping[str, Any],
    tta_config: Mapping[str, Any],
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    signature = _candidate_signature(
        flow=flow,
        seeds=seeds,
        parameter=parameter,
        candidate=candidate,
        source_config=source_config,
        tta_config=tta_config,
    )
    signature_sha256 = _sha256_json(signature)
    aggregate_path = candidate_dir / "aggregate.json"
    cached = _complete_cached_candidate(aggregate_path, signature_sha256)
    if cached is not None:
        return cached

    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    worker_args = SimpleNamespace(
        stream_seed=int(args.stream_seed),
        data_path=str(Path(args.data_path).resolve()),
        device=str(args.device),
        backbone=str(args.backbone),
        gpu_lock_path=str(Path(args.gpu_lock_path).resolve()),
        max_batches=args.max_batches,
    )
    for source_seed in seeds:
        success = None
        for attempt in range(1, int(args.max_attempts) + 1):
            attempt_dir = (
                candidate_dir / f"source_seed_{source_seed}" / f"attempt_{attempt}"
            )
            try:
                success = evaluate_pair(
                    args=worker_args,
                    trial_dir=attempt_dir,
                    dataset=DATASET,
                    flow=flow,
                    source_seed=int(source_seed),
                    source_config=source_config,
                    tta_config=tta_config,
                    expected_source_model_sha256=expected_source_hashes.get(
                        str(source_seed)
                    ),
                )
                break
            except WorkerFailure as exc:
                failures.append(
                    {
                        "source_seed": int(source_seed),
                        "attempt": attempt,
                        "message": str(exc)[:1500],
                        **dict(exc.details),
                    }
                )
        if success is None:
            result = {
                "status": "failed",
                "candidate_signature_sha256": signature_sha256,
                "signature": signature,
                "parameter": parameter,
                "candidate_value": candidate,
                "failures": failures,
                "updated_at": utc_now(),
            }
            atomic_write_json(result, aggregate_path)
            return result
        rows.append(dict(success))

    source_hashes = {
        str(row["source_seed"]): str(row["source_model_sha256"]) for row in rows
    }
    if len(source_hashes) != len(seeds):
        raise RuntimeError("candidate did not produce one source hash per seed")
    full = [float(row["full_f1"]) for row in rows]
    no_ssaw = [float(row["no_ssaw_f1"]) for row in rows]
    delta = [float(row["full_minus_no_ssaw"]) for row in rows]
    result = {
        "status": "complete",
        "candidate_signature_sha256": signature_sha256,
        "signature": signature,
        "dataset": DATASET,
        "scenario": _flow_label(flow),
        "parameter": parameter,
        "candidate_value": candidate,
        "source_seeds": list(map(int, seeds)),
        "source_model_sha256_by_seed": source_hashes,
        "full_f1_mean": statistics.fmean(full),
        "full_f1_std": statistics.stdev(full) if len(full) > 1 else 0.0,
        "no_ssaw_f1_mean": statistics.fmean(no_ssaw),
        "no_ssaw_f1_std": statistics.stdev(no_ssaw) if len(no_ssaw) > 1 else 0.0,
        "full_minus_no_ssaw_mean": statistics.fmean(delta),
        "full_minus_no_ssaw_std": (
            statistics.stdev(delta) if len(delta) > 1 else 0.0
        ),
        "paired_rows": rows,
        "failures": failures,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "updated_at": utc_now(),
    }
    atomic_write_json(result, aggregate_path)
    return result


def _load_initial_profile(path: Path, flow: Sequence[str]) -> dict[str, Any]:
    profiles = load_paper_flow_profiles(path, (DATASET,))
    key = (DATASET, _flow_label(flow))
    if key not in profiles:
        raise RuntimeError(f"initial profile is missing {DATASET}:{key[1]}")
    _, defaults = current_profiles(DATASET)
    result = dict(defaults)
    result.update(profiles[key])
    if float(result.get("ssaw_auxiliary_weight", 0.0)) <= 0.0:
        raise RuntimeError("initial EEG SSAW weight must be positive")
    return result


def _state_signature(args: argparse.Namespace, flow: Sequence[str]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "code_sha256": _code_sha256(),
        "production_code_sha256": production_code_sha256(),
        "dataset": DATASET,
        "flow": list(flow),
        "source_seeds": list(args.source_seeds),
        "stream_seed": int(args.stream_seed),
        "initial_profile_json": str(Path(args.initial_profile_json).resolve()),
        "stage_order": [name for name, _ in STAGES],
        "stage_values": {name: list(values) for name, values in STAGES},
        "f1_tolerance_pp": float(args.f1_tolerance_pp),
    }


def run_flow(args: argparse.Namespace, output_dir: Path, flow: Sequence[str]) -> dict:
    flow_dir = output_dir / "flows" / _flow_slug(flow)
    flow_dir.mkdir(parents=True, exist_ok=True)
    state_path = flow_dir / "state.json"
    signature = _state_signature(args, flow)
    signature_sha256 = _sha256_json(signature)
    source_config, _ = current_profiles(DATASET)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature_sha256") != signature_sha256:
            raise RuntimeError(f"stale EEG tuning state: {state_path}")
    else:
        state = {
            "status": "running",
            "signature": signature,
            "signature_sha256": signature_sha256,
            "dataset": DATASET,
            "flow": list(flow),
            "source_config": dict(source_config),
            "tta_config": _load_initial_profile(
                Path(args.initial_profile_json), flow
            ),
            "source_model_sha256_by_seed": {},
            "next_stage_index": 0,
            "history": [],
            "updated_at": utc_now(),
        }
        atomic_write_json(state, state_path)

    while int(state["next_stage_index"]) < len(STAGES):
        stage_index = int(state["next_stage_index"])
        parameter, grid = STAGES[stage_index]
        current = state["tta_config"][parameter]
        values = _stable_values(grid, current)
        records: list[dict[str, Any]] = []
        stage_dir = flow_dir / f"stage_{stage_index:02d}_{parameter}"
        for value in values:
            candidate_config = deepcopy(state["tta_config"])
            candidate_config[parameter] = value
            candidate_dir = stage_dir / f"candidate_{_value_slug(value)}"
            record = evaluate_candidate(
                args=args,
                candidate_dir=candidate_dir,
                flow=flow,
                seeds=args.source_seeds,
                parameter=parameter,
                candidate=value,
                source_config=state["source_config"],
                tta_config=candidate_config,
                expected_source_hashes=state["source_model_sha256_by_seed"],
            )
            records.append(record)
            pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"signature", "paired_rows", "failures"}
                    }
                    for item in records
                ]
            ).to_csv(stage_dir / "candidate_summary.csv", index=False)
        winner = select_winner(records, f1_tolerance_pp=args.f1_tolerance_pp)
        state["tta_config"][parameter] = winner["candidate_value"]
        if not state["source_model_sha256_by_seed"]:
            state["source_model_sha256_by_seed"] = dict(
                winner["source_model_sha256_by_seed"]
            )
        state["history"].append(
            {
                "stage_index": stage_index,
                "parameter": parameter,
                "previous_value": current,
                "selected_value": winner["candidate_value"],
                "selected_full_f1_mean": winner["full_f1_mean"],
                "selected_no_ssaw_f1_mean": winner["no_ssaw_f1_mean"],
                "selected_full_minus_no_ssaw_mean": winner[
                    "full_minus_no_ssaw_mean"
                ],
                "stage_max_full_f1_mean": winner["stage_max_full_f1_mean"],
                "f1_tolerance_pp": float(args.f1_tolerance_pp),
                "completed_at": utc_now(),
            }
        )
        state["next_stage_index"] = stage_index + 1
        state["updated_at"] = utc_now()
        atomic_write_json(state, state_path)
    state["status"] = "complete"
    state["completed_at"] = utc_now()
    atomic_write_json(state, state_path)
    return state


def _write_profile_json(
    *, args: argparse.Namespace, output_dir: Path, states: Sequence[Mapping[str, Any]]
) -> Path:
    source_path = Path(args.initial_profile_json).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    profiles = dict(payload.get("profiles", payload))
    selected_keys = {name for name, _ in STAGES}
    for state in states:
        key = f"{DATASET}:{_flow_label(state['flow'])}"
        original = dict(profiles[key])
        for name in selected_keys:
            original[name] = state["tta_config"][name]
        if float(original["ssaw_auxiliary_weight"]) <= 0.0:
            raise RuntimeError(f"selected profile has non-positive SSAW weight: {key}")
        profiles[key] = original
    result = {
        "protocol": "paper_flow_profiles_v3_eeg_three_seed_retuned_descriptive",
        "selection_source_seeds": list(args.source_seeds),
        "stream_seed": int(args.stream_seed),
        "source_training_overridden": False,
        "target_labels_used_for_tta_profile_selection": True,
        "confirmatory": False,
        "selection_rule": (
            "sequential per-flow coordinates maximize paired three-seed mean "
            "Full F1; within f1_tolerance_pp maximize Full-minus-NoSSAW"
        ),
        "f1_tolerance_pp": float(args.f1_tolerance_pp),
        "ssaw_weight_strictly_positive": True,
        "base_profile_json": str(source_path),
        "profiles": profiles,
    }
    target = output_dir / "paper_flow_profiles_v3_eeg_retuned.json"
    atomic_write_json(result, target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "optuna" / "eeg_flowwise_three_seed_v1"),
    )
    parser.add_argument(
        "--initial-profile-json",
        default=str(ROOT / "configs" / "paper_flow_profiles_v2.json"),
    )
    parser.add_argument("--source-seeds", default="0,1,2")
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    parser.add_argument("--f1-tolerance-pp", type=float, default=0.10)
    parser.add_argument("--gpu-lock-path", default=str(DEFAULT_GPU_LOCK))
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-flows", type=int, default=None)
    args = parser.parse_args()
    try:
        args.source_seeds = tuple(
            int(item.strip()) for item in args.source_seeds.split(",") if item.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid source seeds: {exc}")
    if (
        not args.source_seeds
        or len(args.source_seeds) != len(set(args.source_seeds))
        or min(args.source_seeds) < 0
    ):
        parser.error("source seeds must be unique non-negative integers")
    if args.f1_tolerance_pp < 0 or args.max_attempts < 1:
        parser.error("tolerance must be non-negative and attempts positive")
    if args.max_flows is not None and args.max_flows < 1:
        parser.error("max flows must be positive")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flows = [tuple(map(str, flow)) for flow in formal_scenario_pairs(DATASET)]
    if len(flows) != 5:
        raise RuntimeError(f"EEG must have five formal flows, found {len(flows)}")
    if args.max_flows is not None:
        flows = flows[: int(args.max_flows)]
    manifest = {
        "protocol": PROTOCOL,
        "code_sha256": _code_sha256(),
        "production_code_sha256": production_code_sha256(),
        "dataset": DATASET,
        "flows": [_flow_label(flow) for flow in flows],
        "source_seeds": list(args.source_seeds),
        "stream_seed": int(args.stream_seed),
        "stage_order": [name for name, _ in STAGES],
        "stage_values": {name: list(values) for name, values in STAGES},
        "flow_specific_tta_profiles": True,
        "source_training_frozen": True,
        "source_checkpoint_reused_across_tta_candidates": True,
        "full_no_ssaw_share_source_checkpoint": True,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_online_decision": False,
        "confirmatory": False,
        "started_at": utc_now(),
    }
    atomic_write_json(manifest, output_dir / "manifest.json")
    states = []
    for flow_index, flow in enumerate(flows):
        atomic_write_json(
            {
                "status": "running",
                "flow_index": flow_index,
                "flow_count": len(flows),
                "current_scenario": _flow_label(flow),
                "updated_at": utc_now(),
            },
            output_dir / "status.json",
        )
        states.append(run_flow(args, output_dir, flow))
    profile_path = _write_profile_json(args=args, output_dir=output_dir, states=states)
    rows = []
    for state in states:
        last = state["history"][-1]
        rows.append(
            {
                "dataset": DATASET,
                "scenario": _flow_label(state["flow"]),
                "source_seeds": ",".join(map(str, args.source_seeds)),
                "full_f1_mean": last["selected_full_f1_mean"],
                "no_ssaw_f1_mean": last["selected_no_ssaw_f1_mean"],
                "full_minus_no_ssaw_mean": last[
                    "selected_full_minus_no_ssaw_mean"
                ],
                **{
                    name: state["tta_config"][name]
                    for name, _ in STAGES
                },
            }
        )
    atomic_write_csv(pd.DataFrame(rows), output_dir / "selected_summary.csv", index=False)
    manifest.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "selected_profile_json": str(profile_path),
            "selected_flow_count": len(states),
        }
    )
    atomic_write_json(manifest, output_dir / "manifest.json")
    atomic_write_json(
        {
            "status": "complete",
            "flow_count": len(states),
            "selected_profile_json": str(profile_path),
            "updated_at": utc_now(),
        },
        output_dir / "status.json",
    )
    print(json.dumps({"status": "complete", "profile": str(profile_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
