"""One-factor-at-a-time sensitivity around the frozen HAR DuSafe profile.

Every profile is evaluated on the same five fixed HAR flows, three independent
source checkpoints, and stream seed 42.  The target labels are used only for
post-hoc reporting; this runner never selects or updates the frozen profile.
Each profile owns a separate result directory so main-table resume keys cannot
silently mix different runtime hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.har_frozen_profile import (  # noqa: E402
    DEVELOPMENT_EFFECT,
    FROZEN_HAR_TTA_PARAMS,
    PROFILE_ID as FROZEN_HAR_PROFILE_ID,
    validate_frozen_har_profile,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir  # noqa: E402
from scripts.run_optuna_stepwise import scenario_pairs  # noqa: E402


PROTOCOL = "HAR frozen-profile one-factor sensitivity v1"
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
EXPECTED_FLOWS = 5
EXPECTED_CELLS_PER_PROFILE = EXPECTED_FLOWS * len(SOURCE_SEEDS)
EXPECTED_SCENARIOS = tuple(
    f"{source}->{target}" for source, target in scenario_pairs("HAR")
)
SENSITIVITY_AXES = {
    "batch_size": (32, 64),
    "learning_rate": (1e-4, 1e-3),
    "steps": (8, 32),
    "ssaw_strength": (2.0, 6.0),
    "ssaw_auxiliary_weight": (0.5, 2.0),
    "ssaw_kl_scale": (0.01, 0.1),
    "confidence_keep_fraction": (0.99, 1.0),
}


def profile_specs() -> list[dict]:
    profiles = [
        {
            "profile_id": "frozen",
            "parameter": "frozen_profile",
            "value": FROZEN_HAR_PROFILE_ID,
            "overrides": {},
        }
    ]
    for parameter, values in SENSITIVITY_AXES.items():
        for index, value in enumerate(values):
            profiles.append(
                {
                    "profile_id": f"{parameter}_{index}",
                    "parameter": parameter,
                    "value": value,
                    "overrides": {parameter: value},
                }
            )
    return profiles


def _encode_override(key: str, value) -> str:
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif value is None:
        encoded = "none"
    else:
        encoded = repr(value)
    return f"{key}={encoded}"


def profile_run_signature(args, profile: dict) -> str:
    payload = {
        "protocol": PROTOCOL,
        "frozen_profile_id": FROZEN_HAR_PROFILE_ID,
        "profile_id": profile["profile_id"],
        "overrides": dict(profile["overrides"]),
        "data_path": str(Path(args.data_path).resolve()),
        "backbone": str(args.backbone),
        "pretrain_cache_dir": str(Path(args.pretrain_cache_dir).resolve()),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "scenarios": list(EXPECTED_SCENARIOS),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"{PROTOCOL}:{hashlib.sha256(encoded).hexdigest()}"


def build_profile_command(args, profile: dict, profile_dir: Path) -> list[str]:
    run_signature = profile_run_signature(args, profile)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_full_main_table.py"),
        "--data-path",
        str(Path(args.data_path).resolve()),
        "--device",
        str(args.device),
        "--backbone",
        str(args.backbone),
        "--datasets",
        "HAR",
        "--methods",
        "DuSafe",
        "--source-seeds",
        ",".join(str(seed) for seed in SOURCE_SEEDS),
        "--stream-seed",
        str(STREAM_SEED),
        "--pretrain-cache-dir",
        str(Path(args.pretrain_cache_dir).resolve()),
        "--eata-fisher-cache-dir",
        str(Path(args.eata_fisher_cache_dir).resolve()),
        "--output-dir",
        str(profile_dir.resolve()),
        "--retry-failures",
        "--run-signature",
        run_signature,
    ]
    for key, value in sorted(profile["overrides"].items()):
        command.extend(("--override", _encode_override(key, value)))
    return command


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_profile_rows(
    frame: pd.DataFrame, profile: dict, expected_signature: str
) -> pd.DataFrame:
    required = {
        "dataset",
        "scenario",
        "method",
        "source_seed",
        "stream_seed",
        "run_signature",
        "status",
        "f1",
        "accuracy",
        "runtime_hparams",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{profile['profile_id']}: missing result columns {missing}"
        )
    selected = frame[
        frame["dataset"].eq("HAR")
        & frame["method"].eq("DuSafe")
        & frame["status"].eq("ok")
        & frame["source_seed"].isin(SOURCE_SEEDS)
        & frame["stream_seed"].eq(STREAM_SEED)
    ].copy()
    key_columns = ["scenario", "source_seed", "stream_seed"]
    if selected.duplicated(key_columns).any():
        raise ValueError(f"{profile['profile_id']}: duplicate result cells")
    if len(selected) != EXPECTED_CELLS_PER_PROFILE:
        failed = frame[~frame["status"].eq("ok")]
        raise ValueError(
            f"{profile['profile_id']}: expected {EXPECTED_CELLS_PER_PROFILE} "
            f"successful cells, found {len(selected)}; failed={len(failed)}"
        )
    if selected["scenario"].nunique() != EXPECTED_FLOWS:
        raise ValueError(f"{profile['profile_id']}: expected five HAR flows")
    if set(selected["scenario"].astype(str)) != set(EXPECTED_SCENARIOS):
        raise ValueError(
            f"{profile['profile_id']}: HAR flow set does not match the protocol"
        )
    expected_hparams = {
        **FROZEN_HAR_TTA_PARAMS,
        **dict(profile["overrides"]),
    }
    for row in selected.to_dict("records"):
        if str(row.get("run_signature", "")) != expected_signature:
            raise ValueError(
                f"{profile['profile_id']}: stale or mismatched run signature"
            )
        runtime = json.loads(str(row["runtime_hparams"]))
        mismatched = {
            key: {"expected": value, "observed": runtime.get(key)}
            for key, value in expected_hparams.items()
            if runtime.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"{profile['profile_id']}: runtime hparams drifted: {mismatched}"
            )
    selected.insert(0, "profile_id", profile["profile_id"])
    selected.insert(1, "sensitivity_parameter", profile["parameter"])
    selected.insert(2, "sensitivity_value", str(profile["value"]))
    return selected


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistics.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def summarize(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_seed = (
        all_rows.groupby(
            [
                "profile_id",
                "sensitivity_parameter",
                "sensitivity_value",
                "source_seed",
            ],
            as_index=False,
        )
        .agg(
            n_flows=("scenario", "nunique"),
            f1=("f1", "mean"),
            accuracy=("accuracy", "mean"),
        )
    )
    summary_rows = []
    for keys, group in per_seed.groupby(
        ["profile_id", "sensitivity_parameter", "sensitivity_value"],
        sort=False,
    ):
        values = group["f1"].to_numpy(dtype=float)
        count = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if count > 1 else float("nan")
        if count > 1:
            half_width = float(
                stats.t.ppf(0.975, count - 1) * std / math.sqrt(count)
            )
        else:
            half_width = float("nan")
        summary_rows.append(
            {
                "profile_id": keys[0],
                "sensitivity_parameter": keys[1],
                "sensitivity_value": keys[2],
                "n_source_seeds": count,
                "n_flows_per_seed_min": int(group["n_flows"].min()),
                "f1_mean": mean,
                "f1_std_across_source_seeds": std,
                "f1_ci95_low": mean - half_width,
                "f1_ci95_high": mean + half_width,
                "accuracy_mean": float(group["accuracy"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    frozen = all_rows[all_rows["profile_id"].eq("frozen")][
        ["scenario", "source_seed", "stream_seed", "f1"]
    ].rename(columns={"f1": "f1_frozen"})
    paired_rows = []
    for profile_id, group in all_rows[
        ~all_rows["profile_id"].eq("frozen")
    ].groupby("profile_id", sort=False):
        merged = group.merge(
            frozen,
            on=["scenario", "source_seed", "stream_seed"],
            how="inner",
            validate="one_to_one",
        )
        merged["f1_delta_vs_frozen"] = merged["f1"] - merged["f1_frozen"]
        seed_delta = merged.groupby("source_seed")[
            "f1_delta_vs_frozen"
        ].mean()
        values = seed_delta.to_numpy(dtype=float)
        paired_rows.append(
            {
                "profile_id": profile_id,
                "sensitivity_parameter": str(
                    group["sensitivity_parameter"].iloc[0]
                ),
                "sensitivity_value": str(group["sensitivity_value"].iloc[0]),
                "mean_f1_delta_vs_frozen": float(values.mean()),
                "exact_source_seed_sign_flip_p": _exact_sign_flip_p(values),
                "n_source_seeds": int(len(values)),
                "n_paired_flow_seed_cells": int(len(merged)),
            }
        )
    return per_seed, summary, pd.DataFrame(paired_rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--eata-fisher-cache-dir",
        default=str(ROOT / "results" / "eata_fisher_cache" / "reviewer_queue_v2"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "reviewer_queue_v2" / "har_frozen_sensitivity_v1"),
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Optional comma-separated profile ids; frozen is added automatically.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_frozen_har_profile()
    output_dir = ensure_dir(args.output_dir)
    run_root = ensure_dir(output_dir / "profile_runs")
    profiles = profile_specs()
    requested = {
        value.strip() for value in str(args.profiles).split(",") if value.strip()
    }
    if requested:
        requested.add("frozen")
        known = {profile["profile_id"] for profile in profiles}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown sensitivity profiles: {unknown}")
        profiles = [
            profile for profile in profiles if profile["profile_id"] in requested
        ]

    status_path = output_dir / "status.json"
    status = {
        "protocol": PROTOCOL,
        "status": "running",
        "profile_ids": [profile["profile_id"] for profile in profiles],
        "profile_run_signatures": {
            profile["profile_id"]: profile_run_signature(args, profile)
            for profile in profiles
        },
        "completed_profiles": [],
    }
    _atomic_json(status_path, status)
    all_frames = []
    for profile in profiles:
        profile_dir = ensure_dir(run_root / profile["profile_id"])
        command = build_profile_command(args, profile, profile_dir)
        print(
            f"[HAR sensitivity] {profile['profile_id']}: {' '.join(command)}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            status.update(
                {
                    "status": "failed",
                    "failed_profile": profile["profile_id"],
                    "returncode": int(completed.returncode),
                }
            )
            _atomic_json(status_path, status)
            return int(completed.returncode)
        raw_path = profile_dir / "per_source_seed_results.csv"
        frame = _validate_profile_rows(
            pd.read_csv(raw_path),
            profile,
            profile_run_signature(args, profile),
        )
        all_frames.append(frame)
        status["completed_profiles"].append(profile["profile_id"])
        _atomic_json(status_path, status)

    all_rows = pd.concat(all_frames, ignore_index=True)
    per_seed, summary, paired = summarize(all_rows)
    atomic_write_csv(all_rows, output_dir / "all_profile_cells.csv", index=False)
    atomic_write_csv(per_seed, output_dir / "per_source_seed.csv", index=False)
    atomic_write_csv(summary, output_dir / "sensitivity_summary.csv", index=False)
    atomic_write_csv(paired, output_dir / "paired_vs_frozen.csv", index=False)
    manifest = {
        "protocol": PROTOCOL,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "frozen_har_profile_id": FROZEN_HAR_PROFILE_ID,
        "frozen_har_tta_hparams": dict(FROZEN_HAR_TTA_PARAMS),
        "source_seeds": list(SOURCE_SEEDS),
        "source_seed_is_independent_unit": True,
        "stream_seed": STREAM_SEED,
        "stream_seed_is_paired_control": True,
        "flows": EXPECTED_FLOWS,
        "scenarios": list(EXPECTED_SCENARIOS),
        "expected_cells_per_profile": EXPECTED_CELLS_PER_PROFILE,
        "profiles": profiles,
        "profile_run_signatures": {
            profile["profile_id"]: profile_run_signature(args, profile)
            for profile in profiles
        },
        "one_factor_at_a_time": True,
        "sensitivity_profile_selection_performed": False,
        "target_labels_used_online": False,
        "sensitivity_target_labels_used_for_selection": False,
        "frozen_profile_original_target_labels_used_for_selection": bool(
            DEVELOPMENT_EFFECT["target_labels_used_for_profile_selection"]
        ),
        "target_labels_used_posthoc_for_sensitivity_reporting": True,
        "outputs": [
            "all_profile_cells.csv",
            "per_source_seed.csv",
            "sensitivity_summary.csv",
            "paired_vs_frozen.csv",
        ],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    status.update({"status": "complete", "manifest": "manifest.json"})
    _atomic_json(status_path, status)
    print(f"Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
