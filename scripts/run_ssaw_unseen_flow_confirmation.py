"""Run the frozen SSAW stable-neighborhood test on untouched HHAR flows.

The calibration flow (HHAR 4->5) is excluded.  One frozen runtime profile is
applied unchanged to the remaining five AdaTime flows, source seeds 0/1/2,
and a Sobol direction bank that is disjoint from calibration/test_v1.  The
underlying causal cells compare confidence-only, matched raw duplication,
random eligible spline views, and hard SSAW from the same batch-start state.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_representative_causal_ablation import (  # noqa: E402
    PROTOCOL as CAUSAL_PROTOCOL,
    aggregate_directory,
)


PROTOCOL = "ssaw_unseen_flow_confirmation_v1"
DEFAULT_PROFILE = ROOT / "configs" / "ssaw_unseen_flow_confirmation_profile_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_evidence_v2" / PROTOCOL
CAUSAL_RUNNER = ROOT / "scripts" / "run_representative_causal_ablation.py"


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != "ssaw_unseen_flow_confirmation_profile_v1":
        raise ValueError("confirmation profile protocol mismatch")
    if payload.get("frozen_before_confirmation") is not True:
        raise ValueError("confirmation profile must be frozen before execution")
    flows = tuple(str(value) for value in payload.get("evaluation_flows", ()))
    if flows != ("5->0", "6->1", "7->4", "8->3", "0->2"):
        raise ValueError("confirmation flow set/order drifted")
    seeds = tuple(int(value) for value in payload.get("source_seeds", ()))
    if seeds != (0, 1, 2):
        raise ValueError("confirmation source-seed set drifted")
    if str(payload.get("heldout_direction_bank")) != "confirmatory_v1":
        raise ValueError("confirmation direction bank drifted")
    return payload


def build_plan(
    *,
    profile_path: str | Path = DEFAULT_PROFILE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    data_path: str | Path = ROOT / "data" / "Dataset",
    device: str = "cuda:0",
    backbone: str = "CNN",
    pretrain_cache_dir: str | Path = ROOT / "results" / "pretrain_cache" / "optuna_stepwise",
) -> dict[str, Any]:
    profile_path = Path(profile_path).expanduser().resolve()
    payload = load_protocol(profile_path)
    output_dir = Path(output_dir).expanduser().resolve()
    cells = []
    for flow in payload["evaluation_flows"]:
        source, target = flow.split("->", 1)
        for seed in payload["source_seeds"]:
            cell_dir = output_dir / "HHAR" / f"flow_{source}_to_{target}" / f"source_seed_{seed}" / "clean"
            command = [
                sys.executable,
                str(CAUSAL_RUNNER),
                "--cell",
                "--dataset", "HHAR",
                "--scenario", flow,
                "--source-seed", str(seed),
                "--stream-seed", str(payload["stream_seed"]),
                "--condition", "clean",
                "--data-path", str(data_path),
                "--device", str(device),
                "--backbone", str(backbone),
                "--pretrain-cache-dir", str(pretrain_cache_dir),
                "--runtime-profile-json", str(profile_path),
                "--evaluation-role", "confirmatory",
                "--heldout-bank-tag", str(payload["heldout_direction_bank"]),
                "--horizons", "5",
                "--output-dir", str(cell_dir),
            ]
            cells.append(
                {
                    "key": f"HHAR:{flow}:source{seed}",
                    "flow": flow,
                    "source_seed": int(seed),
                    "output_dir": str(cell_dir),
                    "command": command,
                }
            )
    return {
        "protocol": PROTOCOL,
        "status": "planned",
        "causal_protocol": CAUSAL_PROTOCOL,
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_file(profile_path),
        "calibration_flow_excluded": payload["calibration_flow"],
        "evaluation_flows": list(payload["evaluation_flows"]),
        "source_seeds": list(payload["source_seeds"]),
        "heldout_direction_bank": payload["heldout_direction_bank"],
        "expected_cells": len(cells),
        "target_labels_used_for_online_updates": False,
        "target_labels_used_for_evaluation_flow_selection": False,
        "target_labels_used_for_runtime_profile_selection_on_evaluation_flows": False,
        "parameter_selection_data_overlap": False,
        "confirmatory": True,
        "preregistered_decision_rule": payload["preregistered_decision_rule"],
        "cells": cells,
    }


def _cell_complete(path: Path) -> bool:
    manifest = path / "manifest.json"
    raw = path / "raw.csv"
    if not manifest.is_file() or not raw.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("protocol") == CAUSAL_PROTOCOL
        and payload.get("status") == "complete"
        and payload.get("protocol_passed") is True
        and payload.get("confirmatory") is True
        and payload.get("parameter_selection_data_overlap") is False
        and payload.get("target_labels_used_for_parameter_selection") is False
        and payload.get("heldout_bank_tag") == "confirmatory_v1"
    )


def _weighted(group: pd.DataFrame, value: str, weight: str) -> float:
    values = pd.to_numeric(group[value], errors="coerce")
    weights = pd.to_numeric(group[weight], errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0.0)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "dataset", "scenario", "source_seed", "variant", "future_macro_f1",
        "heldout_flip_rate", "heldout_worst_margin", "heldout_consistency",
        "heldout_stable_radius_sum", "heldout_stable_radius_admitted_count",
        "heldout_sample_count", "heldout_cap_stable_ray_successes",
        "heldout_cap_stable_ray_total",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"confirmation raw data lacks {missing}")
    unit_rows = []
    keys = ["dataset", "scenario", "source_seed", "variant"]
    for key, group in raw.groupby(keys, sort=True):
        radius_count = pd.to_numeric(
            group["heldout_stable_radius_admitted_count"], errors="coerce"
        ).sum()
        radius_sum = pd.to_numeric(
            group["heldout_stable_radius_sum"], errors="coerce"
        ).sum()
        cap_total = pd.to_numeric(
            group["heldout_cap_stable_ray_total"], errors="coerce"
        ).sum()
        cap_success = pd.to_numeric(
            group["heldout_cap_stable_ray_successes"], errors="coerce"
        ).sum()
        unit_rows.append(
            {
                "dataset": key[0],
                "scenario": key[1],
                "source_seed": int(key[2]),
                "variant": key[3],
                "future_macro_f1": float(pd.to_numeric(group["future_macro_f1"], errors="coerce").mean()),
                "heldout_flip_rate": _weighted(group, "heldout_flip_rate", "heldout_sample_count"),
                "heldout_worst_margin": _weighted(group, "heldout_worst_margin", "heldout_sample_count"),
                "heldout_consistency": _weighted(group, "heldout_consistency", "heldout_sample_count"),
                "heldout_stable_radius": float(radius_sum / radius_count) if radius_count > 0 else math.nan,
                "heldout_cap_stable_ray_fraction": float(cap_success / cap_total) if cap_total > 0 else math.nan,
                "ssaw_training_participation_rate": float(
                    pd.to_numeric(group.get("ssaw_training_participation_rate"), errors="coerce").mean()
                ),
            }
        )
    units = pd.DataFrame(unit_rows)
    variants = set(units["variant"])
    expected = {"confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"}
    if not expected.issubset(variants):
        raise ValueError("confirmation control grid is incomplete")
    pivot = units[units["variant"].isin(expected)].pivot(
        index=["dataset", "scenario", "source_seed"],
        columns="variant",
        values=[
            "future_macro_f1", "heldout_flip_rate", "heldout_worst_margin",
            "heldout_consistency", "heldout_stable_radius",
            "heldout_cap_stable_ray_fraction", "ssaw_training_participation_rate",
        ],
    )
    pivot.columns = [f"{metric}__{variant}" for metric, variant in pivot.columns]
    effects = pivot.reset_index()
    for metric in (
        "future_macro_f1", "heldout_worst_margin", "heldout_consistency",
        "heldout_stable_radius", "heldout_cap_stable_ray_fraction",
    ):
        effects[f"delta_{metric}_hard_vs_confidence"] = (
            effects[f"{metric}__hard_ssaw"] - effects[f"{metric}__confidence_only"]
        )
    effects["flip_reduction_hard_vs_confidence"] = (
        effects["heldout_flip_rate__confidence_only"]
        - effects["heldout_flip_rate__hard_ssaw"]
    )
    effects["relative_flip_reduction_hard_vs_confidence"] = (
        effects["flip_reduction_hard_vs_confidence"]
        / effects["heldout_flip_rate__confidence_only"].clip(lower=1e-12)
    )
    effects["delta_stable_radius_hard_vs_random"] = (
        effects["heldout_stable_radius__hard_ssaw"]
        - effects["heldout_stable_radius__random_eligible_spline"]
    )
    effects["delta_stable_radius_hard_vs_duplicate"] = (
        effects["heldout_stable_radius__hard_ssaw"]
        - effects["heldout_stable_radius__matched_raw_duplicate"]
    )

    stable_column = "delta_heldout_stable_radius_hard_vs_confidence"
    stable_values = effects[stable_column].to_numpy(dtype=float)
    rng = np.random.default_rng(20260823)
    bootstrap = np.asarray([
        rng.choice(stable_values, size=stable_values.size, replace=True).mean()
        for _ in range(50_000)
    ])
    flow_effect = effects.groupby("scenario")[stable_column].mean()
    seed_effect = effects.groupby("source_seed")[stable_column].mean()
    confidence_flip = float(effects["heldout_flip_rate__confidence_only"].mean())
    flip_reduction = float(effects["flip_reduction_hard_vs_confidence"].mean())
    relative_flip = flip_reduction / max(confidence_flip, 1e-12)
    checks = {
        "positive_mean_stable_radius": float(stable_values.mean()) > 0.0,
        "positive_at_least_3_of_5_flows": int(flow_effect.gt(0.0).sum()) >= 3,
        "positive_at_least_2_of_3_seeds": int(seed_effect.gt(0.0).sum()) >= 2,
        "relative_flip_reduction_at_least_5_percent": relative_flip >= 0.05,
        "future_f1_noninferiority_minus_0p2pp": float(
            effects["delta_future_macro_f1_hard_vs_confidence"].mean()
        ) >= -0.002,
        "beats_random_stable_radius": float(
            effects["delta_stable_radius_hard_vs_random"].mean()
        ) > 0.0,
        "beats_duplicate_stable_radius": float(
            effects["delta_stable_radius_hard_vs_duplicate"].mean()
        ) > 0.0,
    }
    report = {
        "protocol": PROTOCOL,
        "status": "complete",
        "confirmatory": True,
        "independent_evaluation_units": int(len(effects)),
        "flows": sorted(effects["scenario"].unique().tolist()),
        "source_seeds": sorted(int(value) for value in effects["source_seed"].unique()),
        "mean_stable_radius_delta": float(stable_values.mean()),
        "stable_radius_delta_cluster_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "positive_flow_count": int(flow_effect.gt(0.0).sum()),
        "positive_seed_count": int(seed_effect.gt(0.0).sum()),
        "mean_future_macro_f1_delta": float(
            effects["delta_future_macro_f1_hard_vs_confidence"].mean()
        ),
        "mean_flip_reduction": flip_reduction,
        "relative_flip_reduction": relative_flip,
        "mean_worst_margin_gain": float(
            effects["delta_heldout_worst_margin_hard_vs_confidence"].mean()
        ),
        "mean_consistency_gain": float(
            effects["delta_heldout_consistency_hard_vs_confidence"].mean()
        ),
        "mean_stable_radius_gain_vs_random": float(
            effects["delta_stable_radius_hard_vs_random"].mean()
        ),
        "mean_stable_radius_gain_vs_duplicate": float(
            effects["delta_stable_radius_hard_vs_duplicate"].mean()
        ),
        "checks": checks,
        "decision": "supports_stable_neighborhood_expansion" if all(checks.values()) else "does_not_meet_preregistered_confirmation_rule",
    }
    return units, effects, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"))
    parser.add_argument("--gpu-lock-path", type=Path, default=ROOT / "results" / ".current_experiment_gpu.lock")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = build_plan(
        profile_path=args.profile_json,
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
        backbone=args.backbone,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(plan, output / "plan.json")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    completed_cells = 0
    for cell in plan["cells"]:
        cell_dir = Path(cell["output_dir"])
        if not _cell_complete(cell_dir):
            lock = (
                wait_for_gpu_experiment_lock(args.gpu_lock_path)
                if str(args.device).lower().startswith("cuda")
                else contextlib.nullcontext()
            )
            with lock:
                completed = subprocess.run(cell["command"], cwd=ROOT, check=False)
            if completed.returncode != 0:
                _atomic_json(
                    {
                        **plan,
                        "status": "failed",
                        "failed_cell": cell["key"],
                        "returncode": int(completed.returncode),
                        "completed_cells": completed_cells,
                    },
                    output / "status.json",
                )
                raise RuntimeError(f"confirmation cell failed: {cell['key']}")
        completed_cells += 1
        _atomic_json(
            {
                **plan,
                "status": "running",
                "completed_cells": completed_cells,
                "current_cell": cell["key"],
            },
            output / "status.json",
        )
    aggregate_directory(output, output)
    raw = pd.read_csv(output / "raw.csv")
    units, effects, report = summarize(raw)
    _atomic_csv(units, output / "unit_metrics.csv")
    _atomic_csv(effects, output / "paired_effects.csv")
    _atomic_json(report, output / "confirmation_report.json")
    _atomic_json(
        {
            **plan,
            "status": "complete",
            "completed_cells": completed_cells,
            "decision": report["decision"],
        },
        output / "status.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "build_plan", "load_protocol", "summarize"]
