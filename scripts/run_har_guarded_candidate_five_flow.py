"""HAR five-flow validation, coordinate tuning, and coupling ablation.

Only HAR is in scope.  One dataset-level profile is shared by all five flows.
Target labels are never passed to the online adapter, but Macro-F1 from the
same five target flows is used for parameter selection.  The outputs are
therefore explicitly target-selected/descriptive rather than confirmatory.

Each GPU cell runs in an isolated child process and acquires the repository
GPU lock.  A completed ``summary.json`` is published only after its CSV
artifacts, allowing safe resume after OOM, native crashes, or power loss.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe_guarded_candidate import (  # noqa: E402
    GUARDED_CANDIDATE_VARIANTS,
)
from configs.data_model_configs import HAR  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_har_guarded_candidate_diagnostic import (  # noqa: E402
    COUNTERFACTUAL_COLUMNS,
    _atomic_csv,
    _atomic_json,
    _run_variant,
)


PROTOCOL = "har_guarded_candidate_five_flow_v1_fixed_source_anchor_admission"
HAR_FLOWS = tuple((str(source), str(target)) for source, target in HAR.scenarios)
PAIR_VARIANTS = ("Full", "No-SSAW")
ABLATION_VARIANTS = (
    "Full",
    "No-SSAW",
    "No-Anchor-Admission",
    "No-Admission-No-SSAW",
    "Confidence-Only-Admission",
    "Semantic-Only-Admission",
)
DEFAULT_SOURCE_SEEDS = (1, 2, 3)
DEFAULT_STREAM_SEED = 42
DEFAULT_OUTPUT_DIR = ROOT / "results" / "har_guarded_candidate_five_flow"
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"

BASELINE_PROFILE = {
    "learning_rate": 3.325e-4,
    "steps": 23,
    "batch_size": 48,
    "ssaw_auxiliary_weight": 1.25,
    "ssaw_strength": 4.0,
    "candidate_guard_fraction": 0.25,
    "candidate_backtracking_scale": 0.5,
}

STEP_VALUES = (1, 2, 4, 8)
LEARNING_RATE_VALUES = (1e-5, 3e-5, 1e-4, 3.325e-4, 1e-3, 3e-3)
BATCH_SIZE_VALUES = (24, 48, 96)
SSAW_WEIGHT_VALUES = (0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0)


def _hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug_float(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def _parse_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = value.split(",")
    else:
        pieces = value
    parsed = tuple(int(piece) for piece in pieces if str(piece).strip())
    if not parsed:
        raise ValueError("at least one source seed is required")
    return parsed


def _profile(profile_id: str, base: Mapping[str, object], **changes) -> dict:
    values = dict(base)
    values.pop("profile_id", None)
    values.update(changes)
    return {"profile_id": str(profile_id), **values}


def _profile_hparams(profile: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "learning_rate",
        "steps",
        "batch_size",
        "ssaw_auxiliary_weight",
        "ssaw_strength",
        "candidate_guard_fraction",
        "candidate_backtracking_scale",
    )
    return {key: profile[key] for key in keys}


def _cell_dir(
    output_dir: Path,
    stage: str,
    profile_id: str,
    scenario: tuple[str, str],
    source_seed: int,
    variant: str,
) -> Path:
    safe_variant = variant.lower().replace("-", "_")
    return (
        output_dir
        / "cells"
        / stage
        / profile_id
        / f"flow_{scenario[0]}_to_{scenario[1]}"
        / f"source_seed_{int(source_seed)}"
        / safe_variant
    )


def _cell_signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "stage": spec["stage"],
        "profile": spec["profile"],
        "scenario": spec["scenario"],
        "source_seed": int(spec["source_seed"]),
        "stream_seed": int(spec["stream_seed"]),
        "variant": spec["variant"],
        "condition": "clean",
        "max_batches": spec.get("max_batches"),
        "pretrain_cache_dir": spec["pretrain_cache_dir"],
        "backbone": spec["backbone"],
        "target_labels_used_for_online_decision": False,
    }


def _cell_complete(cell_dir: Path, signature_hash: str) -> bool:
    summary_path = cell_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "ok"
        and payload.get("signature_hash") == signature_hash
        and (cell_dir / "batch_diagnostics.csv").is_file()
        and (cell_dir / "rejected_counterfactuals.csv").is_file()
    )


def _worker(spec_path: Path) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    signature = _cell_signature(spec)
    signature_hash = _hash(signature)
    if _cell_complete(cell_dir, signature_hash):
        return 0
    scenario = tuple(str(value) for value in spec["scenario"])
    worker_args = SimpleNamespace(
        data_path=spec["data_path"],
        device=spec["device"],
        backbone=spec["backbone"],
        stream_seed=int(spec["stream_seed"]),
        pretrain_cache_dir=Path(spec["pretrain_cache_dir"]),
        steps=None,
        batch_size=None,
        corruption_seed=1,
        max_batches=spec.get("max_batches"),
    )
    result = None
    try:
        lock_context = (
            wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
            if str(spec["device"]).lower().startswith("cuda")
            else None
        )
        if lock_context is None:
            result, batch_rows, counterfactual_rows = _run_variant(
                args=worker_args,
                condition="clean",
                source_seed=int(spec["source_seed"]),
                variant=str(spec["variant"]),
                scenario=scenario,
                hparam_overrides=_profile_hparams(spec["profile"]),
            )
        else:
            with lock_context:
                result, batch_rows, counterfactual_rows = _run_variant(
                    args=worker_args,
                    condition="clean",
                    source_seed=int(spec["source_seed"]),
                    variant=str(spec["variant"]),
                    scenario=scenario,
                    hparam_overrides=_profile_hparams(spec["profile"]),
                )
        result = {
            **result,
            "protocol": PROTOCOL,
            "stage": spec["stage"],
            "profile_id": spec["profile"]["profile_id"],
            "signature_hash": signature_hash,
            "target_labels_used_for_online_decision": False,
            "target_labels_used_for_parameter_selection": True,
            "evaluation_partition": "target_selected_evaluation",
            "confirmatory": False,
        }
        for row in batch_rows:
            row.update(
                {
                    "protocol": PROTOCOL,
                    "stage": spec["stage"],
                    "profile_id": spec["profile"]["profile_id"],
                }
            )
        for row in counterfactual_rows:
            row.update(
                {
                    "protocol": PROTOCOL,
                    "stage": spec["stage"],
                    "profile_id": spec["profile"]["profile_id"],
                }
            )
        _atomic_csv(pd.DataFrame(batch_rows), cell_dir / "batch_diagnostics.csv")
        counterfactual_columns = tuple(COUNTERFACTUAL_COLUMNS) + (
            "protocol",
            "stage",
            "profile_id",
        )
        _atomic_csv(
            pd.DataFrame(counterfactual_rows, columns=counterfactual_columns),
            cell_dir / "rejected_counterfactuals.csv",
        )
        _atomic_json(result, cell_dir / "summary.json")
        return 0
    except BaseException as exc:
        failure = {
            "status": "failed",
            "protocol": PROTOCOL,
            "stage": spec.get("stage"),
            "profile_id": spec.get("profile", {}).get("profile_id"),
            "scenario": f"{scenario[0]}->{scenario[1]}",
            "source_seed": int(spec["source_seed"]),
            "variant": spec["variant"],
            "signature_hash": signature_hash,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower(),
        }
        _atomic_json(failure, cell_dir / "summary.json")
        print(f"worker failed: {failure}", file=sys.stderr, flush=True)
        return 1
    finally:
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _write_status(output_dir: Path, **values) -> None:
    payload = {
        "protocol": PROTOCOL,
        "updated_at_unix": time.time(),
        **values,
    }
    _atomic_json(payload, output_dir / "status.json")


def _run_cell(spec: Mapping[str, object], *, fail_fast: bool) -> dict:
    cell_dir = Path(spec["cell_dir"])
    signature_hash = _hash(_cell_signature(spec))
    if not _cell_complete(cell_dir, signature_hash):
        cell_dir.mkdir(parents=True, exist_ok=True)
        spec_path = cell_dir / "worker_spec.json"
        _atomic_json(dict(spec), spec_path)
        log_path = cell_dir / "worker.log"
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)]
        with log_path.open("a", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 and fail_fast:
            raise RuntimeError(
                f"HAR worker failed with return code {completed.returncode}: {cell_dir}"
            )
    summary_path = cell_dir / "summary.json"
    if not summary_path.is_file():
        return {
            "status": "failed",
            "error": "worker did not publish summary.json",
            "cell_dir": str(cell_dir),
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _evaluate_profiles(
    *,
    args,
    output_dir: Path,
    stage: str,
    profiles: Sequence[Mapping[str, object]],
    source_seeds: Sequence[int],
    variants: Sequence[str],
) -> pd.DataFrame:
    expected = len(profiles) * len(HAR_FLOWS) * len(source_seeds) * len(variants)
    completed_count = 0
    rows: list[dict] = []
    for profile in profiles:
        for scenario in HAR_FLOWS:
            for source_seed in source_seeds:
                for variant in variants:
                    if variant not in GUARDED_CANDIDATE_VARIANTS:
                        raise ValueError(f"unknown ablation variant {variant!r}")
                    cell_dir = _cell_dir(
                        output_dir,
                        stage,
                        str(profile["profile_id"]),
                        scenario,
                        source_seed,
                        variant,
                    )
                    spec = {
                        "protocol": PROTOCOL,
                        "stage": stage,
                        "profile": dict(profile),
                        "scenario": list(scenario),
                        "source_seed": int(source_seed),
                        "stream_seed": int(args.stream_seed),
                        "variant": variant,
                        "data_path": str(Path(args.data_path).resolve()),
                        "device": args.device,
                        "backbone": args.backbone,
                        "pretrain_cache_dir": str(Path(args.pretrain_cache_dir).resolve()),
                        "gpu_lock_path": str(Path(args.gpu_lock_path).resolve()),
                        "max_batches": args.max_batches,
                        "cell_dir": str(cell_dir.resolve()),
                    }
                    _write_status(
                        output_dir,
                        status="running",
                        phase=stage,
                        completed_cells=completed_count,
                        expected_cells=expected,
                        current_cell={
                            "profile": profile["profile_id"],
                            "scenario": f"{scenario[0]}->{scenario[1]}",
                            "source_seed": int(source_seed),
                            "variant": variant,
                        },
                    )
                    result = _run_cell(spec, fail_fast=args.fail_fast)
                    completed_count += 1
                    rows.append(result)
                    if result.get("status") != "ok" and args.fail_fast:
                        raise RuntimeError(f"failed HAR cell: {result}")
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, output_dir / f"{stage}_raw.csv")
    failed = frame.loc[~frame["status"].eq("ok")]
    if not failed.empty:
        raise RuntimeError(
            f"HAR stage {stage!r} has {len(failed)} failed cells; "
            f"resume after inspecting {output_dir / (stage + '_raw.csv')}"
        )
    return frame


def _validate_pairing(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame.loc[frame["status"].eq("ok")].copy()
    required = {"Full", "No-SSAW"}
    keys = ["profile_id", "scenario", "source_seed"]
    paired_rows = []
    for key, group in ok.groupby(keys, dropna=False):
        variants = set(group["variant"].astype(str))
        if not required.issubset(variants):
            continue
        selected = group.set_index("variant")
        full = selected.loc["Full"]
        no_ssaw = selected.loc["No-SSAW"]
        if str(full["source_model_sha256"]) != str(no_ssaw["source_model_sha256"]):
            raise RuntimeError(f"source checkpoint mismatch in paired cell {key}")
        paired_rows.append(
            {
                "profile_id": key[0],
                "scenario": key[1],
                "source_seed": int(key[2]),
                "full_f1": float(full["post_stream_macro_f1"]),
                "no_ssaw_f1": float(no_ssaw["post_stream_macro_f1"]),
                "full_minus_no_ssaw": float(full["post_stream_macro_f1"])
                - float(no_ssaw["post_stream_macro_f1"]),
                "full_admission_rate": float(
                    full["fixed_source_anchor_admission_rate"]
                ),
                "no_ssaw_admission_rate": float(
                    no_ssaw["fixed_source_anchor_admission_rate"]
                ),
                "full_final_skip_rate": float(full["final_skip_rate"]),
                "no_ssaw_final_skip_rate": float(no_ssaw["final_skip_rate"]),
            }
        )
    return pd.DataFrame(paired_rows)


def _profile_summary(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    paired = _validate_pairing(frame)
    if paired.empty:
        _atomic_csv(paired, output_path)
        return paired
    summary = (
        paired.groupby("profile_id", dropna=False)
        .agg(
            paired_units=("full_minus_no_ssaw", "size"),
            full_f1_mean=("full_f1", "mean"),
            full_f1_std=("full_f1", "std"),
            no_ssaw_f1_mean=("no_ssaw_f1", "mean"),
            no_ssaw_f1_std=("no_ssaw_f1", "std"),
            full_minus_no_ssaw_mean=("full_minus_no_ssaw", "mean"),
            full_minus_no_ssaw_std=("full_minus_no_ssaw", "std"),
            positive_pair_fraction=("full_minus_no_ssaw", lambda x: float((x > 0).mean())),
            worst_pair_delta=("full_minus_no_ssaw", "min"),
            full_admission_rate=("full_admission_rate", "mean"),
            full_final_skip_rate=("full_final_skip_rate", "mean"),
        )
        .reset_index()
    )
    _atomic_csv(summary, output_path)
    return summary


def _select_profile(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        raise RuntimeError("no complete paired profiles are available for selection")
    maximum_full = float(summary["full_f1_mean"].max())
    shortlist = summary.loc[
        summary["full_f1_mean"].ge(maximum_full - 0.005)
    ].copy()
    selected = shortlist.sort_values(
        ["full_minus_no_ssaw_mean", "full_f1_mean", "positive_pair_fraction"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "profile_id": str(selected["profile_id"]),
        "selection_rule": (
            "retain profiles within 0.5 percentage points of the best Full "
            "Macro-F1, then maximize paired Full-minus-No-SSAW mean"
        ),
        "best_full_f1_in_stage": maximum_full,
        "selected_full_f1_mean": float(selected["full_f1_mean"]),
        "selected_no_ssaw_f1_mean": float(selected["no_ssaw_f1_mean"]),
        "selected_full_minus_no_ssaw_mean": float(
            selected["full_minus_no_ssaw_mean"]
        ),
        "selected_positive_pair_fraction": float(
            selected["positive_pair_fraction"]
        ),
    }


def _lookup(profiles: Sequence[Mapping[str, object]], profile_id: str) -> dict:
    for profile in profiles:
        if str(profile["profile_id"]) == str(profile_id):
            return dict(profile)
    raise KeyError(profile_id)


def _top_finalists(summary: pd.DataFrame, count: int = 3) -> list[str]:
    best_full = float(summary["full_f1_mean"].max())
    near = summary.loc[summary["full_f1_mean"].ge(best_full - 0.015)].copy()
    ranked = near.sort_values(
        ["full_minus_no_ssaw_mean", "full_f1_mean"],
        ascending=[False, False],
    )
    identifiers = list(ranked["profile_id"].astype(str).head(count))
    best_id = str(
        summary.sort_values("full_f1_mean", ascending=False).iloc[0]["profile_id"]
    )
    if best_id not in identifiers:
        identifiers = ([best_id] + identifiers)[:count]
    return identifiers


def _ablation_summary(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ok = frame.loc[frame["status"].eq("ok")].copy()
    summary = (
        ok.groupby("variant", dropna=False)
        .agg(
            units=("post_stream_macro_f1", "size"),
            macro_f1_mean=("post_stream_macro_f1", "mean"),
            macro_f1_std=("post_stream_macro_f1", "std"),
            admission_rate_mean=("fixed_source_anchor_admission_rate", "mean"),
            final_skip_rate_mean=("final_skip_rate", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    _atomic_csv(summary, output_dir / "ablation_summary.csv")
    values = summary.set_index("variant")["macro_f1_mean"].to_dict()
    needed = set(ABLATION_VARIANTS)
    if not needed.issubset(values):
        missing = sorted(needed - set(values))
        raise RuntimeError(f"ablation is incomplete; missing {missing}")
    coupling = pd.DataFrame(
        [
            {
                "effect": "SSAW_given_anchor_admission",
                "macro_f1_delta": values["Full"] - values["No-SSAW"],
            },
            {
                "effect": "SSAW_without_anchor_admission",
                "macro_f1_delta": values["No-Anchor-Admission"]
                - values["No-Admission-No-SSAW"],
            },
            {
                "effect": "anchor_admission_given_SSAW",
                "macro_f1_delta": values["Full"]
                - values["No-Anchor-Admission"],
            },
            {
                "effect": "anchor_admission_without_SSAW",
                "macro_f1_delta": values["No-SSAW"]
                - values["No-Admission-No-SSAW"],
            },
            {
                "effect": "admission_x_SSAW_interaction",
                "macro_f1_delta": values["Full"]
                - values["No-SSAW"]
                - values["No-Anchor-Admission"]
                + values["No-Admission-No-SSAW"],
            },
            {
                "effect": "semantic_branch_within_admission",
                "macro_f1_delta": values["Full"]
                - values["Confidence-Only-Admission"],
            },
            {
                "effect": "confidence_branch_within_admission",
                "macro_f1_delta": values["Full"]
                - values["Semantic-Only-Admission"],
            },
        ]
    )
    _atomic_csv(coupling, output_dir / "coupling_effects.csv")
    return summary


def run(args) -> None:
    if tuple(HAR_FLOWS) != (
        ("2", "11"),
        ("6", "23"),
        ("7", "13"),
        ("9", "18"),
        ("12", "16"),
    ):
        raise RuntimeError(f"HAR flow registry drifted: {HAR_FLOWS}")
    source_seeds = _parse_ints(args.source_seeds)
    if any(seed not in DEFAULT_SOURCE_SEEDS for seed in source_seeds):
        raise ValueError("formal HAR source seeds must be drawn from 1,2,3")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "dataset": "HAR",
        "flows": [f"{source}->{target}" for source, target in HAR_FLOWS],
        "dataset_level_profile_shared_across_flows": True,
        "source_seeds": list(source_seeds),
        "tuning_screen_source_seeds": [1],
        "stream_seed": int(args.stream_seed),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "evaluation_partition": "target_selected_evaluation",
        "confirmatory": False,
        "selection_non_inferiority_band": 0.005,
        "coordinate_order": ["steps", "learning_rate", "batch_size", "ssaw_auxiliary_weight"],
        "baseline_profile": BASELINE_PROFILE,
        "search_space": {
            "steps": list(STEP_VALUES),
            "learning_rate": list(LEARNING_RATE_VALUES),
            "batch_size": list(BATCH_SIZE_VALUES),
            "ssaw_auxiliary_weight": list(SSAW_WEIGHT_VALUES),
        },
        "ablation_variants": list(ABLATION_VARIANTS),
    }
    _atomic_json(manifest, output_dir / "manifest.json")

    baseline = _profile("baseline_current", BASELINE_PROFILE)
    validation_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="validation_current",
        profiles=[baseline],
        source_seeds=source_seeds,
        variants=PAIR_VARIANTS,
    )
    _profile_summary(validation_raw, output_dir / "validation_current_summary.csv")
    if args.stop_after == "validation":
        _write_status(
            output_dir,
            status="partial_complete",
            phase="validation_complete",
            stop_after="validation",
        )
        return

    current = dict(baseline)
    stage_selections = []

    step_profiles = [
        _profile(f"steps_{value}", current, steps=value)
        for value in STEP_VALUES
    ]
    step_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="tune_steps",
        profiles=step_profiles,
        source_seeds=(1,),
        variants=PAIR_VARIANTS,
    )
    step_summary = _profile_summary(step_raw, output_dir / "tune_steps_summary.csv")
    step_selection = _select_profile(step_summary)
    stage_selections.append({"stage": "steps", **step_selection})
    current = _lookup(step_profiles, step_selection["profile_id"])

    lr_profiles = [
        _profile(
            f"lr_{_slug_float(value)}",
            current,
            learning_rate=value,
        )
        for value in LEARNING_RATE_VALUES
    ]
    lr_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="tune_learning_rate",
        profiles=lr_profiles,
        source_seeds=(1,),
        variants=PAIR_VARIANTS,
    )
    lr_summary = _profile_summary(
        lr_raw, output_dir / "tune_learning_rate_summary.csv"
    )
    lr_selection = _select_profile(lr_summary)
    stage_selections.append({"stage": "learning_rate", **lr_selection})
    current = _lookup(lr_profiles, lr_selection["profile_id"])

    batch_profiles = [
        _profile(f"batch_{value}", current, batch_size=value)
        for value in BATCH_SIZE_VALUES
    ]
    batch_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="tune_batch_size",
        profiles=batch_profiles,
        source_seeds=(1,),
        variants=PAIR_VARIANTS,
    )
    batch_summary = _profile_summary(
        batch_raw, output_dir / "tune_batch_size_summary.csv"
    )
    batch_selection = _select_profile(batch_summary)
    stage_selections.append({"stage": "batch_size", **batch_selection})
    current = _lookup(batch_profiles, batch_selection["profile_id"])

    weight_profiles = [
        _profile(
            f"weight_{_slug_float(value)}",
            current,
            ssaw_auxiliary_weight=value,
        )
        for value in SSAW_WEIGHT_VALUES
    ]
    weight_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="tune_ssaw_weight",
        profiles=weight_profiles,
        source_seeds=(1,),
        variants=PAIR_VARIANTS,
    )
    weight_summary = _profile_summary(
        weight_raw, output_dir / "tune_ssaw_weight_summary.csv"
    )
    weight_selection = _select_profile(weight_summary)
    stage_selections.append({"stage": "ssaw_auxiliary_weight", **weight_selection})

    finalist_ids = _top_finalists(weight_summary, count=3)
    finalist_profiles = [_lookup(weight_profiles, value) for value in finalist_ids]
    finalist_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="validate_finalists_three_seeds",
        profiles=finalist_profiles,
        source_seeds=source_seeds,
        variants=PAIR_VARIANTS,
    )
    finalist_summary = _profile_summary(
        finalist_raw, output_dir / "validate_finalists_three_seeds_summary.csv"
    )
    final_selection = _select_profile(finalist_summary)
    selected_profile = _lookup(finalist_profiles, final_selection["profile_id"])
    selection_payload = {
        "protocol": PROTOCOL,
        "selected_profile": selected_profile,
        "final_selection": final_selection,
        "coordinate_stage_selections": stage_selections,
        "target_labels_used_for_parameter_selection": True,
        "evaluation_partition": "target_selected_evaluation",
        "confirmatory": False,
    }
    _atomic_json(selection_payload, output_dir / "selected_profile.json")
    if args.stop_after == "tuning":
        _write_status(
            output_dir,
            status="partial_complete",
            phase="tuning_complete",
            stop_after="tuning",
            selected_profile=selected_profile,
        )
        return

    ablation_profile = {
        **selected_profile,
        "profile_id": "selected_profile_ablation",
    }
    ablation_raw = _evaluate_profiles(
        args=args,
        output_dir=output_dir,
        stage="ablation_five_flow_three_seed",
        profiles=[ablation_profile],
        source_seeds=source_seeds,
        variants=ABLATION_VARIANTS,
    )
    _ablation_summary(ablation_raw, output_dir)
    _write_status(
        output_dir,
        status="complete",
        phase="complete",
        selected_profile=selected_profile,
        expected_flows=5,
        final_source_seeds=list(source_seeds),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=False, default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK_PATH)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("validation", "tuning", "complete"),
        default="complete",
    )
    parser.add_argument("--worker-spec", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_spec is not None:
        return _worker(args.worker_spec)
    if args.max_batches is not None and int(args.max_batches) < 1:
        raise ValueError("--max-batches must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
