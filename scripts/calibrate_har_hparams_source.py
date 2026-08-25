"""Target-free HAR SSAW hyperparameter calibration.

This runner evaluates only held-out source-domain streams.  It uses source
labels after inference to score clean and deterministic 50% signal-freeze
Macro-F1; labels never enter an online update.  The profile grid is fixed
before execution and the selection rule is deterministic:

1. retain profiles whose clean and corrupted F1 deltas versus the paired
   no-SSAW source baseline are both at least the configured floor;
2. require a non-trivial but bounded SSAW/raw objective-loss ratio;
3. maximize the worst-condition F1 delta; then
4. prefer the profile closest to the requested loss-ratio target.

If no profile satisfies both deltas, retain profiles with corrupted delta at
least -0.1 points and maximize clean delta.  The output is resumable: every
completed cell is atomically appended to ``raw.csv``.
"""

from __future__ import annotations

import argparse
import gc
import json
import msvcrt
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from dataloader.corruption_transforms import CORRUPTION_REGISTRY  # noqa: E402
from scripts.run_controlled_safety_benchmark import (  # noqa: E402
    deterministic_mask_fn,
)
from scripts.run_optuna_stepwise import scenario_pairs  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    BatchTransformLoader,
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


GRID_STRENGTHS = (4.0,)
GRID_AUXILIARY_WEIGHTS = (1.0, 4.0, 6.0, 8.0, 12.0)
GRID_KL_SCALES = (0.05,)
CONDITIONS = ("clean", "signal_freeze_moderate")
DEFAULT_F1_FLOOR = -0.001
DEFAULT_MIN_LOSS_RATIO = 0.03
DEFAULT_MAX_LOSS_RATIO = 0.06
DEFAULT_TARGET_LOSS_RATIO = 0.04


def _gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"GPU lock is busy: {path}") from exc
    return handle


def _profile_rows(args: argparse.Namespace) -> list[dict]:
    rows = []
    for strength in args.strengths:
        for auxiliary_weight in args.auxiliary_weights:
            for kl_scale in args.kl_scales:
                rows.append(
                    {
                        "profile": (
                            f"full_s{strength:g}_a{auxiliary_weight:g}_"
                            f"k{kl_scale:g}"
                        ),
                        "variant": "full",
                        "strength": float(strength),
                        "auxiliary_weight": float(auxiliary_weight),
                        "kl_scale": float(kl_scale),
                    }
                )
    rows.append(
        {
            "profile": "no_ssaw",
            "variant": "no_ssaw",
            "strength": 0.0,
            "auxiliary_weight": 0.0,
            "kl_scale": 0.0,
        }
    )
    return rows


def _run_cell(
    *,
    args: argparse.Namespace,
    source_domain: str,
    profile: dict,
    condition: str,
    source_seed: int,
) -> dict:
    hparams = get_hparams_class("HAR")()
    source_config = {
        **hparams.alg_hparams["NoAdap"],
        **hparams.source_train_params,
    }
    tta_config = {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
    }
    if profile["variant"] == "full":
        tta_config.update(
            {
                "ssaw_strength": profile["strength"],
                "ssaw_auxiliary_weight": profile["auxiliary_weight"],
                "ssaw_kl_scale": profile["kl_scale"],
            }
        )
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset="HAR",
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="har_source_hparam_calibration",
        # The stream order is fixed; independent calibration replicates are
        # source checkpoints, not duplicated test-time RNG seeds.
        seed=args.stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        # This profile is the paired whole-SSAW ablation.  It leaves the raw
        # pseudo-label adaptation path and both source gates unchanged.
        ablation_mode=(
            None if profile["variant"] == "full" else "no_ssaw"
        ),
    )
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        # Using source_domain as both IDs ensures this calibration never loads
        # another domain.  The source test loader is the held-out stream used
        # for post-hoc source-only scoring.
        adapter, source_model = create_tta_model(
            trainer,
            source_domain,
            source_domain,
            run_seed=args.stream_seed,
        )
        # Rebuild the held-out source stream with the deployment batch size.
        # Reusing src_test_dl directly would silently retain the source
        # training batch size (16), changing TTBN statistics and the online
        # update protocol relative to target evaluation (48 for HAR).
        trainer.trg_whole_dl = DataLoader(
            trainer.src_test_dl.dataset,
            batch_size=int(tta_config["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        if condition != "clean":
            trainer.trg_whole_dl = BatchTransformLoader(
                trainer.trg_whole_dl,
                CORRUPTION_REGISTRY["signal_freeze"],
                "moderate",
                sample_mask_fn=deterministic_mask_fn(
                    args.corruption_fraction, args.corruption_seed
                ),
                meta={
                    "corruption_type": "signal_freeze",
                    "severity": "moderate",
                },
                transform_seed=args.corruption_seed + 20_000,
            )
        metrics = trainer.calculate_metrics(adapter)
        diagnostics = dict(
            getattr(trainer, "last_batch_log_summary", {}) or {}
        )
        return {
            "dataset": "HAR",
            "source_domain": str(source_domain),
            "source_seed": int(source_seed),
            "stream_seed": int(args.stream_seed),
            "profile": profile["profile"],
            "variant": profile["variant"],
            "strength": float(profile["strength"]),
            "auxiliary_weight": float(profile["auxiliary_weight"]),
            "kl_scale": float(profile["kl_scale"]),
            "condition": condition,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "diag_raw_ce_loss": float(
                diagnostics.get("raw_ce_loss", float("nan"))
            ),
            "diag_ssaw_weighted_consistency_loss": float(
                diagnostics.get(
                    "ssaw_weighted_consistency_loss", float("nan")
                )
            ),
            "diag_ssaw_realized_consistency_ratio": float(
                diagnostics.get(
                    "ssaw_realized_consistency_ratio", float("nan")
                )
            ),
            "diag_ssaw_label_flip_rate": float(
                diagnostics.get("ssaw_label_flip_rate", float("nan"))
            ),
            "diag_ssaw_training_participation_rate": float(
                diagnostics.get(
                    "ssaw_training_participation_rate", float("nan")
                )
            ),
            "diag_ssaw_admitted_participation_rate": float(
                diagnostics.get(
                    "ssaw_admitted_participation_rate", float("nan")
                )
            ),
            "target_data_used": False,
        }
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)
        del trainer, adapter, source_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _select(
    frame: pd.DataFrame,
    *,
    f1_floor: float = DEFAULT_F1_FLOOR,
    min_loss_ratio: float = DEFAULT_MIN_LOSS_RATIO,
    max_loss_ratio: float = DEFAULT_MAX_LOSS_RATIO,
    target_loss_ratio: float = DEFAULT_TARGET_LOSS_RATIO,
) -> tuple[pd.DataFrame, dict]:
    baseline = frame[frame["variant"].eq("no_ssaw")]
    baseline = (
        baseline.groupby("condition", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "baseline_f1"})
    )
    full = frame[frame["variant"].eq("full")].copy()
    diagnostic_columns = [
        "diag_raw_ce_loss",
        "diag_ssaw_weighted_consistency_loss",
        "diag_ssaw_realized_consistency_ratio",
        "diag_ssaw_label_flip_rate",
        "diag_ssaw_training_participation_rate",
        "diag_ssaw_admitted_participation_rate",
    ]
    missing_diagnostics = sorted(
        set(diagnostic_columns) - set(full.columns)
    )
    if missing_diagnostics:
        raise ValueError(
            "Loss-ratio calibration requires diagnostics: "
            f"{missing_diagnostics}"
        )
    summary = (
        full.groupby(
            ["profile", "strength", "auxiliary_weight", "kl_scale", "condition"],
            as_index=False,
        ).agg(
            f1=("f1", "mean"),
            raw_ce_loss=("diag_raw_ce_loss", "mean"),
            weighted_ssaw_loss=(
                "diag_ssaw_weighted_consistency_loss",
                "mean",
            ),
            mean_batch_loss_ratio=(
                "diag_ssaw_realized_consistency_ratio",
                "mean",
            ),
            label_flip_rate=("diag_ssaw_label_flip_rate", "mean"),
            training_participation_rate=(
                "diag_ssaw_training_participation_rate",
                "mean",
            ),
            admitted_participation_rate=(
                "diag_ssaw_admitted_participation_rate",
                "mean",
            ),
        )
        .merge(baseline, on="condition", how="left")
    )
    summary["delta_vs_no_ssaw"] = summary["f1"] - summary["baseline_f1"]
    summary["objective_loss_ratio"] = (
        summary["weighted_ssaw_loss"]
        / summary["raw_ce_loss"].clip(lower=1e-12)
    )
    f1_wide = summary.pivot_table(
        index=["profile", "strength", "auxiliary_weight", "kl_scale"],
        columns="condition",
        values="delta_vs_no_ssaw",
    ).reset_index()
    ratio_wide = summary.pivot_table(
        index=["profile", "strength", "auxiliary_weight", "kl_scale"],
        columns="condition",
        values="objective_loss_ratio",
    ).reset_index()
    ratio_wide = ratio_wide.rename(
        columns={condition: f"{condition}_loss_ratio" for condition in CONDITIONS}
    )
    diagnostic_means = (
        summary.groupby(
            ["profile", "strength", "auxiliary_weight", "kl_scale"],
            as_index=False,
        )
        .agg(
            label_flip_rate=("label_flip_rate", "mean"),
            training_participation_rate=("training_participation_rate", "mean"),
            admitted_participation_rate=(
                "admitted_participation_rate", "mean"
            ),
            mean_batch_loss_ratio=("mean_batch_loss_ratio", "mean"),
        )
    )
    wide = f1_wide.merge(
        ratio_wide,
        on=["profile", "strength", "auxiliary_weight", "kl_scale"],
        how="left",
        validate="one_to_one",
    ).merge(
        diagnostic_means,
        on=["profile", "strength", "auxiliary_weight", "kl_scale"],
        how="left",
        validate="one_to_one",
    )
    for condition in CONDITIONS:
        if condition not in wide:
            wide[condition] = float("nan")
        ratio_column = f"{condition}_loss_ratio"
        if ratio_column not in wide:
            wide[ratio_column] = float("nan")
    wide["min_delta"] = wide[list(CONDITIONS)].min(axis=1)
    ratio_columns = [f"{condition}_loss_ratio" for condition in CONDITIONS]
    wide["loss_ratio_mean"] = wide[ratio_columns].mean(axis=1)
    wide["loss_ratio_min"] = wide[ratio_columns].min(axis=1)
    wide["loss_ratio_max"] = wide[ratio_columns].max(axis=1)
    wide["loss_ratio_target_distance"] = (
        wide["loss_ratio_mean"] - float(target_loss_ratio)
    ).abs()
    wide["meets_dual_floor"] = wide[list(CONDITIONS)].ge(
        float(f1_floor)
    ).all(axis=1)
    wide["meets_loss_ratio_band"] = (
        wide["loss_ratio_mean"].ge(float(min_loss_ratio))
        & wide["loss_ratio_mean"].le(float(max_loss_ratio))
    )
    eligible = wide[
        wide["meets_dual_floor"] & wide["meets_loss_ratio_band"]
    ]
    if not eligible.empty:
        selected = eligible.sort_values(
            ["min_delta", "loss_ratio_target_distance", "clean"],
            ascending=[False, True, False],
        ).iloc[0]
        rule = "dual_f1_floor_and_ratio_band_then_max_min_f1"
    else:
        fallback = wide[wide["meets_dual_floor"]]
        if fallback.empty:
            fallback = wide
        selected = fallback.sort_values(
            ["loss_ratio_target_distance", "min_delta", "clean"],
            ascending=[True, False, False],
        ).iloc[0]
        rule = "fallback_closest_ratio_then_max_min_f1"
    return wide, {
        "selected_profile": str(selected["profile"]),
        "selection_rule": rule,
        "floor_delta": float(f1_floor),
        "min_loss_ratio": float(min_loss_ratio),
        "max_loss_ratio": float(max_loss_ratio),
        "target_loss_ratio": float(target_loss_ratio),
        "selected_loss_ratio_mean": float(selected["loss_ratio_mean"]),
        "selected_loss_ratio_min": float(selected["loss_ratio_min"]),
        "selected_loss_ratio_max": float(selected["loss_ratio_max"]),
        "selected_clean_delta": float(selected["clean"]),
        "selected_corruption_delta": float(
            selected["signal_freeze_moderate"]
        ),
        "selected_min_delta": float(selected["min_delta"]),
        "target_data_used": False,
    }


def _completed_keys(existing: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    """Return resumable cell keys, keeping source checkpoints independent.

    ``stream_seed`` is intentionally absent: it is a fixed paired stream for
    every source checkpoint, while ``source_seed`` identifies the independent
    pre-trained source model and must be part of the cell identity.
    """
    if existing.empty:
        return set()
    if "source_seed" not in existing:
        # Do not silently treat a legacy test-time-seed file as a valid v3
        # resume state.  The old protocol reused the same source checkpoint.
        return set()
    return set(
        zip(
            existing["source_domain"].astype(str),
            existing["profile"].astype(str),
            existing["condition"].astype(str),
            existing["source_seed"].astype(int),
        )
    )


def _write_json_atomic(payload: dict, destination: Path) -> None:
    """Write a small calibration manifest without exposing partial JSON."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "calibration" / "har_hparams_source_v1"),
    )
    parser.add_argument(
        "--strengths",
        default=",".join(str(v) for v in GRID_STRENGTHS),
    )
    parser.add_argument(
        "--auxiliary-weights",
        default=",".join(str(v) for v in GRID_AUXILIARY_WEIGHTS),
    )
    parser.add_argument(
        "--kl-scales", default=",".join(str(v) for v in GRID_KL_SCALES)
    )
    parser.add_argument("--corruption-fraction", type=float, default=0.5)
    parser.add_argument("--corruption-seed", type=int, default=1)
    parser.add_argument("--f1-floor", type=float, default=DEFAULT_F1_FLOOR)
    parser.add_argument(
        "--min-loss-ratio", type=float, default=DEFAULT_MIN_LOSS_RATIO
    )
    parser.add_argument(
        "--max-loss-ratio", type=float, default=DEFAULT_MAX_LOSS_RATIO
    )
    parser.add_argument(
        "--target-loss-ratio", type=float, default=DEFAULT_TARGET_LOSS_RATIO
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    args.auxiliary_weights = [
        float(x) for x in args.auxiliary_weights.split(",") if x.strip()
    ]
    args.kl_scales = [float(x) for x in args.kl_scales.split(",") if x.strip()]
    args.source_seeds = [
        int(x) for x in str(args.source_seeds).split(",") if x.strip()
    ]
    if not args.source_seeds:
        raise ValueError("--source-seeds must not be empty")
    if not 0.0 <= args.min_loss_ratio <= args.max_loss_ratio:
        raise ValueError("loss-ratio bounds must satisfy 0 <= min <= max")
    if not args.min_loss_ratio <= args.target_loss_ratio <= args.max_loss_ratio:
        raise ValueError("target loss ratio must lie inside the requested band")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.csv"
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    completed = _completed_keys(existing)
    profiles = _profile_rows(args)
    source_domains = [str(domain) for domain, _ in scenario_pairs("HAR")]
    manifest = {
        "protocol": "HAR source-only additive SSAW weight calibration v4",
        "independent_unit": (
            "source checkpoint identified by source_seed; stream_seed is a "
            "fixed paired control"
        ),
        "source_domains": source_domains,
        "source_seeds": [int(value) for value in args.source_seeds],
        "stream_seed": int(args.stream_seed),
        "profiles": profiles,
        "conditions": list(CONDITIONS),
        "corruption_fraction": float(args.corruption_fraction),
        "corruption_seed": int(args.corruption_seed),
        "selection_rule": {
            "clean_and_corruption_floor_absolute": float(args.f1_floor),
            "loss_ratio_band": [
                float(args.min_loss_ratio),
                float(args.max_loss_ratio),
            ],
            "target_loss_ratio": float(args.target_loss_ratio),
            "priority": [
                "maximize minimum source F1 delta",
                "minimize distance to target SSAW/raw objective-loss ratio",
                "maximize clean source F1 delta",
            ],
            "fallback": "closest loss ratio among dual-F1-safe profiles",
        },
        "target_labels_used_for_selection": False,
        "expected_cells": int(
            len(source_domains)
            * len(args.source_seeds)
            * len(profiles)
            * len(CONDITIONS)
        ),
        "status": "running",
        "outputs": {
            "raw": raw_path.name,
            "summary": "summary.csv",
            "selected_profile": "selected_profile.json",
        },
    }
    _write_json_atomic(manifest, output_dir / "manifest.json")
    lock = _gpu_lock(ROOT / "results" / ".current_experiment_gpu.lock")
    try:
        rows = existing.to_dict(orient="records") if not existing.empty else []
        for source_seed in args.source_seeds:
            for source_domain, _ in scenario_pairs("HAR"):
                for profile in profiles:
                    for condition in CONDITIONS:
                        key = (
                            str(source_domain),
                            profile["profile"],
                            condition,
                            int(source_seed),
                        )
                        if key in completed:
                            continue
                        print(
                            f"[HAR source grid] source={source_domain} "
                            f"source_seed={source_seed} profile={profile['profile']} "
                            f"condition={condition}",
                            flush=True,
                        )
                        row = _run_cell(
                            args=args,
                            source_domain=source_domain,
                            profile=profile,
                            condition=condition,
                            source_seed=int(source_seed),
                        )
                        rows.append(row)
                        completed.add(key)
                        atomic_write_csv(pd.DataFrame(rows), raw_path, index=False)
        frame = pd.DataFrame(rows)
        summary, selection = _select(
            frame,
            f1_floor=args.f1_floor,
            min_loss_ratio=args.min_loss_ratio,
            max_loss_ratio=args.max_loss_ratio,
            target_loss_ratio=args.target_loss_ratio,
        )
        atomic_write_csv(summary, output_dir / "summary.csv", index=False)
        _write_json_atomic(selection, output_dir / "selected_profile.json")
        manifest.update(
            {
                "status": "complete",
                "completed_cells": int(len(frame)),
                "selected_profile": selection,
            }
        )
        _write_json_atomic(manifest, output_dir / "manifest.json")
        print(json.dumps(selection, indent=2), flush=True)
        return 0
    finally:
        try:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
