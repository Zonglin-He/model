"""Select a bounded HAR orientation severity using source data only.

The calibration never uses target-stream labels or target-stream metrics.  For
each of the five configured HAR source domains it evaluates a frozen source
checkpoint under a small grid of maximum SO(3) angles and records:

* view label preservation relative to the raw source prediction;
* source-label accuracy (post-hoc source calibration evidence);
* raw-to-view predictive KL;
* distance in the frozen source semantic feature space; and
* normalized input RMS.

The selected strength is the largest candidate satisfying the source-only
flip, KL, and semantic-distance constraints.  The production default is
written to ``configs/tta_hparams_new.py`` separately so the calibration
artifact remains an auditable measurement rather than a runtime target-data
tuning loop.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import msvcrt
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe import _extract_features  # noqa: E402
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.run_optuna_stepwise import scenario_pairs  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    cleanup_trainer,
    create_tta_model,
    build_trainer,
    extract_primary_tensor,
)


def _features(module, inputs):
    values = module(inputs)
    if isinstance(values, (tuple, list)):
        values = values[0]
    return values


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


def _run_source_domain(
    *,
    source_domain: tuple[str, str],
    strengths: list[float],
    args: argparse.Namespace,
) -> list[dict]:
    dataset = "HAR"
    hparams = get_hparams_class(dataset)()
    source_config = {
        **hparams.alg_hparams["NoAdap"],
        **hparams.source_train_params,
    }
    tta_config = {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
    }
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="har_orientation_source_calibration",
        seed=args.seed,
        source_seed=args.source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    adapter = source_model = None
    rows: list[dict] = []
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        # ``create_tta_model`` loads a target loader as part of the common
        # trainer setup, but no target tensor, label, prediction, or metric is
        # consumed below.  All calibration arrays come from src_train_dl.
        adapter, source_model = create_tta_model(
            trainer,
            source_domain[0],
            source_domain[1],
            run_seed=args.seed,
        )
        source_batches = [
            (
                extract_primary_tensor(data).detach().cpu(),
                labels.detach().view(-1).cpu(),
            )
            for data, labels, _ in trainer.src_train_dl
        ]
        if not source_batches:
            raise RuntimeError(f"No source batches for domain {source_domain[0]}")

        for strength in strengths:
            adapter.ssaw.strength = float(strength)
            flip_values: list[torch.Tensor] = []
            kl_values: list[torch.Tensor] = []
            semantic_values: list[torch.Tensor] = []
            rms_values: list[torch.Tensor] = []
            raw_accuracy: list[torch.Tensor] = []
            view_accuracy: list[torch.Tensor] = []
            with torch.inference_mode():
                for inputs_cpu, labels_cpu in source_batches:
                    inputs = inputs_cpu.to(args.device)
                    adapter.ssaw.clear_cached_view()
                    adapter.ssaw(
                        inputs,
                        adapter.model,
                        normalization_mean=adapter.source_normalization_mean,
                        normalization_std=adapter.source_normalization_std,
                    )
                    views = adapter.ssaw.last_view_inputs
                    metadata = adapter.ssaw.last_metadata
                    raw_prediction = adapter.ssaw.last_reference_logits.argmax(
                        dim=1
                    )
                    view_logits = []
                    for view in views:
                        with adapter.ssaw._preserved_bn_buffers(adapter.model):
                            view_logits.append(
                                adapter.model.classifier(
                                    _extract_features(adapter.model, view)
                                )
                            )
                    view_prediction = torch.stack(view_logits).argmax(dim=2)
                    flip_values.append(
                        (
                            view_prediction
                            != raw_prediction.unsqueeze(0)
                        ).float().cpu()
                    )
                    kl_values.append(
                        torch.as_tensor(
                            metadata["selected_kl_by_view"]
                        ).float().cpu()
                    )
                    source_extractor = adapter.source_semantic_feature_extractor
                    raw_features = F.normalize(
                        _features(source_extractor, inputs).flatten(1),
                        dim=1,
                    )
                    view_features = F.normalize(
                        _features(
                            source_extractor,
                            views.reshape(-1, *views.shape[2:]),
                        ).flatten(1),
                        dim=1,
                    ).reshape(views.shape[0], inputs.shape[0], -1)
                    semantic_values.append(
                        (
                            1.0
                            - (
                                view_features
                                * raw_features.unsqueeze(0)
                            ).sum(dim=-1)
                        ).cpu()
                    )
                    residual = views - inputs.unsqueeze(0)
                    rms_values.append(
                        (
                            residual.square().mean(dim=(-2, -1)).sqrt()
                            / inputs.unsqueeze(0)
                            .square()
                            .mean(dim=(-2, -1))
                            .sqrt()
                            .clamp_min(1e-8)
                        ).cpu()
                    )
                    raw_accuracy.append(
                        (raw_prediction.cpu() == labels_cpu.unsqueeze(0))
                        .float()
                        .expand(views.shape[0], -1)
                    )
                    view_accuracy.append(
                        (view_prediction.cpu() == labels_cpu.unsqueeze(0))
                        .float()
                    )

            def _aggregate(values: list[torch.Tensor]) -> list[float]:
                return torch.cat(values, dim=1).mean(dim=1).tolist()

            flip = _aggregate(flip_values)
            kl = _aggregate(kl_values)
            semantic = _aggregate(semantic_values)
            rms = _aggregate(rms_values)
            raw_acc = _aggregate(raw_accuracy)
            view_acc = _aggregate(view_accuracy)
            sample_count = int(torch.cat(flip_values, dim=1).shape[1])
            for view_index in range(len(flip)):
                rows.append(
                    {
                        "dataset": dataset,
                        "source_domain": source_domain[0],
                        "strength_deg": float(strength),
                        "view_role": (
                            "positive"
                            if view_index < len(flip) // 2
                            else "inverse"
                        ),
                        "label_flip_rate": float(flip[view_index]),
                        "kl_mean": float(kl[view_index]),
                        "semantic_distance_mean": float(
                            semantic[view_index]
                        ),
                        "relative_rms_mean": float(rms[view_index]),
                        "raw_source_accuracy": float(raw_acc[view_index]),
                        "view_source_accuracy": float(view_acc[view_index]),
                        "samples": sample_count,
                    }
                )
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)
        del trainer, adapter, source_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _select(
    frame: pd.DataFrame,
    *,
    max_flip: float,
    max_kl: float,
    max_semantic: float,
) -> dict:
    grouped = (
        frame.groupby("strength_deg", sort=True)[
            [
                "label_flip_rate",
                "kl_mean",
                "semantic_distance_mean",
                "relative_rms_mean",
                "raw_source_accuracy",
                "view_source_accuracy",
            ]
        ]
        .mean()
        .reset_index()
    )
    grouped["meets_constraints"] = (
        grouped["label_flip_rate"].le(max_flip)
        & grouped["kl_mean"].le(max_kl)
        & grouped["semantic_distance_mean"].le(max_semantic)
    )
    eligible = grouped[grouped["meets_constraints"]]
    selected = (
        eligible.sort_values("strength_deg").iloc[-1]
        if not eligible.empty
        else grouped.sort_values(
            ["label_flip_rate", "kl_mean", "semantic_distance_mean"]
        ).iloc[0]
    )
    return {
        "selected_strength_deg": float(selected["strength_deg"]),
        "max_label_flip": float(max_flip),
        "max_kl": float(max_kl),
        "max_semantic_distance": float(max_semantic),
        "selected_metrics": {
            key: float(selected[key])
            for key in (
                "label_flip_rate",
                "kl_mean",
                "semantic_distance_mean",
                "relative_rms_mean",
                "raw_source_accuracy",
                "view_source_accuracy",
            )
        },
        "candidate_summary": grouped.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "calibration" / "har_orientation_source_v1"),
    )
    parser.add_argument(
        "--strengths",
        default="0,1,2,4,6,8,10,12,15",
        help="Maximum total SO(3) angles in degrees.",
    )
    parser.add_argument("--max-label-flip", type=float, default=0.01)
    parser.add_argument("--max-kl", type=float, default=0.02)
    parser.add_argument("--max-semantic-distance", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.strengths = [
        float(value.strip())
        for value in str(args.strengths).split(",")
        if value.strip()
    ]
    if not args.strengths or any(value < 0.0 or value > 90.0 for value in args.strengths):
        raise ValueError("--strengths must contain values in [0, 90]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = _gpu_lock(ROOT / "results" / ".current_experiment_gpu.lock")
    try:
        rows: list[dict] = []
        for source_domain in scenario_pairs("HAR"):
            print(
                f"[HAR source calibration] source={source_domain[0]}",
                flush=True,
            )
            rows.extend(
                _run_source_domain(
                    source_domain=source_domain,
                    strengths=args.strengths,
                    args=args,
                )
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "source_calibration.csv", index=False)
        selection = _select(
            frame,
            max_flip=args.max_label_flip,
            max_kl=args.max_kl,
            max_semantic=args.max_semantic_distance,
        )
        selection.update(
            {
                "dataset": "HAR",
                "source_seed": int(args.source_seed),
                "source_domains": [pair[0] for pair in scenario_pairs("HAR")],
                "target_labels_used": False,
                "target_metrics_used": False,
                "orientation_definition": "bounded axis-angle SO(3); strength is maximum total angle in degrees",
            }
        )
        (output_dir / "selected_strength.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
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

