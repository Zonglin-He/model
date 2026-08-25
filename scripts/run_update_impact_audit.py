"""Local counterfactual audit of each accepted batch update on the next batch."""

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.corruption_transforms import CORRUPTION_REGISTRY
from scripts.run_controlled_safety_benchmark import (
    deterministic_mask_fn,
    parse_overrides,
)
from scripts.supplementary_utils import (
    BatchTransformLoader,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    move_data_to_device,
)


SCENARIOS = {"EEG": ("16", "1"), "HAR": ("12", "16"), "FD": ("2", "3")}


def parse_list(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


@contextmanager
def protocol_inference(model):
    """Disable autograd without switching TTBN to source running statistics."""
    del model
    with torch.inference_mode():
        yield


def primary(data):
    return data[0] if isinstance(data, (tuple, list)) else data


def raw_logits(model, data):
    x = primary(data)
    features = model.feature_extractor(x)
    if isinstance(features, (tuple, list)):
        features = features[0]
    return model.classifier(features)


def prediction_metrics(values, labels):
    predictions = values.argmax(dim=1)
    probabilities = values.softmax(dim=1)
    top = values.topk(k=min(2, values.size(1)), dim=1).values
    margins = top[:, 0] if top.size(1) == 1 else top[:, 0] - top[:, 1]
    return {
        "accuracy": float((predictions == labels).float().mean().item()),
        "f1": float(
            f1_score(
                labels.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        ),
        "cross_entropy": float(F.cross_entropy(values, labels).item()),
        "true_class_probability": float(
            probabilities.gather(1, labels[:, None]).mean().item()
        ),
        "logit_margin": float(margins.mean().item()),
        "predictions": predictions,
    }


def run_job(args, dataset, method, corruption, severity, source_seed):
    src_id, trg_id = SCENARIOS[dataset]
    stream_seed = int(args.stream_seed)
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method=method,
        backbone=args.backbone,
        exp_name=f"update_impact_{method}_{source_seed}",
        seed=stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    if getattr(args, "overrides", None):
        trainer.set_runtime_hparams(dict(args.overrides))
    tta_model = pre_trained_model = None
    try:
        tta_model, pre_trained_model = create_tta_model(
            trainer, src_id, trg_id, run_seed=stream_seed
        )
        corruption_seed = (
            int(source_seed)
            if args.corruption_seed is None
            else int(args.corruption_seed)
        )
        mask_builder = deterministic_mask_fn(
            args.corruption_fraction, corruption_seed
        )
        transformed = BatchTransformLoader(
            trainer.trg_whole_dl,
            CORRUPTION_REGISTRY[corruption],
            severity,
            sample_mask_fn=mask_builder,
            meta={"corruption_type": corruption, "severity": severity},
            transform_seed=corruption_seed + 20_000,
        )
        batches = list(transformed)
        rows = []
        online_logits = []
        online_labels = []
        for batch_index, (batch, next_batch) in enumerate(zip(batches[:-1], batches[1:])):
            data, labels, indices = batch
            next_data, next_labels, _ = next_batch
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            next_data = move_data_to_device(next_data, trainer.device)
            next_labels = next_labels.view(-1).long().to(trainer.device)
            corrupted = mask_builder(data, labels, indices, batch_index, len(batches)).to(torch.bool)

            with protocol_inference(tta_model.model):
                next_before_logits = raw_logits(tta_model.model, next_data)
            before_metrics = prediction_metrics(next_before_logits, next_labels)
            trainable_before = {
                name: parameter.detach().clone()
                for name, parameter in tta_model.model.named_parameters()
                if parameter.requires_grad
            }
            current_logits = tta_model(
                {
                    "data": data,
                    "labels": labels,
                    "meta": {
                        "corruption_mask": corrupted.tolist(),
                        "corruption_type": corruption,
                        "severity": severity,
                    },
                }
            )
            online_logits.append(current_logits.detach().cpu())
            online_labels.append(labels.detach().cpu())
            gate = dict(getattr(tta_model, "_last_gate_log", {}) or {})
            selected = gate.get(
                "active_mask", gate.get("selected_mask", torch.ones_like(labels, dtype=torch.bool))
            )
            selected = torch.as_tensor(selected, dtype=torch.bool, device=labels.device).view(-1)
            current_predictions = current_logits.detach().argmax(dim=1)
            selected_correct = selected & (current_predictions == labels)
            selected_corrupted = selected & corrupted.to(labels.device)
            selected_unsafe = selected & (
                (current_predictions != labels) | corrupted.to(labels.device)
            )

            squared_delta = 0.0
            for name, parameter in tta_model.model.named_parameters():
                if name in trainable_before:
                    squared_delta += float(
                        torch.sum((parameter.detach() - trainable_before[name]) ** 2).item()
                    )
            with protocol_inference(tta_model.model):
                next_after_logits = raw_logits(tta_model.model, next_data)
            after_metrics = prediction_metrics(next_after_logits, next_labels)
            changed = before_metrics["predictions"] != after_metrics["predictions"]
            became_correct = changed & (after_metrics["predictions"] == next_labels)
            became_wrong = changed & (before_metrics["predictions"] == next_labels)
            rows.append({
                "dataset": dataset,
                "scenario": f"{src_id}->{trg_id}",
                "method": method,
                "corruption": corruption,
                "severity": severity,
                "source_seed": source_seed,
                "stream_seed": stream_seed,
                "corruption_seed": corruption_seed,
                "batch_index": batch_index,
                "selected_count": int(selected.sum().item()),
                "selected_correct_count": int(selected_correct.sum().item()),
                "selected_incorrect_count": int((selected & ~selected_correct).sum().item()),
                "selected_corrupted_count": int(selected_corrupted.sum().item()),
                "selected_unsafe_count": int(selected_unsafe.sum().item()),
                "parameter_delta_l2": math.sqrt(squared_delta),
                "next_accuracy_before": before_metrics["accuracy"],
                "next_accuracy_after": after_metrics["accuracy"],
                "next_accuracy_delta": after_metrics["accuracy"] - before_metrics["accuracy"],
                "next_f1_before": before_metrics["f1"],
                "next_f1_after": after_metrics["f1"],
                "next_f1_delta": after_metrics["f1"] - before_metrics["f1"],
                "next_cross_entropy_before": before_metrics["cross_entropy"],
                "next_cross_entropy_after": after_metrics["cross_entropy"],
                "next_cross_entropy_improvement": before_metrics["cross_entropy"] - after_metrics["cross_entropy"],
                "next_true_probability_delta": after_metrics["true_class_probability"] - before_metrics["true_class_probability"],
                "next_logit_margin_delta": after_metrics["logit_margin"] - before_metrics["logit_margin"],
                "next_prediction_change_rate": float(changed.float().mean().item()),
                "next_became_correct_rate": float(became_correct.float().mean().item()),
                "next_became_wrong_rate": float(became_wrong.float().mean().item()),
            })

        # The final batch has no next-batch intervention outcome, but its
        # pre-update prediction belongs in the online stream metric.
        last_data, last_labels, _ = batches[-1]
        last_data = move_data_to_device(last_data, trainer.device)
        last_labels = last_labels.view(-1).long().to(trainer.device)
        final_logits = tta_model({"data": last_data, "labels": last_labels, "meta": {}})
        online_logits.append(final_logits.detach().cpu())
        online_labels.append(last_labels.detach().cpu())
        all_logits = torch.cat(online_logits)
        all_labels = torch.cat(online_labels)
        online_f1 = f1_score(
            all_labels.numpy(), all_logits.argmax(dim=1).numpy(), average="macro", zero_division=0
        )
        frame = pd.DataFrame(rows)
        frame["online_f1"] = float(online_f1)
        return frame
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--methods", default="DuSafe")
    parser.add_argument("--corruptions", default="signal_freeze,saturation")
    parser.add_argument("--severities", default="moderate,severe")
    parser.add_argument("--source_seeds", default="1,2,3")
    parser.add_argument("--stream_seed", type=int, default=42)
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Runtime key=value override applied to every requested method.",
    )
    parser.add_argument("--corruption_fraction", type=float, default=0.5)
    parser.add_argument(
        "--corruption_seed",
        type=int,
        default=None,
        help=(
            "Fixed corruption mask/transform seed shared across source "
            "checkpoints. Omit only to reproduce legacy source-seed-derived "
            "corruptions."
        ),
    )
    parser.add_argument(
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "reviewer_rerun" / "update_impact"),
    )
    args = parser.parse_args()
    args.overrides = parse_overrides(args.override)
    output_dir = ensure_dir(args.output_dir)
    raw_path = output_dir / "batch_counterfactuals.csv"
    rows = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    completed = set()
    if not rows.empty:
        completed = set(
            zip(
                rows["dataset"], rows["method"], rows["corruption"], rows["severity"],
                rows["source_seed"].astype(int), rows["stream_seed"].astype(int),
                rows.get(
                    "corruption_seed",
                    rows["source_seed"],
                ).fillna(rows["source_seed"]).astype(int),
            )
        )
    for dataset in parse_list(args.datasets):
        for method in parse_list(args.methods):
            for corruption in parse_list(args.corruptions):
                if corruption not in CORRUPTION_REGISTRY:
                    raise ValueError(f"Unknown corruption: {corruption}")
                for severity in parse_list(args.severities):
                    for source_seed in parse_list(args.source_seeds, int):
                        key = (
                            dataset, method, corruption, severity,
                            source_seed, int(args.stream_seed),
                            int(
                                source_seed
                                if args.corruption_seed is None
                                else args.corruption_seed
                            ),
                        )
                        if key in completed:
                            continue
                        print(f"[Update impact] {key}", flush=True)
                        job = run_job(
                            args, dataset, method, corruption, severity, source_seed
                        )
                        rows = pd.concat([rows, job], ignore_index=True)
                        rows.to_csv(raw_path, index=False)
                        completed.add(key)
    rows["impact"] = np.select(
        [rows["next_cross_entropy_improvement"] > 1e-9, rows["next_cross_entropy_improvement"] < -1e-9],
        ["beneficial", "harmful"],
        default="neutral",
    )
    selected_denominator = rows["selected_count"].replace(0, np.nan)
    rows["selected_safe_count"] = (
        rows["selected_count"] - rows["selected_unsafe_count"]
    )
    rows["selected_safe_fraction"] = (
        rows["selected_safe_count"] / selected_denominator
    ).fillna(0.0)
    rows["selected_unsafe_fraction"] = (
        rows["selected_unsafe_count"] / selected_denominator
    ).fillna(0.0)
    rows["selected_incorrect_fraction"] = (
        rows["selected_incorrect_count"] / selected_denominator
    ).fillna(0.0)
    rows["selected_corrupted_fraction"] = (
        rows["selected_corrupted_count"] / selected_denominator
    ).fillna(0.0)
    rows["accepted_unsafe"] = rows["selected_unsafe_count"] > 0
    rows["unsafe_fraction_band"] = pd.cut(
        rows["selected_unsafe_fraction"],
        bins=[-1e-12, 0.0, 0.10, 0.25, 1.0],
        labels=["none", "low_(0,0.10]", "medium_(0.10,0.25]", "high_(0.25,1]"],
        include_lowest=True,
    )
    rows.to_csv(raw_path, index=False)
    summary = (
        rows.groupby(
            [
                "dataset", "method", "corruption", "severity",
                "corruption_seed", "accepted_unsafe",
            ],
            as_index=False,
        )
        .agg(
            jobs=("source_seed", "nunique"),
            batches=("batch_index", "count"),
            selected_count_mean=("selected_count", "mean"),
            parameter_delta_l2_mean=("parameter_delta_l2", "mean"),
            next_ce_improvement_mean=("next_cross_entropy_improvement", "mean"),
            next_accuracy_delta_mean=("next_accuracy_delta", "mean"),
            next_f1_delta_mean=("next_f1_delta", "mean"),
            harmful_update_rate=("impact", lambda values: float((values == "harmful").mean())),
            beneficial_update_rate=("impact", lambda values: float((values == "beneficial").mean())),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    risk_band_summary = (
        rows.groupby(
            ["dataset", "corruption_seed", "unsafe_fraction_band"],
            observed=True,
            as_index=False,
        )
        .agg(
            batches=("batch_index", "count"),
            source_seeds=("source_seed", "nunique"),
            unsafe_fraction_mean=("selected_unsafe_fraction", "mean"),
            next_ce_improvement_mean=("next_cross_entropy_improvement", "mean"),
            next_accuracy_delta_mean=("next_accuracy_delta", "mean"),
            next_f1_delta_mean=("next_f1_delta", "mean"),
            harmful_update_rate=(
                "impact", lambda values: float((values == "harmful").mean())
            ),
            beneficial_update_rate=(
                "impact", lambda values: float((values == "beneficial").mean())
            ),
        )
    )
    risk_band_summary.to_csv(
        output_dir / "risk_band_summary.csv", index=False
    )

    association_rows = []
    for (dataset, corruption_seed), group in rows.groupby(
        ["dataset", "corruption_seed"]
    ):
        for risk_measure in (
            "selected_unsafe_fraction",
            "selected_incorrect_fraction",
            "selected_corrupted_fraction",
        ):
            association_rows.append({
                "dataset": dataset,
                "corruption_seed": int(corruption_seed),
                "risk_measure": risk_measure,
                "batches": int(len(group)),
                "batch_level_spearman_with_next_ce_improvement": float(
                    group[risk_measure].corr(
                        group["next_cross_entropy_improvement"],
                        method="spearman",
                    )
                ),
                "interpretation_scope": (
                    "descriptive batch-level association; batches share stream "
                    "history and are not independent significance units"
                ),
            })
    pd.DataFrame(association_rows).to_csv(
        output_dir / "risk_effect_associations.csv", index=False
    )
    manifest = {
        "intervention": "for batch t, compare batch t+1 under identical weights immediately before versus immediately after the accepted update",
        "history_control": "both counterfactuals share all adaptation history through batch t-1",
        "outcome": "next-batch accuracy, Macro-F1, cross-entropy, true-class probability, logit margin",
        "scope": "local batch-update effect; it does not identify an individual sample's Shapley contribution",
        "corruption_annotations": "deterministic masks independent of model predictions",
        "source_seeds": parse_list(args.source_seeds, int),
        "stream_seed": int(args.stream_seed),
        "corruption_seed": args.corruption_seed,
        "runtime_overrides": dict(args.overrides),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
