"""Compare each production SSAW update with its local raw-only counterfactual.

For every inner step, both branches start from identical model and optimizer
state.  The Full branch uses DuSafe's current raw anchor plus physical-view
auxiliary gradient; the shadow branch uses the same raw confidence-and-semantic
admission and raw loss without SSAW.  Full remains the online history.  Target
labels are used only for post-hoc outcomes and never for either update.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe import DuSafe, _extract_features  # noqa: E402
from scripts.diagnose_ssaw_pipeline import FORMAL_MANIFESTS  # noqa: E402
from scripts.run_optuna_stepwise import scenario_pairs  # noqa: E402
from scripts.run_ssaw_internal_ablation import load_json  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    move_data_to_device,
)


def _primary(data):
    return data[0] if isinstance(data, (tuple, list)) else data


def _clone_state(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _restore(model, optimizer, model_state, optimizer_state) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def _logits(model, data) -> torch.Tensor:
    return model(_primary(data))


def _outcomes(full_logits, base_logits, labels, prefix: str) -> dict:
    full_predictions = full_logits.argmax(dim=1)
    base_predictions = base_logits.argmax(dim=1)
    full_correct = full_predictions.eq(labels)
    base_correct = base_predictions.eq(labels)
    class_indices = list(range(full_logits.size(1)))
    labels_cpu = labels.detach().cpu().numpy()
    full_macro_f1 = f1_score(
        labels_cpu,
        full_predictions.detach().cpu().numpy(),
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    base_macro_f1 = f1_score(
        labels_cpu,
        base_predictions.detach().cpu().numpy(),
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    return {
        f"{prefix}_full_ce": float(F.cross_entropy(full_logits, labels).item()),
        f"{prefix}_base_ce": float(F.cross_entropy(base_logits, labels).item()),
        f"{prefix}_ce_improvement_full_vs_base": float(
            F.cross_entropy(base_logits, labels).item()
            - F.cross_entropy(full_logits, labels).item()
        ),
        f"{prefix}_accuracy_delta_full_vs_base": float(
            full_correct.float().mean().item()
            - base_correct.float().mean().item()
        ),
        f"{prefix}_macro_f1_delta_full_vs_base": float(
            full_macro_f1 - base_macro_f1
        ),
        f"{prefix}_prediction_difference_rate": float(
            full_predictions.ne(base_predictions).float().mean().item()
        ),
        f"{prefix}_full_only_correct_count": int(
            (full_correct & (~base_correct)).sum().item()
        ),
        f"{prefix}_base_only_correct_count": int(
            (base_correct & (~full_correct)).sum().item()
        ),
    }


def _parameter_distance(left, right) -> float:
    squared = 0.0
    for name in left:
        if torch.is_floating_point(left[name]):
            squared += float((left[name] - right[name]).pow(2).sum().item())
    return squared**0.5


def _base_forward_and_adapt(adapter, data):
    """Execute the no-SSAW raw gate/update from the current shadow state."""

    raw_inputs = _primary(data)
    bn_snapshot = adapter._snapshot_bn_buffers(adapter.model)
    with torch.no_grad():
        raw_features = _extract_features(adapter.model, raw_inputs)
        raw_logits = adapter.model.classifier(raw_features)
        pseudo_labels = raw_logits.argmax(dim=1)
        raw_nll = -raw_logits.log_softmax(dim=1).gather(
            1, pseudo_labels[:, None]
        ).squeeze(1)
        semantic_mask, _, _ = DuSafe._source_semantic_decision(
            adapter, raw_inputs, pseudo_labels
        )
    adapter._restore_bn_buffers(bn_snapshot)
    confidence_mask = (
        raw_nll.le(adapter.confidence_nll_threshold)
        if adapter.enable_confidence_gate
        else torch.ones_like(semantic_mask)
    )
    admission_mask = confidence_mask & semantic_mask
    raw_train_logits = adapter.model.classifier(
        _extract_features(adapter.model, raw_inputs)
    )
    if admission_mask.any():
        loss = F.cross_entropy(
            raw_train_logits[admission_mask],
            pseudo_labels.detach()[admission_mask],
        )
    else:
        loss = raw_inputs.sum() * 0.0
    update = adapter._apply_update(
        adapter.model,
        adapter.optimizer,
        loss,
        admission_mask,
    )
    return pseudo_labels, admission_mask, update


def _prepare(args):
    dataset = args.dataset.upper()
    requested = tuple(args.scenario.replace("->", ",").split(","))
    requested = tuple(piece.strip() for piece in requested)
    if len(requested) != 2 or requested not in scenario_pairs(dataset):
        raise ValueError(f"Unknown {dataset} scenario: {args.scenario}")

    tuning_path = args.tuning_dir / dataset / "state.json"
    tuning_state = load_json(tuning_path) if tuning_path.is_file() else None
    source_seed = int(
        tuning_state.get("signature", {}).get("source_seed", 1)
        if tuning_state is not None
        else 1
    )
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="ssaw_update_counterfactual",
        seed=args.test_time_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    if tuning_state is not None:
        trainer.source_hparams.update(dict(tuning_state["source_config"]))

    formal_path = FORMAL_MANIFESTS.get(dataset)
    if formal_path is not None and formal_path.is_file():
        formal_manifest = load_json(formal_path)
        tta_config = dict(formal_manifest["effective_tta_configs"][dataset])
        args.config_source = str(formal_path)
    else:
        # A newly integrated dataset has neither an Optuna state nor an
        # archived formal-ablation manifest yet.  Its checked-in dataset-level
        # configuration is still a valid diagnostic starting point.  Record
        # the fallback explicitly so this probe cannot be mistaken for a
        # tuned/formal result.
        tta_config = dict(trainer.hparams)
        args.config_source = "dataset_default"
    trainer.set_runtime_hparams(tta_config)
    adapter, source_model = create_tta_model(
        trainer,
        requested[0],
        requested[1],
        run_seed=args.test_time_seed,
    )
    return dataset, requested, source_seed, trainer, adapter, source_model


def run_step_counterfactual(args) -> pd.DataFrame:
    dataset, _, source_seed, trainer, adapter, source_model = _prepare(args)
    try:
        batches = list(trainer.trg_whole_dl)
        rows = []
        for batch_index, batch in enumerate(batches):
            data, labels, target_indices = batch
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            next_data = next_labels = None
            if batch_index + 1 < len(batches):
                next_data = move_data_to_device(
                    batches[batch_index + 1][0], trainer.device
                )
                next_labels = (
                    batches[batch_index + 1][1]
                    .view(-1)
                    .long()
                    .to(trainer.device)
                )
            model_inputs = {
                "data": data,
                "labels": labels,
                "meta": {"trg_idx": torch.as_tensor(target_indices).view(-1).tolist()},
            }

            for inner_step in range(adapter.steps):
                model_before = _clone_state(adapter.model)
                optimizer_before = copy.deepcopy(adapter.optimizer.state_dict())

                adapter.forward_and_adapt(
                    model_inputs,
                    adapter.model,
                    adapter.optimizer,
                    target_indices,
                )
                state = adapter._last_gate_log
                batch_log = adapter._last_batch_log
                pseudo_labels = state["pseudo_labels"].to(labels.device)
                base_mask = state["base_admission_mask"].to(labels.device).bool()
                full_mask = state["admission_mask"].to(labels.device).bool()
                veto = state["ssaw_veto_mask"].to(labels.device).bool()
                consistency = state["ssaw_consistency_mask"].to(
                    labels.device
                ).bool()
                # The current production algorithm has no rescue path and
                # keeps raw admission identical with and without SSAW.
                rescue = torch.zeros_like(veto)

                model_full = _clone_state(adapter.model)
                optimizer_full = copy.deepcopy(adapter.optimizer.state_dict())
                full_current_logits = _logits(adapter.model, data)
                full_next_logits = (
                    None if next_data is None else _logits(adapter.model, next_data)
                )

                _restore(
                    adapter.model,
                    adapter.optimizer,
                    model_before,
                    optimizer_before,
                )
                raw_train_logits = adapter.model(data)
                if base_mask.any():
                    base_loss = F.cross_entropy(
                        raw_train_logits[base_mask],
                        pseudo_labels.detach()[base_mask],
                    )
                else:
                    base_loss = data.sum() * 0.0
                base_update = adapter._apply_update(
                    adapter.model,
                    adapter.optimizer,
                    base_loss,
                    base_mask,
                )
                model_base = _clone_state(adapter.model)
                base_current_logits = _logits(adapter.model, data)
                base_next_logits = (
                    None if next_data is None else _logits(adapter.model, next_data)
                )

                row = {
                    "dataset": dataset,
                    "scenario": args.scenario,
                    "source_seed": source_seed,
                    "test_time_seed": int(args.test_time_seed),
                    "batch_index": batch_index,
                    "inner_step": inner_step,
                    "sample_count": int(labels.numel()),
                    "base_admitted_count": int(base_mask.sum().item()),
                    "full_admitted_count": int(full_mask.sum().item()),
                    "veto_count": int(veto.sum().item()),
                    "rescue_count": int(rescue.sum().item()),
                    "veto_wrong_count": int(
                        (veto & pseudo_labels.ne(labels)).sum().item()
                    ),
                    "rescue_wrong_count": 0,
                    "ssaw_consistency_count": int(consistency.sum().item()),
                    "ssaw_consistency_wrong_count": int(
                        (consistency & pseudo_labels.ne(labels)).sum().item()
                    ),
                    "ssaw_label_flip_count": int(
                        state["ssaw_label_flip"].sum().item()
                    ),
                    "ssaw_gradient_available": float(
                        batch_log.get("ssaw_gradient_available", 0.0)
                    ),
                    "ssaw_gradient_applied": float(
                        batch_log.get("ssaw_gradient_applied", 0.0)
                    ),
                    "ssaw_consistency_loss": float(
                        batch_log.get("ssaw_consistency_loss", 0.0)
                    ),
                    "ssaw_weighted_consistency_loss": float(
                        batch_log.get("ssaw_weighted_consistency_loss", 0.0)
                    ),
                    "ssaw_realized_consistency_ratio": float(
                        batch_log.get("ssaw_realized_consistency_ratio", 0.0)
                    ),
                    "ssaw_prediction_kl_mean": float(
                        batch_log.get("ssaw_prediction_kl_mean", 0.0)
                    ),
                    "ssaw_training_participation_rate": float(
                        batch_log.get("ssaw_training_participation_rate", 0.0)
                    ),
                    "base_update_committed": bool(base_update["committed"]),
                    "parameter_l2_full_vs_base": _parameter_distance(
                        model_full, model_base
                    ),
                    **_outcomes(
                        full_current_logits,
                        base_current_logits,
                        labels,
                        "current",
                    ),
                }
                if next_data is not None:
                    row.update(
                        _outcomes(
                            full_next_logits,
                            base_next_logits,
                            next_labels,
                            "next",
                        )
                    )
                rows.append(row)

                _restore(
                    adapter.model,
                    adapter.optimizer,
                    model_full,
                    optimizer_full,
                )
        return pd.DataFrame(rows)
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def run_batch_counterfactual(args) -> pd.DataFrame:
    """Isolate the cumulative effect of all inner SSAW decisions in a batch."""

    dataset, _, source_seed, trainer, adapter, source_model = _prepare(args)
    try:
        batches = list(trainer.trg_whole_dl)
        rows = []
        for batch_index, batch in enumerate(batches):
            data, labels, target_indices = batch
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            next_data = next_labels = None
            if batch_index + 1 < len(batches):
                next_data = move_data_to_device(
                    batches[batch_index + 1][0], trainer.device
                )
                next_labels = (
                    batches[batch_index + 1][1]
                    .view(-1)
                    .long()
                    .to(trainer.device)
                )
            model_inputs = {
                "data": data,
                "labels": labels,
                "meta": {"trg_idx": torch.as_tensor(target_indices).view(-1).tolist()},
            }

            model_before = _clone_state(adapter.model)
            optimizer_before = copy.deepcopy(adapter.optimizer.state_dict())
            veto_count = rescue_count = 0
            veto_wrong_count = rescue_wrong_count = 0
            consistency_count = consistency_wrong_count = 0
            label_flip_count = 0
            gradient_available_steps = gradient_applied_steps = 0
            consistency_losses = []
            weighted_consistency_losses = []
            realized_consistency_ratios = []
            prediction_kls = []
            participation_rates = []
            for _ in range(adapter.steps):
                adapter.forward_and_adapt(
                    model_inputs,
                    adapter.model,
                    adapter.optimizer,
                    target_indices,
                )
                state = adapter._last_gate_log
                batch_log = adapter._last_batch_log
                pseudo_labels = state["pseudo_labels"].to(labels.device)
                veto = state["ssaw_veto_mask"].to(labels.device).bool()
                consistency = state["ssaw_consistency_mask"].to(
                    labels.device
                ).bool()
                rescue = torch.zeros_like(veto)
                veto_count += int(veto.sum().item())
                veto_wrong_count += int(
                    (veto & pseudo_labels.ne(labels)).sum().item()
                )
                consistency_count += int(consistency.sum().item())
                consistency_wrong_count += int(
                    (consistency & pseudo_labels.ne(labels)).sum().item()
                )
                label_flip_count += int(
                    state["ssaw_label_flip"].sum().item()
                )
                gradient_available_steps += int(
                    bool(batch_log.get("ssaw_gradient_available", 0.0))
                )
                gradient_applied_steps += int(
                    bool(batch_log.get("ssaw_gradient_applied", 0.0))
                )
                consistency_losses.append(
                    float(batch_log.get("ssaw_consistency_loss", 0.0))
                )
                weighted_consistency_losses.append(
                    float(
                        batch_log.get("ssaw_weighted_consistency_loss", 0.0)
                    )
                )
                realized_consistency_ratios.append(
                    float(
                        batch_log.get("ssaw_realized_consistency_ratio", 0.0)
                    )
                )
                prediction_kls.append(
                    float(batch_log.get("ssaw_prediction_kl_mean", 0.0))
                )
                participation_rates.append(
                    float(
                        batch_log.get("ssaw_training_participation_rate", 0.0)
                    )
                )

            model_full = _clone_state(adapter.model)
            optimizer_full = copy.deepcopy(adapter.optimizer.state_dict())
            full_current_logits = _logits(adapter.model, data)
            full_next_logits = (
                None if next_data is None else _logits(adapter.model, next_data)
            )

            _restore(
                adapter.model,
                adapter.optimizer,
                model_before,
                optimizer_before,
            )
            base_admitted_count = 0
            base_committed_count = 0
            for _ in range(adapter.steps):
                _, base_mask, base_update = _base_forward_and_adapt(adapter, data)
                base_admitted_count += int(base_mask.sum().item())
                base_committed_count += int(bool(base_update["committed"]))
            model_base = _clone_state(adapter.model)
            base_current_logits = _logits(adapter.model, data)
            base_next_logits = (
                None if next_data is None else _logits(adapter.model, next_data)
            )

            row = {
                "dataset": dataset,
                "scenario": args.scenario,
                "source_seed": source_seed,
                "test_time_seed": int(args.test_time_seed),
                "batch_index": batch_index,
                "inner_steps": int(adapter.steps),
                "sample_count": int(labels.numel()),
                "base_admitted_count": base_admitted_count,
                "base_committed_steps": base_committed_count,
                "veto_count": veto_count,
                "rescue_count": rescue_count,
                "veto_wrong_count": veto_wrong_count,
                "rescue_wrong_count": rescue_wrong_count,
                "ssaw_consistency_count": consistency_count,
                "ssaw_consistency_wrong_count": consistency_wrong_count,
                "ssaw_label_flip_count": label_flip_count,
                "ssaw_gradient_available_steps": gradient_available_steps,
                "ssaw_gradient_applied_steps": gradient_applied_steps,
                "ssaw_consistency_loss_mean": float(
                    sum(consistency_losses) / max(1, len(consistency_losses))
                ),
                "ssaw_weighted_consistency_loss_mean": float(
                    sum(weighted_consistency_losses)
                    / max(1, len(weighted_consistency_losses))
                ),
                "ssaw_realized_consistency_ratio_mean": float(
                    sum(realized_consistency_ratios)
                    / max(1, len(realized_consistency_ratios))
                ),
                "ssaw_prediction_kl_mean": float(
                    sum(prediction_kls) / max(1, len(prediction_kls))
                ),
                "ssaw_training_participation_rate_mean": float(
                    sum(participation_rates) / max(1, len(participation_rates))
                ),
                "parameter_l2_full_vs_base": _parameter_distance(
                    model_full, model_base
                ),
                **_outcomes(
                    full_current_logits,
                    base_current_logits,
                    labels,
                    "current",
                ),
            }
            if next_data is not None:
                row.update(
                    _outcomes(
                        full_next_logits,
                        base_next_logits,
                        next_labels,
                        "next",
                    )
                )
            rows.append(row)
            _restore(
                adapter.model,
                adapter.optimizer,
                model_full,
                optimizer_full,
            )
        return pd.DataFrame(rows)
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--dataset", required=True, choices=("EEG", "HAR", "FD", "HHAR")
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--test-time-seed", type=int, default=1)
    parser.add_argument(
        "--granularity",
        choices=("step", "batch"),
        default="step",
        help="Shadow one update or all inner updates of the current batch.",
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4",
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "ssaw_update_counterfactual_v1",
    )
    args = parser.parse_args(argv)
    output_dir = ensure_dir(args.output_dir)
    frame = (
        run_step_counterfactual(args)
        if args.granularity == "step"
        else run_batch_counterfactual(args)
    )
    output_path = output_dir / (
        f"{args.dataset}_{args.scenario.replace('->', '_to_')}_"
        f"seed{args.test_time_seed}_{args.granularity}.csv"
    )
    atomic_write_csv(frame, output_path, index=False)
    gradient_column = (
        "ssaw_gradient_applied"
        if args.granularity == "step"
        else "ssaw_gradient_applied_steps"
    )
    effect_mask = frame[gradient_column].gt(0) | frame[
        "parameter_l2_full_vs_base"
    ].gt(1e-12)
    intervention = frame[effect_mask]
    summary = {
        "rows": int(len(frame)),
        "intervention_rows": int(len(intervention)),
        "no_intervention_max_parameter_distance": float(
            frame.loc[
                ~effect_mask,
                "parameter_l2_full_vs_base",
            ].max()
            if (~effect_mask).any()
            else float("nan")
        ),
        "intervention_next_ce_improvement_mean": float(
            intervention["next_ce_improvement_full_vs_base"].mean()
        ),
        "intervention_next_accuracy_delta_mean": float(
            intervention["next_accuracy_delta_full_vs_base"].mean()
        ),
        "intervention_next_macro_f1_delta_mean": float(
            intervention["next_macro_f1_delta_full_vs_base"].mean()
        ),
        "intervention_next_full_only_correct": int(
            intervention["next_full_only_correct_count"].sum()
        ),
        "intervention_next_base_only_correct": int(
            intervention["next_base_only_correct_count"].sum()
        ),
        "scope": (
            f"local {args.granularity}-level counterfactual; rows share online "
            "history and are not independent significance units"
        ),
        "intervention_definition": (
            "production SSAW auxiliary gradient applied or parameter distance "
            "from the paired raw-only update exceeds 1e-12"
        ),
        "target_labels_used_for_updates": False,
        "config_source": getattr(args, "config_source", "unknown"),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
