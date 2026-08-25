from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.demo_dataloader import data_generator_demo, whole_targe_data_generator_demo
from configs.data_model_configs import get_dataset_class, validate_scenario
from configs.dusafe_ablation import resolve_dusafe_ablation
from configs.tta_hparams_new import get_hparams_class
from algorithms.get_tta_class import get_algorithm_class

from models.da_models import get_backbone_class
from pre_train_model.pre_train_model import PreTrainModel
from pre_train_model.build import pre_train_model
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


_STEP_SAFETY_COUNT_KEYS = (
    "decision_count",
    "correct_count",
    "wrong_count",
    "admitted_count",
    "admitted_correct_count",
    "admitted_wrong_count",
    "active_count",
    "active_correct_count",
    "active_wrong_count",
    "corrupted_count",
    "admitted_corrupted_count",
    "active_corrupted_count",
    "admitted_unsafe_count",
    "active_unsafe_count",
    "clean_correct_count",
    "admission_rejected_clean_correct_count",
    "inactive_clean_correct_count",
)


_GATE_CONTRIBUTION_KEYS = (
    "decision_count",
    "clean_correct_count",
    "corrupted_count",
    "wrong_count",
    "confidence_rejected_count",
    "confidence_rejected_clean_correct_count",
    "confidence_rejected_corrupted_count",
    "confidence_rejected_wrong_count",
    "semantic_rejected_count",
    "semantic_rejected_clean_correct_count",
    "semantic_rejected_corrupted_count",
    "semantic_rejected_wrong_count",
    "commit_guard_rejected_count",
    "commit_guard_rejected_clean_correct_count",
    "commit_guard_rejected_corrupted_count",
    "commit_guard_rejected_wrong_count",
)


def _empty_gate_contribution_counts():
    """Return additive step-by-sample gate contribution counters."""
    return {key: 0 for key in _GATE_CONTRIBUTION_KEYS}


def _count_gate_contributions(
    *,
    pseudo_labels,
    confidence_masks,
    semantic_masks,
    base_admission_masks,
    admission_masks,
    active_masks,
    labels,
    corrupted,
):
    """Count disjoint gate/commit decisions at ``[step, sample]`` grain.

    Confidence rejection is attributed first, semantic rejection second, and
    the commit guard last. SSAW eligibility is an auxiliary-loss decision, not
    a raw-update rejection category.
    """
    tensors = [
        torch.as_tensor(value, dtype=dtype).cpu()
        for value, dtype in (
            (pseudo_labels, torch.long),
            (confidence_masks, torch.bool),
            (semantic_masks, torch.bool),
            (base_admission_masks, torch.bool),
            (admission_masks, torch.bool),
            (active_masks, torch.bool),
        )
    ]
    (
        pseudo_labels,
        confidence_masks,
        semantic_masks,
        base_admission_masks,
        admission_masks,
        active_masks,
    ) = tensors
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
    corrupted = torch.as_tensor(corrupted, dtype=torch.bool).view(-1).cpu()
    expected_shape = (pseudo_labels.size(0), labels.numel())
    if pseudo_labels.dim() != 2 or tuple(pseudo_labels.shape) != expected_shape:
        raise RuntimeError("TTA gate tensors must have shape [steps, batch]")
    for tensor in (
        confidence_masks,
        semantic_masks,
        base_admission_masks,
        admission_masks,
        active_masks,
    ):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError("TTA gate tensors must have shape [steps, batch]")
    if corrupted.numel() != labels.numel():
        raise RuntimeError("TTA corruption mask length does not match labels")
    if (active_masks & (~admission_masks)).any():
        raise RuntimeError("A committed update cannot bypass admission")
    # Validate that the logged base gate is exactly the production path.  This
    # catches diagnostic drift without changing the policy itself.
    expected_base = confidence_masks & semantic_masks
    if not torch.equal(expected_base, base_admission_masks):
        raise RuntimeError(
            "Logged base admission masks do not match confidence/semantic gates"
        )
    if not torch.equal(base_admission_masks, admission_masks):
        raise RuntimeError("Logged admission masks do not match confidence/semantic gates")

    correct = pseudo_labels.eq(labels.unsqueeze(0))
    clean_correct = correct & (~corrupted.unsqueeze(0))
    corrupted_steps = corrupted.unsqueeze(0).expand_as(correct)
    wrong = ~correct
    confidence_rejected = ~confidence_masks
    semantic_rejected = confidence_masks & (~semantic_masks)
    # The commit guard acts only after admission and is disjoint from the two
    # admission checks.
    commit_guard_rejected = admission_masks & (~active_masks)
    categories = {
        "confidence_rejected": confidence_rejected,
        "semantic_rejected": semantic_rejected,
        "commit_guard_rejected": commit_guard_rejected,
    }
    counts = _empty_gate_contribution_counts()
    counts["decision_count"] = int(correct.numel())
    counts["clean_correct_count"] = int(clean_correct.sum().item())
    counts["corrupted_count"] = int(corrupted_steps.sum().item())
    counts["wrong_count"] = int(wrong.sum().item())
    for name, mask in categories.items():
        counts[f"{name}_count"] = int(mask.sum().item())
        counts[f"{name}_clean_correct_count"] = int(
            (mask & clean_correct).sum().item()
        )
        counts[f"{name}_corrupted_count"] = int(
            (mask & corrupted_steps).sum().item()
        )
        counts[f"{name}_wrong_count"] = int((mask & wrong).sum().item())
    return counts


def _empty_step_safety_counts():
    """Return additive counters for step-by-sample safety decisions."""
    return {key: 0 for key in _STEP_SAFETY_COUNT_KEYS}


def _count_step_safety_decisions(
    *,
    pseudo_labels,
    admission_masks,
    active_masks,
    labels,
    corrupted,
):
    """Count safety outcomes at the common ``[inner_step, sample]`` grain."""
    pseudo_labels = torch.as_tensor(pseudo_labels, dtype=torch.long).cpu()
    admission_masks = torch.as_tensor(
        admission_masks, dtype=torch.bool
    ).cpu()
    active_masks = torch.as_tensor(active_masks, dtype=torch.bool).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
    corrupted = torch.as_tensor(
        corrupted, dtype=torch.bool
    ).view(-1).cpu()

    expected_shape = (pseudo_labels.size(0), labels.numel())
    if (
        pseudo_labels.dim() != 2
        or tuple(pseudo_labels.shape) != expected_shape
        or tuple(admission_masks.shape) != expected_shape
        or tuple(active_masks.shape) != expected_shape
    ):
        raise RuntimeError(
            "TTA inner-step safety tensors must have shape [steps, batch]"
        )
    if corrupted.numel() != labels.numel():
        raise RuntimeError(
            "TTA corruption mask length does not match the evaluated batch"
        )
    if (active_masks & (~admission_masks)).any():
        raise RuntimeError(
            "A committed sample update cannot bypass the admission decision"
        )

    correct = pseudo_labels.eq(labels.unsqueeze(0))
    wrong = ~correct
    corrupted_steps = corrupted.unsqueeze(0).expand_as(correct)
    clean_correct = correct & (~corrupted_steps)
    admitted_unsafe = admission_masks & (wrong | corrupted_steps)
    active_unsafe = active_masks & (wrong | corrupted_steps)

    tensors = {
        "decision_count": torch.ones_like(correct),
        "correct_count": correct,
        "wrong_count": wrong,
        "admitted_count": admission_masks,
        "admitted_correct_count": admission_masks & correct,
        "admitted_wrong_count": admission_masks & wrong,
        "active_count": active_masks,
        "active_correct_count": active_masks & correct,
        "active_wrong_count": active_masks & wrong,
        "corrupted_count": corrupted_steps,
        "admitted_corrupted_count": admission_masks & corrupted_steps,
        "active_corrupted_count": active_masks & corrupted_steps,
        "admitted_unsafe_count": admitted_unsafe,
        "active_unsafe_count": active_unsafe,
        "clean_correct_count": clean_correct,
        "admission_rejected_clean_correct_count": (
            (~admission_masks) & clean_correct
        ),
        "inactive_clean_correct_count": (~active_masks) & clean_correct,
    }
    return {key: int(value.sum().item()) for key, value in tensors.items()}


def _safe_ratio(numerator, denominator):
    if int(denominator) == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _binary_mcc(tp, fp, tn, fn):
    denominator = float(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ) ** 0.5
    if denominator == 0.0:
        return float("nan")
    return float(tp * tn - fp * fn) / denominator


def _summarize_step_safety(counts):
    """Build canonical safety metrics from additive step-level counts."""
    total = counts["decision_count"]
    correct = counts["correct_count"]
    wrong = counts["wrong_count"]
    admitted = counts["admitted_count"]
    active = counts["active_count"]
    corrupted = counts["corrupted_count"]
    clean_correct = counts["clean_correct_count"]

    admitted_correct = counts["admitted_correct_count"]
    admitted_wrong = counts["admitted_wrong_count"]
    active_correct = counts["active_correct_count"]
    active_wrong = counts["active_wrong_count"]
    admission_rejected_wrong = wrong - admitted_wrong
    admission_rejected_correct = correct - admitted_correct
    inactive_wrong = wrong - active_wrong
    inactive_correct = correct - active_correct
    admission_correct_acceptance = _safe_ratio(admitted_correct, correct)
    update_correct_acceptance = _safe_ratio(active_correct, correct)
    admission_wrong_rejection = _safe_ratio(
        admission_rejected_wrong, wrong
    )
    update_wrong_rejection = _safe_ratio(inactive_wrong, wrong)

    summary = {
        # Version and additive counts make the aggregation grain auditable.
        "safety_metric_version": 2.0,
        "step_decision_count": float(total),
        "step_correct_candidate_count": float(correct),
        "step_wrong_candidate_count": float(wrong),
        "step_admitted_count": float(admitted),
        "step_committed_sample_update_count": float(active),
        # Canonical update-level metrics. Each observation is one sample at
        # one inner adaptation step.
        "coverage": _safe_ratio(active, total),
        "committed_update_coverage": _safe_ratio(active, total),
        "accepted_pseudo_label_accuracy": _safe_ratio(
            active_correct, active
        ),
        "wrong_update_rate": _safe_ratio(active_wrong, active),
        "wrong_rejection_recall": update_wrong_rejection,
        "correct_false_rejection_rate": _safe_ratio(
            inactive_correct, correct
        ),
        "correct_acceptance_rate": update_correct_acceptance,
        "dangerous_rejection_recall": update_wrong_rejection,
        "unsafe_update_rate": _safe_ratio(
            counts["active_unsafe_count"], active
        ),
        "corruption_rejection_recall": _safe_ratio(
            corrupted - counts["active_corrupted_count"], corrupted
        ),
        "clean_correct_false_rejection_rate": _safe_ratio(
            counts["inactive_clean_correct_count"], clean_correct
        ),
        "committed_corruption_rate": _safe_ratio(
            counts["active_corrupted_count"], active
        ),
        "update_gate_balanced_accuracy": (
            float("nan")
            if update_correct_acceptance != update_correct_acceptance
            or update_wrong_rejection != update_wrong_rejection
            else 0.5
            * (update_correct_acceptance + update_wrong_rejection)
        ),
        "update_gate_mcc": _binary_mcc(
            active_correct,
            active_wrong,
            inactive_wrong,
            inactive_correct,
        ),
        # Admission metrics isolate the gate from optimizer rollback/failure.
        "admission_coverage": _safe_ratio(admitted, total),
        "admitted_pseudo_label_accuracy": _safe_ratio(
            admitted_correct, admitted
        ),
        "admission_wrong_rejection_recall": admission_wrong_rejection,
        "admission_correct_false_rejection_rate": _safe_ratio(
            admission_rejected_correct, correct
        ),
        "admission_correct_acceptance_rate": admission_correct_acceptance,
        "admission_dangerous_rejection_recall": (
            admission_wrong_rejection
        ),
        "admission_unsafe_rate": _safe_ratio(
            counts["admitted_unsafe_count"], admitted
        ),
        "admission_corruption_rejection_recall": _safe_ratio(
            corrupted - counts["admitted_corrupted_count"], corrupted
        ),
        "admission_clean_correct_false_rejection_rate": _safe_ratio(
            counts["admission_rejected_clean_correct_count"],
            clean_correct,
        ),
        "admitted_corruption_rate": _safe_ratio(
            counts["admitted_corrupted_count"], admitted
        ),
        "admission_gate_balanced_accuracy": (
            float("nan")
            if admission_correct_acceptance
            != admission_correct_acceptance
            or admission_wrong_rejection != admission_wrong_rejection
            else 0.5
            * (admission_correct_acceptance + admission_wrong_rejection)
        ),
        "admission_gate_mcc": _binary_mcc(
            admitted_correct,
            admitted_wrong,
            admission_rejected_wrong,
            admission_rejected_correct,
        ),
    }
    return summary


@torch.inference_mode()
def _predict_after_adaptation(tta_model, model_inputs):
    """Re-evaluate a batch after every requested inner update is complete."""
    adapted_model = getattr(tta_model, "model", None)
    if adapted_model is None:
        raise RuntimeError(
            "Post-update evaluation requires a TTA wrapper with a model"
        )

    # A read-only metric pass must not add an extra BN-state update. Batch-mode
    # DuSafe is stateless, while this snapshot also protects frozen/running
    # variants and keeps the evaluator usable for protocol comparisons.
    bn_snapshots = []
    cached_bn_mutability = getattr(
        adapted_model, "_dusafe_bn_buffers_may_update_cache", None
    )
    if cached_bn_mutability is not False:
        for module in adapted_model.modules():
            if (
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                and module.training
                and module.track_running_stats
            ):
                bn_snapshots.append(
                    (
                        module,
                        None
                        if module.running_mean is None
                        else module.running_mean.detach().clone(),
                        None
                        if module.running_var is None
                        else module.running_var.detach().clone(),
                        None
                        if module.num_batches_tracked is None
                        else module.num_batches_tracked.detach().clone(),
                    )
                )
    try:
        adapted_inputs = (
            model_inputs.get("data")
            if isinstance(model_inputs, dict)
            else model_inputs
        )
        predictions = adapted_model(adapted_inputs)
    finally:
        for module, mean, variance, batches in bn_snapshots:
            if mean is not None and module.running_mean is not None:
                module.running_mean.copy_(mean)
            if variance is not None and module.running_var is not None:
                module.running_var.copy_(variance)
            if batches is not None and module.num_batches_tracked is not None:
                module.num_batches_tracked.copy_(batches)
    if not torch.is_tensor(predictions) or predictions.dim() != 2:
        raise RuntimeError(
            "Post-update model prediction must have shape [batch, classes]"
        )
    return predictions


class TTAAbstractTrainer(object):
    """Shared fixed-source preparation, evaluation, and result handling."""

    def __init__(self, args):
        self.da_method = args.da_method 
        self.dataset = args.dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)
        self.algorithm_registry = str(
            getattr(args, "algorithm_registry", "production")
        ).strip().lower()
        if self.algorithm_registry not in {"production", "benchmark"}:
            raise ValueError(
                "algorithm_registry must be 'production' or 'benchmark'"
            )
        self.ablation_mode = getattr(args, "ablation_mode", None)
        self._ablation_name = None
        self._ablation_hparams = {}

        self.run_description = f"{args.da_method}_{args.exp_name}"
        self.experiment_description = args.dataset

        self.home_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.num_runs = args.num_runs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self._base_alg_hparams = dict(
            self.hparams_class.alg_hparams[self.da_method]
        )
        self._runtime_hparam_overrides = {}
        self._train_params = dict(self.hparams_class.train_params)
        # Source training must not inherit target-scenario or method-specific
        # adaptation settings. This profile is shared by every TTA method.
        source_alg_hparams = dict(self.hparams_class.alg_hparams.get("NoAdap", {}))
        source_train_params = dict(
            getattr(
                self.hparams_class,
                "source_train_params",
                self._train_params,
            )
        )
        self.source_hparams = {
            **source_alg_hparams,
            **source_train_params,
        }
        if self.ablation_mode:
            ablation_payload = resolve_dusafe_ablation(self.ablation_mode)
            self._ablation_name = ablation_payload["name"]
            self._ablation_hparams = dict(ablation_payload["overrides"])
        self.hparams = {**self._base_alg_hparams, **self._train_params}

        self._backbone_attr_names = (
            "times_hidden_channels",
            "times_num_layers",
            "times_patch_lens",
            "times_dropout",
            "times_ffn_expansion",
        )
        self._dataset_backbone_defaults = {
            attr: deepcopy(getattr(self.dataset_configs, attr))
            for attr in self._backbone_attr_names
            if hasattr(self.dataset_configs, attr)
        }

        self.num_classes = self.dataset_configs.num_classes
        # Multiclass evaluation metrics.
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)

        # Cache latest experiment metrics so external tools (e.g., Optuna) can read them
        self.scenario_metrics = {}
        self.last_table_results = None
        self.last_table_risks = None

    def initialize_pretrained_model(self):
        backbone_fe = get_backbone_class(self.backbone)
        pretrained_model = PreTrainModel(
            backbone_fe, self.dataset_configs, self.source_hparams
        )
        pretrained_model = pretrained_model.to(self.device)

        return pretrained_model

    def pre_train(self):
        backbone_fe = get_backbone_class(self.backbone)
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        non_adapted_model_state, pre_trained_model = pre_train_model(
            backbone_fe,
            self.dataset_configs,
            self.source_hparams,
            self.src_train_dl,
            self.pre_loss_avg_meters,
            self.logger,
            self.device,
        )

        return non_adapted_model_state, pre_trained_model

    def evaluate(self, test_loader, tta_model):
        """Run evaluation and keep cached tensors on CPU to avoid GPU bloat."""
        configure_graph_workload = getattr(
            tta_model, "configure_candidate_graph_workload", None
        )
        if callable(configure_graph_workload):
            loader_batch_size = getattr(test_loader, "batch_size", None)
            loader_dataset = getattr(test_loader, "dataset", None)
            if loader_batch_size and loader_dataset is not None:
                full_batch_count = len(loader_dataset) // int(loader_batch_size)
                configure_graph_workload(
                    expected_full_batch_searches=(
                        full_batch_count
                        * max(1, int(getattr(tta_model, "steps", 1)))
                    )
                )
        preds_list, pre_final_preds_list, labels_list = [], [], []
        gate_log_rows = []
        candidate_hash_rows = []
        safety_rows = []
        step_safety_counts = _empty_step_safety_counts()
        gate_contribution_counts = _empty_gate_contribution_counts()
        self.last_gate_contribution_summary = {}
        loader_meta = getattr(test_loader, "meta", None)
        source_only_eval = self.da_method == "NoAdap"
        record_per_sample_evidence = bool(
            self.hparams.get("record_per_sample_evidence", True)
        )
        if source_only_eval:
            tta_model.eval()

        for batch_index, (data, labels, trg_idx) in enumerate(test_loader):
            if isinstance(data, list):
                data = [tensor.float().to(self.device) for tensor in data]
            else:
                data = data.float().to(self.device)
            labels = labels.view(-1).long()
            if record_per_sample_evidence:
                labels = labels.to(self.device)

            batch_meta = {}
            if record_per_sample_evidence:
                batch_meta["trg_idx"] = (
                    trg_idx.detach().cpu().tolist()
                    if torch.is_tensor(trg_idx)
                    else trg_idx
                )
                if isinstance(loader_meta, dict):
                    batch_meta.update(loader_meta)
            model_inputs = {
                "data": data,
                "labels": labels,
                "meta": batch_meta,
            }
            if source_only_eval:
                with torch.inference_mode():
                    pre_final_predictions = tta_model(model_inputs)
                predictions = pre_final_predictions
            else:
                pre_final_predictions = tta_model(model_inputs)
                predictions = _predict_after_adaptation(
                    tta_model, model_inputs
                )
            preds_list.append(predictions.detach().cpu())
            pre_final_preds_list.append(
                pre_final_predictions.detach().cpu()
            )
            labels_list.append(labels.cpu())
            batch_gate_log = getattr(tta_model, "_last_batch_log", None)
            if isinstance(batch_gate_log, dict) and batch_gate_log:
                numeric_row = {}
                for key, value in batch_gate_log.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        numeric_row[key] = float(value)
                if numeric_row:
                    gate_log_rows.append(numeric_row)

            # Production main-table evaluation only needs post-update logits,
            # labels, and compact batch summaries. Everything below builds
            # target-label-dependent per-sample evidence and is intentionally
            # reserved for safety/mechanism runners.
            if not record_per_sample_evidence:
                continue

            sample_gate_log = getattr(tta_model, "_last_gate_log", {})
            candidate_hash_rows.append(
                {
                    "batch_index": int(batch_index),
                    "candidate_sha256": sample_gate_log.get(
                        "ssaw_candidate_sha256"
                    ),
                    "test_time_seed": getattr(tta_model, "test_time_seed", None),
                    "effective_ssaw_seed": getattr(
                        tta_model, "ssaw_effective_sobol_seed", None
                    ),
                }
            )
            predicted_labels = predictions.detach().argmax(dim=1).cpu()
            pre_final_predicted_labels = (
                pre_final_predictions.detach().argmax(dim=1).cpu()
            )
            confidence = predictions.detach().softmax(dim=1).amax(dim=1).cpu()
            pre_final_confidence = (
                pre_final_predictions.detach()
                .softmax(dim=1)
                .amax(dim=1)
                .cpu()
            )
            post_update_true_label_nll = (
                -predictions.detach().log_softmax(dim=1)
                .gather(1, labels[:, None])
                .squeeze(1)
                .cpu()
            )
            pre_final_update_true_label_nll = (
                -pre_final_predictions.detach().log_softmax(dim=1)
                .gather(1, labels[:, None])
                .squeeze(1)
                .cpu()
            )
            top2_logits = predictions.detach().topk(k=min(2, predictions.size(1)), dim=1).values.cpu()
            pre_final_top2_logits = pre_final_predictions.detach().topk(
                k=min(2, pre_final_predictions.size(1)), dim=1
            ).values.cpu()
            if top2_logits.size(1) == 1:
                logit_margin = top2_logits[:, 0]
            else:
                logit_margin = top2_logits[:, 0] - top2_logits[:, 1]
            if pre_final_top2_logits.size(1) == 1:
                pre_final_logit_margin = pre_final_top2_logits[:, 0]
            else:
                pre_final_logit_margin = (
                    pre_final_top2_logits[:, 0]
                    - pre_final_top2_logits[:, 1]
                )
            if source_only_eval:
                selected_mask = torch.zeros_like(predicted_labels, dtype=torch.bool)
                admission_mask = selected_mask.clone()
            else:
                selected_mask = sample_gate_log.get(
                    "active_mask",
                    sample_gate_log.get("selected_mask", torch.ones_like(predicted_labels, dtype=torch.bool)),
                )
                selected_mask = torch.as_tensor(selected_mask, dtype=torch.bool).view(-1).cpu()
                admission_mask = sample_gate_log.get(
                    "admission_mask", selected_mask
                )
                admission_mask = torch.as_tensor(
                    admission_mask, dtype=torch.bool
                ).view(-1).cpu()
            diagnostic_vectors = {}
            for diagnostic_name in (
                "raw_entropy",
                "ssaw_entropy",
                "ssaw_entropy_shift",
                "kl_divergence",
                "ssaw_feature_distance",
                "ssaw_vote_agreement",
                "ssaw_label_preserving_count",
                "ssaw_selected_nll",
                "ssaw_selected_margin",
                "ssaw_entropy_rise",
                "ssaw_selected_radius",
                "ssaw_selected_sign",
                "ssaw_selected_direction",
                "ssaw_endpoint_flip_fraction",
                "ssaw_raw_pseudo_margin",
                "ssaw_selected_margin_drop",
                "ssaw_selected_normalized_margin_ratio",
                "ssaw_gathered_actual_margin",
                "ssaw_gathered_actual_margin_drop",
                "ssaw_gathered_actual_normalized_margin_ratio",
                "source_semantic_prediction",
                "source_semantic_margin",
                "raw_top1_nll",
            ):
                diagnostic_value = sample_gate_log.get(diagnostic_name)
                if diagnostic_value is None:
                    diagnostic_vectors[diagnostic_name] = torch.full(
                        (labels.numel(),), float("nan"), dtype=torch.float32
                    )
                else:
                    diagnostic_tensor = torch.as_tensor(
                        diagnostic_value, dtype=torch.float32
                    ).view(-1).cpu()
                    if diagnostic_tensor.numel() != labels.numel():
                        raise RuntimeError(
                            f"TTA diagnostic '{diagnostic_name}' length does not match the evaluated batch"
                        )
                    diagnostic_vectors[diagnostic_name] = diagnostic_tensor
            boolean_diagnostic_vectors = {}
            for diagnostic_name in (
                "ssaw_label_flip",
                "ssaw_view_selected_mask",
                "source_semantic_router_mask",
                "ssaw_router_mask",
                "ssaw_backtracking_used",
                "ssaw_final_skip",
                "ssaw_gathered_actual_label_flip",
                "ssaw_gathered_training_mask",
            ):
                diagnostic_value = sample_gate_log.get(diagnostic_name)
                if diagnostic_value is None:
                    diagnostic_tensor = torch.full(
                        (labels.numel(),), False, dtype=torch.bool
                    )
                else:
                    diagnostic_tensor = torch.as_tensor(
                        diagnostic_value, dtype=torch.bool
                    ).view(-1).cpu()
                    if diagnostic_tensor.numel() != labels.numel():
                        raise RuntimeError(
                            f"TTA diagnostic '{diagnostic_name}' length does not match the evaluated batch"
                        )
                boolean_diagnostic_vectors[diagnostic_name] = diagnostic_tensor
            corruption_mask = batch_meta.get("corruption_mask", [False] * labels.numel())
            corruption_mask = torch.as_tensor(corruption_mask, dtype=torch.bool).view(-1).cpu()
            sample_indices = torch.as_tensor(trg_idx).view(-1).cpu()
            labels_cpu_batch = labels.detach().cpu()
            if selected_mask.numel() != labels_cpu_batch.numel():
                raise RuntimeError("TTA selected mask length does not match the evaluated batch")
            if admission_mask.numel() != labels_cpu_batch.numel():
                raise RuntimeError("TTA admission mask length does not match the evaluated batch")
            inner_pseudo_labels = sample_gate_log.get("inner_pseudo_labels")
            inner_admission_masks = sample_gate_log.get(
                "inner_admission_masks"
            )
            inner_active_masks = sample_gate_log.get("inner_active_masks")
            inner_confidence_masks = sample_gate_log.get(
                "inner_confidence_masks"
            )
            inner_semantic_masks = sample_gate_log.get("inner_semantic_masks")
            inner_base_admission_masks = sample_gate_log.get(
                "inner_base_admission_masks"
            )
            gate_masks_available = all(
                value is not None
                for value in (
                    inner_pseudo_labels,
                    inner_admission_masks,
                    inner_active_masks,
                    inner_confidence_masks,
                    inner_semantic_masks,
                    inner_base_admission_masks,
                )
            )
            if gate_masks_available:
                inner_pseudo_labels = torch.as_tensor(
                    inner_pseudo_labels, dtype=torch.long
                ).cpu()
                inner_admission_masks = torch.as_tensor(
                    inner_admission_masks, dtype=torch.bool
                ).cpu()
                inner_active_masks = torch.as_tensor(
                    inner_active_masks, dtype=torch.bool
                ).cpu()
                inner_confidence_masks = torch.as_tensor(
                    inner_confidence_masks, dtype=torch.bool
                ).cpu()
                inner_semantic_masks = torch.as_tensor(
                    inner_semantic_masks, dtype=torch.bool
                ).cpu()
                inner_base_admission_masks = torch.as_tensor(
                    inner_base_admission_masks, dtype=torch.bool
                ).cpu()
                expected_shape = (
                    inner_pseudo_labels.size(0),
                    labels_cpu_batch.numel(),
                )
                if (
                    inner_pseudo_labels.dim() != 2
                    or tuple(inner_pseudo_labels.shape) != expected_shape
                    or tuple(inner_admission_masks.shape) != expected_shape
                    or tuple(inner_active_masks.shape) != expected_shape
                    or tuple(inner_confidence_masks.shape) != expected_shape
                    or tuple(inner_semantic_masks.shape) != expected_shape
                    or tuple(inner_base_admission_masks.shape) != expected_shape
                ):
                    raise RuntimeError(
                        "TTA inner-step safety/gate tensors must have shape "
                        "[steps, batch]"
                    )
                inner_correct = inner_pseudo_labels.eq(
                    labels_cpu_batch.unsqueeze(0)
                )
                candidate_pseudo_label_correct = inner_correct.all(dim=0)
                admission_pseudo_label_correct = (
                    (~inner_admission_masks) | inner_correct
                ).all(dim=0)
                update_pseudo_label_correct = (
                    (~inner_active_masks) | inner_correct
                ).all(dim=0)
            else:
                single_pseudo_labels = sample_gate_log.get(
                    "pseudo_labels", predicted_labels
                )
                single_pseudo_labels = torch.as_tensor(
                    single_pseudo_labels, dtype=torch.long
                ).view(-1).cpu()
                if single_pseudo_labels.numel() != labels_cpu_batch.numel():
                    raise RuntimeError(
                        "TTA pseudo-label vector length does not match the "
                        "evaluated batch"
                    )
                candidate_pseudo_label_correct = single_pseudo_labels.eq(
                    labels_cpu_batch
                )
                admission_pseudo_label_correct = (
                    (~admission_mask) | candidate_pseudo_label_correct
                )
                update_pseudo_label_correct = (
                    (~selected_mask) | candidate_pseudo_label_correct
                )
                inner_pseudo_labels = single_pseudo_labels.unsqueeze(0)
                inner_admission_masks = admission_mask.unsqueeze(0)
                inner_active_masks = selected_mask.unsqueeze(0)
                inner_confidence_masks = inner_admission_masks.clone()
                inner_semantic_masks = inner_admission_masks.clone()
                inner_base_admission_masks = inner_admission_masks.clone()

            if gate_masks_available:
                batch_gate_contributions = _count_gate_contributions(
                    pseudo_labels=inner_pseudo_labels,
                    confidence_masks=inner_confidence_masks,
                    semantic_masks=inner_semantic_masks,
                    base_admission_masks=inner_base_admission_masks,
                    admission_masks=inner_admission_masks,
                    active_masks=inner_active_masks,
                    labels=labels_cpu_batch,
                    corrupted=corruption_mask,
                )
                for key, value in batch_gate_contributions.items():
                    gate_contribution_counts[key] += value

            batch_step_counts = _count_step_safety_decisions(
                pseudo_labels=inner_pseudo_labels,
                admission_masks=inner_admission_masks,
                active_masks=inner_active_masks,
                labels=labels_cpu_batch,
                corrupted=corruption_mask,
            )
            for key, value in batch_step_counts.items():
                step_safety_counts[key] += value
            for sample_offset in range(labels_cpu_batch.numel()):
                safety_rows.append({
                    "batch_index": int(batch_index),
                    "sample_index": int(sample_indices[sample_offset].item()),
                    "label": int(labels_cpu_batch[sample_offset].item()),
                    "pseudo_label": int(
                        inner_pseudo_labels[-1, sample_offset].item()
                    ),
                    "prediction": int(predicted_labels[sample_offset].item()),
                    "post_update_prediction": int(
                        predicted_labels[sample_offset].item()
                    ),
                    "pre_final_update_prediction": int(
                        pre_final_predicted_labels[sample_offset].item()
                    ),
                    "correct": bool(predicted_labels[sample_offset] == labels_cpu_batch[sample_offset]),
                    "post_update_correct": bool(
                        predicted_labels[sample_offset]
                        == labels_cpu_batch[sample_offset]
                    ),
                    "pre_final_update_correct": bool(
                        pre_final_predicted_labels[sample_offset]
                        == labels_cpu_batch[sample_offset]
                    ),
                    "candidate_pseudo_label_correct": bool(
                        candidate_pseudo_label_correct[sample_offset].item()
                    ),
                    "admission_pseudo_label_correct": bool(
                        admission_pseudo_label_correct[sample_offset].item()
                    ),
                    "update_pseudo_label_correct": bool(
                        update_pseudo_label_correct[sample_offset].item()
                    ),
                    "admitted": bool(admission_mask[sample_offset].item()),
                    "selected": bool(selected_mask[sample_offset].item()),
                    "corrupted": bool(corruption_mask[sample_offset].item()),
                    "confidence": float(confidence[sample_offset].item()),
                    "pre_final_update_confidence": float(
                        pre_final_confidence[sample_offset].item()
                    ),
                    "post_update_true_label_nll": float(
                        post_update_true_label_nll[sample_offset].item()
                    ),
                    "pre_final_update_true_label_nll": float(
                        pre_final_update_true_label_nll[sample_offset].item()
                    ),
                    "logit_margin": float(logit_margin[sample_offset].item()),
                    "pre_final_update_logit_margin": float(
                        pre_final_logit_margin[sample_offset].item()
                    ),
                    "raw_entropy": float(diagnostic_vectors["raw_entropy"][sample_offset].item()),
                    "ssaw_entropy": float(diagnostic_vectors["ssaw_entropy"][sample_offset].item()),
                    "ssaw_entropy_shift": float(
                        diagnostic_vectors["ssaw_entropy_shift"][sample_offset].item()
                    ),
                    "kl_divergence": float(
                        diagnostic_vectors["kl_divergence"][sample_offset].item()
                    ),
                    "ssaw_label_flip": bool(
                        boolean_diagnostic_vectors["ssaw_label_flip"][sample_offset].item()
                    ),
                    "ssaw_view_selected": bool(
                        boolean_diagnostic_vectors[
                            "ssaw_view_selected_mask"
                        ][sample_offset].item()
                    ),
                    "source_semantic_router_agree": bool(
                        boolean_diagnostic_vectors[
                            "source_semantic_router_mask"
                        ][sample_offset].item()
                    ),
                    "ssaw_router_selected": bool(
                        boolean_diagnostic_vectors[
                            "ssaw_router_mask"
                        ][sample_offset].item()
                    ),
                    "ssaw_backtracking_used": bool(
                        boolean_diagnostic_vectors[
                            "ssaw_backtracking_used"
                        ][sample_offset].item()
                    ),
                    "ssaw_final_skip": bool(
                        boolean_diagnostic_vectors[
                            "ssaw_final_skip"
                        ][sample_offset].item()
                    ),
                    "ssaw_gathered_actual_label_flip": bool(
                        boolean_diagnostic_vectors[
                            "ssaw_gathered_actual_label_flip"
                        ][sample_offset].item()
                    ),
                    "ssaw_gathered_training_selected": bool(
                        boolean_diagnostic_vectors[
                            "ssaw_gathered_training_mask"
                        ][sample_offset].item()
                    ),
                    "ssaw_feature_distance": float(
                        diagnostic_vectors["ssaw_feature_distance"][sample_offset].item()
                    ),
                    "ssaw_vote_agreement": float(
                        diagnostic_vectors["ssaw_vote_agreement"][sample_offset].item()
                    ),
                    "ssaw_label_preserving_count": float(
                        diagnostic_vectors["ssaw_label_preserving_count"][sample_offset].item()
                    ),
                    "ssaw_selected_nll": float(
                        diagnostic_vectors["ssaw_selected_nll"][sample_offset].item()
                    ),
                    "ssaw_selected_margin": float(
                        diagnostic_vectors["ssaw_selected_margin"][sample_offset].item()
                    ),
                    "ssaw_entropy_rise": float(
                        diagnostic_vectors["ssaw_entropy_rise"][sample_offset].item()
                    ),
                    "ssaw_selected_radius": float(
                        diagnostic_vectors[
                            "ssaw_selected_radius"
                        ][sample_offset].item()
                    ),
                    "ssaw_selected_sign": float(
                        diagnostic_vectors[
                            "ssaw_selected_sign"
                        ][sample_offset].item()
                    ),
                    "ssaw_selected_direction": float(
                        diagnostic_vectors[
                            "ssaw_selected_direction"
                        ][sample_offset].item()
                    ),
                    "ssaw_endpoint_flip_fraction": float(
                        diagnostic_vectors[
                            "ssaw_endpoint_flip_fraction"
                        ][sample_offset].item()
                    ),
                    "ssaw_raw_pseudo_margin": float(
                        diagnostic_vectors[
                            "ssaw_raw_pseudo_margin"
                        ][sample_offset].item()
                    ),
                    "ssaw_selected_margin_drop": float(
                        diagnostic_vectors[
                            "ssaw_selected_margin_drop"
                        ][sample_offset].item()
                    ),
                    "ssaw_selected_normalized_margin_ratio": float(
                        diagnostic_vectors[
                            "ssaw_selected_normalized_margin_ratio"
                        ][sample_offset].item()
                    ),
                    "ssaw_gathered_actual_margin": float(
                        diagnostic_vectors[
                            "ssaw_gathered_actual_margin"
                        ][sample_offset].item()
                    ),
                    "ssaw_gathered_actual_margin_drop": float(
                        diagnostic_vectors[
                            "ssaw_gathered_actual_margin_drop"
                        ][sample_offset].item()
                    ),
                    "ssaw_gathered_actual_normalized_margin_ratio": float(
                        diagnostic_vectors[
                            "ssaw_gathered_actual_normalized_margin_ratio"
                        ][sample_offset].item()
                    ),
                    "source_semantic_prediction": float(
                        diagnostic_vectors[
                            "source_semantic_prediction"
                        ][sample_offset].item()
                    ),
                    "source_semantic_margin": float(
                        diagnostic_vectors[
                            "source_semantic_margin"
                        ][sample_offset].item()
                    ),
                    "raw_top1_nll": float(
                        diagnostic_vectors["raw_top1_nll"][sample_offset].item()
                    ),
                })

        self.full_preds = torch.cat(preds_list)
        self.full_pre_final_update_preds = torch.cat(pre_final_preds_list)
        self.full_labels = torch.cat(labels_list)
        self.loss = F.cross_entropy(self.full_preds, self.full_labels)
        self.pre_final_update_loss = F.cross_entropy(
            self.full_pre_final_update_preds, self.full_labels
        )
        if gate_log_rows:
            gate_df = pd.DataFrame(gate_log_rows)
            self.last_batch_log_records = gate_df.copy()
            self.last_batch_log_summary = {
                col: float(gate_df[col].mean())
                for col in gate_df.columns
            }
        else:
            self.last_batch_log_records = pd.DataFrame()
            self.last_batch_log_summary = {}
        self.last_candidate_hash_records = pd.DataFrame(candidate_hash_rows)
        self.last_safety_records = pd.DataFrame(safety_rows)
        self.last_step_safety_counts = dict(step_safety_counts)
        if self.last_safety_records.empty:
            self.last_safety_summary = {}
        else:
            records = self.last_safety_records
            selected = records["selected"].astype(bool)
            admitted = records["admitted"].astype(bool)
            correct = records["correct"].astype(bool)
            candidate_correct = records.get(
                "candidate_pseudo_label_correct", records["correct"]
            ).astype(bool)
            admission_correct = records.get(
                "admission_pseudo_label_correct", records["correct"]
            ).astype(bool)
            update_correct = records.get(
                "update_pseudo_label_correct", records["correct"]
            ).astype(bool)
            corrupted = records["corrupted"].astype(bool)
            accepted_count = int(selected.sum())
            admitted_count = int(admitted.sum())
            corrupted_count = int(corrupted.sum())
            clean_correct = (~corrupted) & candidate_correct
            final_path_summary = {
                "coverage": float(selected.mean()),
                "committed_update_coverage": float(selected.mean()),
                "accepted_pseudo_label_accuracy": (
                    float(update_correct[selected].mean()) if accepted_count else float("nan")
                ),
                "wrong_update_rate": (
                    float((~update_correct)[selected].mean()) if accepted_count else float("nan")
                ),
                "wrong_rejection_recall": (
                    float((~selected)[~candidate_correct].mean()) if int((~candidate_correct).sum()) else float("nan")
                ),
                "correct_false_rejection_rate": (
                    float((~selected)[candidate_correct].mean()) if int(candidate_correct.sum()) else float("nan")
                ),
                "admission_coverage": float(admitted.mean()),
                "admitted_pseudo_label_accuracy": (
                    float(admission_correct[admitted].mean()) if admitted_count else float("nan")
                ),
                "admission_wrong_rejection_recall": (
                    float((~admitted)[~candidate_correct].mean()) if int((~candidate_correct).sum()) else float("nan")
                ),
                "admission_correct_false_rejection_rate": (
                    float((~admitted)[candidate_correct].mean()) if int(candidate_correct.sum()) else float("nan")
                ),
                "unsafe_update_rate": (
                    float(((~update_correct) | corrupted)[selected].mean()) if accepted_count else float("nan")
                ),
                "corruption_rejection_recall": (
                    float((~selected)[corrupted].mean()) if corrupted_count else float("nan")
                ),
                "clean_correct_false_rejection_rate": (
                    float((~selected)[clean_correct].mean()) if int(clean_correct.sum()) else float("nan")
                ),
                "admission_corruption_rejection_recall": (
                    float((~admitted)[corrupted].mean()) if corrupted_count else float("nan")
                ),
                "admission_clean_correct_false_rejection_rate": (
                    float((~admitted)[clean_correct].mean()) if int(clean_correct.sum()) else float("nan")
                ),
                "admitted_corruption_rate": (
                    float(corrupted[admitted].mean()) if admitted_count else float("nan")
                ),
                "committed_corruption_rate": (
                    float(corrupted[selected].mean()) if accepted_count else float("nan")
                ),
            }
            final_path_summary["correct_acceptance_rate"] = (
                1.0 - final_path_summary["correct_false_rejection_rate"]
                if final_path_summary["correct_false_rejection_rate"]
                == final_path_summary["correct_false_rejection_rate"]
                else float("nan")
            )
            final_path_summary["dangerous_rejection_recall"] = (
                final_path_summary["wrong_rejection_recall"]
            )
            self.last_safety_summary = _summarize_step_safety(
                step_safety_counts
            )
            self.last_gate_contribution_summary = {
                key: float(value)
                for key, value in gate_contribution_counts.items()
            }
            denominator_map = {
                "confidence_rejected": "decision_count",
                "semantic_rejected": "decision_count",
                "commit_guard_rejected": None,
            }
            for name, denominator_key in denominator_map.items():
                if name == "commit_guard_rejected":
                    denominator = step_safety_counts["admitted_count"]
                else:
                    denominator = gate_contribution_counts[denominator_key]
                self.last_gate_contribution_summary[
                    f"{name}_rate"
                ] = _safe_ratio(
                    gate_contribution_counts[f"{name}_count"], denominator
                )
                clean_denominator = gate_contribution_counts["clean_correct_count"]
                corrupted_denominator = gate_contribution_counts["corrupted_count"]
                wrong_denominator = gate_contribution_counts["wrong_count"]
                self.last_gate_contribution_summary.update(
                    {
                        f"{name}_clean_correct_fpr": _safe_ratio(
                            gate_contribution_counts[
                                f"{name}_clean_correct_count"
                            ],
                            clean_denominator,
                        ),
                        f"{name}_corrupted_recall": _safe_ratio(
                            gate_contribution_counts[
                                f"{name}_corrupted_count"
                            ],
                            corrupted_denominator,
                        ),
                        f"{name}_wrong_rejection_recall": _safe_ratio(
                            gate_contribution_counts[f"{name}_wrong_count"],
                            wrong_denominator,
                        ),
                    }
                )
            self.last_safety_summary.update(
                {
                    f"final_path_{key}": value
                    for key, value in final_path_summary.items()
                }
            )

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        if self.algorithm_registry == "benchmark":
            from configs.benchmark_baselines import get_benchmark_hparams_class

            hparams_class = get_benchmark_hparams_class(self.dataset)
        else:
            hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def set_runtime_hparams(self, overrides):
        """Set temporary sweep/CLI overrides shared by the active dataset."""
        self._runtime_hparam_overrides.update(dict(overrides or {}))

    def set_test_time_seed(self, seed):
        """Expose the active deployment seed to stochastic TTA components."""
        seed = int(seed)
        self._current_run_seed = seed
        self.set_runtime_hparams({"test_time_seed": seed})
        return seed

    def set_scenario_hparams(self, src_id, trg_id):
        """Refresh one dataset-level setting for any source-to-target pair."""
        validate_scenario(self.dataset, src_id, trg_id)
        combined = {**self._base_alg_hparams, **self._train_params}
        combined.update(self._runtime_hparam_overrides)
        if self._ablation_hparams:
            combined.update(self._ablation_hparams)
        self._apply_backbone_overrides(combined)
        self.hparams = combined
        return self.hparams

    def _apply_backbone_overrides(self, hparams):
        """Reset dataset backbone params to defaults, then apply overrides if provided."""
        for attr, value in self._dataset_backbone_defaults.items():
            setattr(self.dataset_configs, attr, deepcopy(value))
        for attr in self._backbone_attr_names:
            if attr in hparams:
                setattr(self.dataset_configs, attr, deepcopy(hparams[attr]))

    def get_tta_model_class(self):
        if self.algorithm_registry == "benchmark":
            from benchmark_baselines.registry import get_algorithm_class as get_benchmark_algorithm_class

            tta_model_class = get_benchmark_algorithm_class(self.da_method)
        else:
            variant = None
            if self.da_method == "DuSafe":
                variant = dict(getattr(self, "hparams", {}) or {}).get(
                    "dusafe_variant"
                )
            tta_model_class = get_algorithm_class(
                self.da_method,
                variant=variant,
            )

        return tta_model_class

    def load_data_demo(self, src_id, trg_id, run_id=0, source_seed=None):
        source_seed = int(run_id if source_seed is None else source_seed)
        normalization_reference = str(
            self.hparams.get("normalization_reference", "split")
        ).strip().lower()
        if normalization_reference not in {"split", "source"}:
            raise ValueError(
                "normalization_reference must be 'split' or 'source'"
            )
        self.src_train_dl = data_generator_demo(
            self.data_path,
            src_id,
            self.dataset_configs,
            self.source_hparams,
            "train",
            seed_id=source_seed,
        )
        source_normalization_stats = (
            self.src_train_dl.dataset.normalization_stats
            if normalization_reference == "source"
            else None
        )
        self.src_test_dl = data_generator_demo(
            self.data_path,
            src_id,
            self.dataset_configs,
            self.source_hparams,
            "test",
            seed_id=source_seed,
            normalization_stats=source_normalization_stats,
        )
        self.trg_whole_dl = whole_targe_data_generator_demo(
            self.data_path,
            trg_id,
            self.dataset_configs,
            self.hparams,
            seed_id=run_id,
            normalization_stats=source_normalization_stats,
        )

    def save_tables_to_file(self, table, name):
        table.to_csv(os.path.join(self.exp_log_dir, f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, non_adapted):
        torch.save(
            {
                "non_adapted": non_adapted,
                "source_signal_metadata": getattr(
                    self, "source_signal_metadata", None
                ),
                "source_confidence_metadata": getattr(
                    self, "source_confidence_metadata", None
                ),
                "source_semantic_metadata": getattr(
                    self, "source_semantic_metadata", None
                ),
            },
            os.path.join(home_path, log_dir, "checkpoint.pt"),
        )

    def calculate_metrics(self, tta_model):
        # Main metrics use a read-only forward after all requested inner
        # updates. The legacy pre-final-update values remain available for
        # protocol reconciliation.
        self.evaluate(self.trg_whole_dl, tta_model)
        labels_cpu = self.full_labels.cpu()

        def metric_tuple(predictions, risk):
            predictions = predictions.cpu()
            predicted_labels = predictions.argmax(dim=1)
            values = (
                self.ACC(predicted_labels, labels_cpu).item(),
                self.F1(predicted_labels, labels_cpu).item(),
                self.AUROC(predictions, labels_cpu).item(),
                float(risk),
            )
            self.ACC.reset()
            self.F1.reset()
            self.AUROC.reset()
            return values

        post_update = metric_tuple(self.full_preds, self.loss.item())
        pre_final_update = metric_tuple(
            self.full_pre_final_update_preds,
            self.pre_final_update_loss.item(),
        )
        self.last_prediction_metric_summary = {
            "prediction_metric_version": 2.0,
            "post_update_accuracy": float(post_update[0]),
            "post_update_macro_f1": float(post_update[1]),
            "post_update_auroc": float(post_update[2]),
            "post_update_risk": float(post_update[3]),
            "pre_final_update_accuracy": float(pre_final_update[0]),
            "pre_final_update_macro_f1": float(pre_final_update[1]),
            "pre_final_update_auroc": float(pre_final_update[2]),
            "pre_final_update_risk": float(pre_final_update[3]),
        }
        return post_update

    def append_results_to_tables(self, table, scenario, run_id, metrics, seed=None):
        row = [scenario]
        if "seed" in table.columns:
            row.append(seed if seed is not None else getattr(self, "seed", None))
        row.append(run_id)

        if isinstance(metrics, float):
            row.append(metrics)
        elif isinstance(metrics, tuple):
            row.extend(metrics)

        # Create new dataframes for each row
        results_df = pd.DataFrame([row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table

    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        metric_start_idx = 3 if "seed" in columns else 2
        metric_cols = columns[metric_start_idx:]
        avg_metrics = [table[metric].mean() for metric in metric_cols]
        std_metrics = [table[metric].std() for metric in metric_cols]

        # Create dataframes for mean and std values
        prefix = ['mean', '-']
        prefix_std = ['std', '-']
        if "seed" in columns:
            prefix.insert(1, '-')
            prefix_std.insert(1, '-')
        mean_metrics_df = pd.DataFrame([prefix + avg_metrics], columns=columns)
        std_metrics_df = pd.DataFrame([prefix_std + std_metrics], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        if hasattr(table, "map"):
            table = table.map(format_func)
        else:
            table = table.applymap(format_func)

        return table
