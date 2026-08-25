"""Hard-physical-view candidate updates with a disjoint raw guard set.

This module is intentionally separate from :mod:`algorithms.dusafe`.  It is
the first-run implementation of the candidate/guard design and therefore does
not change the archived production DuSafe results.  Target labels are never
read here.  A runner may pair the detached rejected-candidate logits with
labels afterwards for an explicitly offline counterfactual audit.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.dusafe import DuSafe, _extract_features, _extract_primary_tensor


FIXED_SOURCE_ANCHOR_ADMISSION_MODES = (
    "joint",
    "confidence_only",
    "semantic_only",
    "all",
)


@torch.no_grad()
def fixed_source_anchor_admission(
    *,
    raw_top1_nll: torch.Tensor,
    pseudo_labels: torch.Tensor,
    confidence_nll_threshold: torch.Tensor,
    semantic_predictions: torch.Tensor,
    mode: str = "joint",
) -> Dict[str, torch.Tensor]:
    """Return one fixed-source anchor-admission decision.

    ``joint`` is the production definition.  The other modes exist only for
    dedicated ablation classes below; they do not introduce extra deployment
    gates.  The continuous source statistic is the confidence quantile.  The
    semantic term is a discrete agreement with frozen source prototypes.
    """

    mode = str(mode).strip().lower()
    if mode not in FIXED_SOURCE_ANCHOR_ADMISSION_MODES:
        raise ValueError(
            "fixed_source_anchor_admission_mode must be one of "
            f"{FIXED_SOURCE_ANCHOR_ADMISSION_MODES}, got {mode!r}"
        )
    raw_top1_nll = torch.as_tensor(raw_top1_nll).view(-1)
    pseudo_labels = torch.as_tensor(
        pseudo_labels, device=raw_top1_nll.device, dtype=torch.long
    ).view(-1)
    semantic_predictions = torch.as_tensor(
        semantic_predictions, device=raw_top1_nll.device, dtype=torch.long
    ).view(-1)
    if not (
        raw_top1_nll.numel()
        == pseudo_labels.numel()
        == semantic_predictions.numel()
    ):
        raise ValueError("anchor-admission inputs have different lengths")

    confidence_threshold = torch.as_tensor(
        confidence_nll_threshold,
        device=raw_top1_nll.device,
        dtype=raw_top1_nll.dtype,
    )
    if confidence_threshold.numel() != 1:
        raise ValueError("confidence_nll_threshold must be scalar")
    source_calibrated_confidence = raw_top1_nll.le(confidence_threshold)
    source_semantic_agreement = semantic_predictions.eq(pseudo_labels)

    if mode == "joint":
        anchor_admission = (
            source_calibrated_confidence & source_semantic_agreement
        )
    elif mode == "confidence_only":
        anchor_admission = source_calibrated_confidence
    elif mode == "semantic_only":
        anchor_admission = source_semantic_agreement
    else:
        anchor_admission = torch.ones_like(pseudo_labels, dtype=torch.bool)
    return {
        "source_calibrated_confidence": source_calibrated_confidence,
        "source_semantic_agreement": source_semantic_agreement,
        "anchor_admission_mask": anchor_admission,
    }


def predictive_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL(reference || candidate)`` without reducing samples."""

    reference_log_prob = reference_logits.detach().log_softmax(dim=-1)
    reference_prob = reference_log_prob.exp()
    candidate_log_prob = candidate_logits.log_softmax(dim=-1)
    return (
        reference_prob * (reference_log_prob - candidate_log_prob)
    ).sum(dim=-1)


@torch.no_grad()
def select_hard_physical_views(
    reference_logits: torch.Tensor,
    candidate_logits_by_view: torch.Tensor,
    view_inputs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Select the maximum-KL physical view independently for every sample."""

    if candidate_logits_by_view.dim() != 3:
        raise ValueError("candidate logits must have shape [V, B, K]")
    if view_inputs.dim() != 4:
        raise ValueError("physical views must have shape [V, B, C, T]")
    view_count, batch_size, _ = candidate_logits_by_view.shape
    if view_count < 2 or view_count % 2:
        raise ValueError("hard selection requires positive/inverse view pairs")
    if reference_logits.shape[0] != batch_size:
        raise ValueError("reference and candidate batch sizes differ")
    if tuple(view_inputs.shape[:2]) != (view_count, batch_size):
        raise ValueError("view inputs and view logits have different leading shapes")

    reference_log_prob = reference_logits.log_softmax(dim=1)
    reference_prob = reference_log_prob.exp()
    candidate_log_prob = candidate_logits_by_view.log_softmax(dim=2)
    kl_by_view = (
        reference_prob.unsqueeze(0)
        * (reference_log_prob.unsqueeze(0) - candidate_log_prob)
    ).sum(dim=2)
    selected_index = kl_by_view.argmax(dim=0)
    sample_index = torch.arange(batch_size, device=view_inputs.device)
    selected_inputs = view_inputs[selected_index, sample_index]
    selected_logits = candidate_logits_by_view[selected_index, sample_index]
    selected_kl = kl_by_view[selected_index, sample_index]
    reference_labels = reference_logits.argmax(dim=1)
    selected_labels = selected_logits.argmax(dim=1)
    pair_count = view_count // 2
    return {
        "selected_index": selected_index,
        "selected_inputs": selected_inputs,
        "selected_logits": selected_logits,
        "selected_kl": selected_kl,
        "selected_label_flip": selected_labels.ne(reference_labels),
        "selected_positive": selected_index.lt(pair_count),
        "kl_by_view": kl_by_view,
    }


@torch.no_grad()
def stratified_anchor_split(
    admitted_mask: torch.Tensor,
    pseudo_labels: torch.Tensor,
    sample_ids: Optional[torch.Tensor] = None,
    *,
    guard_fraction: float = 0.25,
    split_seed: int = 271828,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split admitted anchors into optimization and guard.

    Every pseudo-label stratum containing at least two anchors contributes at
    least one guard sample and retains at least one optimization sample.
    Singleton strata remain optimization samples.  If all strata are
    singletons, a deterministic global fallback supplies one guard sample.
    """

    admitted_mask = torch.as_tensor(admitted_mask, dtype=torch.bool).view(-1)
    pseudo_labels = torch.as_tensor(
        pseudo_labels, dtype=torch.long, device=admitted_mask.device
    ).view(-1)
    if admitted_mask.numel() != pseudo_labels.numel():
        raise ValueError("admission mask and pseudo-label lengths differ")
    if not 0.0 < float(guard_fraction) < 1.0:
        raise ValueError("guard_fraction must lie in (0, 1)")
    if sample_ids is None:
        sample_ids = torch.arange(
            admitted_mask.numel(), device=admitted_mask.device, dtype=torch.long
        )
    else:
        sample_ids = torch.as_tensor(
            sample_ids, dtype=torch.long, device=admitted_mask.device
        ).view(-1)
    if sample_ids.numel() != admitted_mask.numel():
        raise ValueError("sample_ids and admission mask lengths differ")

    optimization = admitted_mask.clone()
    guard = torch.zeros_like(admitted_mask)
    admitted_indices = admitted_mask.nonzero(as_tuple=False).flatten()
    if admitted_indices.numel() < 2:
        return optimization, guard

    # The integer hash makes the split independent of loader ordering while
    # remaining reproducible from target indices and a fixed, label-free seed.
    hashed = (
        sample_ids * 1_103_515_245
        + int(split_seed) * 12_345
        + 1_013_904_223
    ).remainder(2_147_483_647)
    for label in pseudo_labels[admitted_indices].unique(sorted=True):
        class_indices = admitted_indices[
            pseudo_labels[admitted_indices].eq(label)
        ]
        class_count = int(class_indices.numel())
        if class_count < 2:
            continue
        order = torch.argsort(hashed[class_indices], stable=True)
        guard_count = int(math.floor(class_count * float(guard_fraction) + 0.5))
        guard_count = max(1, min(class_count - 1, guard_count))
        selected = class_indices[order[:guard_count]]
        guard[selected] = True
        optimization[selected] = False

    if not guard.any():
        order = torch.argsort(hashed[admitted_indices], stable=True)
        selected = admitted_indices[order[0]]
        guard[selected] = True
        optimization[selected] = False
    return optimization, guard


class DuSafeGuardedCandidate(DuSafe):
    """Train a candidate on hard views, then certify it on raw guard anchors."""

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(configs, hparams, model, optimizer)
        explicit_admission_mode = hparams.get(
            "fixed_source_anchor_admission_mode"
        )
        if explicit_admission_mode is None:
            if self.enable_confidence_gate and self.enable_source_semantic_gate:
                explicit_admission_mode = "joint"
            elif self.enable_confidence_gate:
                explicit_admission_mode = "confidence_only"
            elif self.enable_source_semantic_gate:
                explicit_admission_mode = "semantic_only"
            else:
                explicit_admission_mode = "all"
        self.fixed_source_anchor_admission_mode = str(
            explicit_admission_mode
        ).strip().lower()
        if (
            self.fixed_source_anchor_admission_mode
            not in FIXED_SOURCE_ANCHOR_ADMISSION_MODES
        ):
            raise ValueError(
                "unsupported fixed-source anchor-admission mode: "
                f"{self.fixed_source_anchor_admission_mode!r}"
            )
        self.guard_fraction = float(hparams.get("candidate_guard_fraction", 0.25))
        if not 0.0 < self.guard_fraction < 1.0:
            raise ValueError("candidate_guard_fraction must lie in (0, 1)")
        self.guard_split_seed = int(
            hparams.get("candidate_guard_split_seed", 271828)
        )
        self.backtracking_scale = float(
            hparams.get("candidate_backtracking_scale", 0.5)
        )
        if not 0.0 < self.backtracking_scale < 1.0:
            raise ValueError("candidate_backtracking_scale must lie in (0, 1)")
        self.record_gradient_diagnostics = bool(
            hparams.get("candidate_record_gradient_diagnostics", True)
        )
        if self.enable_ssaw and (
            not self.ssaw.antithetic or self.ssaw.antithetic_pairs < 1
        ):
            raise ValueError(
                "guarded hard-view selection requires antithetic SSAW views"
            )
        self._last_candidate_attempts: list[Dict[str, object]] = []
        self._last_rejected_candidate_logits: list[Dict[str, object]] = []

    @staticmethod
    def _trainable_parameters(model) -> list[torch.Tensor]:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def _raw_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        bn_snapshot = self._snapshot_bn_buffers(self.model)
        try:
            features = _extract_features(self.model, inputs)
            return self.model.classifier(features).detach()
        finally:
            self._restore_bn_buffers(bn_snapshot)

    @torch.no_grad()
    def _guard_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate a guard subset with fixed source BN statistics.

        HAR normally uses test-time batch statistics.  Using the full target
        batch for certification would make guard predictions depend on
        optimization/non-admitted samples.  This path temporarily freezes only
        BatchNorm modules, evaluates the disjoint guard subset, and restores
        every training flag afterwards.  Dropout is already disabled by
        :meth:`DuSafe.configure_model`.
        """

        batch_norm_states = []
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.running_mean is None or module.running_var is None:
                    raise RuntimeError(
                        "guard certification requires source BatchNorm buffers"
                    )
                batch_norm_states.append((module, bool(module.training)))
                module.training = False
        try:
            features = _extract_features(self.model, inputs)
            return self.model.classifier(features).detach()
        finally:
            for module, training in batch_norm_states:
                module.training = training

    @torch.no_grad()
    def predict_raw(self, batch_data) -> torch.Tensor:
        """Read-only raw prediction used by the diagnostic runner."""

        return self._raw_logits(_extract_primary_tensor(batch_data))

    def _capture_candidate_state(self) -> Dict[str, object]:
        state = {
            "model": {
                name: tensor.detach().clone()
                for name, tensor in self.model.state_dict().items()
            },
            "optimizer": (
                None
                if self.optimizer is None
                else copy.deepcopy(self.optimizer.state_dict())
            ),
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": None,
        }
        model_device = next(self.model.parameters()).device
        if model_device.type == "cuda":
            state["cuda_rng"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
        return state

    def _restore_candidate_state(self, snapshot: Mapping[str, object]) -> None:
        self.model.load_state_dict(snapshot["model"], strict=True)
        if self.optimizer is not None and snapshot["optimizer"] is not None:
            self.optimizer.load_state_dict(snapshot["optimizer"])
            self.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(snapshot["cpu_rng"])
        cuda_rng = snapshot.get("cuda_rng")
        if cuda_rng is not None and next(self.model.parameters()).is_cuda:
            torch.cuda.set_rng_state_all(cuda_rng)

    @staticmethod
    def _gradient_norm(gradients) -> torch.Tensor:
        values = [gradient for gradient in gradients if gradient is not None]
        if not values:
            return torch.tensor(0.0)
        device = values[0].device
        squared = torch.zeros((), device=device, dtype=torch.float32)
        for gradient in values:
            squared = squared + gradient.detach().float().square().sum()
        return squared.sqrt()

    @staticmethod
    def _gradient_cosine(left, right) -> torch.Tensor:
        numerator = None
        left_sq = None
        right_sq = None
        for left_value, right_value in zip(left, right):
            if left_value is None or right_value is None:
                continue
            current_dot = (
                left_value.detach().float() * right_value.detach().float()
            ).sum()
            current_left = left_value.detach().float().square().sum()
            current_right = right_value.detach().float().square().sum()
            numerator = current_dot if numerator is None else numerator + current_dot
            left_sq = current_left if left_sq is None else left_sq + current_left
            right_sq = current_right if right_sq is None else right_sq + current_right
        if numerator is None:
            return torch.tensor(float("nan"))
        return numerator / (left_sq.sqrt() * right_sq.sqrt()).clamp_min(1e-12)

    def _run_candidate_attempt(
        self,
        *,
        raw_inputs: torch.Tensor,
        optimization_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        guard_reference_labels: torch.Tensor,
        selected_view_inputs: Optional[torch.Tensor],
        learning_rate_scale: float,
        attempt_index: int,
    ) -> Dict[str, object]:
        parameters = self._trainable_parameters(self.model)
        raw_opt = raw_inputs[optimization_mask]
        pseudo_opt = pseudo_labels[optimization_mask].detach()
        teacher_opt = teacher_logits[optimization_mask].detach()
        view_opt = (
            None
            if selected_view_inputs is None
            else selected_view_inputs[optimization_mask]
        )
        original_learning_rates = []
        if self.optimizer is not None:
            original_learning_rates = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
            for group, learning_rate in zip(
                self.optimizer.param_groups, original_learning_rates
            ):
                group["lr"] = learning_rate * float(learning_rate_scale)

        raw_norms = []
        ssaw_norms = []
        weighted_ratios = []
        gradient_cosines = []
        raw_losses = []
        ssaw_losses = []
        finite = True
        try:
            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)
                raw_logits = self.model.classifier(
                    _extract_features(self.model, raw_opt)
                )
                raw_loss = F.cross_entropy(raw_logits, pseudo_opt)
                use_ssaw = bool(
                    self.enable_ssaw
                    and self.ssaw_auxiliary_weight > 0.0
                    and view_opt is not None
                )
                raw_gradients = torch.autograd.grad(
                    raw_loss,
                    parameters,
                    allow_unused=True,
                )
                if use_ssaw:
                    view_logits = self.model.classifier(
                        _extract_features(self.model, view_opt)
                    )
                    ssaw_loss = predictive_kl(teacher_opt, view_logits).mean()
                    ssaw_gradients = torch.autograd.grad(
                        ssaw_loss,
                        parameters,
                        allow_unused=True,
                    )
                else:
                    ssaw_loss = raw_loss.detach() * 0.0
                    ssaw_gradients = tuple(None for _ in parameters)

                raw_norm = self._gradient_norm(raw_gradients).to(raw_loss.device)
                ssaw_norm = self._gradient_norm(ssaw_gradients).to(raw_loss.device)
                weighted_ssaw_norm = ssaw_norm * float(self.ssaw_auxiliary_weight)
                ratio = weighted_ssaw_norm / raw_norm.clamp_min(1e-12)
                cosine = self._gradient_cosine(raw_gradients, ssaw_gradients).to(
                    raw_loss.device
                )
                for parameter, raw_gradient, ssaw_gradient in zip(
                    parameters, raw_gradients, ssaw_gradients
                ):
                    combined = None
                    if raw_gradient is not None:
                        combined = raw_gradient.detach()
                    if ssaw_gradient is not None:
                        weighted = ssaw_gradient.detach() * float(
                            self.ssaw_auxiliary_weight
                        )
                        combined = weighted if combined is None else combined + weighted
                    parameter.grad = combined

                objective_values = [raw_loss.detach(), ssaw_loss.detach()]
                finite = bool(
                    self._tensors_are_finite(objective_values)
                    and self._tensors_are_finite(
                        [parameter.grad for parameter in parameters]
                    )
                )
                if not finite:
                    break
                self.optimizer.step()
                finite = self._tensors_are_finite(parameters)
                raw_losses.append(float(raw_loss.detach().item()))
                ssaw_losses.append(float(ssaw_loss.detach().item()))
                if self.record_gradient_diagnostics:
                    raw_norms.append(float(raw_norm.detach().item()))
                    ssaw_norms.append(float(ssaw_norm.detach().item()))
                    weighted_ratios.append(float(ratio.detach().item()))
                    gradient_cosines.append(float(cosine.detach().item()))
                if not finite:
                    break
        finally:
            if self.optimizer is not None:
                for group, learning_rate in zip(
                    self.optimizer.param_groups, original_learning_rates
                ):
                    group["lr"] = learning_rate

        candidate_logits = None
        guard_flip_count = 0
        guard_check_skipped = not finite
        guard_flip_mask = torch.zeros_like(guard_mask)
        if finite:
            candidate_guard_logits = self._guard_logits(raw_inputs[guard_mask])
            candidate_guard_labels = candidate_guard_logits.argmax(dim=1)
            guard_local_flips = candidate_guard_labels.ne(guard_reference_labels)
            guard_flip_mask[guard_mask] = guard_local_flips
            guard_flip_count = int(guard_local_flips.sum().item())
        passed = bool(finite and guard_flip_count == 0)
        # Full-batch logits are retained only for a rejected candidate and are
        # never consulted by the commit decision.
        if finite and not passed:
            candidate_logits = self._raw_logits(raw_inputs)
        mean = lambda values: float(sum(values) / len(values)) if values else float("nan")
        return {
            "attempt_index": int(attempt_index),
            "learning_rate_scale": float(learning_rate_scale),
            "finite": bool(finite),
            "passed": passed,
            "guard_flip_count": int(guard_flip_count),
            "guard_check_skipped": bool(guard_check_skipped),
            "guard_flip_mask": guard_flip_mask.detach().cpu(),
            "candidate_logits": (
                None if candidate_logits is None else candidate_logits.detach().cpu()
            ),
            "raw_loss_mean": mean(raw_losses),
            "ssaw_loss_mean": mean(ssaw_losses),
            "raw_gradient_norm_mean": mean(raw_norms),
            "ssaw_gradient_norm_mean": mean(ssaw_norms),
            "weighted_ssaw_to_raw_gradient_ratio_mean": mean(weighted_ratios),
            "raw_ssaw_gradient_cosine_mean": mean(gradient_cosines),
            "completed_inner_steps": int(len(raw_losses)),
        }

    def _initial_batch_state(
        self, raw_inputs: torch.Tensor
    ) -> Dict[str, object]:
        raw_logits = self._raw_logits(raw_inputs)
        pseudo_labels = raw_logits.argmax(dim=1)
        raw_nll = -raw_logits.log_softmax(dim=1).gather(
            1, pseudo_labels[:, None]
        ).squeeze(1)
        _semantic_mask, semantic_prediction, semantic_margin = (
            self._source_semantic_decision(raw_inputs, pseudo_labels)
        )
        if self.enable_confidence_gate:
            if not self.source_confidence_reference_ready:
                raise RuntimeError("source confidence metadata was not loaded")
            confidence_threshold = self.confidence_nll_threshold
        else:
            confidence_threshold = raw_nll.new_tensor(float("inf"))
        admission = fixed_source_anchor_admission(
            raw_top1_nll=raw_nll,
            pseudo_labels=pseudo_labels,
            confidence_nll_threshold=confidence_threshold,
            semantic_predictions=semantic_prediction,
            mode=self.fixed_source_anchor_admission_mode,
        )
        return {
            "raw_logits": raw_logits,
            "pseudo_labels": pseudo_labels,
            "raw_nll": raw_nll,
            "semantic_mask": admission["source_semantic_agreement"],
            "semantic_prediction": semantic_prediction,
            "semantic_margin": semantic_margin,
            "confidence_mask": admission["source_calibrated_confidence"],
            "source_calibrated_confidence": admission[
                "source_calibrated_confidence"
            ],
            "source_semantic_agreement": admission[
                "source_semantic_agreement"
            ],
            "anchor_admission_mask": admission["anchor_admission_mask"],
            # Compatibility alias for existing diagnostics.
            "admission_mask": admission["anchor_admission_mask"],
        }

    def _hard_view_state(
        self, raw_inputs: torch.Tensor, raw_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        self.ssaw.clear_cached_view()
        prepared = self.ssaw.prepare_view_inputs(
            raw_inputs,
            normalization_mean=self.source_normalization_mean,
            normalization_std=self.source_normalization_std,
            reuse_cached_view=False,
        )
        view_inputs = torch.as_tensor(prepared["view_inputs"])
        view_logits = []
        for current_view in view_inputs:
            view_logits.append(self._raw_logits(current_view))
        return select_hard_physical_views(
            raw_logits,
            torch.stack(view_logits),
            view_inputs,
        )

    def forward(self, inputs, trg_idx=None):
        raw_inputs = _extract_primary_tensor(inputs)
        if not torch.is_tensor(raw_inputs) or raw_inputs.dim() != 3:
            raise ValueError("DuSafeGuardedCandidate expects [B, C, T]")
        initial = self._initial_batch_state(raw_inputs)
        raw_logits = initial["raw_logits"]
        pseudo_labels = initial["pseudo_labels"]
        admission_mask = initial["anchor_admission_mask"]
        sample_ids = (
            torch.arange(raw_inputs.size(0), device=raw_inputs.device)
            if trg_idx is None
            else torch.as_tensor(
                trg_idx, device=raw_inputs.device, dtype=torch.long
            ).view(-1)
        )
        optimization_mask, guard_mask = stratified_anchor_split(
            admission_mask,
            pseudo_labels,
            sample_ids,
            guard_fraction=self.guard_fraction,
            split_seed=self.guard_split_seed,
        )

        hard_view = None
        selected_view_inputs = None
        selected_kl = raw_inputs.new_zeros(raw_inputs.size(0))
        selected_flip = torch.zeros_like(admission_mask)
        selected_positive = torch.zeros_like(admission_mask)
        if self.enable_ssaw:
            if (
                self.source_normalization_mean.numel() != raw_inputs.size(1)
                or self.source_normalization_std.numel() != raw_inputs.size(1)
            ):
                raise RuntimeError("fixed source normalization statistics are required")
            hard_view = self._hard_view_state(raw_inputs, raw_logits)
            selected_view_inputs = hard_view["selected_inputs"].detach()
            selected_kl = hard_view["selected_kl"].detach()
            selected_flip = hard_view["selected_label_flip"].detach()
            selected_positive = hard_view["selected_positive"].detach()

        eligible = bool(
            self.enable_adaptation
            and self.optimizer is not None
            and optimization_mask.any()
            and guard_mask.any()
        )
        snapshot = None
        attempts: list[Dict[str, object]] = []
        committed = False
        first_commit = False
        rescued = False
        if eligible:
            guard_reference_logits = self._guard_logits(raw_inputs[guard_mask])
            guard_reference_labels = guard_reference_logits.argmax(dim=1)
            guard_reference_mismatch_count = int(
                guard_reference_labels.ne(pseudo_labels[guard_mask]).sum().item()
            )
            snapshot = self._capture_candidate_state()
            try:
                first = self._run_candidate_attempt(
                    raw_inputs=raw_inputs,
                    optimization_mask=optimization_mask,
                    guard_mask=guard_mask,
                    pseudo_labels=pseudo_labels,
                    teacher_logits=raw_logits,
                    guard_reference_labels=guard_reference_labels,
                    selected_view_inputs=selected_view_inputs,
                    learning_rate_scale=1.0,
                    attempt_index=1,
                )
            except BaseException:
                self._restore_candidate_state(snapshot)
                raise
            attempts.append(first)
            if bool(first["passed"]):
                committed = True
                first_commit = True
            else:
                self._restore_candidate_state(snapshot)
                try:
                    second = self._run_candidate_attempt(
                        raw_inputs=raw_inputs,
                        optimization_mask=optimization_mask,
                        guard_mask=guard_mask,
                        pseudo_labels=pseudo_labels,
                        teacher_logits=raw_logits,
                        guard_reference_labels=guard_reference_labels,
                        selected_view_inputs=selected_view_inputs,
                        learning_rate_scale=self.backtracking_scale,
                        attempt_index=2,
                    )
                except BaseException:
                    self._restore_candidate_state(snapshot)
                    raise
                attempts.append(second)
                if bool(second["passed"]):
                    committed = True
                    rescued = True
                else:
                    self._restore_candidate_state(snapshot)

        else:
            guard_reference_mismatch_count = 0

        rejected = []
        for attempt in attempts:
            if not bool(attempt["passed"]):
                candidate_logits = attempt.get("candidate_logits")
                if candidate_logits is not None:
                    rejected.append(
                        {
                            "attempt_index": int(attempt["attempt_index"]),
                            "learning_rate_scale": float(
                                attempt["learning_rate_scale"]
                            ),
                            "candidate_logits": candidate_logits,
                            "guard_flip_count": int(attempt["guard_flip_count"]),
                        }
                    )
        self._last_candidate_attempts = attempts
        self._last_rejected_candidate_logits = rejected

        final_skip = bool(eligible and not committed)
        active_mask = optimization_mask if committed else torch.zeros_like(admission_mask)
        first_guard_flips = (
            int(attempts[0]["guard_flip_count"]) if attempts else 0
        )
        retry_guard_flips = (
            int(attempts[1]["guard_flip_count"]) if len(attempts) > 1 else 0
        )
        # Headline hard-view diagnostics use the samples that actually enter
        # the SSAW loss.  Views evaluated for guard/non-admitted samples are
        # not credited as training participation.
        selected_mask = (
            optimization_mask
            if self.enable_ssaw
            else torch.zeros_like(admission_mask)
        )
        selected_count = int(selected_mask.sum().item())
        selected_positive_count = int((selected_positive & selected_mask).sum().item())
        selected_flip_count = int((selected_flip & selected_mask).sum().item())
        selected_kl_sum = float(selected_kl[selected_mask].sum().item()) if selected_count else 0.0
        attempt_for_grad = attempts[-1] if attempts else {}

        repeat_count = max(1, int(self.steps))
        repeat = lambda value: value.unsqueeze(0).repeat(repeat_count, 1).detach().cpu()
        zero_mask = torch.zeros_like(admission_mask)
        self._last_gate_log = {
            "pseudo_labels": pseudo_labels.detach().cpu(),
            "fixed_source_anchor_admission_mode": (
                self.fixed_source_anchor_admission_mode
            ),
            "source_calibrated_confidence": initial[
                "source_calibrated_confidence"
            ].detach().cpu(),
            "source_semantic_agreement": initial[
                "source_semantic_agreement"
            ].detach().cpu(),
            "anchor_admission_mask": admission_mask.detach().cpu(),
            # Legacy names remain read-only aliases for archived analyzers.
            "confidence_mask": initial["confidence_mask"].detach().cpu(),
            "semantic_mask": initial["semantic_mask"].detach().cpu(),
            "base_admission_mask": admission_mask.detach().cpu(),
            "admission_mask": admission_mask.detach().cpu(),
            "optimization_mask": optimization_mask.detach().cpu(),
            "guard_mask": guard_mask.detach().cpu(),
            "active_mask": active_mask.detach().cpu(),
            "selected_mask": active_mask.detach().cpu(),
            "ssaw_veto_mask": zero_mask.detach().cpu(),
            "ssaw_view_selected_mask": selected_mask.detach().cpu(),
            "ssaw_label_flip": selected_flip.detach().cpu(),
            "ssaw_selected_kl": selected_kl.detach().cpu(),
            "ssaw_selected_positive": selected_positive.detach().cpu(),
            "raw_top1_nll": initial["raw_nll"].detach().cpu(),
            "source_semantic_prediction": initial["semantic_prediction"].detach().cpu(),
            "source_semantic_margin": initial["semantic_margin"].detach().cpu(),
            "inner_pseudo_labels": repeat(pseudo_labels),
            "inner_confidence_masks": repeat(initial["confidence_mask"]),
            "inner_semantic_masks": repeat(initial["semantic_mask"]),
            "inner_base_admission_masks": repeat(admission_mask),
            "inner_admission_masks": repeat(admission_mask),
            "inner_active_masks": repeat(active_mask),
            "inner_ssaw_veto_masks": repeat(zero_mask),
            "inner_step_count": repeat_count,
        }
        self._last_batch_log = {
            "sample_count": float(raw_inputs.size(0)),
            "fixed_source_anchor_admission_count": float(
                admission_mask.sum().item()
            ),
            "fixed_source_anchor_admission_rate": float(
                admission_mask.float().mean().item()
            ),
            "admitted_count": float(admission_mask.sum().item()),
            "optimization_count": float(optimization_mask.sum().item()),
            "guard_count": float(guard_mask.sum().item()),
            "candidate_eligible": float(eligible),
            "first_attempt_commit": float(first_commit),
            "backtracking_attempted": float(len(attempts) == 2),
            "backtracking_rescue": float(rescued),
            "final_skip": float(final_skip),
            "candidate_committed": float(committed),
            "first_guard_flip_count": float(first_guard_flips),
            "retry_guard_flip_count": float(retry_guard_flips),
            "guard_flip_count": float(first_guard_flips + retry_guard_flips),
            "guard_reference_mismatch_count": float(
                guard_reference_mismatch_count
            ),
            "guard_check_skipped_count": float(
                sum(bool(item.get("guard_check_skipped", False)) for item in attempts)
            ),
            "selected_view_count": float(selected_count),
            "selected_positive_count": float(selected_positive_count),
            "selected_negative_count": float(selected_count - selected_positive_count),
            "selected_view_label_flip_count": float(selected_flip_count),
            "selected_kl_sum": float(selected_kl_sum),
            "raw_gradient_norm_mean": float(
                attempt_for_grad.get("raw_gradient_norm_mean", float("nan"))
            ),
            "ssaw_gradient_norm_mean": float(
                attempt_for_grad.get("ssaw_gradient_norm_mean", float("nan"))
            ),
            "weighted_ssaw_to_raw_gradient_ratio_mean": float(
                attempt_for_grad.get(
                    "weighted_ssaw_to_raw_gradient_ratio_mean", float("nan")
                )
            ),
            "raw_ssaw_gradient_cosine_mean": float(
                attempt_for_grad.get("raw_ssaw_gradient_cosine_mean", float("nan"))
            ),
        }
        # The pre-update raw logits preserve the trainer's pre-final-output
        # semantics.  Post-update metrics must call ``predict_raw``.
        return raw_logits.detach()


def _ablation_hparams(
    hparams: Mapping[str, object],
    *,
    enable_ssaw: bool,
    enable_confidence: bool,
    enable_semantic: bool,
    admission_mode: str,
) -> Dict[str, object]:
    values = dict(hparams)
    values.update(
        {
            "enable_ssaw": bool(enable_ssaw),
            "enable_confidence_gate": bool(enable_confidence),
            "enable_source_semantic_gate": bool(enable_semantic),
            "fixed_source_anchor_admission_mode": str(admission_mode),
        }
    )
    return values


class DuSafeGuardedCandidateNoSSAW(DuSafeGuardedCandidate):
    """Dedicated whole-branch ablation: no physical views or SSAW loss."""

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(
            configs,
            _ablation_hparams(
                hparams,
                enable_ssaw=False,
                enable_confidence=True,
                enable_semantic=True,
                admission_mode="joint",
            ),
            model,
            optimizer,
        )

    def _hard_view_state(self, *args, **kwargs):
        raise AssertionError("No-SSAW ablation must not construct physical views")


class DuSafeGuardedCandidateConfidenceOnly(DuSafeGuardedCandidate):
    """Internal admission ablation without the source-semantic branch."""

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(
            configs,
            _ablation_hparams(
                hparams,
                enable_ssaw=True,
                enable_confidence=True,
                enable_semantic=False,
                admission_mode="confidence_only",
            ),
            model,
            optimizer,
        )


class DuSafeGuardedCandidateSemanticOnly(DuSafeGuardedCandidate):
    """Internal admission ablation without the confidence-quantile branch."""

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(
            configs,
            _ablation_hparams(
                hparams,
                enable_ssaw=True,
                enable_confidence=False,
                enable_semantic=True,
                admission_mode="semantic_only",
            ),
            model,
            optimizer,
        )


class DuSafeGuardedCandidateNoAnchorAdmission(DuSafeGuardedCandidate):
    """SSAW-only factorial cell: every raw prediction becomes an anchor."""

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(
            configs,
            _ablation_hparams(
                hparams,
                enable_ssaw=True,
                enable_confidence=False,
                enable_semantic=False,
                admission_mode="all",
            ),
            model,
            optimizer,
        )


class DuSafeGuardedCandidateNoAdmissionNoSSAW(
    DuSafeGuardedCandidateNoSSAW
):
    """Neither-factor factorial cell: raw-only updates on every sample."""

    def __init__(self, configs, hparams, model, optimizer):
        DuSafeGuardedCandidate.__init__(
            self,
            configs,
            _ablation_hparams(
                hparams,
                enable_ssaw=False,
                enable_confidence=False,
                enable_semantic=False,
                admission_mode="all",
            ),
            model,
            optimizer,
        )

    def _hard_view_state(self, *args, **kwargs):
        raise AssertionError(
            "No-admission/No-SSAW ablation must not construct physical views"
        )


GUARDED_CANDIDATE_VARIANTS: Dict[str, Type[DuSafeGuardedCandidate]] = {
    "Full": DuSafeGuardedCandidate,
    "No-SSAW": DuSafeGuardedCandidateNoSSAW,
    "Confidence-Only-Admission": DuSafeGuardedCandidateConfidenceOnly,
    "Semantic-Only-Admission": DuSafeGuardedCandidateSemanticOnly,
    "No-Anchor-Admission": DuSafeGuardedCandidateNoAnchorAdmission,
    "No-Admission-No-SSAW": DuSafeGuardedCandidateNoAdmissionNoSSAW,
}


__all__ = [
    "FIXED_SOURCE_ANCHOR_ADMISSION_MODES",
    "GUARDED_CANDIDATE_VARIANTS",
    "DuSafeGuardedCandidate",
    "DuSafeGuardedCandidateNoSSAW",
    "DuSafeGuardedCandidateConfidenceOnly",
    "DuSafeGuardedCandidateSemanticOnly",
    "DuSafeGuardedCandidateNoAnchorAdmission",
    "DuSafeGuardedCandidateNoAdmissionNoSSAW",
    "fixed_source_anchor_admission",
    "predictive_kl",
    "select_hard_physical_views",
    "stratified_anchor_split",
]
