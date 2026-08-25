"""Small, auditable time-series adapters for the benchmark-only registry.

The update rules preserve the characteristic operation of each cited method
while adapting the official image-stream call shape to this repository's
``BaseTestTimeAlgorithm`` contract: ``forward(inputs, trg_idx=None)`` receives
either a tensor or ``{"data": tensor}``, and returns one logits tensor.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm


def extract_primary_tensor(batch_data):
    if isinstance(batch_data, dict):
        batch_data = batch_data.get("data", batch_data)
    if isinstance(batch_data, (tuple, list)):
        return batch_data[0]
    return batch_data


def softmax_entropy(logits):
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def teacher_cross_entropy(student_logits, teacher_logits):
    teacher_prob = teacher_logits.detach().softmax(dim=1)
    return -(teacher_prob * student_logits.log_softmax(dim=1)).sum(dim=1)


def configure_norm_adaptation(model, train_all=False, running_stats=False):
    model.train()
    model.requires_grad_(bool(train_all))
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if running_stats:
                module.track_running_stats = True
                module.momentum = 0.2
            else:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    return model


def clone_frozen(model):
    cloned = deepcopy(model)
    cloned.requires_grad_(False)
    cloned.eval()
    return cloned


@torch.no_grad()
def update_ema(teacher, student, momentum):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.mul_(float(momentum)).add_(student_param, alpha=1.0 - float(momentum))


def smooth_amplitude_augment(x, sigma=0.1, control_points=8, noise_std=0.005):
    if x.dim() != 3:
        raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
    batch, channels, timesteps = x.shape
    points = max(2, min(int(control_points), timesteps))
    log_scale = torch.randn(batch, channels, points, device=x.device, dtype=x.dtype)
    log_scale = log_scale * torch.log1p(torch.as_tensor(max(0.0, sigma), device=x.device, dtype=x.dtype))
    curve = F.interpolate(log_scale, size=timesteps, mode="linear", align_corners=True).exp()
    low = max(0.05, 1.0 - 3.0 * max(0.0, sigma))
    high = 1.0 + 3.0 * max(0.0, sigma)
    augmented = x * curve.clamp(low, high)
    if noise_std > 0:
        scale = x.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
        augmented = augmented + torch.randn_like(x) * scale * float(noise_std)
    return augmented


def _safe_selected_mask(x, value=True):
    return torch.full((x.size(0),), bool(value), dtype=torch.bool, device=x.device)


class SAM(torch.optim.Optimizer):
    """Minimal SAM wrapper used by SAR and SoTTA."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.data.clone()
                factor = torch.pow(parameter, 2) if group["adaptive"] else 1.0
                parameter.add_(factor * parameter.grad * scale.to(parameter))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self, zero_grad=False):
        for group in self.param_groups:
            for parameter in group["params"]:
                old = self.state[parameter].get("old_p")
                if old is not None:
                    parameter.data.copy_(old)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.restore(zero_grad=False)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        if not self.param_groups or not self.param_groups[0]["params"]:
            return torch.tensor(0.0)
        device = self.param_groups[0]["params"][0].device
        values = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    factor = torch.abs(parameter) if group["adaptive"] else 1.0
                    values.append((factor * parameter.grad).norm().to(device))
        return torch.norm(torch.stack(values), p=2) if values else torch.tensor(0.0, device=device)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class Tent(BaseTestTimeAlgorithm):
    """TENT entropy minimization over normalization affine parameters."""

    def __init__(self, configs, hparams, model, optimizer):
        self.episodic = bool(hparams.get("episodic", False))
        super().__init__(configs, hparams, model, optimizer)
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer else None

    def configure_model(self, model):
        return configure_norm_adaptation(model, train_all=False)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del trg_idx
        if self.episodic:
            self.reset()
        x = extract_primary_tensor(batch_data)
        logits = model(x)
        entropy = softmax_entropy(logits)
        loss = entropy.mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        selected = _safe_selected_mask(x)
        self._selected_counter += int(selected.sum())
        self._last_gate_log = {"selected_mask": selected.detach().cpu()}
        self._last_batch_log = {
            "selected_pass_rate": 1.0,
            "batch_entropy": float(entropy.mean().detach()),
            "total_loss": float(loss.detach()),
        }
        return logits


class EATA(BaseTestTimeAlgorithm):
    """EATA entropy/diversity filtering with Fisher-style anchoring."""

    def __init__(self, configs, hparams, model, optimizer):
        self.adapt_keywords = tuple(hparams.get("adapt_keywords", ("classifier", "adapter")))
        self.e_margin = float(hparams.get("e_margin", math.log(configs.num_classes) * 0.4))
        self.d_margin = float(hparams.get("d_margin", 0.05))
        self.fisher_alpha = float(hparams.get("fisher_alpha", 2000.0))
        # The current benchmark profile intentionally has no source Fisher
        # diagonal.  EATA's Fisher anchor is therefore disabled by default;
        # silently replacing it with an unscaled L2 penalty changes EATA.
        self.fisher_enabled = bool(hparams.get("fisher_enabled", False))
        self.grad_clip = float(hparams.get("grad_clip", 5.0))
        self.current_model_probs = None
        super().__init__(configs, hparams, model, optimizer)
        self.theta0 = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.fishers = {}
        if self.fisher_enabled:
            fisher_state = hparams.get("fisher_state")
            fisher_path = hparams.get("fisher_path")
            if fisher_state is None and isinstance(fisher_path, str) and os.path.exists(fisher_path):
                fisher_state = torch.load(fisher_path, map_location="cpu")
            if isinstance(fisher_state, dict) and "fishers" in fisher_state:
                self.fisher_metadata = {
                    key: value
                    for key, value in fisher_state.items()
                    if key != "fishers"
                }
                fisher_state = fisher_state["fishers"]
            else:
                self.fisher_metadata = {}
            if not isinstance(fisher_state, dict):
                raise ValueError(
                    "EATA fisher_enabled=True requires fisher_state or "
                    "an existing fisher_path"
                )
            for name, value in fisher_state.items():
                if isinstance(value, (list, tuple)):
                    diagonal, reference = value[0], value[1]
                else:
                    diagonal, reference = value, self.theta0.get(name)
                if reference is None:
                    continue
                self.fishers[name] = (
                    torch.as_tensor(diagonal).detach().clone(),
                    torch.as_tensor(reference).detach().clone(),
                )
            if not self.fishers:
                raise ValueError(
                    "EATA fisher_enabled=True provided no Fisher entries "
                    "matching trainable parameters"
                )
        else:
            self.fisher_metadata = {}
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        for name, parameter in model.named_parameters():
            if any(keyword in name for keyword in self.adapt_keywords):
                parameter.requires_grad_(True)
        return model

    def _regularizer(self, model):
        if not self.fisher_enabled:
            return next(model.parameters()).sum() * 0.0
        result = None
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.fishers:
                continue
            diagonal, reference = self.fishers[name]
            term = (
                diagonal.to(parameter.device)
                * (parameter - reference.to(parameter.device)).square()
            ).sum()
            result = term if result is None else result + term
        if result is None:
            return next(model.parameters()).sum() * 0.0
        return result * self.fisher_alpha

    @staticmethod
    def _extract_batch_views(batch_data):
        if isinstance(batch_data, dict):
            batch_data = batch_data.get("data", batch_data)
        if isinstance(batch_data, (list, tuple)):
            raw_data = batch_data[0]
            aug_data = batch_data[1] if len(batch_data) > 1 else raw_data
            return raw_data, aug_data
        return batch_data, batch_data

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del trg_idx
        x, augmented = self._extract_batch_views(batch_data)
        raw_features, _ = model.feature_extractor(x)
        raw_logits = model.classifier(raw_features)
        aug_features, _ = model.feature_extractor(augmented)
        aug_logits = model.classifier(aug_features)
        probs = raw_logits.softmax(dim=1)
        entropy = softmax_entropy(raw_logits)
        first = torch.where(entropy < self.e_margin)[0]
        selected_indices = torch.arange(first.numel(), device=x.device)
        if self.current_model_probs is not None and first.numel():
            reference = self.current_model_probs.to(x.device)
            diversity = 1.0 - F.cosine_similarity(reference.unsqueeze(0), probs[first], dim=1)
            selected_indices = torch.where(diversity > self.d_margin)[0]
        selected = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if first.numel() and selected_indices.numel():
            selected[first[selected_indices]] = True
        if selected.any():
            selected_indices = first[selected_indices]
            selected_entropy = softmax_entropy(aug_logits[selected_indices])
            coeff = torch.exp(
                -(entropy[selected_indices].detach() - self.e_margin)
            )
            loss_ent = (selected_entropy * coeff).mean()
        else:
            loss_ent = raw_logits.sum() * 0.0
        loss_reg = self._regularizer(model)
        loss = loss_ent + loss_reg
        if optimizer is not None and selected.any():
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if selected.any():
                mean_probs = probs[selected].mean(dim=0)
                self.current_model_probs = mean_probs if self.current_model_probs is None else 0.9 * self.current_model_probs + 0.1 * mean_probs
        self._selected_counter += int(selected.sum())
        self._last_gate_log = {"selected_mask": selected.detach().cpu()}
        self._last_batch_log = {
            "entropy_gate_pass_rate": float((entropy < self.e_margin).float().mean()),
            "selected_pass_rate": float(selected.float().mean()),
            "batch_entropy": float(entropy.mean().detach()),
            "loss_ent": float(loss_ent.detach()),
            "loss_reg": float(loss_reg.detach()),
            "fisher_enabled": float(self.fisher_enabled),
            "total_loss": float(loss.detach()),
        }
        return 0.5 * (raw_logits + aug_logits)


class SAR(BaseTestTimeAlgorithm):
    """SAR entropy filtering with a sharpness-aware two-step update."""

    def __init__(self, configs, hparams, model, optimizer):
        self.margin_e0 = float(hparams.get("sar_margin_e0", -1.0))
        self.reset_constant_em = float(hparams.get("sar_reset_constant_em", 0.2))
        self.sar_rho = float(hparams.get("sar_rho", 0.05))
        self.sar_adaptive = bool(hparams.get("sar_adaptive", False))
        self.sar_base_optimizer = str(hparams.get("sar_base_optimizer", "sgd")).lower()
        self.episodic = bool(hparams.get("episodic", False))
        self.ema = None
        super().__init__(configs, hparams, model, optimizer)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        base = torch.optim.SGD if self.sar_base_optimizer == "sgd" else torch.optim.Adam
        kwargs = {"lr": float(hparams["learning_rate"]), "weight_decay": float(hparams.get("weight_decay", 0.0))}
        if base is torch.optim.SGD:
            kwargs["momentum"] = float(hparams.get("momentum", 0.9))
        self.optimizer = SAM(parameters, base, rho=self.sar_rho, adaptive=self.sar_adaptive, **kwargs) if parameters else None
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer else None

    def configure_model(self, model):
        return configure_norm_adaptation(model, train_all=False)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        self.ema = None

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del trg_idx
        if self.episodic:
            self.reset()
        x = extract_primary_tensor(batch_data)
        logits = model(x)
        entropy = softmax_entropy(logits)
        margin = self.margin_e0 if self.margin_e0 > 0 else 0.4 * math.log(max(2, self.configs.num_classes))
        first_mask = entropy < margin
        selected = torch.zeros_like(first_mask)
        reset_flag = False
        if optimizer is not None and first_mask.any():
            optimizer.zero_grad(set_to_none=True)
            entropy[first_mask].mean().backward()
            optimizer.first_step(zero_grad=True)
            second_logits = model(x)
            second_entropy = softmax_entropy(second_logits)
            second_mask = second_entropy[first_mask] < margin
            indices = torch.where(first_mask)[0]
            if second_mask.any():
                selected[indices[second_mask]] = True
                second_entropy[indices[second_mask]].mean().backward()
                optimizer.second_step(zero_grad=True)
                value = float(second_entropy[indices[second_mask]].mean().detach())
                self.ema = value if self.ema is None else 0.9 * self.ema + 0.1 * value
            else:
                optimizer.restore(zero_grad=True)
        if self.ema is not None and self.ema < self.reset_constant_em:
            reset_flag = True
            self.reset()
        self._selected_counter += int(selected.sum())
        self._last_gate_log = {"selected_mask": selected.detach().cpu(), "reset_flag": reset_flag}
        self._last_batch_log = {
            "entropy_gate_pass_rate": float(first_mask.float().mean()),
            "selected_pass_rate": float(selected.float().mean()),
            "batch_entropy": float(entropy.mean().detach()),
            "reset_flag": float(reset_flag),
            "ema": float(self.ema) if self.ema is not None else float("nan"),
        }
        return logits


def domain_contrastive_loss(features, labels, temperature=0.6):
    features = F.normalize(features, dim=1)
    similarities = features @ features.t() / max(float(temperature), 1e-6)
    eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positives = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~eye
    logits = similarities.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0.0
    return -(log_prob[valid] * positives[valid]).sum(dim=1).div(counts[valid]).mean()


class ACCUPOfficial(BaseTestTimeAlgorithm):
    """ACCUP prototype/support update and augmented contrastive objective."""

    def __init__(self, configs, hparams, model, optimizer):
        self.filter_k = int(hparams.get("filter_K", -1))
        self.tau = float(hparams.get("tau", 1.0))
        self.temperature = float(hparams.get("temperature", 0.6))
        self.num_classes = int(configs.num_classes)
        super().__init__(configs, hparams, model, optimizer)
        self.featurizer = self.model.feature_extractor
        self.classifier = self.model.classifier
        layer = getattr(self.classifier, "logits", self.classifier)
        if not hasattr(layer, "weight"):
            raise ValueError("ACCUPOfficial requires a linear classifier")
        supports = layer.weight.detach().clone()
        warmup_logits = self.classifier(supports)
        self.supports = supports
        self.labels = F.one_hot(warmup_logits.argmax(dim=1), num_classes=self.num_classes).float()
        self.ents = softmax_entropy(warmup_logits).detach()
        self.cls_scores = warmup_logits.softmax(dim=1).detach()
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
        if hasattr(model, "feature_extractor"):
            for module in model.feature_extractor.modules():
                if isinstance(module, nn.Conv1d):
                    module.requires_grad_(True)
        return model

    def _select_supports(self):
        predicted = self.labels.argmax(dim=1)
        if self.filter_k == -1:
            indices = torch.arange(len(self.ents), device=self.ents.device)
        else:
            selected = []
            all_indices = torch.arange(len(self.ents), device=self.ents.device)
            for class_index in range(self.num_classes):
                class_indices = all_indices[predicted == class_index]
                if class_indices.numel():
                    selected.append(class_indices[torch.argsort(self.ents[class_indices])[: self.filter_k]])
            if not selected:
                raise RuntimeError("ACCUPOfficial prototype memory contains no class support")
            else:
                indices = torch.cat(selected)
        self.supports = self.supports[indices].detach()
        self.labels = self.labels[indices].detach()
        self.ents = self.ents[indices].detach()
        self.cls_scores = self.cls_scores[indices].detach()
        return self.supports, self.labels

    def _prototype_logits(self, features, supports, labels):
        centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ supports
        return self.tau * F.normalize(features, dim=1) @ F.normalize(centroids, dim=1).T

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del trg_idx
        if isinstance(batch_data, dict):
            batch_data = batch_data.get("data", batch_data)
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0]
            augmented = batch_data[1] if len(batch_data) > 1 else x
        else:
            x = batch_data
            augmented = x
        raw_features, _ = model.feature_extractor(x)
        raw_logits = model.classifier(raw_features)
        augmented_features, _ = model.feature_extractor(augmented)
        augmented_logits = model.classifier(augmented_features)
        ensemble_features = 0.5 * (raw_features + augmented_features)
        ensemble_logits = 0.5 * (raw_logits + augmented_logits)
        pseudo = F.one_hot(ensemble_logits.argmax(dim=1), num_classes=self.num_classes).float()
        ensemble_entropy = softmax_entropy(ensemble_logits)
        with torch.no_grad():
            self.supports = torch.cat([self.supports.to(x.device), ensemble_features.detach()])
            self.labels = torch.cat([self.labels.to(x.device), pseudo.detach()])
            self.ents = torch.cat([self.ents.to(x.device), ensemble_entropy.detach()])
            self.cls_scores = torch.cat([self.cls_scores.to(x.device), ensemble_logits.softmax(dim=1).detach()])
            supports, labels = self._select_supports()
            prototype_logits = self._prototype_logits(ensemble_features.detach(), supports, labels)
            prototype_entropy = softmax_entropy(prototype_logits)
            use_prototype = prototype_entropy < ensemble_entropy.detach()
            classifier_scores = ensemble_logits.softmax(dim=1)
            selected_predictions = torch.where(
                use_prototype.unsqueeze(1),
                prototype_logits,
                classifier_scores.detach(),
            )
            pseudo_labels = selected_predictions.argmax(dim=1)
        contrastive_features = torch.cat([raw_logits, augmented_logits, ensemble_logits], dim=0)
        loss = domain_contrastive_loss(contrastive_features, pseudo_labels.repeat(3), self.temperature)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        selected = _safe_selected_mask(x)
        self._selected_counter += int(selected.sum())
        self._last_gate_log = {
            "selected_mask": selected.detach().cpu(),
            "selected_indices": selected.detach().cpu().tolist(),
        }
        self._last_batch_log = {
            "selected_pass_rate": 1.0,
            "prototype_prediction_rate": float(use_prototype.float().mean()),
            "batch_entropy": float(ensemble_entropy.mean().detach()),
            "contrastive_loss": float(loss.detach()),
        }
        return selected_predictions


class CoTTA(BaseTestTimeAlgorithm):
    """EMA teacher, augmentation averaging, and stochastic restoration."""

    def __init__(self, configs, hparams, model, optimizer):
        self.mt_alpha = float(hparams.get("cotta_mt_alpha", 0.999))
        self.restore_probability = float(hparams.get("cotta_restore_probability", 0.01))
        self.anchor_threshold = float(hparams.get("cotta_anchor_threshold", 0.9))
        self.num_augmentations = max(1, int(hparams.get("cotta_num_augmentations", 8)))
        self.augmentation_sigma = float(hparams.get("cotta_augmentation_sigma", 0.1))
        self.augmentation_control_points = int(hparams.get("cotta_control_points", 8))
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        super().__init__(configs, hparams, model, optimizer)
        self.source_state = deepcopy(self.model.state_dict())
        self.teacher = clone_frozen(self.model)
        self.anchor = clone_frozen(self.model)

    def configure_model(self, model):
        return configure_norm_adaptation(model, train_all=bool(self.hparams.get("cotta_train_all", True)))

    @torch.no_grad()
    def _teacher_prediction(self, x):
        self.teacher.eval()
        self.anchor.eval()
        confidence = self.anchor(x).softmax(dim=1).amax(dim=1)
        standard = self.teacher(x)
        use_augmented = bool(confidence.mean() < self.anchor_threshold)
        if not use_augmented:
            return standard, confidence, False
        predictions = [self.teacher(smooth_amplitude_augment(x, self.augmentation_sigma, self.augmentation_control_points)) for _ in range(self.num_augmentations)]
        return torch.stack(predictions).mean(dim=0), confidence, True

    @torch.no_grad()
    def _stochastic_restore(self):
        restored = 0
        if self.restore_probability <= 0:
            return restored
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and name in self.source_state:
                mask = torch.rand_like(parameter) < self.restore_probability
                restored += int(mask.sum())
                parameter.copy_(torch.where(mask, self.source_state[name].to(parameter.device), parameter))
        return restored

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del trg_idx
        x = extract_primary_tensor(batch_data)
        model.train()
        student_logits = model(x)
        teacher_logits, confidence, used_augmented = self._teacher_prediction(x)
        loss = teacher_cross_entropy(student_logits, teacher_logits).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        update_ema(self.teacher, model, self.mt_alpha)
        restored = self._stochastic_restore()
        selected = _safe_selected_mask(x)
        self._selected_counter += int(selected.sum())
        self._last_gate_log = {"selected_mask": selected.detach().cpu()}
        self._last_batch_log = {
            "selected_pass_rate": 1.0,
            "loss": float(loss.detach()),
            "anchor_confidence": float(confidence.mean()),
            "augmentation_averaging": float(used_augmented),
            "restored_parameters": float(restored),
        }
        return teacher_logits.detach()


class _UniformConfidenceMemory:
    def __init__(self, capacity, num_classes, threshold):
        self.capacity = max(1, int(capacity))
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.data = defaultdict(list)

    def __len__(self):
        return sum(len(values) for values in self.data.values())

    def _largest(self):
        largest = max((len(values) for values in self.data.values()), default=0)
        return [label for label, values in self.data.items() if len(values) == largest and values]

    def add(self, x, label, confidence):
        if confidence < self.threshold:
            return False
        if len(self) >= self.capacity:
            largest = self._largest()
            if largest:
                victim = random.choice(largest)
                self.data[victim].pop(random.randrange(len(self.data[victim])))
        self.data[int(label)].append((x.detach().cpu(), float(confidence)))
        return True

    def tensors(self, device):
        values = [value[0] for label in sorted(self.data) for value in self.data[label]]
        return torch.stack(values).to(device) if values else None


class SoTTA(BaseTestTimeAlgorithm):
    """High-confidence uniform-class memory with SAM entropy updates."""

    def __init__(self, configs, hparams, model, optimizer):
        self.memory = _UniformConfidenceMemory(hparams.get("sotta_memory_size", 64), configs.num_classes, hparams.get("sotta_confidence_threshold", 0.9))
        self.update_frequency = max(1, int(hparams.get("sotta_update_frequency", 64)))
        self.temperature = float(hparams.get("sotta_temperature", 1.0))
        self.sam_rho = float(hparams.get("sotta_rho", 0.05))
        self.seen = 0
        self.num_updates = 0
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        super().__init__(configs, hparams, model, optimizer)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = SAM(parameters, torch.optim.SGD, rho=self.sam_rho, lr=float(hparams["learning_rate"]), momentum=float(hparams.get("momentum", 0.9)), weight_decay=float(hparams.get("weight_decay", 0.0))) if parameters else None

    def configure_model(self, model):
        return configure_norm_adaptation(model, train_all=False, running_stats=True)

    def _sam_update(self, memory_x):
        if self.optimizer is None or memory_x is None:
            return None
        self.model.train()
        first = self.model(memory_x)
        loss_first = softmax_entropy(first / self.temperature).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_first.backward()
        self.optimizer.first_step(zero_grad=True)
        second = self.model(memory_x)
        loss_second = softmax_entropy(second / self.temperature).mean()
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)
        self.num_updates += 1
        return float(loss_second.detach())

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del optimizer, trg_idx
        x = extract_primary_tensor(batch_data)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            confidence, labels = logits.softmax(dim=1).max(dim=1)
        accepted = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for index in range(x.size(0)):
            accepted[index] = self.memory.add(x[index], int(labels[index]), float(confidence[index]))
        previous = self.seen // self.update_frequency
        self.seen += x.size(0)
        update_loss = None
        if self.seen // self.update_frequency > previous:
            update_loss = self._sam_update(self.memory.tensors(x.device))
        self._selected_counter += int(accepted.sum())
        self._last_gate_log = {
            "selected_mask": accepted.detach().cpu(),
            "confidence_mask": accepted.detach().cpu(),
        }
        self._last_batch_log = {
            "selected_pass_rate": float(accepted.float().mean()),
            "mean_confidence": float(confidence.mean()),
            "memory_occupancy": float(len(self.memory)),
            "num_updates": float(self.num_updates),
            "loss": float("nan") if update_loss is None else update_loss,
        }
        return logits


class RobustBatchNorm1d(nn.Module):
    def __init__(self, source_bn, alpha):
        super().__init__()
        if source_bn.running_mean is None or source_bn.running_var is None:
            raise ValueError("RoTTA requires source BatchNorm running statistics")
        self.alpha = float(alpha)
        self.eps = float(source_bn.eps)
        self.register_buffer("source_mean", source_bn.running_mean.detach().clone())
        self.register_buffer("source_var", source_bn.running_var.detach().clone())
        self.weight = nn.Parameter(source_bn.weight.detach().clone()) if source_bn.affine else None
        self.bias = nn.Parameter(source_bn.bias.detach().clone()) if source_bn.affine else None

    def forward(self, x):
        if x.dim() not in {2, 3}:
            raise ValueError("RobustBatchNorm1d expects 2-D or 3-D input")
        if self.training:
            reduce_dims = (0,) if x.dim() == 2 else (0, 2)
            batch_var, batch_mean = torch.var_mean(x, dim=reduce_dims, unbiased=False)
            self.source_mean.copy_((1.0 - self.alpha) * self.source_mean + self.alpha * batch_mean.detach())
            self.source_var.copy_((1.0 - self.alpha) * self.source_var + self.alpha * batch_var.detach())
        shape = (1, -1) if x.dim() == 2 else (1, -1, 1)
        output = (x - self.source_mean.view(*shape)) / torch.sqrt(self.source_var.view(*shape) + self.eps)
        if self.weight is not None:
            output = output * self.weight.view(*shape)
        if self.bias is not None:
            output = output + self.bias.view(*shape)
        return output


def _replace_batch_norm(module, alpha):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, RobustBatchNorm1d(child, alpha))
        else:
            _replace_batch_norm(child, alpha)


class _CSTUMemory:
    def __init__(self, capacity, num_classes, lambda_t=1.0, lambda_u=1.0):
        self.capacity = max(1, int(capacity))
        self.num_classes = int(num_classes)
        self.per_class = self.capacity / self.num_classes
        self.lambda_t = float(lambda_t)
        self.lambda_u = float(lambda_u)
        self.data = defaultdict(list)

    def __len__(self):
        return sum(len(values) for values in self.data.values())

    def _score(self, age, uncertainty):
        time_score = 1.0 / (1.0 + math.exp(-float(age) / self.capacity))
        uncertainty_score = float(uncertainty) / math.log(max(2, self.num_classes))
        return self.lambda_t * time_score + self.lambda_u * uncertainty_score

    def add(self, x, label, uncertainty):
        label = int(label)
        new_score = self._score(0, uncertainty)
        if len(self.data[label]) < self.per_class and len(self) < self.capacity:
            keep = True
        else:
            labels = [label] if len(self.data[label]) >= self.per_class else list(range(self.num_classes))
            victim = None
            for candidate in labels:
                for index, item in enumerate(self.data[candidate]):
                    score = self._score(item[2], item[1])
                    if victim is None or score >= victim[0]:
                        victim = (score, candidate, index)
            keep = victim is None or victim[0] > new_score
            if keep and victim is not None:
                self.data[victim[1]].pop(victim[2])
        if keep:
            self.data[label].append([x.detach().cpu(), float(uncertainty), 0])
        for values in self.data.values():
            for item in values:
                item[2] += 1
        return bool(keep)

    def tensors_and_ages(self, device):
        items = [item for label in range(self.num_classes) for item in self.data[label]]
        if not items:
            return None, None
        x = torch.stack([item[0] for item in items]).to(device)
        ages = torch.tensor([item[2] / self.capacity for item in items], device=device, dtype=x.dtype)
        return x, ages


class RoTTA(BaseTestTimeAlgorithm):
    """CSTU memory, robust BN, EMA teacher, and timeliness weighting."""

    def __init__(self, configs, hparams, model, optimizer):
        self.num_classes = int(configs.num_classes)
        self.memory = _CSTUMemory(hparams.get("rotta_memory_size", 64), self.num_classes, hparams.get("rotta_lambda_t", 1.0), hparams.get("rotta_lambda_u", 1.0))
        self.update_frequency = max(1, int(hparams.get("rotta_update_frequency", 64)))
        self.alpha = float(hparams.get("rotta_bn_alpha", 0.05))
        self.nu = float(hparams.get("rotta_nu", 0.001))
        self.augmentation_sigma = float(hparams.get("rotta_augmentation_sigma", 0.1))
        self.current_instance = 0
        self.num_updates = 0
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        super().__init__(configs, hparams, model, optimizer)
        self.teacher = clone_frozen(self.model)

    def configure_model(self, model):
        model.eval()
        model.requires_grad_(False)
        _replace_batch_norm(model, self.alpha)
        for module in model.modules():
            if isinstance(module, RobustBatchNorm1d):
                module.requires_grad_(True)
        return model

    def _update_model(self, device):
        memory_x, ages = self.memory.tensors_and_ages(device)
        if memory_x is None or self.optimizer is None:
            return None
        self.model.train()
        strong = smooth_amplitude_augment(memory_x, self.augmentation_sigma)
        teacher_logits = self.teacher(memory_x)
        student_logits = self.model(strong)
        weights = torch.sigmoid(-ages)
        loss = (teacher_cross_entropy(student_logits, teacher_logits) * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        update_ema(self.teacher, self.model, 1.0 - self.nu)
        self.num_updates += 1
        return float(loss.detach())

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del optimizer, trg_idx
        x = extract_primary_tensor(batch_data)
        model.eval()
        self.teacher.eval()
        with torch.no_grad():
            logits = self.teacher(x)
            probabilities = logits.softmax(dim=1)
            labels = probabilities.argmax(dim=1)
            uncertainty = softmax_entropy(logits)
        accepted = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for index in range(x.size(0)):
            accepted[index] = self.memory.add(x[index], int(labels[index]), float(uncertainty[index]))
            self.current_instance += 1
            if self.current_instance % self.update_frequency == 0:
                update_loss = self._update_model(x.device)
            else:
                update_loss = None
        self._selected_counter += int(accepted.sum())
        self._last_gate_log = {"selected_mask": accepted.detach().cpu()}
        self._last_batch_log = {
            "selected_pass_rate": float(accepted.float().mean()),
            "memory_occupancy": float(len(self.memory)),
            "uncertainty": float(uncertainty.mean()),
            "num_updates": float(self.num_updates),
            "loss": float("nan") if update_loss is None else update_loss,
        }
        return logits


def entropy_of_opinion(logits, evidence_prior=None):
    norm = torch.norm(logits, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    scaled = logits / norm * norm.detach()
    evidence = torch.exp(scaled.clamp(max=50.0))
    prior = float(logits.size(1) if evidence_prior is None else evidence_prior)
    denominator = evidence.sum(dim=1, keepdim=True) + prior
    belief = evidence / denominator
    uncertainty = torch.full_like(denominator, prior) / denominator
    opinion = torch.cat([belief, uncertainty], dim=1).clamp_min(1e-7)
    return -(opinion * opinion.log()).sum(dim=1), uncertainty.squeeze(1)


class COME(BaseTestTimeAlgorithm):
    """COME entropy-of-opinion objective with TENT normalization scope."""

    def __init__(self, configs, hparams, model, optimizer):
        configured_prior = hparams.get("come_evidence_prior")
        self.evidence_prior = float(configs.num_classes if configured_prior in (None, "auto", "num_classes") else configured_prior)
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        super().__init__(configs, hparams, model, optimizer)

    def configure_model(self, model):
        return configure_norm_adaptation(model, train_all=False)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del trg_idx
        x = extract_primary_tensor(batch_data)
        model.train()
        logits = model(x)
        entropy, uncertainty = entropy_of_opinion(logits, self.evidence_prior)
        loss = entropy.mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        selected = _safe_selected_mask(x)
        self._selected_counter += int(selected.sum())
        self._last_gate_log = {"selected_mask": selected.detach().cpu()}
        self._last_batch_log = {
            "selected_pass_rate": 1.0,
            "opinion_entropy": float(entropy.mean().detach()),
            "opinion_uncertainty": float(uncertainty.mean().detach()),
            "loss": float(loss.detach()),
        }
        return logits


class _NotePBRS:
    """Pseudo-balanced reservoir used by the official NOTE online script.

    The upstream implementation stores feature/label/domain tuples in a
    PBRS object.  The shared trainer has no target labels, so this adapter
    stores the model's pseudo label and retains the same class-balanced
    replacement rule.  The returned boolean is the actual memory admission
    decision and is exposed to the safety evaluator.
    """

    def __init__(self, capacity, num_classes):
        self.capacity = max(1, int(capacity))
        self.num_classes = int(num_classes)
        self.data = defaultdict(list)
        self.counter = [0] * self.num_classes

    def __len__(self):
        return sum(len(values) for values in self.data.values())

    def _largest_classes(self):
        largest = max(
            (len(self.data[label]) for label in range(self.num_classes)),
            default=0,
        )
        return [
            label
            for label in range(self.num_classes)
            if len(self.data[label]) == largest and self.data[label]
        ]

    def add(self, x, pseudo_label):
        label = int(pseudo_label)
        self.counter[label] += 1
        keep = True
        if len(self) >= self.capacity:
            largest = self._largest_classes()
            if label not in largest:
                victim_label = random.choice(largest)
                self.data[victim_label].pop(
                    random.randrange(len(self.data[victim_label]))
                )
            else:
                class_count = len(self.data[label])
                seen_for_class = self.counter[label]
                if random.random() <= class_count / max(1, seen_for_class):
                    self.data[label].pop(random.randrange(class_count))
                else:
                    keep = False
        if keep:
            self.data[label].append(x.detach().cpu())
        return bool(keep)

    def tensors(self, device):
        values = [
            item
            for label in range(self.num_classes)
            for item in self.data[label]
        ]
        if not values:
            return None
        return torch.stack(values).to(device)


class NOTE(BaseTestTimeAlgorithm):
    """NOTE BN/entropy adaptation with the official PBRS stream memory.

    There is no local historical NOTE port.  This is an isolated adapter of
    the cloned TaesikGong/NOTE implementation: learned BN statistics, a
    pseudo-balanced memory, periodic entropy updates, and no target-label
    access.  Image-only IABN is not imported; the time-series model's native
    BatchNorm/InstanceNorm modules are the documented substitution.
    """

    def __init__(self, configs, hparams, model, optimizer):
        self.memory_size = max(1, int(hparams.get("note_memory_size", 64)))
        self.update_frequency = max(
            1, int(hparams.get("note_update_frequency", 64))
        )
        self.bn_momentum = float(hparams.get("note_bn_momentum", 0.01))
        self.use_learned_stats = bool(
            hparams.get("note_use_learned_stats", True)
        )
        self.epochs = max(1, int(hparams.get("note_epoch", 1)))
        self.temperature = float(hparams.get("note_temperature", 1.0))
        self.memory = _NotePBRS(self.memory_size, int(configs.num_classes))
        self.seen = 0
        self.num_updates = 0
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        super().__init__(configs, hparams, model, optimizer)

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)
        reference_device = next(model.parameters()).device
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = self.use_learned_stats
                if self.use_learned_stats:
                    module.momentum = self.bn_momentum
                    if module.running_mean is None:
                        module.running_mean = torch.zeros(
                            module.num_features, device=reference_device
                        )
                    if module.running_var is None:
                        module.running_var = torch.ones(
                            module.num_features, device=reference_device
                        )
                else:
                    module.running_mean = None
                    module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
            elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
        return model

    def _adapt_memory(self, memory_x):
        if self.optimizer is None or memory_x is None or memory_x.numel() == 0:
            return None
        # The upstream code switches to eval for a one-sample memory to avoid
        # BatchNorm's invalid-statistics path, then minimizes entropy on the
        # memory for one epoch (the configured NOTE default).
        if memory_x.size(0) == 1:
            self.model.eval()
        else:
            self.model.train()
        loss_value = None
        for _ in range(self.epochs):
            predictions = self.model(memory_x)
            loss = softmax_entropy(predictions / self.temperature).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach())
        self.num_updates += 1
        return loss_value

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        del optimizer, trg_idx
        x = extract_primary_tensor(batch_data)
        # NOTE evaluates the current FIFO batch before the periodic update.
        # This preserves the official online order and prevents target labels
        # from entering either the memory or the update objective.
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probabilities = logits.softmax(dim=1)
            pseudo_labels = probabilities.argmax(dim=1)

        accepted = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for index in range(x.size(0)):
            accepted[index] = self.memory.add(
                x[index], int(pseudo_labels[index].item())
            )
        previous_bucket = self.seen // self.update_frequency
        self.seen += x.size(0)
        current_bucket = self.seen // self.update_frequency
        update_loss = None
        if current_bucket > previous_bucket:
            update_loss = self._adapt_memory(self.memory.tensors(x.device))

        self._selected_counter += int(accepted.sum())
        self._last_gate_log = {
            "selected_mask": accepted.detach().cpu(),
            "memory_admission_mask": accepted.detach().cpu(),
        }
        self._last_batch_log = {
            "selected_pass_rate": float(accepted.float().mean()),
            "memory_occupancy": float(len(self.memory)),
            "num_updates": float(self.num_updates),
            "mean_confidence": float(probabilities.amax(dim=1).mean()),
            "loss": float("nan") if update_loss is None else update_loss,
        }
        return logits


__all__ = [
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
    "RobustBatchNorm1d",
    "entropy_of_opinion",
]
