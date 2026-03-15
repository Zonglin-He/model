import math
import os
from collections import deque
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm
from utils.utils import safe_torch_load, softmax_entropy_from_logits


class NuSTAR_ActiveSearch:
    """Piecewise amplitude search with natural cubic spline upsampling."""

    def __init__(self, num_control_points: int = 10, num_candidates: int = 16, sigma: float = 0.1):
        self.num_control_points = max(2, int(num_control_points))
        self.num_candidates = max(0, int(num_candidates))
        self.sigma = float(sigma)
        self.last_warp_curve = None
        self.last_warp_curves = None
        self.last_metadata = None

    def _sample_control_points(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        sigma: float,
    ) -> torch.Tensor:
        base = torch.ones(
            batch_size,
            self.num_candidates,
            self.num_control_points,
            device=device,
            dtype=dtype,
        )
        if sigma <= 0.0 or self.num_candidates <= 0:
            return base
        noise = torch.randn_like(base) * sigma + 1.0
        return noise.clamp(1.0 - 3.0 * sigma, 1.0 + 3.0 * sigma)

    @staticmethod
    def _natural_cubic_spline_upsample(k: torch.Tensor, target_len: int) -> torch.Tensor:
        if k.dim() != 2:
            raise ValueError(f"Expected control tensor with shape [N, M], got {tuple(k.shape)}")
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        if k.size(1) == 1:
            return k.repeat(1, target_len)

        device = k.device
        dtype = k.dtype
        num_candidates, num_ctrl = k.shape
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        y = k.to(dtype=work_dtype)

        ctrl_x = torch.linspace(
            0.0,
            float(target_len - 1),
            num_ctrl,
            device=device,
            dtype=work_dtype,
        )
        h = ctrl_x[1:] - ctrl_x[:-1]

        second = torch.zeros(num_candidates, num_ctrl, device=device, dtype=work_dtype)
        if num_ctrl > 2:
            rhs = 6.0 * (
                (y[:, 2:] - y[:, 1:-1]) / h[1:].unsqueeze(0)
                - (y[:, 1:-1] - y[:, :-2]) / h[:-1].unsqueeze(0)
            )
            system = torch.zeros(num_ctrl - 2, num_ctrl - 2, device=device, dtype=work_dtype)
            diag = 2.0 * (h[:-1] + h[1:])
            system.diagonal().copy_(diag)
            if num_ctrl - 3 > 0:
                system.diagonal(offset=1).copy_(h[1:-1])
                system.diagonal(offset=-1).copy_(h[1:-1])
            solution = torch.linalg.solve(system.unsqueeze(0).expand(num_candidates, -1, -1), rhs)
            second[:, 1:-1] = solution

        eval_x = torch.linspace(
            0.0,
            float(target_len - 1),
            target_len,
            device=device,
            dtype=work_dtype,
        )
        interval_idx = torch.bucketize(eval_x, ctrl_x[1:-1], right=False)
        interval_idx = interval_idx.clamp(max=num_ctrl - 2)

        x0 = ctrl_x[interval_idx]
        x1 = ctrl_x[interval_idx + 1]
        h_eval = x1 - x0
        delta0 = x1 - eval_x
        delta1 = eval_x - x0

        y0 = y[:, interval_idx]
        y1 = y[:, interval_idx + 1]
        m0 = second[:, interval_idx]
        m1 = second[:, interval_idx + 1]

        term0 = m0 * (delta0 ** 3) / (6.0 * h_eval)
        term1 = m1 * (delta1 ** 3) / (6.0 * h_eval)
        term2 = (y0 - m0 * (h_eval ** 2) / 6.0) * (delta0 / h_eval)
        term3 = (y1 - m1 * (h_eval ** 2) / 6.0) * (delta1 / h_eval)
        curves = term0 + term1 + term2 + term3
        return curves.to(dtype=dtype)

    @staticmethod
    def _extract_features(model, x: torch.Tensor) -> torch.Tensor:
        feats = model.feature_extractor(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, model, sigma: Optional[float] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")

        sigma = self.sigma if sigma is None else float(sigma)
        batch_size, channels, target_len = x.shape
        if self.num_candidates <= 0 or sigma <= 0.0:
            warp = torch.ones(batch_size, 1, target_len, device=x.device, dtype=x.dtype)
            self.last_warp_curves = warp.detach().cpu()
            self.last_warp_curve = warp[:1].detach().cpu()
            self.last_metadata = {
                "mode": "identity",
                "curve": warp.squeeze(1).detach().cpu(),
                "control_points": torch.ones(batch_size, 1, device=x.device, dtype=x.dtype).cpu(),
            }
            return x

        controls = self._sample_control_points(batch_size, x.device, x.dtype, sigma)
        flat_controls = controls.reshape(batch_size * self.num_candidates, self.num_control_points)
        upsampled = self._natural_cubic_spline_upsample(flat_controls, target_len)
        warps = upsampled.reshape(batch_size, self.num_candidates, target_len)
        warped_x = x.unsqueeze(1) * warps.unsqueeze(2)
        flat_x = warped_x.reshape(batch_size * self.num_candidates, channels, target_len)

        feats = self._extract_features(model, flat_x)
        logits = model.classifier(feats)
        entropy = softmax_entropy_from_logits(logits).reshape(batch_size, self.num_candidates)
        best_idx = entropy.argmax(dim=1)
        best_warp = warps[torch.arange(batch_size, device=x.device), best_idx]

        self.last_warp_curves = best_warp.unsqueeze(1).detach().cpu()
        self.last_warp_curve = best_warp[:1].unsqueeze(1).detach().cpu()
        self.last_metadata = {
            "mode": "piecewise_search",
            "curve": best_warp.detach().cpu(),
            "control_points": controls[torch.arange(batch_size, device=x.device), best_idx].detach().cpu(),
            "score": entropy[torch.arange(batch_size, device=x.device), best_idx].detach().cpu(),
        }
        return x * best_warp.unsqueeze(1)


class ACCUP(BaseTestTimeAlgorithm):
    """NuSTAR-compatible ACCUP implementation."""

    def __init__(self, configs, hparams, model, optimizer):
        self.num_classes = int(configs.num_classes)
        self.last_adv_metadata = None
        self._last_gate_log = {}
        self._last_batch_log = {}
        self._zero_active_streak = 0
        super().__init__(configs, hparams, model, optimizer)

        self.featurizer = self.model.feature_extractor
        self.classifier = self.model.classifier

        default_sigmas = hparams.get("adv_sigmas", [hparams.get("adv_sigma", 0.1)])
        self.adv_sigmas = self._build_adv_sigmas(default_sigmas)
        adv_sigma = hparams.get("adv_sigma", None)
        if adv_sigma is None:
            adv_sigma = max((abs(s) for s in self.adv_sigmas), default=0.0)
        self.adv_sigma = float(adv_sigma)
        self.adv_ctrl_points = int(
            hparams.get(
                "adv_ctrl_points",
                hparams.get("adv_num_control_points", hparams.get("num_control_points", hparams.get("adv_control_points", 10))),
            )
        )
        default_candidates = 16 if self.adv_sigma > 0.0 else 0
        self.adv_num_candidates = int(hparams.get("adv_num_candidates", default_candidates))
        self.enable_ssaw = bool(hparams.get("enable_ssaw", True))
        self.active_search = NuSTAR_ActiveSearch(
            num_control_points=self.adv_ctrl_points,
            num_candidates=self.adv_num_candidates,
            sigma=self.adv_sigma,
        )

        self.enable_stat_gate = bool(hparams.get("enable_stat_gate", True))
        self.enable_semantic_gate = bool(hparams.get("enable_semantic_gate", True))
        self.enable_consistency_gate = bool(hparams.get("enable_consistency_gate", True))
        self.enable_gate_relaxation = bool(hparams.get("enable_gate_relaxation", False))
        self.gate_relaxation_patience = max(1, int(hparams.get("gate_relaxation_patience", 4)))
        self.sem_relax_step = float(hparams.get("sem_relax_step", 0.05))
        self.cons_relax_step = float(hparams.get("cons_relax_step", 0.05))
        self.max_relax_steps = max(0, int(hparams.get("max_relax_steps", 4)))

        self.sem_thresh = float(hparams.get("sem_thresh", 0.5))
        self.cons_thresh = float(hparams.get("cons_thresh", 0.5))
        self.proto_momentum = float(hparams.get("proto_momentum", 0.9))
        self.include_warmup_support = bool(hparams.get("include_warmup_support", False))
        self.warmup_min = max(1, int(hparams.get("warmup_min", 1)))

        self.stat_quantile = float(hparams.get("stat_quantile", hparams.get("entropy_quantile", 0.7)))
        self.stat_window = int(hparams.get("stat_window", hparams.get("entropy_hist_len", 512)))
        self.stat_min_history = int(hparams.get("stat_min_history", 32))
        self.stat_min_entropy = float(hparams.get("stat_min_entropy", 0.0))
        self.entropy_history = deque(maxlen=max(1, self.stat_window))

        self.prototypes: Optional[torch.Tensor] = None
        self.proto_counts: Optional[torch.Tensor] = None
        self._init_prototypes_from_model()

        self.lambda_reg = float(hparams.get("lambda_reg", hparams.get("fisher_alpha", 0.0)))
        self.max_fisher_updates = int(hparams.get("max_fisher_updates", -1))
        self.use_online_fisher = bool(hparams.get("online_fisher", True))
        self._online_fisher = None
        self._fisher_samples = 0
        self._fisher_updates = 0
        self.fishers = hparams.get("fisher_state", None)
        fisher_path = hparams.get("fisher_path")
        if self.fishers is None and fisher_path and os.path.exists(fisher_path):
            self.fishers = safe_torch_load(fisher_path, map_location="cpu")
        self.theta_src = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._selected_counter = 0
        self.eata_memory = getattr(self, "eata_memory", None)

    def set_eata_memory(self, memory):
        self.eata_memory = memory

    def _build_adv_sigmas(self, sigmas: Iterable[float]) -> List[float]:
        if isinstance(sigmas, (int, float)):
            sigmas = [float(sigmas)]
        elif isinstance(sigmas, str):
            sigmas = [float(s.strip()) for s in sigmas.split(",") if s.strip()]

        seen = set()
        ordered = []
        for sigma in sigmas:
            sigma = float(sigma)
            for value in (0.0, abs(sigma), -abs(sigma)):
                if value not in seen:
                    ordered.append(value)
                    seen.add(value)
        return ordered or [0.0]

    def _init_prototypes_from_model(self):
        if not self.include_warmup_support:
            self.prototypes = None
            self.proto_counts = None
            return

        init_proto = None
        if hasattr(self.classifier, "logits") and hasattr(self.classifier.logits, "weight"):
            warmup = self.classifier.logits.weight.data.detach()
            if warmup.dim() == 2 and warmup.size(0) == self.num_classes:
                init_proto = F.normalize(warmup, dim=1)
        elif hasattr(self.classifier, "weight"):
            warmup = self.classifier.weight.data.detach()
            if warmup.dim() == 2 and warmup.size(0) == self.num_classes:
                init_proto = F.normalize(warmup, dim=1)

        if init_proto is None:
            self.prototypes = None
            self.proto_counts = None
            return

        self.prototypes = init_proto
        self.proto_counts = torch.full(
            (self.num_classes,),
            self.warmup_min,
            dtype=torch.long,
            device=init_proto.device,
        )

    def _ensure_prototypes(self, feats: torch.Tensor):
        feat_dim = feats.size(1)
        device = feats.device
        if self.prototypes is None or self.prototypes.numel() == 0 or self.prototypes.size(1) != feat_dim:
            self.prototypes = torch.zeros(self.num_classes, feat_dim, device=device)
            self.proto_counts = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        else:
            self.prototypes = self.prototypes.to(device)
            self.proto_counts = self.proto_counts.to(device)

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)

        freeze_bn_stats = bool(self.hparams.get("freeze_bn_stats", True))
        train_bn_affine = bool(self.hparams.get("train_bn_affine", True))
        train_full_backbone = bool(self.hparams.get("train_full_backbone", True))
        train_backbone_modules = self.hparams.get("train_backbone_modules", None)
        train_classifier = bool(self.hparams.get("train_classifier", True))

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if freeze_bn_stats:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                else:
                    module.track_running_stats = True
                if train_bn_affine:
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

        if hasattr(model, "feature_extractor"):
            if train_full_backbone:
                for param in model.feature_extractor.parameters():
                    param.requires_grad_(True)
            elif train_backbone_modules:
                target_names = train_backbone_modules
                if not isinstance(target_names, (list, tuple, set)):
                    target_names = [target_names]
                target_names = {str(name) for name in target_names}
                for name, module in model.feature_extractor.named_modules():
                    if name in target_names:
                        for param in module.parameters():
                            param.requires_grad_(True)

        if train_classifier and hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad_(True)

        return model

    @staticmethod
    def _extract_primary_tensor(batch_data):
        if isinstance(batch_data, dict):
            data = batch_data.get("data")
            return data[0] if isinstance(data, (list, tuple)) else data
        if isinstance(batch_data, (list, tuple)):
            return batch_data[0]
        return batch_data

    @staticmethod
    def _extract_features(model, x: torch.Tensor) -> torch.Tensor:
        feats = model.feature_extractor(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    def _entropy_threshold(self, entropy: torch.Tensor) -> torch.Tensor:
        if len(self.entropy_history) >= self.stat_min_history:
            history = torch.tensor(list(self.entropy_history), device=entropy.device, dtype=entropy.dtype)
            threshold = torch.quantile(history, self.stat_quantile)
        else:
            threshold = entropy.new_tensor(math.log(max(2, self.num_classes)))
        return torch.clamp(threshold, min=self.stat_min_entropy)

    def _update_entropy_history(self, entropy: torch.Tensor):
        self.entropy_history.extend(entropy.detach().cpu().tolist())

    def _relaxation_level(self) -> int:
        if not self.enable_gate_relaxation or self._zero_active_streak < self.gate_relaxation_patience:
            return 0
        return min(self.max_relax_steps, self._zero_active_streak // self.gate_relaxation_patience)

    def _effective_gate_thresholds(self):
        level = self._relaxation_level()
        sem_thresh = self.sem_thresh - level * self.sem_relax_step
        cons_thresh = self.cons_thresh + level * self.cons_relax_step
        return sem_thresh, cons_thresh, level

    def get_adversarial_view(self, x: torch.Tensor, model) -> torch.Tensor:
        if not self.enable_ssaw or self.adv_sigma <= 0.0 or self.adv_num_candidates <= 0:
            self.last_adv_metadata = {
                "mode": "disabled",
                "curve": torch.ones(x.size(0), x.size(-1), device=x.device, dtype=x.dtype).cpu(),
            }
            return x
        adv_view = self.active_search(x, model, sigma=self.adv_sigma)
        self.last_adv_metadata = self.active_search.last_metadata
        return adv_view

    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor):
        self.update_prototypes_with_weights(feats, labels, weights=None)

    def update_prototypes_with_weights(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        if feats.numel() == 0:
            return
        self._ensure_prototypes(feats)
        feats = F.normalize(feats, dim=1)
        for cls_idx in range(self.num_classes):
            mask = labels == cls_idx
            if not torch.any(mask):
                continue
            if weights is None:
                class_mean = feats[mask].mean(dim=0)
            else:
                class_weights = weights[mask].clamp_min(0.0)
                if float(class_weights.sum().item()) == 0.0:
                    continue
                class_weights = class_weights / class_weights.sum()
                class_mean = (feats[mask] * class_weights.unsqueeze(1)).sum(dim=0)
            if int(self.proto_counts[cls_idx].item()) == 0:
                self.prototypes[cls_idx] = class_mean
            else:
                self.prototypes[cls_idx] = (
                    self.proto_momentum * self.prototypes[cls_idx]
                    + (1.0 - self.proto_momentum) * class_mean
                )
            self.prototypes[cls_idx] = F.normalize(self.prototypes[cls_idx], dim=0)
            self.proto_counts[cls_idx] += int(mask.sum().item())

    def _maybe_update_online_fisher(self, model, logits: torch.Tensor):
        if not self.use_online_fisher or self.lambda_reg <= 0 or logits is None or not logits.requires_grad:
            return
        if self.max_fisher_updates >= 0 and self._fisher_updates >= self.max_fisher_updates:
            return

        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            return
        if self._online_fisher is None:
            self._online_fisher = {n: torch.zeros_like(p) for n, p in trainable}

        names, params = zip(*trainable)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        fisher_loss = -(probs * log_probs).sum(dim=1).mean()
        grads = torch.autograd.grad(fisher_loss, params, retain_graph=True, allow_unused=True)
        any_update = False
        for name, grad in zip(names, grads):
            if grad is None:
                continue
            self._online_fisher[name] = self._online_fisher[name].to(grad.device)
            self._online_fisher[name] += grad.detach() ** 2
            any_update = True
        if any_update:
            self._fisher_samples += logits.size(0)
            self._fisher_updates += 1

    def _fisher_regularizer(self, model) -> torch.Tensor:
        device = next(model.parameters()).device
        if self.lambda_reg <= 0:
            return torch.zeros([], device=device)

        reg = None
        if isinstance(self.fishers, dict) and self.fishers:
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.fishers:
                    continue
                item = self.fishers[name]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    diag, theta_prev = item[0], item[1]
                else:
                    diag = item
                    theta_prev = self.theta_src.get(name)
                if theta_prev is None:
                    continue
                term = (diag.to(device) * (param - theta_prev.to(device)) ** 2).sum()
                reg = term if reg is None else reg + term
            if reg is not None:
                return self.lambda_reg * reg

        if self._online_fisher and self._fisher_samples > 0:
            normalizer = float(self._fisher_samples)
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self._online_fisher:
                    continue
                theta_prev = self.theta_src.get(name, param.detach()).to(device)
                diag = (self._online_fisher[name] / normalizer).to(device)
                term = (diag * (param - theta_prev) ** 2).sum()
                reg = term if reg is None else reg + term
            if reg is not None:
                return self.lambda_reg * reg

        return torch.zeros([], device=device)

    def _forward_and_adapt_impl(self, raw_data: torch.Tensor, model, optimizer):
        x_adv = self.get_adversarial_view(raw_data, model)

        raw_feats = self._extract_features(model, raw_data)
        raw_logits = model.classifier(raw_feats)
        raw_probs = F.softmax(raw_logits, dim=1)
        raw_entropy = softmax_entropy_from_logits(raw_logits)

        adv_feats = self._extract_features(model, x_adv)
        adv_logits = model.classifier(adv_feats)
        adv_probs = F.softmax(adv_logits, dim=1)
        adv_entropy = softmax_entropy_from_logits(adv_logits)

        self._maybe_update_online_fisher(model, raw_logits)
        self._ensure_prototypes(raw_feats)

        pred_labels = raw_probs.argmax(dim=1)
        p_bar = 0.5 * (raw_probs + adv_probs)
        stat_entropy = -(p_bar * p_bar.clamp_min(1e-8).log()).sum(dim=1)
        entropy_threshold = self._entropy_threshold(stat_entropy.detach())
        mask_stat = stat_entropy <= entropy_threshold if self.enable_stat_gate else torch.ones_like(pred_labels, dtype=torch.bool)
        self._update_entropy_history(stat_entropy)

        sem_thresh, cons_thresh, relaxation_level = self._effective_gate_thresholds()
        feat_norm = F.normalize(raw_feats.detach(), dim=1)
        proto_vecs = self.prototypes[pred_labels]
        proto_norm = F.normalize(proto_vecs.detach(), dim=1)
        proto_ready = self.proto_counts[pred_labels] >= self.warmup_min
        proto_nonzero = proto_vecs.detach().abs().sum(dim=1) > 0
        cos_sim = F.cosine_similarity(feat_norm, proto_norm, dim=1)
        if self.enable_semantic_gate:
            mask_sem = (~proto_ready) | (~proto_nonzero) | (cos_sim >= sem_thresh)
        else:
            mask_sem = torch.ones_like(pred_labels, dtype=torch.bool)

        log_adv = adv_probs.detach().clamp_min(1e-8).log()
        kl_div = F.kl_div(log_adv, raw_probs.detach(), reduction="none").sum(dim=1)
        if self.enable_consistency_gate:
            mask_cons = kl_div <= cons_thresh
        else:
            mask_cons = torch.ones_like(pred_labels, dtype=torch.bool)

        active_mask = mask_stat & mask_sem & mask_cons
        active_count = int(active_mask.sum().item())
        self._selected_counter += active_count
        self._zero_active_streak = 0 if active_count > 0 else (self._zero_active_streak + 1)

        reg_loss = self._fisher_regularizer(model)
        loss_adv = adv_entropy[active_mask].mean() if active_mask.any() else adv_entropy.new_zeros([])
        total_loss = loss_adv + reg_loss

        if optimizer is not None and total_loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

        quality = 1.0 - (stat_entropy.detach() / math.log(max(2, self.num_classes)))
        quality = quality.clamp(min=0.0, max=1.0)
        with torch.no_grad():
            if active_mask.any():
                self.update_prototypes_with_weights(
                    raw_feats.detach()[active_mask],
                    pred_labels.detach()[active_mask],
                    quality[active_mask],
                )
            elif (self.proto_counts is not None) and torch.any(self.proto_counts < self.warmup_min):
                warmup_mask = mask_stat & mask_cons
                if warmup_mask.any():
                    self.update_prototypes_with_weights(
                        raw_feats.detach()[warmup_mask],
                        pred_labels.detach()[warmup_mask],
                        quality[warmup_mask],
                    )
            if self.eata_memory is not None:
                self.eata_memory.push(raw_feats.detach(), raw_probs.detach())

        self._last_gate_log = {
            "mask_stat": mask_stat.detach().cpu(),
            "mask_sem": mask_sem.detach().cpu(),
            "mask_cons": mask_cons.detach().cpu(),
            "active_mask": active_mask.detach().cpu(),
            "stat_indices": mask_stat.detach().cpu().tolist(),
            "sem_indices": mask_sem.detach().cpu().tolist(),
            "cons_indices": mask_cons.detach().cpu().tolist(),
            "active_indices": active_mask.detach().cpu().tolist(),
            "entropy_threshold": float(entropy_threshold.detach().item()),
            "sem_threshold": float(sem_thresh),
            "cons_threshold": float(cons_thresh),
            "relaxation_level": int(relaxation_level),
        }
        self._last_batch_log = {
            "stat_gate_pass_rate": float(mask_stat.float().mean().item()),
            "sem_gate_pass_rate": float(mask_sem.float().mean().item()),
            "cons_gate_pass_rate": float(mask_cons.float().mean().item()),
            "active_gate_pass_rate": float(active_mask.float().mean().item()),
            "fisher_reg_value": float(reg_loss.detach().item()),
            "batch_entropy": float(raw_entropy.mean().item()),
            "adv_entropy": float(adv_entropy.mean().item()),
            "kl_mean": float(kl_div.mean().item()),
            "sem_threshold": float(sem_thresh),
            "cons_threshold": float(cons_thresh),
            "relaxation_level": int(relaxation_level),
        }
        return {
            "raw_logits": raw_logits,
            "raw_probs": raw_probs,
            "pred_labels": pred_labels,
            "raw_feats": raw_feats,
            "mask_stat": mask_stat,
            "mask_sem": mask_sem,
            "mask_cons": mask_cons,
            "active_mask": active_mask,
            "raw_entropy": raw_entropy,
            "adv_entropy": adv_entropy,
            "stat_entropy": stat_entropy,
            "kl_div": kl_div,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data = self._extract_primary_tensor(batch_data)
        outputs = self._forward_and_adapt_impl(raw_data, model, optimizer)
        return outputs["raw_logits"]


NuSTAR = ACCUP
