import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm
from utils.utils import safe_torch_load


def _softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)


def _update_probs_momentum(
    current: Optional[torch.Tensor],
    new_probs: torch.Tensor,
    momentum: float = 0.9,
) -> Optional[torch.Tensor]:
    if new_probs.numel() == 0:
        return current
    mean_new = new_probs.mean(dim=0)
    if current is None:
        return mean_new
    return momentum * current + (1.0 - momentum) * mean_new


class EATA(BaseTestTimeAlgorithm):
    """Minimal EATA baseline aligned with the ACCUP trainer interface."""

    def __init__(self, configs, hparams, model, optimizer):
        self.adapt_keywords = tuple(hparams.get("adapt_keywords", ("classifier", "adapter")))
        super(EATA, self).__init__(configs, hparams, model, optimizer)

        self.featurizer = model.feature_extractor
        self.classifier = model.classifier
        self.num_classes = configs.num_classes

        self.e_margin = float(hparams.get("e_margin", math.log(self.num_classes) * 0.40))
        self.d_margin = float(hparams.get("d_margin", 0.05))
        self.fisher_alpha = float(hparams.get("fisher_alpha", 2000.0))
        self.grad_clip = float(hparams.get("grad_clip", 5.0))
        self.current_model_probs: Optional[torch.Tensor] = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}

        self._freeze_all_and_unfreeze_keywords(self.model, self.adapt_keywords)
        self.theta0 = {n: p.detach().clone() for n, p in self._iter_trainable_named_params(self.model)}

        self.fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        if "fisher_state" in hparams and isinstance(hparams["fisher_state"], dict):
            self.fishers = {}
            for k, v in hparams["fisher_state"].items():
                diag, theta = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v, self.theta0.get(k, None))
                if theta is None:
                    continue
                self.fishers[k] = (diag.detach().clone(), theta.detach().clone())
        elif "fisher_path" in hparams and isinstance(hparams["fisher_path"], str) and os.path.exists(hparams["fisher_path"]):
            raw = safe_torch_load(hparams["fisher_path"], map_location="cpu")
            self.fishers = {}
            for k, v in raw.items():
                if isinstance(v, (list, tuple)):
                    diag, theta = v[0], v[1]
                else:
                    diag, theta = v, self.theta0.get(k, None)
                if theta is None:
                    continue
                self.fishers[k] = (diag.detach().clone(), theta.detach().clone())

    @staticmethod
    def _extract_batch_views(batch_data):
        if isinstance(batch_data, dict):
            batch_data = batch_data.get("data", batch_data)
        if isinstance(batch_data, (list, tuple)):
            raw_data = batch_data[0]
            aug_data = batch_data[1] if len(batch_data) > 1 else batch_data[0]
            return raw_data, aug_data
        return batch_data, batch_data

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in self.adapt_keywords):
                param.requires_grad_(True)
        return model

    @torch.enable_grad()
    def _forward_and_adapt_impl(self, batch_data, model, optimizer):
        raw_data, aug_data = self._extract_batch_views(batch_data)

        raw_feat, _ = model.feature_extractor(raw_data)
        raw_logits = model.classifier(raw_feat)
        aug_feat, _ = model.feature_extractor(aug_data)
        aug_logits = model.classifier(aug_feat)
        probs_raw = raw_logits.softmax(dim=1)

        entropy = _softmax_entropy_from_logits(raw_logits)
        ids1 = torch.where(entropy < self.e_margin)[0]
        ent_sel = entropy[ids1]

        ids2 = torch.arange(ids1.numel(), device=ids1.device)
        diversity = entropy.new_empty((0,))
        if self.current_model_probs is not None and ids1.numel() > 0:
            ref_probs = self.current_model_probs.to(probs_raw.device)
            cos = F.cosine_similarity(ref_probs.unsqueeze(0), probs_raw[ids1], dim=1)
            diversity = (1.0 - cos).clamp(min=0.0)
            ids2 = torch.where(diversity > self.d_margin)[0]
            ent_sel = ent_sel[ids2]

        if ent_sel.numel() > 0:
            coeff = torch.exp(-(ent_sel.detach() - self.e_margin))
            loss_ent = (_softmax_entropy_from_logits(aug_logits[ids1][ids2]) * coeff).mean()
        else:
            loss_ent = raw_logits.new_zeros([])

        loss_reg = self._regularizer(model)
        loss = loss_ent + loss_reg

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            if ids1.numel() > 0 and ids2.numel() > 0:
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._iter_trainable_params(model), self.grad_clip)
                optimizer.step()

        with torch.no_grad():
            picked = probs_raw[ids1][ids2] if (ids1.numel() > 0 and ids2.numel() > 0) else probs_raw.new_zeros((0, probs_raw.size(1)))
            self.current_model_probs = _update_probs_momentum(self.current_model_probs, picked)
            self.num_samples_update_1 += int(ids1.numel())
            self.num_samples_update_2 += int(ids2.numel())

        entropy_mask = torch.zeros_like(entropy, dtype=torch.bool)
        entropy_mask[ids1] = True
        diversity_mask = torch.zeros_like(entropy, dtype=torch.bool)
        if ids1.numel() > 0:
            diversity_mask[ids1[ids2]] = True
        selected_mask = diversity_mask.clone()
        self._selected_counter += int(selected_mask.sum().item())

        self._last_gate_log = {
            "mask_entropy": entropy_mask.detach().cpu(),
            "mask_diversity": diversity_mask.detach().cpu(),
            "selected_mask": selected_mask.detach().cpu(),
            "entropy_indices": entropy_mask.detach().cpu().tolist(),
            "diversity_indices": diversity_mask.detach().cpu().tolist(),
            "selected_indices": selected_mask.detach().cpu().tolist(),
        }
        self._last_batch_log = {
            "entropy_gate_pass_rate": float(entropy_mask.float().mean().item()),
            "diversity_gate_pass_rate": float(diversity_mask.float().mean().item()),
            "selected_pass_rate": float(selected_mask.float().mean().item()),
            "batch_entropy": float(entropy.mean().item()),
            "loss_ent": float(loss_ent.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "total_loss": float(loss.detach().item()),
        }

        ensemble_logits = (raw_logits + aug_logits) * 0.5
        return {
            "raw_logits": raw_logits,
            "aug_logits": aug_logits,
            "ensemble_logits": ensemble_logits,
            "probs_raw": probs_raw,
            "entropy": entropy,
            "diversity": diversity,
            "entropy_mask": entropy_mask,
            "diversity_mask": diversity_mask,
            "selected_mask": selected_mask,
            "loss_ent": loss_ent,
            "loss_reg": loss_reg,
            "total_loss": loss,
        }

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        outputs = self._forward_and_adapt_impl(batch_data, model, optimizer)
        return outputs["ensemble_logits"]

    def _iter_trainable_named_params(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _iter_trainable_params(self, model):
        for _, param in self._iter_trainable_named_params(model):
            yield param

    def _freeze_all_and_unfreeze_keywords(self, model, keywords):
        model.train()
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in keywords):
                param.requires_grad_(True)

    def _regularizer(self, model) -> torch.Tensor:
        reg = None
        if self.fishers is not None:
            for name, param in self._iter_trainable_named_params(model):
                if name in self.fishers:
                    diag, theta_prev = self.fishers[name]
                    diag = diag.to(param.device)
                    theta_prev = theta_prev.to(param.device)
                    term = (diag * (param - theta_prev) ** 2).sum()
                    reg = term if reg is None else (reg + term)
            if reg is not None:
                return reg * self.fisher_alpha

        for name, param in self._iter_trainable_named_params(model):
            theta0 = self.theta0.get(name, None)
            if theta0 is None:
                continue
            term = ((param - theta0.to(param.device)) ** 2).sum()
            reg = term if reg is None else (reg + term)
        return reg if reg is not None else torch.zeros([], device=next(model.parameters()).device)
