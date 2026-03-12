import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm
from utils.utils import EATAMemory, softmax_entropy_from_logits


class ACCUP(BaseTestTimeAlgorithm):
    """
    NuSTAR: adversarial amplitude search + triple gate reliability before entropy minimization.
    """

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(configs, hparams, model, optimizer)

        self.num_classes = configs.num_classes
        self.adv_sigmas = self._parse_sigmas(hparams.get("adv_sigmas", [0.0]))
        self.sem_thresh = float(hparams.get("sem_thresh", 0.5))
        self.cons_thresh = float(hparams.get("cons_thresh", 0.5))
        self.proto_momentum = float(hparams.get("proto_momentum", 0.9))
        self.entropy_quantile = float(hparams.get("entropy_quantile", 0.7))
        self.entropy_hist_len = int(hparams.get("entropy_hist_len", 256))
        self.lambda_reg = float(hparams.get("lambda_reg", 0.0))

        mem_len = int(hparams.get("memory_size", 4096))
        mem_device = hparams.get("device", "cpu")
        self.eata_memory = getattr(self, "eata_memory", None) or EATAMemory(maxlen=mem_len, device=mem_device)

        prototypes = self._init_prototypes()
        self.register_buffer("prototypes", prototypes)
        self._entropy_history = None
        self._fisher_diag = None
        self._fisher_count = 0
        self._theta0 = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def _parse_sigmas(sigmas):
        if isinstance(sigmas, (int, float)):
            sigmas = [float(sigmas)]
        elif isinstance(sigmas, str):
            sigmas = [float(s.strip()) for s in sigmas.split(",") if s.strip()]
        return [float(s) for s in sigmas] if sigmas else [0.0]

    def _init_prototypes(self):
        device = next(self.model.parameters()).device
        warmup = None

        if hasattr(self.model, "classifier"):
            if hasattr(self.model.classifier, "logits"):
                warmup = self.model.classifier.logits.weight.data.detach()
            elif hasattr(self.model.classifier, "weight"):
                warmup = self.model.classifier.weight.data.detach()

        if warmup is not None:
            feat_dim = warmup.shape[1]
            prototypes = warmup.clone()
        else:
            feat_dim = getattr(self.model.classifier, "in_features", None)
            if feat_dim is None:
                feat_dim = getattr(self.configs, "final_out_channels", 1) * getattr(self.configs, "features_len", 1)
            prototypes = torch.zeros(self.num_classes, int(feat_dim), device=device)

        prototypes = F.normalize(prototypes, dim=1)
        return prototypes.to(device)

    def set_eata_memory(self, memory):
        self.eata_memory = memory

    def configure_model(self, model):
        """Unfreeze adaptation parameters (default: BN affine + backbone + classifier)."""
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

        if train_full_backbone:
            for param in model.feature_extractor.parameters():
                param.requires_grad_(True)
        elif train_backbone_modules:
            target_names = set(
                train_backbone_modules if isinstance(train_backbone_modules, (list, tuple, set)) else [train_backbone_modules]
            )
            for name, module in model.feature_extractor.named_modules():
                if name in target_names:
                    for param in module.parameters():
                        param.requires_grad_(True)
        else:
            for name, module in model.feature_extractor.named_children():
                if name in ("conv_block1", "conv_block2", "conv_block3"):
                    for sub_module in module.children():
                        if isinstance(sub_module, nn.Conv1d):
                            sub_module.requires_grad_(True)

        if train_classifier and hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad_(True)

        return model

    def get_adversarial_view(self, x, model):
        factors = [0.0]
        for sigma in self.adv_sigmas:
            if sigma == 0:
                continue
            factors.extend([sigma, -sigma])

        factors_tensor = torch.tensor(factors, device=x.device, dtype=x.dtype)
        batch_size = x.size(0)
        best_entropy = torch.full((batch_size,), float("-inf"), device=x.device, dtype=x.dtype)
        best_factor = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        for factor in factors_tensor:
            x_view = x * (1.0 + factor)
            with torch.no_grad():
                feats, _ = model.feature_extractor(x_view)
                logits = model.classifier(feats)
                ent = softmax_entropy_from_logits(logits)
            better = ent > best_entropy
            best_entropy = torch.where(better, ent, best_entropy)
            best_factor = torch.where(better, torch.full_like(best_factor, factor), best_factor)

        reshape_dims = (batch_size,) + (1,) * (x.dim() - 1)
        return x * (1.0 + best_factor.view(reshape_dims))

    @torch.no_grad()
    def update_prototypes(self, feats, labels):
        if feats.numel() == 0:
            return
        feats = F.normalize(feats, dim=1)
        proto = self.prototypes
        momentum = self.proto_momentum
        for cls in labels.unique():
            cls = int(cls)
            mask = labels == cls
            if not mask.any():
                continue
            cls_feat = feats[mask].mean(dim=0, keepdim=True)
            updated = momentum * proto[cls : cls + 1] + (1.0 - momentum) * cls_feat
            proto[cls : cls + 1] = F.normalize(updated, dim=1)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data

        x_adv = self.get_adversarial_view(raw_data, model)

        with torch.no_grad():
            raw_feats, _ = model.feature_extractor(raw_data)
            raw_logits = model.classifier(raw_feats)
            raw_probs = torch.softmax(raw_logits, dim=1)
            raw_preds = raw_probs.argmax(dim=1)
            raw_entropy = softmax_entropy_from_logits(raw_logits)

        adv_feats, _ = model.feature_extractor(x_adv)
        adv_logits = model.classifier(adv_feats)
        adv_probs = torch.softmax(adv_logits, dim=1)
        adv_entropy = softmax_entropy_from_logits(adv_logits)

        # Gate 1: statistical (entropy quantile over history)
        mask_stat = self._statistical_gate(raw_entropy)

        # Gate 2: semantic consistency with prototypes
        feat_norm = F.normalize(raw_feats.detach(), dim=1)
        proto_available = (self.prototypes.abs().sum(dim=1) > 0).to(feat_norm.device)
        proto_norm = F.normalize(self.prototypes.to(feat_norm.device), dim=1)
        proto_for_sample = proto_norm[raw_preds]
        has_proto = proto_available[raw_preds]
        sim = (feat_norm * proto_for_sample).sum(dim=1)
        mask_sem = torch.where(has_proto, sim >= self.sem_thresh, torch.ones_like(sim, dtype=torch.bool))

        # Gate 3: prediction consistency under attack (KL raw || adv)
        mask_cons, kl = self._consistency_gate(raw_probs, adv_probs)

        active_mask = mask_stat & mask_sem & mask_cons
        mask_float = active_mask.float()

        loss = (adv_entropy * mask_float).mean()
        reg = self._fisher_regularizer(model, raw_logits, raw_preds)
        total_loss = loss + reg

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            if active_mask.any():
                self.update_prototypes(raw_feats[active_mask], raw_preds[active_mask])
            if self.eata_memory is not None:
                self.eata_memory.push(raw_feats, raw_probs)
            try:
                self._selected_counter = getattr(self, "_selected_counter", 0) + int(active_mask.sum().item())
            except Exception:
                pass

        return raw_logits

    def _update_entropy_history(self, entropy: torch.Tensor):
        ent_detached = entropy.detach()
        if self._entropy_history is None:
            self._entropy_history = ent_detached[-self.entropy_hist_len :]
        else:
            self._entropy_history = torch.cat([self._entropy_history, ent_detached], dim=0)[-self.entropy_hist_len :]
        return self._entropy_history

    def _statistical_gate(self, entropy: torch.Tensor):
        history = self._update_entropy_history(entropy)
        if history.numel() == 0:
            threshold = math.log(max(2, self.num_classes))
        else:
            threshold = torch.quantile(history, self.entropy_quantile).item()
        return entropy <= threshold

    def _consistency_gate(self, raw_probs, adv_probs):
        adv_probs_safe = adv_probs.clamp_min(1e-8)
        raw_probs_safe = raw_probs.clamp_min(1e-8)
        kl = F.kl_div(raw_probs_safe.log(), adv_probs_safe, reduction="none").sum(dim=1)
        return kl <= self.cons_thresh, kl

    def _fisher_regularizer(self, model, raw_logits, raw_preds):
        if self.lambda_reg <= 0.0 or self._theta0 is None:
            return raw_logits.new_zeros(())

        params = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]

        log_probs = torch.log_softmax(raw_logits, dim=1)
        fisher_loss = F.nll_loss(log_probs, raw_preds, reduction="mean")
        grads = torch.autograd.grad(fisher_loss, params, retain_graph=True, allow_unused=True)

        if self._fisher_diag is None:
            self._fisher_diag = {n: torch.zeros_like(p) for n, p in zip(names, params)}

        for name, grad in zip(names, grads):
            if grad is None:
                continue
            self._fisher_diag[name] = self._fisher_diag[name] + grad.detach() ** 2
        self._fisher_count += 1

        reg = None
        normalizer = float(max(1, self._fisher_count))
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self._theta0 or name not in self._fisher_diag:
                continue
            diag = self._fisher_diag[name] / normalizer
            delta = p - self._theta0[name].to(p.device)
            term = (diag * delta * delta).sum()
            reg = term if reg is None else reg + term

        if reg is None:
            return raw_logits.new_zeros(())
        return self.lambda_reg * reg
