import math

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from algorithms.accup import ACCUP
from utils.utils import softmax_entropy_from_logits


class ACCUPInstrumented(ACCUP):
    """ACCUP variant with detailed logging for supplementary experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_log = []
        self.gate_log = []
        self._batch_idx = 0
        self._samples_seen = 0
        self._proto_prev = None
        self._fisher_reg_history = []
        self.last_adv_metadata = None
        self.adv_num_candidates = int(self.hparams.get("adv_num_candidates", 0))
        self.adv_control_points = int(self.hparams.get("adv_control_points", 4))
        self.enable_piecewise_adv = bool(self.hparams.get("enable_piecewise_adv", self.adv_num_candidates > 0))
        adv_sigma = self.hparams.get("adv_sigma", None)
        if adv_sigma is None:
            adv_sigma = max([abs(s) for s in self.adv_sigmas], default=0.0)
        self.adv_sigma = float(adv_sigma)

    def _extract_batch_parts(self, batch_data):
        if isinstance(batch_data, dict):
            raw = batch_data.get("data")
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            labels = batch_data.get("labels")
            meta = dict(batch_data.get("meta", {}))
            return raw, labels, meta
        if isinstance(batch_data, (list, tuple)):
            return batch_data[0], None, {}
        return batch_data, None, {}

    def get_adversarial_view(self, x, model):
        if self.enable_piecewise_adv and self.adv_num_candidates > 0 and self.adv_sigma > 0:
            return self._piecewise_adversarial_view(x, model)
        return self._global_factor_adversarial_view(x, model)

    def _global_factor_adversarial_view(self, x, model):
        factors = [0.0]
        for sigma in self.adv_sigmas:
            if sigma == 0:
                continue
            factors.extend([sigma, -sigma])

        factors_tensor = torch.tensor(factors, device=x.device, dtype=x.dtype)
        batch_size, _, steps = x.shape
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

        curve = best_factor.view(batch_size, 1).expand(batch_size, steps)
        self.last_adv_metadata = {
            "mode": "global_factor",
            "curve": curve.detach().cpu(),
            "control_points": best_factor.view(batch_size, 1).detach().cpu(),
            "score": best_entropy.detach().cpu(),
        }
        return x * (1.0 + best_factor.view(batch_size, 1, 1))

    def _piecewise_adversarial_view(self, x, model):
        batch_size, _, steps = x.shape
        device = x.device
        dtype = x.dtype
        num_candidates = max(1, self.adv_num_candidates)
        num_ctrl = max(2, self.adv_control_points)
        best_entropy = torch.full((batch_size,), float("-inf"), device=device, dtype=dtype)
        best_curve = torch.zeros(batch_size, steps, device=device, dtype=dtype)
        best_ctrl = torch.zeros(batch_size, num_ctrl, device=device, dtype=dtype)

        candidates = [torch.zeros(batch_size, num_ctrl, device=device, dtype=dtype)]
        for _ in range(num_candidates):
            candidates.append(torch.randn(batch_size, num_ctrl, device=device, dtype=dtype) * self.adv_sigma)

        for ctrl in candidates:
            curve = F.interpolate(ctrl.unsqueeze(1), size=steps, mode="linear", align_corners=True).squeeze(1)
            x_view = x * (1.0 + curve.unsqueeze(1))
            with torch.no_grad():
                feats, _ = model.feature_extractor(x_view)
                logits = model.classifier(feats)
                ent = softmax_entropy_from_logits(logits)
            better = ent > best_entropy
            best_entropy = torch.where(better, ent, best_entropy)
            best_curve = torch.where(better.view(batch_size, 1), curve, best_curve)
            best_ctrl = torch.where(better.view(batch_size, 1), ctrl, best_ctrl)

        self.last_adv_metadata = {
            "mode": "piecewise_search",
            "curve": best_curve.detach().cpu(),
            "control_points": best_ctrl.detach().cpu(),
            "score": best_entropy.detach().cpu(),
        }
        return x * (1.0 + best_curve.unsqueeze(1))

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, labels, meta = self._extract_batch_parts(batch_data)
        proto_snapshot = self.prototypes.detach().clone() if self.prototypes is not None else None

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

        mask_stat = self._statistical_gate(raw_entropy)

        feat_norm = F.normalize(raw_feats.detach(), dim=1)
        proto_available = (self.prototypes.abs().sum(dim=1) > 0).to(feat_norm.device)
        proto_norm = F.normalize(self.prototypes.to(feat_norm.device), dim=1)
        proto_for_sample = proto_norm[raw_preds]
        has_proto = proto_available[raw_preds]
        sim = (feat_norm * proto_for_sample).sum(dim=1)
        mask_sem = torch.where(has_proto, sim >= self.sem_thresh, torch.ones_like(sim, dtype=torch.bool))

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
            self._selected_counter = getattr(self, "_selected_counter", 0) + int(active_mask.sum().item())

        proto_drift = float("nan")
        if proto_snapshot is not None and self.prototypes is not None:
            proto_drift = float(torch.norm(self.prototypes.detach() - proto_snapshot).item())

        batch_f1 = float("nan")
        if labels is not None:
            try:
                batch_f1 = float(
                    f1_score(
                        labels.detach().cpu().numpy(),
                        raw_preds.detach().cpu().numpy(),
                        average="macro",
                    )
                )
            except Exception:
                batch_f1 = float("nan")

        batch_size = int(raw_data.size(0))
        self._samples_seen += batch_size
        gate_acceptance_rate = float(active_mask.float().mean().item())
        fisher_reg_value = float(reg.detach().item())
        self._fisher_reg_history.append(fisher_reg_value)

        gate_log = {
            "batch_idx": self._batch_idx,
            "B": batch_size,
            "stat_pass": int(mask_stat.sum().item()),
            "sem_pass": int(mask_sem.sum().item()),
            "cons_pass": int(mask_cons.sum().item()),
            "all_pass": int(active_mask.sum().item()),
            "stat_and_sem": int((mask_stat & mask_sem).sum().item()),
            "stat_and_cons": int((mask_stat & mask_cons).sum().item()),
            "sem_and_cons": int((mask_sem & mask_cons).sum().item()),
            "stat_indices": mask_stat.detach().cpu().tolist(),
            "sem_indices": mask_sem.detach().cpu().tolist(),
            "cons_indices": mask_cons.detach().cpu().tolist(),
            "active_indices": active_mask.detach().cpu().tolist(),
        }
        self.gate_log.append(gate_log)

        stream_item = {
            "batch_idx": self._batch_idx,
            "samples_seen": self._samples_seen,
            "gate_acceptance_rate": gate_acceptance_rate,
            "stat_gate_pass": float(mask_stat.float().mean().item()),
            "sem_gate_pass": float(mask_sem.float().mean().item()),
            "cons_gate_pass": float(mask_cons.float().mean().item()),
            "proto_drift_norm": proto_drift,
            "fisher_reg_value": fisher_reg_value,
            "batch_f1": batch_f1,
            "batch_entropy": float(raw_entropy.mean().item()),
            "adv_entropy": float(adv_entropy.mean().item()),
            "kl_mean": float(kl.mean().item()),
            "loss": float(loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "corruption_phase": meta.get("corruption_phase", "unknown"),
            "corruption_type": meta.get("corruption_type"),
            "severity": meta.get("severity"),
        }
        self.stream_log.append(stream_item)
        self._batch_idx += 1
        return raw_logits
