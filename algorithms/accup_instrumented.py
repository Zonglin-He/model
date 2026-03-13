import torch
from sklearn.metrics import f1_score

from algorithms.accup import ACCUP


class ACCUPInstrumented(ACCUP):
    """ACCUP variant with detailed logging for supplementary experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_log = []
        self.gate_log = []
        self._batch_idx = 0
        self._samples_seen = 0
        self._fisher_reg_history = []

    def _extract_batch_parts(self, batch_data):
        if isinstance(batch_data, dict):
            raw = self._extract_primary_tensor(batch_data)
            labels = batch_data.get("labels")
            meta = dict(batch_data.get("meta", {}))
            return raw, labels, meta
        return self._extract_primary_tensor(batch_data), None, {}

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, labels, meta = self._extract_batch_parts(batch_data)
        proto_snapshot = self.prototypes.detach().clone() if self.prototypes is not None else None

        outputs = self._forward_and_adapt_impl(raw_data, model, optimizer)
        raw_logits = outputs["raw_logits"]
        pred_labels = outputs["pred_labels"]
        raw_entropy = outputs["raw_entropy"]
        adv_entropy = outputs["adv_entropy"]
        kl_div = outputs["kl_div"]
        reg_loss = outputs["reg_loss"]
        total_loss = outputs["total_loss"]
        active_mask = outputs["active_mask"]
        mask_stat = outputs["mask_stat"]
        mask_sem = outputs["mask_sem"]
        mask_cons = outputs["mask_cons"]

        proto_drift = float("nan")
        if proto_snapshot is not None and self.prototypes is not None:
            proto_drift = float(torch.norm(self.prototypes.detach() - proto_snapshot).item())

        batch_f1 = float("nan")
        if labels is not None:
            try:
                batch_f1 = float(
                    f1_score(
                        labels.detach().cpu().numpy(),
                        pred_labels.detach().cpu().numpy(),
                        average="macro",
                    )
                )
            except Exception:
                batch_f1 = float("nan")

        batch_size = int(raw_data.size(0))
        self._samples_seen += batch_size
        fisher_reg_value = float(reg_loss.detach().item())
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
            "sem_threshold": self._last_gate_log.get("sem_threshold"),
            "cons_threshold": self._last_gate_log.get("cons_threshold"),
            "entropy_threshold": self._last_gate_log.get("entropy_threshold"),
            "relaxation_level": self._last_gate_log.get("relaxation_level", 0),
        }
        self.gate_log.append(gate_log)

        stream_item = {
            "batch_idx": self._batch_idx,
            "samples_seen": self._samples_seen,
            "gate_acceptance_rate": float(active_mask.float().mean().item()),
            "stat_gate_pass": float(mask_stat.float().mean().item()),
            "sem_gate_pass": float(mask_sem.float().mean().item()),
            "cons_gate_pass": float(mask_cons.float().mean().item()),
            "proto_drift_norm": proto_drift,
            "fisher_reg_value": fisher_reg_value,
            "batch_f1": batch_f1,
            "batch_entropy": float(raw_entropy.mean().item()),
            "adv_entropy": float(adv_entropy.mean().item()),
            "kl_mean": float(kl_div.mean().item()),
            "loss": float((total_loss - reg_loss).detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "corruption_phase": meta.get("corruption_phase", "unknown"),
            "corruption_type": meta.get("corruption_type"),
            "severity": meta.get("severity"),
            "sem_threshold": self._last_batch_log.get("sem_threshold"),
            "cons_threshold": self._last_batch_log.get("cons_threshold"),
            "relaxation_level": self._last_batch_log.get("relaxation_level", 0),
        }
        self.stream_log.append(stream_item)
        self._batch_idx += 1
        return raw_logits
