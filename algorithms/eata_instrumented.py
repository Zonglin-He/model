import torch
from sklearn.metrics import f1_score

from algorithms.eata_accup import EATA


class EATAInstrumented(EATA):
    """EATA variant with per-batch logging for failure analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_log = []
        self.gate_log = []
        self._batch_idx = 0
        self._samples_seen = 0

    @staticmethod
    def _extract_labels_meta(batch_data):
        if isinstance(batch_data, dict):
            return batch_data.get("labels"), dict(batch_data.get("meta", {}))
        return None, {}

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        labels, meta = self._extract_labels_meta(batch_data)
        raw_data, _ = self._extract_batch_views(batch_data)
        outputs = self._forward_and_adapt_impl(batch_data, model, optimizer)

        selected_mask = outputs["selected_mask"]
        entropy_mask = outputs["entropy_mask"]
        diversity_mask = outputs["diversity_mask"]
        ensemble_logits = outputs["ensemble_logits"]
        pred_labels = ensemble_logits.argmax(dim=1)

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

        self.gate_log.append(
            {
                "batch_idx": self._batch_idx,
                "B": batch_size,
                "entropy_pass": int(entropy_mask.sum().item()),
                "diversity_pass": int(diversity_mask.sum().item()),
                "selected": int(selected_mask.sum().item()),
                "entropy_indices": entropy_mask.detach().cpu().tolist(),
                "diversity_indices": diversity_mask.detach().cpu().tolist(),
                "selected_indices": selected_mask.detach().cpu().tolist(),
            }
        )
        self.stream_log.append(
            {
                "batch_idx": self._batch_idx,
                "samples_seen": self._samples_seen,
                "entropy_gate_pass": float(entropy_mask.float().mean().item()),
                "diversity_gate_pass": float(diversity_mask.float().mean().item()),
                "selected_rate": float(selected_mask.float().mean().item()),
                "batch_f1": batch_f1,
                "batch_entropy": float(outputs["entropy"].mean().item()),
                "loss_ent": float(outputs["loss_ent"].detach().item()),
                "loss_reg": float(outputs["loss_reg"].detach().item()),
                "total_loss": float(outputs["total_loss"].detach().item()),
                "corruption_phase": meta.get("corruption_phase", "unknown"),
                "corruption_type": meta.get("corruption_type"),
                "severity": meta.get("severity"),
            }
        )
        self._batch_idx += 1
        return ensemble_logits
