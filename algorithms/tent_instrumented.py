import torch
from sklearn.metrics import f1_score

from algorithms.tent_tta import Tent, softmax_entropy


class TentInstrumented(Tent):
    """Tent with per-batch logging for failure analysis."""

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
        raw_data = self._extract_primary_tensor(batch_data)
        logits = super().forward_and_adapt(batch_data, model, optimizer)
        entropy = softmax_entropy(logits.detach())
        preds = logits.argmax(dim=1)

        batch_f1 = float("nan")
        if labels is not None:
            try:
                batch_f1 = float(
                    f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
                )
            except Exception:
                batch_f1 = float("nan")

        selected_mask = self._last_gate_log["selected_mask"]
        batch_size = int(raw_data.size(0))
        self._samples_seen += batch_size
        self.gate_log.append(
            {
                "batch_idx": self._batch_idx,
                "B": batch_size,
                "selected": int(selected_mask.sum().item()),
                "selected_indices": selected_mask.tolist(),
            }
        )
        self.stream_log.append(
            {
                "batch_idx": self._batch_idx,
                "samples_seen": self._samples_seen,
                "selected_rate": 1.0,
                "batch_f1": batch_f1,
                "batch_entropy": float(entropy.mean().item()),
                "total_loss": self._last_batch_log["total_loss"],
                "corruption_phase": meta.get("corruption_phase", "unknown"),
                "corruption_type": meta.get("corruption_type"),
                "severity": meta.get("severity"),
            }
        )
        self._batch_idx += 1
        return logits
