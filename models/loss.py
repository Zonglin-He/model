"""Supervised source-training loss."""

import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, device, epsilon=0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)
        self.device = device
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, labels):
        log_probabilities = self.log_softmax(logits)
        targets = torch.zeros_like(log_probabilities).scatter_(
            1, labels.unsqueeze(1), 1
        )
        targets = (
            (1.0 - self.epsilon) * targets
            + self.epsilon / self.num_classes
        )
        return (-targets * log_probabilities).sum(dim=1).mean()


__all__ = ["CrossEntropyLabelSmooth"]
