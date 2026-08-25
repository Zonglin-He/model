"""Minimal base class for online test-time adaptation."""

import torch
import torch.nn as nn


class BaseTestTimeAlgorithm(torch.nn.Module):
    def __init__(self, configs, hparams, model, optimizer):
        super().__init__()
        self.configs = configs
        self.hparams = hparams
        self.model = self.configure_model(model)
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = optimizer(parameters) if parameters else None
        self.steps = int(self.hparams["steps"])
        if self.steps < 1:
            raise ValueError("steps must be at least 1")

    def configure_model(self, model):
        raise NotImplementedError

    def forward_and_adapt(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, inputs, trg_idx=None):
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(
                inputs, self.model, self.optimizer, trg_idx
            )
        return outputs


__all__ = ["BaseTestTimeAlgorithm"]
