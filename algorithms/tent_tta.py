from copy import deepcopy

import torch
import torch.nn as nn

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class Tent(BaseTestTimeAlgorithm):
    """Time-series Tent baseline using norm affine parameters."""

    def __init__(self, configs, hparams, model, optimizer):
        self.episodic = bool(hparams.get("episodic", False))
        super(Tent, self).__init__(configs, hparams, model, optimizer)
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None

    def collect_params(self, model: nn.Module):
        params = []
        names = []
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for param_name, param in module.named_parameters():
                    if param_name in ("weight", "bias") and param.requires_grad:
                        params.append(param)
                        names.append(f"{module_name}.{param_name}")
        return params, names

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                module.requires_grad_(True)
        return model

    @staticmethod
    def _extract_primary_tensor(batch_data):
        if isinstance(batch_data, dict):
            batch_data = batch_data.get("data", batch_data)
        if isinstance(batch_data, (list, tuple)):
            return batch_data[0]
        return batch_data

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        if self.episodic:
            self.reset()

        raw_data = self._extract_primary_tensor(batch_data)
        logits = model(raw_data)
        entropy = softmax_entropy(logits)
        loss = entropy.mean()

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        selected_mask = torch.ones_like(entropy, dtype=torch.bool)
        self._selected_counter += int(selected_mask.sum().item())
        self._last_gate_log = {
            "selected_mask": selected_mask.detach().cpu(),
            "selected_indices": selected_mask.detach().cpu().tolist(),
        }
        self._last_batch_log = {
            "selected_pass_rate": 1.0,
            "batch_entropy": float(entropy.mean().item()),
            "total_loss": float(loss.detach().item()),
        }
        return logits
