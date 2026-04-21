import math
from copy import deepcopy

import torch
import torch.nn as nn

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                e_w = (torch.pow(param, 2) if group["adaptive"] else 1.0) * param.grad * scale.to(param)
                param.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self, zero_grad=False):
        for group in self.param_groups:
            for param in group["params"]:
                if "old_p" in self.state[param]:
                    param.data = self.state[param]["old_p"]
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.restore(zero_grad=False)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    factor = torch.abs(param) if group["adaptive"] else 1.0
                    norms.append((factor * param.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAR(BaseTestTimeAlgorithm):
    """Time-series SAR baseline with SAM updates and entropy filtering."""

    def __init__(self, configs, hparams, model, optimizer):
        self.margin_e0 = float(hparams.get("sar_margin_e0", -1.0))
        self.reset_constant_em = float(hparams.get("sar_reset_constant_em", 0.2))
        self.sar_rho = float(hparams.get("sar_rho", 0.05))
        self.sar_adaptive = bool(hparams.get("sar_adaptive", False))
        self.sar_base_optimizer = str(hparams.get("sar_base_optimizer", "sgd")).lower()
        self.episodic = bool(hparams.get("episodic", False))
        self.ema = None
        super(SAR, self).__init__(configs, hparams, model, optimizer)
        self._rebuild_sam_optimizer()
        self._selected_counter = 0
        self._last_gate_log = {}
        self._last_batch_log = {}
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None

    def _rebuild_sam_optimizer(self):
        if self.optimizer is None:
            return
        params = []
        for group in self.optimizer.param_groups:
            params.extend(group["params"])
        base_cls = torch.optim.SGD if self.sar_base_optimizer == "sgd" else torch.optim.Adam
        kwargs = {
            "lr": self.hparams["learning_rate"],
            "weight_decay": self.hparams["weight_decay"],
        }
        if base_cls is torch.optim.SGD:
            kwargs["momentum"] = self.hparams.get("momentum", 0.9)
        self.optimizer = SAM(params, base_cls, rho=self.sar_rho, adaptive=self.sar_adaptive, **kwargs)

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
        self.ema = None

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        if self.episodic:
            self.reset()

        raw_data = self._extract_primary_tensor(batch_data)
        logits = model(raw_data)
        entropy = softmax_entropy(logits)
        margin = self.margin_e0 if self.margin_e0 > 0 else 0.4 * math.log(max(2, self.configs.num_classes))
        mask_first = entropy < margin
        selected_mask = torch.zeros_like(mask_first)
        ema_before = self.ema
        reset_flag = False

        if optimizer is not None and mask_first.any():
            optimizer.zero_grad()
            loss_first = entropy[mask_first].mean()
            loss_first.backward()
            optimizer.first_step(zero_grad=True)

            logits_second = model(raw_data)
            entropy_second = softmax_entropy(logits_second)[mask_first]
            candidate_indices = torch.where(mask_first)[0]
            mask_second_local = entropy_second < margin
            if mask_second_local.any():
                selected_indices = candidate_indices[mask_second_local]
                selected_mask[selected_indices] = True
                loss_second = entropy_second[mask_second_local].mean()
                self.ema = loss_second.item() if self.ema is None else (0.9 * self.ema + 0.1 * loss_second.item())
                loss_second.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.restore(zero_grad=True)

        if self.ema is not None and self.ema < self.reset_constant_em:
            reset_flag = True
            self.reset()

        self._selected_counter += int(selected_mask.sum().item())
        self._last_gate_log = {
            "mask_entropy": mask_first.detach().cpu(),
            "selected_mask": selected_mask.detach().cpu(),
            "entropy_indices": mask_first.detach().cpu().tolist(),
            "selected_indices": selected_mask.detach().cpu().tolist(),
            "reset_flag": reset_flag,
            "ema_before": ema_before,
            "ema_after": self.ema,
        }
        self._last_batch_log = {
            "entropy_gate_pass_rate": float(mask_first.float().mean().item()),
            "selected_pass_rate": float(selected_mask.float().mean().item()),
            "batch_entropy": float(entropy.mean().item()),
            "reset_flag": float(reset_flag),
            "ema": float(self.ema) if self.ema is not None else float("nan"),
        }
        return logits
