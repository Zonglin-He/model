"""Shared utilities used by the DuSafe training and evaluation path."""

from __future__ import annotations

import logging
import os
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import torch


try:
    import torch.serialization as _torch_serialization

    _torch_serialization.add_safe_globals([np.ndarray])
except Exception:
    pass


def safe_torch_load(*args, **kwargs):
    """Load legacy tensor checkpoints explicitly with pickle enabled."""
    kwargs.setdefault("weights_only", False)
    kwargs.setdefault("pickle_module", pickle)
    return torch.load(*args, **kwargs)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(log_path, level=logging.DEBUG):
    logger = logging.getLogger(log_path)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    del data_type, da_method
    log_dir = os.path.join(
        exp_log_dir, f"{src_id}_to_{tgt_id}_run_{run_id}"
    )
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    return _logger(os.path.join(log_dir, f"logs_{timestamp}.log")), log_dir


def softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


__all__ = [
    "AverageMeter",
    "fix_randomness",
    "safe_torch_load",
    "softmax_entropy_from_logits",
    "starting_logs",
]
