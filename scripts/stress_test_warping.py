import argparse
import csv
import os
import sys
from typing import Dict, Any, Optional

import numpy as np
import torch
from scipy.interpolate import CubicSpline

# 将项目根目录加入 sys.path，便于直接运行脚本
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.data_model_configs import EEG, HAR, FD
from dataloader.dataloader import data_generator
from models.da_models import classifier
from models.timesnet_backbone import TimesNet
from utils.utils import safe_torch_load
import glob


def get_dataset_config(name: str):
    if name == "EEG":
        return EEG()
    if name == "HAR":
        return HAR()
    if name == "FD":
        return FD()
    raise ValueError(f"Unknown dataset: {name}")


def manual_magnitude_warp(x: np.ndarray, sigma: float, knots: int = 4) -> np.ndarray:
    """Apply magnitude warping with a cubic spline curve."""
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x_np.shape}")

    # Ensure channel-first layout: (C, L)
    transpose_back = False
    if x_np.shape[0] > x_np.shape[1]:
        x_np = x_np.T
        transpose_back = True
    C, L = x_np.shape
    steps = np.linspace(0, L - 1, num=knots + 2)
    orig = np.arange(L)

    warped = np.zeros_like(x_np, dtype=np.float32)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(C, knots + 2))
    for c in range(C):
        spline = CubicSpline(steps, random_warps[c])
        curve = spline(orig)
        warped[c] = x_np[c] * curve

    return warped.T if transpose_back else warped


def load_pretrained_model(ckpt_path: str, configs) -> torch.nn.Module:
    backbone = TimesNet(configs)
    head = classifier(configs)

    class TimesNetClassifier(torch.nn.Module):
        def __init__(self, fe, cls):
            super().__init__()
            self.feature_extractor = fe
            self.classifier = cls

        def forward(self, x):
            feat, _ = self.feature_extractor(x)
            return self.classifier(feat)

    model = TimesNetClassifier(backbone, head)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = safe_torch_load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(raw)
    _load_state_dict_flex(model, state_dict)
    return model


def _extract_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("non_adapted", "model_dict", "state_dict", "network", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError("Unrecognized checkpoint format; expected a state dict or dict containing a state dict.")


def _load_state_dict_flex(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except Exception:
        pass
    if hasattr(model, "network"):
        try:
            model.network.load_state_dict(state_dict, strict=True)
            return
        except Exception:
            pass
    model.load_state_dict(state_dict, strict=False)


def _resolve_checkpoint(args) -> str:
    if args.checkpoint:
        return args.checkpoint

    default = os.path.join("results", "pretrain_logs", args.dataset, f"train_{args.source_domain}", "checkpoint.pt")
    if os.path.exists(default):
        return default

    # Fallback: try to find any checkpoint containing train_{source_domain}
    pattern = os.path.join("results", "**", f"*{args.source_domain}*", "checkpoint.pt")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"找不到预训练权重，请用 --checkpoint 指定路径。默认尝试过: {default}"
    )


def run_stress_test(args):
    device = torch.device(args.device)
    configs = get_dataset_config(args.dataset)

    # Build model and load weights
    ckpt_path = _resolve_checkpoint(args)
    model = load_pretrained_model(ckpt_path, configs).to(device)
    model.eval()

    # Minimal hparams for dataloader
    hparams = {"batch_size": 32, "drop_last_test": False}
    data_path = os.path.join(args.data_path, args.dataset)
    test_loader = data_generator(data_path, str(args.source_domain), configs, hparams, "test")

    sigmas = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = []

    with torch.no_grad():
        for sigma in sigmas:
            correct = 0
            total = 0
            for batch_x, batch_y, _ in test_loader:
                batch_np = batch_x.cpu().numpy()
                warped_list = []
                for sample in batch_np:
                    warped = manual_magnitude_warp(sample, sigma=sigma, knots=4)
                    warped_list.append(warped)
                warped_batch = torch.from_numpy(np.stack(warped_list)).float().to(device)
                labels = batch_y.to(device)

                logits = model(warped_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

            acc = 100.0 * correct / total if total > 0 else 0.0
            results.append((sigma, acc))
            print(f"Sigma: {sigma:.2f} | Accuracy: {acc:.2f}%")

    out_path = f"stress_test_results_{args.dataset}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "accuracy"])
        writer.writerows(results)
    print(f"Results saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Label-Preservation Stress Test for magnitude warping.")
    parser.add_argument("--dataset", type=str, required=True, choices=["EEG", "HAR", "FD"])
    parser.add_argument("--source_domain", type=str, required=True, help="Source domain ID (e.g., 0)")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--data_path", type=str, default=r"D:\PyCharm Project\ACCUP + EATA\data\Dataset",
                        help="Root path containing dataset folders")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint. "
                        "If omitted, tries results/pretrain_logs/{dataset}/train_{source}/checkpoint.pt")
    return parser.parse_args()


if __name__ == "__main__":
    run_stress_test(parse_args())
