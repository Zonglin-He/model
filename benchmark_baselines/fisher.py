"""Source-only diagonal Fisher preparation for the benchmark EATA adapter.

The implementation follows the cloned official EATA ``main.py`` protocol:
the model is configured for EATA's trainable normalization/classifier scope,
source inputs are assigned model pseudo-labels, and squared gradients of the
pseudo-label CE loss are averaged over at most the configured source samples.
The source labels are never read by this calibration pass.
"""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


FISHER_CACHE_VERSION = 1


def _configure_fisher_model(model, adapt_keywords):
    fisher_model = deepcopy(model)
    fisher_model.train()
    fisher_model.requires_grad_(False)
    for module in fisher_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    keywords = tuple(adapt_keywords)
    parameters = []
    names = []
    for name, parameter in fisher_model.named_parameters():
        if any(keyword in name for keyword in keywords):
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(name)
    if not parameters:
        raise RuntimeError(
            "EATA source Fisher calibration found no trainable parameters "
            f"for adapt_keywords={keywords!r}"
        )
    return fisher_model, parameters, names


def _source_loader_without_shuffle(source_loader):
    dataset = source_loader.dataset
    batch_size = int(getattr(source_loader, "batch_size", 0) or 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def _primary_source_tensor(batch):
    if isinstance(batch, dict):
        batch = batch.get("data", batch)
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if not torch.is_tensor(batch):
        raise TypeError("EATA Fisher source batches must contain a tensor")
    return batch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fisher_cache_path(cache_dir, dataset, source_seed, source_checkpoint_sha256, samples):
    cache_dir = Path(cache_dir)
    return cache_dir / (
        f"{dataset}_src{int(source_seed)}_"
        f"{source_checkpoint_sha256[:16]}_n{int(samples)}.pt"
    )


def _valid_cache(payload, source_checkpoint_sha256, samples, adapt_keywords):
    if not isinstance(payload, dict) or "fishers" not in payload:
        return False
    if int(payload.get("cache_version", -1)) != FISHER_CACHE_VERSION:
        return False
    if payload.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        return False
    if int(payload.get("requested_samples", -1)) != int(samples):
        return False
    return tuple(payload.get("adapt_keywords", ())) == tuple(adapt_keywords)


def ensure_source_fisher(
    *,
    model,
    source_loader,
    cache_dir,
    dataset,
    source_seed,
    source_checkpoint_sha256,
    samples=2000,
    adapt_keywords=("classifier", "adapter"),
):
    """Load or compute a source Fisher cache and return auditable metadata."""
    samples = max(1, int(samples))
    adapt_keywords = tuple(adapt_keywords)
    cache_path = fisher_cache_path(
        cache_dir,
        dataset,
        source_seed,
        source_checkpoint_sha256,
        samples,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        load_started = time.perf_counter()
        payload = torch.load(cache_path, map_location="cpu")
        cache_valid = _valid_cache(
            payload,
            source_checkpoint_sha256,
            samples,
            adapt_keywords,
        )
        load_seconds = time.perf_counter() - load_started
        if cache_valid:
            cache_bytes = cache_path.stat().st_size
            return {
                "fisher_enabled": True,
                "fisher_cache_path": str(cache_path),
                "fisher_cache_hash": sha256_file(cache_path),
                "fisher_cache_bytes": int(cache_bytes),
                "fisher_cache_hit": True,
                "fisher_compute_seconds": float(
                    payload.get("compute_wall_seconds", 0.0)
                ),
                "fisher_load_seconds": float(load_seconds),
                "fisher_samples": int(payload["sample_count"]),
                "fisher_batches": int(payload["batch_count"]),
                "fisher_source_checkpoint_sha256": source_checkpoint_sha256,
                "fisher_parameter_count": int(
                    sum(int(value[0].numel()) for value in payload["fishers"].values())
                ),
            }

    started = time.perf_counter()
    fisher_model, parameters, names = _configure_fisher_model(
        model, adapt_keywords
    )
    device = next(fisher_model.parameters()).device
    accumulators = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in zip(names, parameters)
    }
    loader = _source_loader_without_shuffle(source_loader)
    sample_count = 0
    batch_count = 0
    fisher_model.train()
    for batch in loader:
        inputs = _primary_source_tensor(batch).float()
        remaining = samples - sample_count
        if remaining <= 0:
            break
        inputs = inputs[:remaining].to(device)
        if inputs.numel() == 0:
            continue
        logits = fisher_model(inputs)
        pseudo_labels = logits.detach().argmax(dim=1)
        loss = F.cross_entropy(logits, pseudo_labels)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
        )
        for name, gradient in zip(names, gradients):
            if gradient is not None:
                accumulators[name].add_(gradient.detach().square())
        sample_count += int(inputs.size(0))
        batch_count += 1
        if sample_count >= samples:
            break
    if sample_count == 0 or batch_count == 0:
        raise RuntimeError("EATA source Fisher calibration received no samples")

    fishers = {
        name: [
            (accumulator / float(batch_count)).detach().cpu(),
            parameter.detach().cpu().clone(),
        ]
        for name, accumulator, parameter in zip(
            names,
            accumulators.values(),
            parameters,
        )
    }
    compute_seconds = time.perf_counter() - started
    payload = {
        "cache_version": FISHER_CACHE_VERSION,
        "method": "EATA",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "dataset": str(dataset),
        "source_seed": int(source_seed),
        "requested_samples": int(samples),
        "sample_count": int(sample_count),
        "batch_count": int(batch_count),
        "adapt_keywords": list(adapt_keywords),
        "compute_wall_seconds": float(compute_seconds),
        "fishers": fishers,
    }
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(cache_path)
    cache_bytes = cache_path.stat().st_size
    return {
        "fisher_enabled": True,
        "fisher_cache_path": str(cache_path),
        "fisher_cache_hash": sha256_file(cache_path),
        "fisher_cache_bytes": int(cache_bytes),
        "fisher_cache_hit": False,
        "fisher_compute_seconds": float(compute_seconds),
        "fisher_load_seconds": 0.0,
        "fisher_samples": int(sample_count),
        "fisher_batches": int(batch_count),
        "fisher_source_checkpoint_sha256": source_checkpoint_sha256,
        "fisher_parameter_count": int(
            sum(int(value[0].numel()) for value in fishers.values())
        ),
    }


__all__ = [
    "FISHER_CACHE_VERSION",
    "ensure_source_fisher",
    "fisher_cache_path",
    "sha256_file",
]
