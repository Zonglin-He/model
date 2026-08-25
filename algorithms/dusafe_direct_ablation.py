"""Direct coarse ablations requested for the final DuSafe component study.

Each current runner changes exactly one remaining component while retaining
the selected flow hyperparameters, optimizer, trainable parameter scope,
candidate budget, and update schedule. Semantic routing is no longer a
production component; its old class remains importable only for historical
artifact compatibility.
"""

from __future__ import annotations

import torch

from algorithms.dusafe_replacement_ablation import (
    GenericJitterSSAWReplacement,
)
from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
)


class DirectFull(ConfidenceAdmittedSplineResidualKL):
    runner_name = "D0_full"


class NoConfidenceAdmission(ConfidenceAdmittedSplineResidualKL):
    """Remove confidence admission by accepting every raw sample."""

    runner_name = "D1_no_confidence"

    def _confidence_admission_mask(
        self,
        raw_top1_nll: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        del pseudo_labels
        return torch.ones_like(raw_top1_nll, dtype=torch.bool)


class NoSemanticRouter(ConfidenceAdmittedSplineResidualKL):
    """Archived no-op kept only for loading historical experiment code."""

    runner_name = "D2_no_semantic_router"

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["enable_source_semantic_router"] = False
        super().__init__(configs, effective, model, optimizer)

    def _ssaw_training_router_mask(
        self,
        confidence_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        del confidence_mask, source_semantic_mask
        raw_margin = torch.as_tensor(
            self.ssaw.last_metadata["raw_pseudo_margin"],
            device=pseudo_labels.device,
            dtype=torch.float32,
        )
        selected_margin = torch.as_tensor(
            self.ssaw.last_metadata["selected_margin"],
            device=pseudo_labels.device,
            dtype=torch.float32,
        )
        selected_valid = ~torch.as_tensor(
            self.ssaw.last_metadata["ssaw_label_flip"],
            device=pseudo_labels.device,
            dtype=torch.bool,
        )
        return (
            selected_valid
            & selected_margin.gt(0.0)
            & selected_margin.lt(raw_margin)
        )


class GaussianJitterConsistency(GenericJitterSSAWReplacement):
    """Replace the complete spline SSAW branch with 24 Gaussian candidates."""

    runner_name = "D3_gaussian_jitter_consistency"

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["enable_source_semantic_router"] = False
        super().__init__(configs, effective, model, optimizer)


DIRECT_ABLATION_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        DirectFull,
        NoConfidenceAdmission,
        GaussianJitterConsistency,
    )
}


ABLATED_COMPONENT = {
    "D0_full": "none",
    "D1_no_confidence": "confidence_admission",
    "D3_gaussian_jitter_consistency": "spline_ssaw_module",
}


def get_direct_ablation_runner(name: str):
    try:
        return DIRECT_ABLATION_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown direct ablation runner: {name}") from exc


__all__ = [
    "ABLATED_COMPONENT",
    "DIRECT_ABLATION_RUNNERS",
    "get_direct_ablation_runner",
]
