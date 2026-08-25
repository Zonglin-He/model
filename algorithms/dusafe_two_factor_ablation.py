"""Current DuSafe two-factor component ablation.

The four unique cells form a 2x2 design:

``A``
    Frozen-source confidence admission.
``B``
    The complete SSAW physical hard-view branch.

Frozen source-semantic routing and gathered-batch rechecking are not factors;
both were removed from the current production method before this study.
"""

from __future__ import annotations

import torch

from algorithms.dusafe_replacement_ablation import _ReplacementRunner


class _AcceptAllConfidenceMixin:
    """Remove A by admitting every raw prediction."""

    def _confidence_admission_mask(
        self,
        raw_top1_nll: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        del pseudo_labels
        return torch.ones_like(raw_top1_nll, dtype=torch.bool)


class _NoSSAWMixin:
    """Remove B before the adapter constructs its execution path."""

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["enable_ssaw"] = False
        effective["enable_source_semantic_router"] = False
        super().__init__(configs, effective, model, optimizer)
        if self.enable_ssaw:
            raise RuntimeError("two-factor no-SSAW cell constructed SSAW")


class TwoFactorBaseline(
    _AcceptAllConfidenceMixin, _NoSSAWMixin, _ReplacementRunner
):
    """F00: raw pseudo-label adaptation without A or B."""

    runner_name = "F00_baseline"


class TwoFactorConfidenceOnly(_NoSSAWMixin, _ReplacementRunner):
    """F10: Baseline + A."""

    runner_name = "F10_baseline_plus_a_confidence"


class TwoFactorSSAWOnly(_AcceptAllConfidenceMixin, _ReplacementRunner):
    """F01: Baseline + B."""

    runner_name = "F01_baseline_plus_b_ssaw"


class TwoFactorFull(_ReplacementRunner):
    """F11: Baseline + A + B, i.e. the current Full method."""

    runner_name = "F11_full"


TWO_FACTOR_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        TwoFactorBaseline,
        TwoFactorConfidenceOnly,
        TwoFactorSSAWOnly,
        TwoFactorFull,
    )
}


TWO_FACTOR_BITS = {
    "F00_baseline": (0, 0),
    "F10_baseline_plus_a_confidence": (1, 0),
    "F01_baseline_plus_b_ssaw": (0, 1),
    "F11_full": (1, 1),
}


TWO_FACTOR_COMPONENT = {
    "F00_baseline": "neither_confidence_nor_ssaw",
    "F10_baseline_plus_a_confidence": "confidence_only",
    "F01_baseline_plus_b_ssaw": "ssaw_only",
    "F11_full": "confidence_plus_ssaw",
}


def get_two_factor_runner(name: str):
    try:
        return TWO_FACTOR_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown two-factor runner: {name}") from exc


__all__ = [
    "TWO_FACTOR_BITS",
    "TWO_FACTOR_COMPONENT",
    "TWO_FACTOR_RUNNERS",
    "TwoFactorBaseline",
    "TwoFactorConfidenceOnly",
    "TwoFactorSSAWOnly",
    "TwoFactorFull",
    "get_two_factor_runner",
]
