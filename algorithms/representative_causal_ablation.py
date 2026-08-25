"""Experiment-only DuSafe controls for the representative causal panels.

The classes in this module are deliberately kept outside the production
registry.  They share the current production adapter's parameter scope,
candidate pool, confidence metadata, optimizer, and positive SSAW weight;
only the causal-panel decision under test is changed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algorithms.dusafe import SSAWPhysicalView, _extract_features
from algorithms.dusafe_replacement_ablation import (
    _ReplacementRunner,
    ReplacementSplineHardView,
)
from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
    ConfidenceRawOnly,
)


class RepresentativeAcceptAllRaw(ConfidenceRawOnly):
    """Panel A control: raw adaptation with confidence admission removed."""

    runner_name = "accept_all_raw"

    def _confidence_admission_mask(self, raw_top1_nll, pseudo_labels):
        del pseudo_labels
        return torch.ones_like(raw_top1_nll, dtype=torch.bool)


class RepresentativeConfidenceRaw(ConfidenceRawOnly):
    """Panel A/B baseline: the current confidence-only raw update."""

    runner_name = "confidence_only"


class RepresentativeMatchedRawDuplicate(ConfidenceAdmittedSplineResidualKL):
    """Panel B control with the hard-view mask but a raw duplicate loss.

    The candidate pool, selected-view mask, confidence mask, and denominator
    are inherited from the current production path.  Only the auxiliary loss
    is replaced by CE on the raw input, which isolates sample reweighting from
    transformed-view information.
    """

    runner_name = "matched_raw_duplicate"

    def _physical_view_consistency_loss(
        self,
        model,
        raw_inputs,
        raw_target_logits,
        view_selection_mask,
        raw_admission_mask,
        sample_weights,
        view_logits_by_view=None,
    ):
        del sample_weights, view_logits_by_view
        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        # Keep BN buffers fixed while creating the matched raw control.  The
        # production raw-forward has already established the batch state; a
        # second forward is intentionally diagnostic and does not alter it.
        with SSAWPhysicalView._preserved_bn_buffers(model):
            raw_features = _extract_features(model, raw_inputs)
            raw_logits = model.classifier(raw_features)
        per_sample = F.cross_entropy(
            raw_logits, pseudo_labels, reduction="none"
        )
        denominator = raw_admission_mask.float().sum().clamp_min(1.0)
        if not view_selection_mask.any():
            return raw_inputs.sum() * 0.0
        return per_sample[view_selection_mask].sum() / denominator


class RepresentativeRandomEligibleSpline(_ReplacementRunner):
    """Core control: a random label-preserving physical view.

    This is the simple random-view comparator used by the formal core
    ablation.  It shares the physical candidate family and auxiliary objective
    with Full but does not use margin-aware hard selection.  A stricter
    participation-matched random control is evidence-only and is defined in
    ``scripts/run_representative_causal_ablation.py``.
    """

    runner_name = "random_eligible_spline"
    spline_view_class = ReplacementSplineHardView
    spline_selection_mode = "random_label_preserving_candidate"


class RepresentativeHardSSAW(ConfidenceAdmittedSplineResidualKL):
    """Panel B treatment: the current production hard-view SSAW branch."""

    runner_name = "hard_ssaw"

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["record_ssaw_candidate_hash"] = True
        super().__init__(configs, effective, model, optimizer)


REPRESENTATIVE_VARIANTS = {
    cls.runner_name: cls
    for cls in (
        RepresentativeAcceptAllRaw,
        RepresentativeConfidenceRaw,
        RepresentativeMatchedRawDuplicate,
        RepresentativeRandomEligibleSpline,
        RepresentativeHardSSAW,
    )
}

PANEL_A_VARIANTS = ("accept_all_raw", "confidence_only")
PANEL_B_VARIANTS = (
    "confidence_only",
    "matched_raw_duplicate",
    "random_eligible_spline",
    "hard_ssaw",
)


def get_representative_variant(name: str):
    try:
        return REPRESENTATIVE_VARIANTS[str(name).strip()]
    except KeyError as exc:
        raise ValueError(
            f"unknown representative causal variant: {name!r}; "
            f"expected one of {tuple(REPRESENTATIVE_VARIANTS)}"
        ) from exc


__all__ = [
    "PANEL_A_VARIANTS",
    "PANEL_B_VARIANTS",
    "REPRESENTATIVE_VARIANTS",
    "RepresentativeAcceptAllRaw",
    "RepresentativeConfidenceRaw",
    "RepresentativeHardSSAW",
    "RepresentativeMatchedRawDuplicate",
    "RepresentativeRandomEligibleSpline",
    "get_representative_variant",
]
