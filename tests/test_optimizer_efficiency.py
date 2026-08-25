"""Numerical contracts for the low-synchronization gradient-clip wrapper."""

from __future__ import annotations

import math

import torch

from optim.optimizer import GradClipWrapper


def _one_update(record_stats: bool):
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    wrapped = GradClipWrapper(
        torch.optim.SGD([parameter], lr=0.2),
        max_norm=0.3,
        clip_value=0.25,
        record_stats=record_stats,
    )
    loss = (parameter.square() * torch.tensor([3.0, 0.5])).sum()
    wrapped.zero_grad(set_to_none=True)
    loss.backward()
    wrapped.step()
    return parameter.detach().clone(), wrapped


def test_disabling_optimizer_diagnostics_does_not_change_clipped_update():
    without_stats, quiet = _one_update(False)
    with_stats, diagnostic = _one_update(True)

    torch.testing.assert_close(without_stats, with_stats, rtol=0.0, atol=0.0)
    assert math.isnan(quiet.last_pre_clip_grad_norm)
    assert math.isnan(quiet.last_post_clip_grad_norm)
    assert math.isfinite(diagnostic.last_pre_clip_grad_norm)
    assert math.isfinite(diagnostic.last_post_clip_grad_norm)
    assert diagnostic.last_post_clip_grad_norm <= 0.3 + 1e-6


def test_prepared_norm_path_matches_historical_step_exactly():
    initial = torch.tensor([2.0, -1.0])

    def run(prepare: bool):
        parameter = torch.nn.Parameter(initial.clone())
        wrapped = GradClipWrapper(
            torch.optim.Adam([parameter], lr=0.02),
            max_norm=0.3,
            record_stats=False,
        )
        loss = (parameter.square() * torch.tensor([3.0, 0.5])).sum()
        wrapped.zero_grad(set_to_none=True)
        loss.backward()
        norm = wrapped.prepare_gradients_for_step() if prepare else None
        wrapped.step()
        return parameter.detach().clone(), wrapped.state_dict(), norm

    historical_parameter, historical_state, _ = run(False)
    prepared_parameter, prepared_state, prepared_norm = run(True)

    torch.testing.assert_close(
        historical_parameter, prepared_parameter, rtol=0.0, atol=0.0
    )
    assert prepared_norm is not None and torch.isfinite(prepared_norm)
    historical_slots = historical_state["state"][0]
    prepared_slots = prepared_state["state"][0]
    assert historical_slots.keys() == prepared_slots.keys()
    for key in historical_slots:
        torch.testing.assert_close(
            historical_slots[key], prepared_slots[key], rtol=0.0, atol=0.0
        )


def test_value_clip_path_is_not_prepared_before_nonfinite_guard():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    wrapped = GradClipWrapper(
        torch.optim.SGD([parameter], lr=0.1),
        max_norm=0.3,
        clip_value=0.25,
    )
    parameter.grad = torch.tensor([float("inf")])

    assert wrapped.prepare_gradients_for_step() is None
    assert torch.isinf(parameter.grad).all()
