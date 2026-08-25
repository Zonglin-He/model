"""Reusable snapshots and comparisons for CPU DuSafe equivalence tests.

The contract deliberately compares two independently constructed instances
from the same input/state.  It does not encode a particular random draw or a
particular model output, so an optimized implementation can replace one side
of the comparison without changing the test's oracle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def snapshot_value(value: Any) -> Any:
    """Detach tensors and recursively copy supported diagnostic containers."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: snapshot_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot_value(child) for child in value)
    if isinstance(value, list):
        return [snapshot_value(child) for child in value]
    return value


def sobol_state(view: Any) -> dict[str, int | None]:
    """Capture stable seed/index counters without serializing Sobol internals."""

    engine = getattr(view, "_sobol", None)
    generated = getattr(engine, "num_generated", None)
    return {
        "sobol_seed": int(view.sobol_seed),
        "physical_call_index": int(view._physical_call_index),
        "sobol_num_generated": (
            None if generated is None else int(generated)
        ),
    }


def snapshot_ssaw(view: Any) -> dict[str, Any]:
    """Capture SSAW outputs, caches, metadata, and reproducibility counters."""

    tensor_fields = (
        "last_warp_curve",
        "last_view_inputs",
        "last_stress_logits",
        "last_stress_features",
        "last_reference_logits",
        "last_reference_features",
        "_cached_view_inputs",
        "_cached_warp_curve",
        "_cached_controls",
        "_cached_rotation_matrices",
        "_last_rotation_matrix",
    )
    return {
        "config": {
            "num_control_points": int(view.num_control_points),
            "sigma": float(view.sigma),
            "sobol_seed": int(view.sobol_seed),
            "strength": float(view.strength),
            "temporal_mode": str(view.temporal_mode),
            "antithetic": bool(view.antithetic),
            "antithetic_pairs": int(view.antithetic_pairs),
        },
        "sobol": sobol_state(view),
        "tensors": {
            name: snapshot_value(getattr(view, name)) for name in tensor_fields
        },
        "last_metadata": snapshot_value(view.last_metadata),
    }


def snapshot_dusafe(adapter: Any) -> dict[str, Any]:
    """Capture the state required to compare one DuSafe update."""

    optimizer = getattr(adapter, "optimizer", None)
    return {
        # adapter.state_dict includes model and frozen semantic extractor.
        "adapter_state": snapshot_value(adapter.state_dict()),
        "model_state": snapshot_value(adapter.model.state_dict()),
        "optimizer_state": snapshot_value(
            None if optimizer is None else optimizer.state_dict()
        ),
        "ssaw": snapshot_ssaw(adapter.ssaw),
        "gate_log": snapshot_value(adapter._last_gate_log),
        "batch_log": snapshot_value(adapter._last_batch_log),
        "runtime_state": snapshot_value(
            {
                "source_normalization_mean": adapter.source_normalization_mean,
                "source_normalization_std": adapter.source_normalization_std,
                "source_semantic_prototypes": adapter.source_semantic_prototypes,
                "confidence_nll_threshold": adapter.confidence_nll_threshold,
                "source_semantic_reference_ready": (
                    adapter.source_semantic_reference_ready
                ),
                "source_confidence_reference_ready": (
                    adapter.source_confidence_reference_ready
                ),
            }
        ),
        "sobol_config": {
            "base_seed": int(adapter.ssaw_base_sobol_seed),
            "effective_seed": int(adapter.ssaw_effective_sobol_seed),
            "test_time_seed": adapter.test_time_seed,
        },
    }


def assert_equivalent(
    left: Any,
    right: Any,
    *,
    path: str = "root",
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Recursively compare snapshots with exact masks/structure checks."""

    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        if left.dtype == torch.bool or not (left.is_floating_point() or left.is_complex()):
            assert torch.equal(left, right), path
        else:
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=path)
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping) and isinstance(right, Mapping), path
        assert set(left) == set(right), path
        for key in left:
            assert_equivalent(
                left[key], right[key], path=f"{path}[{key!r}]", rtol=rtol, atol=atol
            )
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_equivalent(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(left, float) or isinstance(right, float):
        assert isinstance(left, (int, float)) and isinstance(right, (int, float)), path
        assert math.isclose(
            float(left), float(right), rel_tol=rtol, abs_tol=atol
        ), path
        return
    assert left == right, path


__all__ = [
    "assert_equivalent",
    "snapshot_dusafe",
    "snapshot_ssaw",
    "snapshot_value",
    "sobol_state",
]
