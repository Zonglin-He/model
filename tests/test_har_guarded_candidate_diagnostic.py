from __future__ import annotations

import torch

from scripts.run_har_guarded_candidate_diagnostic import (
    PROTOCOL,
    canonical_conditions,
    deterministic_corruption_mask,
    signal_freeze_at_level,
)


def test_condition_protocol_and_aliases_are_fixed():
    assert PROTOCOL == (
        "har_guarded_candidate_diagnostic_v2_frozen_bn_guard_12_to_16"
    )
    assert canonical_conditions("clean,signal_freeze_moderate,signal_freeze_severe") == (
        "clean",
        "signal_freeze_s3",
        "signal_freeze_s6",
    )
    assert canonical_conditions(corruptions="signal_freeze", severities="s3,s6") == (
        "signal_freeze_s3",
        "signal_freeze_s6",
    )


def test_corruption_mask_is_label_free_and_batch_partition_invariant():
    indices = torch.arange(100)
    full = deterministic_corruption_mask(indices, seed=1, fraction=0.5)
    split = torch.cat(
        [
            deterministic_corruption_mask(indices[:37], seed=1, fraction=0.5),
            deterministic_corruption_mask(indices[37:], seed=1, fraction=0.5),
        ]
    )
    assert torch.equal(full, split)
    assert 35 <= int(full.sum()) <= 65


def test_signal_freeze_uses_exact_physical_levels():
    signal = torch.arange(10, dtype=torch.float32).reshape(1, 1, -1)
    s3 = signal_freeze_at_level(signal, "s3")
    s6 = signal_freeze_at_level(signal, "s6")
    assert torch.equal(s3[..., 8:], signal[..., 7:8].expand_as(s3[..., 8:]))
    assert torch.equal(s6[..., 4:], signal[..., 3:4].expand_as(s6[..., 4:]))
