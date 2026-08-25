import hashlib

import numpy as np
import torch

from dataloader.har_cross_dataset_corruptions import (
    CORRUPTED_INDICES,
    TARGET_SAMPLES,
    blackout,
    corruption_mask_sha256,
    exact_index_stable_mask_fn,
    localized_attenuation,
    physical_corruption_metadata,
    signal_freeze,
    smooth_gain_drift,
)


def test_registered_mask_is_exact_and_order_invariant():
    assert len(CORRUPTED_INDICES) == 55
    make_mask = exact_index_stable_mask_fn(0.5, 314159)
    order = torch.randperm(TARGET_SAMPLES, generator=torch.Generator().manual_seed(9))
    observed = make_mask(None, torch.zeros(TARGET_SAMPLES), order, 0, 1)
    assert set(order[observed].tolist()) == set(CORRUPTED_INDICES)
    mask = np.zeros(TARGET_SAMPLES, dtype=np.uint8)
    mask[list(CORRUPTED_INDICES)] = 1
    assert hashlib.sha256(mask.tobytes()).hexdigest() == corruption_mask_sha256()


def test_blackout_and_freeze_use_registered_discrete_durations():
    inputs = torch.ones(2, 3, 128)
    indices = [0, 1]
    b3 = blackout(inputs, "s3", indices)
    b6 = blackout(inputs, "s6", indices)
    for row in range(2):
        z3 = b3[row, 0].eq(0)
        z6 = b6[row, 0].eq(0)
        assert int(z3.sum()) == 13
        assert int(z6.sum()) == 38
        assert not (z3 & ~z6).any()
    ramp = torch.arange(128, dtype=torch.float32).view(1, 1, -1)
    f3 = signal_freeze(ramp, "s3", [0])
    f6 = signal_freeze(ramp, "s6", [0])
    assert f3[..., 102:].eq(101).all()
    assert f6[..., 51:].eq(50).all()


def test_gain_drift_is_monotone_positive_and_independent_family():
    inputs = torch.ones(4, 1, 128, dtype=torch.float64)
    indices = [0, 1, 2, 3]
    for severity, ratio in (("s3", 1.35), ("s6", 2.0)):
        gain = smooth_gain_drift(inputs, severity, indices)[:, 0]
        assert gain.gt(0).all()
        assert torch.equal(gain[:, 0], torch.ones(4, dtype=torch.float64))
        allowed = torch.tensor([ratio, 1.0 / ratio], dtype=torch.float64)
        assert all(torch.isclose(value, allowed, atol=1e-12).any() for value in gain[:, -1])
        assert all(row.ge(0).all() or row.le(0).all() for row in gain.diff(dim=1))
        assert float(gain.log().diff(dim=1).diff(dim=1).abs().max()) < 1e-12
    metadata = physical_corruption_metadata("smooth_gain_drift", "s6")
    assert metadata["physical_parameters"]["ssaw_sobol_or_spline_reused"] is False


def test_local_attenuation_is_positive_smooth_and_nested():
    inputs = torch.ones(3, 1, 128, dtype=torch.float64)
    indices = [4, 5, 6]
    s3 = localized_attenuation(inputs, "s3", indices)[:, 0]
    s6 = localized_attenuation(inputs, "s6", indices)[:, 0]
    for row in range(3):
        support3 = s3[row].lt(1)
        support6 = s6[row].lt(1)
        assert int(support3.sum()) == 11
        assert int(support6.sum()) == 36
        assert not (support3 & ~support6).any()
        assert s3[row].gt(0).all() and s6[row].gt(0).all()
        assert torch.isclose(s3[row].min(), torch.tensor(0.65, dtype=torch.float64))
        assert torch.isclose(s6[row].min(), torch.tensor(0.20, dtype=torch.float64))
        assert s6[row].le(s3[row] + 1e-12).all()
    m3 = physical_corruption_metadata("localized_attenuation", "s3")
    m6 = physical_corruption_metadata("localized_attenuation", "s6")
    assert m3["physical_parameters"]["ramp_plateau_ramp_samples"] == [3, 7, 3]
    assert m6["physical_parameters"]["ramp_plateau_ramp_samples"] == [9, 20, 9]
