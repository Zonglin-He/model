import hashlib

import numpy as np
import torch

from dataloader.eeg_cross_dataset_corruptions import (
    CORRUPTED_INDICES,
    TARGET_SAMPLES,
    blackout,
    corruption_mask_sha256,
    exact_index_stable_mask_fn,
    localized_attenuation,
    signal_freeze,
    smooth_gain_drift,
)


def test_exact_mask_is_registered_and_order_invariant():
    assert len(CORRUPTED_INDICES) == 283
    assert corruption_mask_sha256() == (
        "4b973ad58d1d7b965eeade71966608bbb9404af90c082271e75f34b605698316"
    )
    make_mask = exact_index_stable_mask_fn(0.5, 314159)
    order = torch.randperm(TARGET_SAMPLES, generator=torch.Generator().manual_seed(7))
    observed = make_mask(None, torch.zeros(TARGET_SAMPLES), order, 0, 1)
    selected = set(order[observed].tolist())
    assert selected == set(CORRUPTED_INDICES)


def test_blackout_and_freeze_have_exact_registered_support():
    inputs = torch.ones(2, 2, 3000)
    indices = [0, 1]
    black_s3 = blackout(inputs, "s3", indices)
    black_s6 = blackout(inputs, "s6", indices)
    for row in range(2):
        support_s3 = black_s3[row, 0].eq(0)
        support_s6 = black_s6[row, 0].eq(0)
        assert int(support_s3.sum()) == 300
        assert int(support_s6.sum()) == 900
        assert not (support_s3 & ~support_s6).any()
    ramp = torch.arange(3000, dtype=torch.float32).view(1, 1, -1)
    freeze_s3 = signal_freeze(ramp, "s3", [0])
    freeze_s6 = signal_freeze(ramp, "s6", [0])
    assert torch.equal(freeze_s3[..., :2400], ramp[..., :2400])
    assert freeze_s3[..., 2400:].eq(2399).all()
    assert torch.equal(freeze_s6[..., :1200], ramp[..., :1200])
    assert freeze_s6[..., 1200:].eq(1199).all()


def test_gain_drift_is_positive_monotone_and_not_a_spline():
    inputs = torch.ones(4, 1, 3000, dtype=torch.float64)
    indices = [0, 1, 2, 3]
    for severity, ratio in (("s3", 1.35), ("s6", 2.0)):
        gain = smooth_gain_drift(inputs, severity, indices)[:, 0]
        assert torch.equal(gain[:, 0], torch.ones(4, dtype=torch.float64))
        endpoints = gain[:, -1]
        allowed = torch.tensor([ratio, 1.0 / ratio], dtype=torch.float64)
        assert all(torch.isclose(value, allowed, atol=1e-12).any() for value in endpoints)
        assert gain.gt(0).all()
        differences = gain.diff(dim=1)
        assert all(row.ge(0).all() or row.le(0).all() for row in differences)
        log_second = gain.log().diff(dim=1).diff(dim=1)
        assert float(log_second.abs().max()) < 1e-12
    signs_s3 = smooth_gain_drift(inputs, "s3", indices)[:, 0, -1].gt(1)
    signs_s6 = smooth_gain_drift(inputs, "s6", indices)[:, 0, -1].gt(1)
    assert torch.equal(signs_s3, signs_s6)


def test_localized_attenuation_is_smooth_positive_and_nested():
    inputs = torch.ones(3, 1, 3000, dtype=torch.float64)
    indices = [4, 5, 6]
    s3 = localized_attenuation(inputs, "s3", indices)[:, 0]
    s6 = localized_attenuation(inputs, "s6", indices)[:, 0]
    for row in range(3):
        support_s3 = s3[row].lt(1)
        support_s6 = s6[row].lt(1)
        assert int(support_s3.sum()) == 298
        assert int(support_s6.sum()) == 898
        assert not (support_s3 & ~support_s6).any()
        assert torch.isclose(s3[row].min(), torch.tensor(0.65, dtype=torch.float64))
        assert torch.isclose(s6[row].min(), torch.tensor(0.20, dtype=torch.float64))
        assert s3[row].gt(0).all() and s6[row].gt(0).all()
        assert s6[row].le(s3[row] + 1e-12).all()
        assert float(s3[row].diff().abs().max()) < 0.02
        assert float(s6[row].diff().abs().max()) < 0.02


def test_mask_fingerprint_is_byte_level_auditable():
    mask = np.zeros(TARGET_SAMPLES, dtype=np.uint8)
    mask[list(CORRUPTED_INDICES)] = 1
    assert hashlib.sha256(mask.tobytes()).hexdigest() == corruption_mask_sha256()
