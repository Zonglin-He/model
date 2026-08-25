import torch

from dataloader.physical_corruption_transforms import (
    PHYSICAL_CORRUPTION_REGISTRY,
    physical_corruption_metadata,
    resolve_severity,
)


def _input():
    return torch.linspace(-2.0, 3.0, 2 * 3 * 128).reshape(2, 3, 128)


def test_every_physical_corruption_has_exact_identity_point():
    inputs = _input()
    for name, transform in PHYSICAL_CORRUPTION_REGISTRY.items():
        torch.testing.assert_close(transform(inputs, "s0"), inputs, rtol=0, atol=0)
        metadata = physical_corruption_metadata(name, "s0")
        assert metadata["normalized_severity"] == 0.0
        assert metadata["severity_name"] == "s0"


def test_transforms_are_reproducible_under_explicit_rng_seed():
    inputs = _input()
    for name in ("blackout", "packet_loss"):
        transform = PHYSICAL_CORRUPTION_REGISTRY[name]
        torch.manual_seed(123)
        first = transform(inputs, "s5")
        torch.manual_seed(123)
        second = transform(inputs, "s5")
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_severity_endpoints_have_expected_physical_ordering():
    inputs = _input()
    assert PHYSICAL_CORRUPTION_REGISTRY["attenuation"](inputs, "s6").abs().mean() < (
        PHYSICAL_CORRUPTION_REGISTRY["attenuation"](inputs, "s1").abs().mean()
    )
    mild_saturation = PHYSICAL_CORRUPTION_REGISTRY["saturation"](inputs, "s1")
    severe_saturation = PHYSICAL_CORRUPTION_REGISTRY["saturation"](inputs, "s6")
    assert severe_saturation.amax() <= mild_saturation.amax()
    mild_freeze = PHYSICAL_CORRUPTION_REGISTRY["signal_freeze"](inputs, "s1")
    severe_freeze = PHYSICAL_CORRUPTION_REGISTRY["signal_freeze"](inputs, "s6")
    assert torch.unique(severe_freeze[..., -64:]).numel() < torch.unique(
        mild_freeze[..., -64:]
    ).numel()


def test_packet_loss_uses_requested_non_overlapping_budget_for_nonzero_data():
    inputs = torch.ones(1, 2, 100)
    transform = PHYSICAL_CORRUPTION_REGISTRY["packet_loss"]
    output = transform(inputs, "s6")
    missing_per_channel = output.eq(0).sum(dim=-1)
    assert missing_per_channel.tolist() == [[30, 30]]


def test_unregistered_severity_fails_closed():
    try:
        resolve_severity("blackout", "moderate")
    except ValueError as exc:
        assert "s0...s6" in str(exc)
    else:
        raise AssertionError("legacy categorical severity was accepted")
