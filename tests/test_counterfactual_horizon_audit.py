import copy

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.counterfactual_horizon_common import (
    DEFAULT_HORIZONS,
    batchnorm_state,
    batchnorm_states_equal,
    classify_impact,
    clone_branch_adapters,
    run_horizon_audit,
    snapshot_state,
    state_hash,
    states_equal,
)
from scripts.run_full_no_ssaw_horizon_audit import parse_horizons


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 2),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.classifier(self.feature_extractor(inputs))


class TinyAdapter:
    def __init__(self, model=None, enable_ssaw=True):
        self.model = TinyModel() if model is None else model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02, momentum=0.9)
        self.enable_ssaw = bool(enable_ssaw)
        self.seen_label_keys = []

    def forward_and_adapt(self, batch_data, model, optimizer, _indices=None):
        # The production common runner must never put true labels in this dict.
        self.seen_label_keys.append(tuple(sorted(batch_data)))
        assert "labels" not in batch_data
        inputs = batch_data["data"]
        logits = model(inputs)
        pseudo = logits.detach().argmax(dim=1)
        loss = F.cross_entropy(logits, pseudo)
        if self.enable_ssaw:
            # A deterministic auxiliary branch gives Full a distinct update
            # while preserving the source-free/pseudo-label nature of the test.
            loss = loss + 0.03 * logits.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {"committed": True}


def _stream(batch_count=8):
    batches = []
    for index in range(batch_count):
        data = torch.tensor(
            [[1.0 + index * 0.03, 0.2], [-1.0 - index * 0.02, -0.3],
             [0.8, 0.4]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        batches.append((data, labels, torch.arange(3) + index * 3))
    return batches


def test_horizon_audit_forks_exact_states_and_excludes_labels_from_updates():
    torch.manual_seed(7)
    full = TinyAdapter()
    branches = clone_branch_adapters(full)
    initial_hash = state_hash(full)
    frame, audit = run_horizon_audit(
        branches["full"],
        _stream(),
        horizons=DEFAULT_HORIZONS,
        no_ssaw_adapter=branches["no_ssaw"],
        no_update_adapter=branches["no_update"],
        num_classes=2,
        metadata={"flow": "synthetic->synthetic", "target_labels_used": False},
    )

    # Starts t=0..n-h-1: 7 + 5 + 3 windows.
    assert len(frame) == 15
    assert set(frame["horizon"]) == set(DEFAULT_HORIZONS)
    assert audit["state_equivalence_passed"] is True
    assert audit["state_equivalence_failures"] == 0
    assert frame["state_branch_state_objects_independent"].all()
    assert frame.filter(like="state_no_update_untouched")["state_no_update_untouched"].all()
    assert frame["target_labels_used"].eq(False).all()
    assert frame["target_labels_used_for_updates"].eq(False).all()
    assert frame["full_vs_no_ssaw_nll_impact"].isin(
        ["beneficial", "harmful", "tied"]
    ).all()
    assert set(audit["summary"][0]) >= {
        "full_vs_no_ssaw_nll_beneficial_fraction",
        "full_vs_no_ssaw_nll_harmful_fraction",
        "full_vs_no_ssaw_nll_tied_fraction",
    }
    # The canonical Full history advances, so its state need not equal the
    # initial state; the branch checks, not an accidental final reset, are the
    # protocol invariant.
    assert state_hash(branches["full"]) != initial_hash
    assert all(keys == ("data",) for keys in branches["full"].seen_label_keys)
    assert all(keys == ("data",) for keys in branches["no_ssaw"].seen_label_keys)


def test_low_memory_snapshot_mode_matches_three_branch_reference():
    torch.manual_seed(19)
    reference = TinyAdapter()
    reference_branches = clone_branch_adapters(reference)
    reference_frame, reference_audit = run_horizon_audit(
        reference_branches["full"],
        _stream(),
        horizons=(1, 3),
        no_ssaw_adapter=reference_branches["no_ssaw"],
        no_update_adapter=reference_branches["no_update"],
        num_classes=2,
    )

    torch.manual_seed(19)
    low_memory = TinyAdapter()
    low_frame, low_audit = run_horizon_audit(
        low_memory,
        _stream(),
        horizons=(1, 3),
        num_classes=2,
        low_memory=True,
        snapshot_cpu=True,
    )

    metric_columns = [
        "batch_index",
        "horizon",
        "no_update_macro_f1",
        "no_ssaw_macro_f1",
        "full_macro_f1",
        "no_update_true_label_nll",
        "no_ssaw_true_label_nll",
        "full_true_label_nll",
    ]
    pd.testing.assert_frame_equal(
        reference_frame[metric_columns].reset_index(drop=True),
        low_frame[metric_columns].reset_index(drop=True),
    )
    assert low_audit["low_memory"] is True
    assert low_audit["state_snapshot_device"] == "cpu"
    assert low_audit["state_equivalence_passed"] is True
    assert low_audit["rng_equivalence_passed"] is True


def test_state_snapshot_includes_optimizer_and_batchnorm_buffers():
    adapter = TinyAdapter()
    before = snapshot_state(adapter)
    before_bn = batchnorm_state(adapter)
    adapter.forward_and_adapt({"data": torch.randn(4, 2)}, adapter.model, adapter.optimizer)
    after = snapshot_state(adapter)
    assert not states_equal(before, after)
    adapter.model.load_state_dict(before.model_state)
    adapter.optimizer.load_state_dict(copy.deepcopy(before.optimizer_state))
    assert states_equal(before, snapshot_state(adapter))
    assert batchnorm_states_equal(before_bn, batchnorm_state(adapter))


def test_impact_labels_and_horizon_parser_are_bounded():
    assert classify_impact(0.1) == "beneficial"
    assert classify_impact(-0.1) == "harmful"
    assert classify_impact(1e-10, tolerance=1e-9) == "tied"
    assert parse_horizons("5,1,3,3") == (1, 3, 5)
    with pytest.raises(ValueError, match="subset"):
        parse_horizons("1,2,5")
