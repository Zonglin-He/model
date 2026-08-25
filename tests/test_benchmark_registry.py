from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algorithms.get_tta_class import get_algorithm_class as get_production_class
from algorithms.dusafe import DuSafe
from benchmark_baselines.registry import get_algorithm_class, list_methods, provenance
from benchmark_baselines.fisher import ensure_source_fisher
from configs.benchmark_baselines import get_benchmark_hparams_class
from optim.optimizer import build_optimizer


class ToyFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, inputs):
        sequence = torch.relu(self.bn(self.conv(inputs)))
        return sequence.mean(dim=-1), sequence


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.logits(inputs)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ToyFeatureExtractor()
        self.classifier = ToyClassifier()

    def forward(self, inputs):
        features, _ = self.feature_extractor(inputs)
        return self.classifier(features)


TOY_CONFIG = SimpleNamespace(num_classes=3)


def test_benchmark_registry_is_explicit_and_production_stays_du_safe():
    assert set(list_methods()) == {
        "Tent",
        "EATA",
        "SAR",
        "ACCUPOfficial",
        "CoTTA",
        "SoTTA",
        "RoTTA",
        "COME",
        "NOTE",
        "DuSafe",
    }
    with __import__("pytest").raises(NotImplementedError):
        get_production_class("Tent")
    assert set(provenance()) == set(list_methods())
    assert get_algorithm_class("DuSafe") is DuSafe
    assert provenance("DuSafe")["port_class"] == "production_reference"
    assert provenance("DuSafe")["selection_semantics"] == (
        "fixed_source_confidence_admission_with_unified_spline_"
        "hard_view_residual_consistency"
    )


def test_all_benchmark_adapters_accept_current_forward_contract(tmp_path):
    benchmark_hparams = get_benchmark_hparams_class("EEG")()
    inputs = torch.randn(4, 2, 16)
    for method in list_methods():
        if method == "DuSafe":
            # DuSafe is a production reference alias, not a baseline adapter;
            # its protocol is covered by the production tests.
            continue
        hparams = dict(benchmark_hparams.alg_hparams[method])
        # Keep this unit test single-batch and CPU-bound while preserving each
        # method's update rule and optimizer construction.
        hparams.update(
            {
                "steps": 1,
                "cotta_num_augmentations": 1,
                "cotta_train_all": False,
                "sotta_memory_size": 8,
                "sotta_update_frequency": 4,
                "rotta_memory_size": 8,
                "rotta_update_frequency": 4,
                "note_memory_size": 8,
                "note_update_frequency": 4,
            }
        )
        model = ToyModel()
        if method == "EATA":
            source_labels = torch.randint(0, 3, (inputs.size(0),))
            source_loader = DataLoader(
                TensorDataset(inputs, source_labels, torch.arange(inputs.size(0))),
                batch_size=2,
                shuffle=False,
            )
            fisher = ensure_source_fisher(
                model=model,
                source_loader=source_loader,
                cache_dir=tmp_path,
                dataset="TOY",
                source_seed=1,
                source_checkpoint_sha256="toy-checkpoint",
                samples=inputs.size(0),
                adapt_keywords=hparams["adapt_keywords"],
            )
            hparams["fisher_enabled"] = True
            hparams["fisher_path"] = fisher["fisher_cache_path"]
        adapter = get_algorithm_class(method)(
            TOY_CONFIG,
            hparams,
            model,
            build_optimizer(hparams),
        )
        outputs = adapter({"data": inputs})
        assert torch.is_tensor(outputs)
        assert tuple(outputs.shape) == (4, 3)
        selected = adapter._last_gate_log["selected_mask"]
        assert tuple(selected.shape) == (4,)
        assert selected.dtype == torch.bool
        if method in {"Tent", "ACCUPOfficial", "CoTTA", "COME", "NOTE"}:
            assert selected.all(), f"{method} must expose its all-update mask"
        if method == "SoTTA":
            assert torch.equal(
                selected, adapter._last_gate_log["confidence_mask"]
            )
        if method == "NOTE":
            assert torch.equal(
                selected, adapter._last_gate_log["memory_admission_mask"]
            )
        if method == "EATA":
            assert adapter.fisher_enabled
            assert adapter.fishers


def test_eata_source_fisher_cache_is_real_and_reusable(tmp_path):
    torch.manual_seed(3)
    model = ToyModel()
    source_loader = DataLoader(
        TensorDataset(torch.randn(6, 2, 16), torch.zeros(6, dtype=torch.long)),
        batch_size=3,
        shuffle=False,
    )
    first = ensure_source_fisher(
        model=model,
        source_loader=source_loader,
        cache_dir=tmp_path,
        dataset="EEG",
        source_seed=1,
        source_checkpoint_sha256="abc123",
        samples=5,
        adapt_keywords=("classifier",),
    )
    assert first["fisher_enabled"]
    assert not first["fisher_cache_hit"]
    assert first["fisher_samples"] == 5
    assert first["fisher_batches"] == 2
    assert first["fisher_cache_bytes"] > 0
    second = ensure_source_fisher(
        model=model,
        source_loader=source_loader,
        cache_dir=tmp_path,
        dataset="EEG",
        source_seed=1,
        source_checkpoint_sha256="abc123",
        samples=5,
        adapt_keywords=("classifier",),
    )
    assert second["fisher_cache_hit"]
    assert second["fisher_cache_hash"] == first["fisher_cache_hash"]
    assert second["fisher_load_seconds"] > 0.0


def test_eata_fisher_regularizer_requires_explicit_diagonal():
    model = ToyModel()
    hparams = dict(get_benchmark_hparams_class("EEG")().alg_hparams["EATA"])
    hparams.update({"steps": 1, "fisher_enabled": True})
    hparams["fisher_state"] = {
        name: [torch.ones_like(parameter), parameter.detach().clone()]
        for name, parameter in model.named_parameters()
        if "classifier" in name
    }
    adapter = get_algorithm_class("EATA")(
        TOY_CONFIG,
        hparams,
        model,
        build_optimizer(hparams),
    )
    for parameter in adapter.model.parameters():
        if parameter.requires_grad:
            parameter.data.add_(0.1)
    assert adapter.fisher_enabled
    assert float(adapter._regularizer(adapter.model).detach()) > 0.0
