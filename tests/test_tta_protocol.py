from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, AUROC, F1Score

from algorithms.dusafe import (
    DuSafe,
    SSAWPhysicalView,
    collect_source_confidence_metadata,
    collect_source_semantic_metadata,
)
from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView
from algorithms.get_tta_class import get_algorithm_class
from configs.dusafe_ablation import ablation_names, resolve_dusafe_ablation
from configs.tta_hparams_new import get_hparams_class
from dataloader.demo_dataloader import Load_Dataset
from scripts.run_controlled_safety_benchmark import admission_risk_score, risk_coverage
from scripts.supplementary_utils import BatchTransformLoader, atomic_write_csv
from trainers.tta_abstract_trainer import (
    TTAAbstractTrainer,
    _count_step_safety_decisions,
    _predict_after_adaptation,
    _summarize_step_safety,
)
from trainers.tta_trainer import TTATrainer


class ToyFeatureExtractor(nn.Module):
    def __init__(self, channels=2, hidden=4):
        super().__init__()
        self.conv = nn.Conv1d(channels, hidden, 3, padding=1)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        sequence = torch.relu(self.bn(self.conv(inputs)))
        return sequence.mean(dim=-1), sequence


class ToyClassifier(nn.Module):
    def __init__(self, hidden=4, classes=3):
        super().__init__()
        self.logits = nn.Linear(hidden, classes)

    def forward(self, inputs):
        return self.logits(inputs)


class ToyModel(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.feature_extractor = ToyFeatureExtractor(channels=channels)
        self.classifier = ToyClassifier()

    def forward(self, inputs):
        features, _ = self.feature_extractor(inputs)
        return self.classifier(features)


class AliasedToyModel(ToyModel):
    """Mirror PreTrainModel's duplicated module registration."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(self.feature_extractor, self.classifier)


def optimizer_factory(hparams):
    return lambda params: torch.optim.SGD(
        params, lr=hparams["learning_rate"]
    )


def common_hparams(**overrides):
    values = {
        "steps": 1,
        "learning_rate": 1e-3,
        "enable_adaptation": True,
        "enable_ssaw": True,
        "ssaw_control_points": 5,
        "ssaw_sigma": 0.20,
        "ssaw_sobol_seed": 1729,
        "ssaw_strength": 10.0,
        "ssaw_kl_scale": 0.02,
        "ssaw_risk_temperature": 1.0,
        "enable_confidence_gate": False,
        "confidence_keep_fraction": 0.90,
        "enable_source_semantic_gate": False,
        "source_semantic_reference_samples": 128,
        "bn_statistics": "batch",
        "adapt_parameter_scope": "feature_extractor",
    }
    values.update(overrides)
    return values


def test_joint_grid_csv_write_is_atomic_and_cleans_temporary_file(tmp_path):
    destination = tmp_path / "raw.csv"
    destination.write_text("old\nvalue\n", encoding="utf-8")
    frame = __import__("pandas").DataFrame(
        [{"setting": "a", "f1": 0.75}, {"setting": "b", "f1": 0.80}]
    )

    atomic_write_csv(frame, destination)

    loaded = __import__("pandas").read_csv(destination)
    assert loaded.to_dict("records") == frame.to_dict("records")
    assert list(tmp_path.glob(".raw.csv.*.tmp")) == []


def test_controlled_audit_no_gate_does_not_fabricate_a_risk_score():
    frame = __import__("pandas").DataFrame(
        {
            "raw_top1_nll": [0.2, 3.0],
        }
    )

    scores, components = admission_risk_score(
        frame,
        confidence_enabled=False,
        confidence_threshold=float("nan"),
    )

    assert __import__("numpy").isnan(scores).all()
    assert components == ["no_continuous_admission_score"]


def test_controlled_audit_semantic_disagreement_reconstructs_binary_gate():
    frame = __import__("pandas").DataFrame(
        {
            "prediction": [1, 2],
            "source_semantic_prediction": [1, 0],
            "raw_top1_nll": [0.2, 0.2],
        }
    )

    scores, components = admission_risk_score(
        frame,
        confidence_enabled=False,
        confidence_threshold=float("nan"),
        semantic_enabled=True,
    )

    assert scores.tolist() == [0.0, 2.0]
    assert components == ["fixed_source_semantic_disagreement"]


def test_controlled_audit_skips_risk_coverage_without_continuous_score():
    frame = __import__("pandas").DataFrame(
        {"correct": [True, False], "admission_risk_score": [float("nan"), float("nan")]}
    )
    assert risk_coverage(
        frame,
        {"risk_coverage_status": "not_available_no_continuous_score"},
    ) == []


def make_method(model=None, **overrides):
    hparams = common_hparams(**overrides)
    method = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        ToyModel() if model is None else model,
        optimizer_factory(hparams),
    )
    if hparams["enable_confidence_gate"]:
        method.load_source_confidence_reference(
            {
                "version": 1,
                "top1_nll": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            }
        )
    channels = method.model.feature_extractor.conv.in_channels
    method.load_source_normalization_reference(
        torch.zeros(channels), torch.ones(channels)
    )
    return method


def test_ssaw_curve_is_positive_bounded_and_reproducible():
    torch.manual_seed(3)
    model = ToyModel().eval()
    inputs = torch.randn(6, 2, 41)
    first = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=7,
    )
    second = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=7,
    )
    reference = {
        "normalization_mean": torch.zeros(2),
        "normalization_std": torch.ones(2),
    }
    output = first(inputs, model, **reference)
    replay = second(inputs, model, **reference)
    curves = first.last_warp_curve
    assert output.shape == inputs.shape
    assert float(curves.min()) >= 0.4 - 1e-6
    assert float(curves.max()) <= 1.6 + 1e-6
    torch.testing.assert_close(output, replay)


def test_ssaw_sequence_is_reproducible_per_test_time_seed_and_varies_across_seeds():
    first = make_method(test_time_seed=1)
    replay = make_method(test_time_seed=1)
    second = make_method(test_time_seed=2)

    assert first.ssaw_base_sobol_seed == 1729
    assert first.ssaw_effective_sobol_seed == replay.ssaw_effective_sobol_seed
    assert first.ssaw_effective_sobol_seed != second.ssaw_effective_sobol_seed
    assert first.ssaw.sobol_seed == first.ssaw_effective_sobol_seed

    model = ToyModel().eval()
    inputs = torch.ones(4, 2, 31)
    reference = {
        "normalization_mean": torch.zeros(2),
        "normalization_std": torch.ones(2),
    }
    same_seed_a = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=first.ssaw_effective_sobol_seed,
    )
    same_seed_b = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=replay.ssaw_effective_sobol_seed,
    )
    other_seed = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=second.ssaw_effective_sobol_seed,
    )
    same_seed_a(inputs, model, **reference)
    same_seed_b(inputs, model, **reference)
    other_seed(inputs, model, **reference)

    torch.testing.assert_close(
        same_seed_a.last_warp_curve,
        same_seed_b.last_warp_curve,
    )
    assert not torch.equal(
        same_seed_a.last_warp_curve,
        other_seed.last_warp_curve,
    )


def test_inner_steps_reuse_one_fixed_ssaw_view_per_batch():
    torch.manual_seed(41)
    method = make_method(steps=3, learning_rate=0.0)
    draw_count = 0
    original = method.ssaw._sensor_calibration_view

    def counted_view(inputs, normalization_mean, normalization_std):
        nonlocal draw_count
        draw_count += 1
        return original(inputs, normalization_mean, normalization_std)

    method.ssaw._sensor_calibration_view = counted_view
    method({"data": torch.randn(8, 2, 31)})

    assert draw_count == 1
    assert method.ssaw.last_metadata["reused_view"] is True
    assert method._last_batch_log["ssaw_view_reused"] == pytest.approx(2.0 / 3.0)






def test_ssaw_controls_follow_a_clipped_normal_not_uniform_noise():
    search = SSAWPhysicalView(
        num_control_points=10,
        sigma=0.2,
        sobol_seed=11,
    )
    _, controls = search._sample_curves(
        128, 2, 64, torch.device("cpu"), torch.float32
    )
    normalized = (controls - 1.0) / 0.2
    assert abs(float(normalized.mean())) < 0.03
    assert 0.98 < float(normalized.std()) < 1.01
    assert float(normalized.abs().max()) <= 3.0 + 1e-5


def test_ssaw_uses_independent_window_constant_gain_per_channel():
    search = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=13,
    )
    curves, controls = search._sample_curves(
        6, 3, 41, torch.device("cpu"), torch.float32
    )
    assert curves.shape == (6, 3, 41)
    assert controls.shape == (6, 3, 5)
    assert not torch.allclose(curves[:, 0], curves[:, 1])
    torch.testing.assert_close(curves, curves[..., :1].expand_as(curves))


def test_smooth_temporal_mode_remains_available_for_ablation():
    search = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=13,
        temporal_mode="smooth",
    )
    curves, _ = search._sample_curves(
        6, 3, 41, torch.device("cpu"), torch.float32
    )
    assert not torch.allclose(curves, curves[..., :1].expand_as(curves))


def test_antithetic_views_are_centered_on_the_raw_signal():
    inputs = torch.randn(6, 2, 41)
    search = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=13,
        antithetic=True,
    )
    search(
        inputs,
        ToyModel().eval(),
        normalization_mean=torch.zeros(2),
        normalization_std=torch.ones(2),
    )
    assert search.last_view_inputs.shape == (2, 6, 2, 41)
    torch.testing.assert_close(search.last_view_inputs.mean(dim=0), inputs)
    assert search.last_metadata["view_count"] == 2
    assert search.last_metadata["ssaw_label_flip_by_view"].shape == (2, 6)


def test_multiple_antithetic_pairs_share_one_center_and_are_cached():
    inputs = torch.randn(6, 2, 41)
    search = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.2,
        sobol_seed=13,
        antithetic=True,
        antithetic_pairs=2,
    )
    kwargs = {
        "normalization_mean": torch.zeros(2),
        "normalization_std": torch.ones(2),
    }
    search(inputs, ToyModel().eval(), **kwargs)
    first = search.last_view_inputs.clone()
    search(
        inputs,
        ToyModel().eval(),
        reuse_cached_view=True,
        **kwargs,
    )
    assert search.last_view_inputs.shape == (4, 6, 2, 41)
    torch.testing.assert_close(search.last_view_inputs.mean(dim=0), inputs)
    torch.testing.assert_close(search.last_view_inputs, first)
    assert search.last_metadata["view_count"] == 4


def test_har_antithetic_orientation_uses_exact_inverse_and_preserves_norms():
    inputs = torch.randn(4, 3, 37)
    mean = torch.tensor([0.4, -0.2, 9.1])
    std = torch.tensor([1.2, 0.8, 1.5])
    search = SSAWPhysicalView(
        num_control_points=3,
        sigma=0.0,
        sobol_seed=31,
        strength=15.0,
        antithetic=True,
    )
    kwargs = {"normalization_mean": mean, "normalization_std": std}
    search(inputs, ToyModel(channels=3).eval(), **kwargs)
    views = search.last_view_inputs
    raw = inputs * std[None, :, None] + mean[None, :, None]
    physical = views * std[None, None, :, None] + mean[None, None, :, None]
    raw_norm = raw.square().sum(dim=1).sqrt()
    view_norm = physical.square().sum(dim=2).sqrt()
    torch.testing.assert_close(
        view_norm,
        raw_norm.unsqueeze(0).expand_as(view_norm),
        rtol=3e-5,
        atol=3e-5,
    )
    assert search._cached_rotation_matrices is not None
    rotation = search._cached_rotation_matrices[0]
    assert torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        torch.eye(3).expand_as(rotation),
        rtol=3e-5,
        atol=3e-5,
    )


def test_har_multiple_antithetic_rotation_pairs_replay_exactly():
    inputs = torch.randn(4, 3, 37)
    search = SSAWPhysicalView(
        num_control_points=3,
        sigma=0.0,
        sobol_seed=31,
        strength=8.0,
        antithetic=True,
        antithetic_pairs=2,
    )
    kwargs = {
        "normalization_mean": torch.tensor([0.4, -0.2, 9.1]),
        "normalization_std": torch.tensor([1.2, 0.8, 1.5]),
    }
    search(inputs, ToyModel(channels=3).eval(), **kwargs)
    first = search.last_view_inputs.clone()
    assert search._cached_rotation_matrices is not None
    assert search._cached_rotation_matrices.shape[:2] == (2, 4)
    search(
        inputs,
        ToyModel(channels=3).eval(),
        reuse_cached_view=True,
        **kwargs,
    )
    torch.testing.assert_close(search.last_view_inputs, first)


def test_sensor_calibration_reduces_to_gain_or_triad_rotation_by_shape():
    torch.manual_seed(23)
    scalar = torch.randn(5, 1, 37)
    scalar_search = SSAWPhysicalView(
        num_control_points=4,
        sigma=0.1,
        sobol_seed=29,
        strength=10.0,
    )
    scalar_search(
        scalar,
        ToyModel(channels=1).eval(),
        normalization_mean=torch.tensor([0.4]),
        normalization_std=torch.tensor([1.2]),
    )
    assert scalar_search.last_view_inputs.shape == (5, 1, 37)

    triad = torch.randn(5, 3, 37)
    mean = torch.tensor([0.4, -0.2, 9.1])
    std = torch.tensor([1.2, 0.8, 1.5])
    triad_search = SSAWPhysicalView(
        num_control_points=3,
        sigma=0.0,
        sobol_seed=31,
        strength=15.0,
    )
    triad_search(
        triad,
        ToyModel(channels=3).eval(),
        normalization_mean=mean,
        normalization_std=std,
    )
    raw = triad * std[None, :, None] + mean[None, :, None]
    view = (
        triad_search.last_view_inputs.cpu() * std[None, :, None]
        + mean[None, :, None]
    )
    raw_norm = raw.square().sum(dim=1).sqrt()
    view_norm = view.square().sum(dim=1).sqrt()
    torch.testing.assert_close(
        view_norm,
        raw_norm,
        rtol=2e-5,
        atol=2e-5,
    )
    raw_gram = raw.transpose(1, 2) @ raw
    view_gram = view.transpose(1, 2) @ view
    torch.testing.assert_close(view_gram, raw_gram, rtol=3e-5, atol=3e-5)


def test_zero_strength_ssaw_is_an_exact_elementwise_identity():
    inputs = torch.randn(5, 3, 37)
    search = SSAWPhysicalView(
        num_control_points=3,
        sigma=0.0,
        sobol_seed=31,
        strength=0.0,
    )
    output = search(
        inputs,
        ToyModel(channels=3).eval(),
        normalization_mean=torch.tensor([0.4, -0.2, 9.1]),
        normalization_std=torch.tensor([1.2, 0.8, 1.5]),
    )
    assert torch.equal(output, inputs)


def test_invalid_ssaw_parameters_are_rejected():
    with pytest.raises(ValueError, match="sigma"):
        SSAWPhysicalView(sigma=1.0 / 3.0)
    with pytest.raises(ValueError, match="strength"):
        SSAWPhysicalView(strength=-0.1)
    with pytest.raises(ValueError, match="strength"):
        SSAWPhysicalView(strength=90.1)
    with pytest.raises(ValueError, match="temporal_mode"):
        SSAWPhysicalView(temporal_mode="fast_drift")
    with pytest.raises(ValueError, match="antithetic_pairs"):
        SSAWPhysicalView(antithetic_pairs=0)


def test_pseudo_label_preserving_view_selection_is_intrinsic_to_ssaw():
    class MeanFeature(nn.Module):
        def forward(self, inputs):
            return inputs.mean(dim=(1, 2), keepdim=False)[:, None]

    class MeanClassifier(nn.Module):
        def forward(self, features):
            zeros = torch.zeros_like(features)
            return torch.cat((features, -features, zeros), dim=1)

    model = nn.Module()
    model.feature_extractor = MeanFeature()
    model.classifier = MeanClassifier()
    inputs = torch.ones(4, 1, 17)
    reference = {
        "normalization_mean": torch.zeros(1),
        "normalization_std": torch.ones(1),
    }

    search = SSAWPhysicalView()

    def negative_view(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        view = -inputs.clone()
        curves = torch.ones_like(view)
        controls = inputs.new_ones(
            inputs.size(0),
            inputs.size(1),
            search.num_control_points,
        )
        return view, curves, controls

    search._sensor_calibration_view = negative_view
    search(inputs, model, **reference)

    assert not search.last_metadata["ssaw_view_selected"].any()
    assert search.last_metadata["ssaw_label_flip"].all()




def method_features(model, inputs):
    features = model.feature_extractor(inputs)
    return features[0] if isinstance(features, (tuple, list)) else features




def test_physical_soft_consistency_replays_the_full_batchnorm_population():
    torch.manual_seed(38)
    method = make_method(learning_rate=0.0)
    inputs = torch.randn(8, 2, 31)
    method({"data": inputs})
    view_inputs = method.ssaw.last_view_inputs.to(inputs)
    selected_mask = method._last_gate_log["ssaw_consistency_mask"].to(
        inputs.device
    )
    raw_logits = method.ssaw.last_reference_logits.to(inputs)
    view_logits = method.model.classifier(
        method._extract_features(method.model, view_inputs)
    )
    raw_log_probabilities = raw_logits.log_softmax(dim=1)
    per_sample = (
        raw_log_probabilities.exp()
        * (raw_log_probabilities - view_logits.log_softmax(dim=1))
    ).sum(dim=1)
    admitted = method._last_gate_log["admission_mask"].to(inputs.device)
    expected = per_sample[selected_mask].sum() / admitted.sum()
    assert method._last_batch_log["ssaw_consistency_loss"] == pytest.approx(
        float(expected.item()), rel=1e-6, abs=1e-6
    )


def test_full_anchors_ssaw_auxiliary_objective_to_raw_ce():
    torch.manual_seed(38)
    method = make_method(learning_rate=0.0)
    inputs = torch.randn(8, 2, 31)
    method({"data": inputs})
    raw_loss = method._last_batch_log["raw_ce_loss"]
    consistency = method._last_batch_log["ssaw_consistency_loss"]
    weighted = method._last_batch_log["ssaw_weighted_consistency_loss"]
    assert weighted == pytest.approx(
        method.ssaw_auxiliary_weight * consistency,
        rel=1e-6,
        abs=1e-6,
    )
    assert method._last_batch_log["adaptation_loss"] == pytest.approx(
        raw_loss + weighted, rel=1e-6, abs=1e-6
    )
    assert method._last_batch_log["ssaw_hard_view_loss"] == pytest.approx(
        consistency, rel=1e-6, abs=1e-6
    )
    assert raw_loss != pytest.approx(consistency, rel=1e-5, abs=1e-5)


def test_ssaw_does_not_replace_raw_view_prediction():
    torch.manual_seed(43)
    method = make_method(learning_rate=0.0)
    inputs = torch.randn(8, 2, 31)
    outputs = method({"data": inputs})
    torch.testing.assert_close(outputs, method.ssaw.last_reference_logits)


def test_physical_view_contributes_to_explicit_weighted_update():
    class SignedFeature(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            return inputs.mean(dim=(1, 2))[:, None] * self.scale

    class SignedClassifier(nn.Module):
        def forward(self, features):
            zeros = torch.zeros_like(features)
            return torch.cat((features, -features, zeros), dim=1)

    def model():
        result = nn.Module()
        result.feature_extractor = SignedFeature()
        result.classifier = SignedClassifier()
        return result

    hparams = common_hparams(learning_rate=0.1)
    full = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        model(),
        optimizer_factory(hparams),
    )
    no_ssaw_hparams = {**hparams, "enable_ssaw": False}
    no_ssaw = DuSafe(
        SimpleNamespace(num_classes=3),
        no_ssaw_hparams,
        model(),
        optimizer_factory(no_ssaw_hparams),
    )
    full.load_source_normalization_reference(torch.zeros(1), torch.ones(1))
    no_ssaw.load_source_normalization_reference(torch.zeros(1), torch.ones(1))

    def stronger_same_label_view(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        view = inputs * 1.5
        return (
            view,
            torch.full_like(view, 1.5),
            inputs.new_full(
                (inputs.size(0), inputs.size(1), full.ssaw.num_control_points),
                1.5,
            ),
        )

    full.ssaw._sensor_calibration_view = stronger_same_label_view
    inputs = torch.ones(4, 1, 17)
    full({"data": inputs})
    no_ssaw({"data": inputs})

    assert full._last_batch_log["adaptation_loss"] == pytest.approx(
        full._last_batch_log["raw_ce_loss"]
        + full.ssaw_auxiliary_weight
        * full._last_batch_log["ssaw_hard_view_loss"]
    )
    assert full._last_batch_log["ssaw_gradient_applied"] == 1.0
    assert full.model.feature_extractor.scale.item() != pytest.approx(
        no_ssaw.model.feature_extractor.scale.item()
    )


def test_no_ssaw_is_raw_view_ce_under_the_same_gate():
    torch.manual_seed(38)
    method = make_method(enable_ssaw=False, learning_rate=0.0)
    inputs = torch.randn(8, 2, 31)
    raw_logits = method({"data": inputs})
    recomputed = method.model.classifier(
        method._extract_features(method.model, inputs)
    )
    expected = F.cross_entropy(
        recomputed, raw_logits.detach().argmax(dim=1)
    )
    assert method._last_batch_log["adaptation_loss"] == pytest.approx(
        float(expected.item()), rel=1e-6, abs=1e-6
    )


def test_pseudo_label_changing_view_skips_only_ssaw_auxiliary():
    class SignedFeature(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            return inputs.mean(dim=(1, 2))[:, None] * self.scale

    class SignedClassifier(nn.Module):
        def forward(self, features):
            zeros = torch.zeros_like(features)
            return torch.cat((features, -features, zeros), dim=1)

    model = nn.Module()
    model.feature_extractor = SignedFeature()
    model.classifier = SignedClassifier()
    hparams = common_hparams(learning_rate=1e-2)
    method = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        model,
        optimizer_factory(hparams),
    )
    method.load_source_normalization_reference(
        torch.zeros(1), torch.ones(1)
    )
    def negative_view(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        return (
            -inputs.clone(),
            torch.ones_like(inputs),
            inputs.new_ones(
                inputs.size(0),
                inputs.size(1),
                method.ssaw.num_control_points,
            ),
        )

    method.ssaw._sensor_calibration_view = negative_view
    method({"data": torch.ones(4, 1, 17)})
    gate = method._last_gate_log
    assert gate["base_admission_mask"].all()
    assert not gate["ssaw_view_selected_mask"].any()
    assert "ssaw_veto_mask" not in gate
    assert gate["admission_mask"].all()
    assert not gate["ssaw_consistency_mask"].any()
    assert method._last_batch_log["raw_ce_loss"] > 0.0
    assert method._last_batch_log["ssaw_consistency_loss"] == 0.0
    assert method._last_batch_log["adaptation_loss"] == pytest.approx(
        method._last_batch_log["raw_ce_loss"]
    )
    assert method._last_batch_log["update_attempted"] == 1.0


def test_every_admitted_ssaw_sample_participates_in_hard_view_training():
    torch.manual_seed(45)
    method = make_method(
        learning_rate=0.0,
        ssaw_sigma=0.0,
        ssaw_strength=0.0,
        ssaw_kl_scale=1e-3,
    )
    inputs = torch.randn(8, 2, 31)

    method({"data": inputs})

    gate = method._last_gate_log
    torch.testing.assert_close(
        gate["ssaw_consistency_mask"], gate["admission_mask"]
    )
    assert method._last_batch_log[
        "ssaw_admitted_participation_rate"
    ] == pytest.approx(1.0)
    assert "ssaw_veto_mask" not in gate


def test_large_kl_without_label_flip_is_not_reweighted_or_vetoed():
    class SignedFeature(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            return inputs.mean(dim=(1, 2))[:, None] * self.scale

    class SignedClassifier(nn.Module):
        def forward(self, features):
            zeros = torch.zeros_like(features)
            return torch.cat((features, -features, zeros), dim=1)

    model = nn.Module()
    model.feature_extractor = SignedFeature()
    model.classifier = SignedClassifier()
    hparams = common_hparams(learning_rate=0.0, ssaw_kl_scale=1e-4)
    method = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        model,
        optimizer_factory(hparams),
    )
    method.load_source_normalization_reference(torch.zeros(1), torch.ones(1))

    def lower_margin_view(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        view = inputs * 0.1
        return (
            view,
            torch.full_like(view, 0.1),
            inputs.new_full(
                (
                    inputs.size(0),
                    inputs.size(1),
                    method.ssaw.num_control_points,
                ),
                0.1,
            ),
        )

    method.ssaw._sensor_calibration_view = lower_margin_view
    method({"data": torch.ones(4, 1, 17)})

    gate = method._last_gate_log
    assert not gate["ssaw_label_flip"].any()
    assert "ssaw_veto_mask" not in gate
    assert gate["admission_mask"].all()
    assert "ssaw_update_weight" not in gate
    assert method._last_batch_log["optimizer_update_scale"] == pytest.approx(1.0)
    assert method._last_batch_log["adaptation_loss"] == pytest.approx(
        method._last_batch_log["raw_ce_loss"]
        + method._last_batch_log["ssaw_weighted_consistency_loss"]
    )


def test_source_confidence_metadata_is_label_independent_and_preserves_model():
    torch.manual_seed(20)
    source = torch.randn(24, 2, 32)
    first_loader = DataLoader(
        TensorDataset(source, torch.zeros(24).long()), batch_size=6
    )
    second_loader = DataLoader(
        TensorDataset(source, torch.arange(24).remainder(3)), batch_size=6
    )
    model = ToyModel().eval()
    before = deepcopy(model.state_dict())
    first = collect_source_confidence_metadata(
        first_loader, model, reference_samples=24
    )
    second = collect_source_confidence_metadata(
        second_loader, model, reference_samples=24
    )
    torch.testing.assert_close(first["top1_nll"], second["top1_nll"])
    assert first["top1_nll"].shape == (24,)
    assert first["source_batch_size"] == 6
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_source_semantic_metadata_is_labelled_fixed_source_and_preserves_model():
    torch.manual_seed(21)
    source = torch.randn(24, 2, 32)
    labels = torch.arange(24).remainder(3)
    loader = DataLoader(TensorDataset(source, labels), batch_size=6)
    model = ToyModel().eval()
    before = deepcopy(model.state_dict())

    metadata = collect_source_semantic_metadata(
        loader,
        model,
        num_classes=3,
        reference_samples=24,
    )

    assert metadata["prototypes"].shape == (3, 4)
    assert metadata["class_counts"].tolist() == [8, 8, 8]
    assert metadata["source_batch_size"] == 6
    torch.testing.assert_close(
        metadata["prototypes"].norm(dim=1), torch.ones(3)
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_source_semantic_output_is_invariant_to_target_split_and_order():
    torch.manual_seed(211)
    source = torch.randn(48, 2, 32)
    source_labels = torch.arange(48).remainder(3)
    source_model = ToyModel().eval()
    metadata = collect_source_semantic_metadata(
        DataLoader(
            TensorDataset(source, source_labels),
            batch_size=6,
            shuffle=False,
        ),
        source_model,
        num_classes=3,
        reference_samples=48,
        bn_statistics="frozen",
    )
    hparams = common_hparams(
        enable_source_semantic_gate=True,
        enable_confidence_gate=False,
        enable_ssaw=False,
        enable_adaptation=False,
        source_semantic_bn_statistics="frozen",
    )
    method = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        deepcopy(source_model),
        optimizer_factory(hparams),
    )
    method.load_source_semantic_reference(metadata)
    target = torch.randn(13, 2, 32)
    pseudo = torch.zeros(13, dtype=torch.long)
    _, full_prediction, full_margin = method._source_semantic_decision(
        target, pseudo
    )
    split_predictions, split_margins = [], []
    for part in (slice(0, 5), slice(5, 13)):
        _, prediction, margin = method._source_semantic_decision(
            target[part], pseudo[part]
        )
        split_predictions.append(prediction)
        split_margins.append(margin)
    assert torch.equal(full_prediction, torch.cat(split_predictions))
    torch.testing.assert_close(full_margin, torch.cat(split_margins))

    permutation = torch.tensor([8, 1, 11, 0, 5, 12, 3, 9, 2, 7, 4, 10, 6])
    _, permuted_prediction, permuted_margin = method._source_semantic_decision(
        target[permutation], pseudo[permutation]
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(full_prediction, permuted_prediction[inverse])
    torch.testing.assert_close(full_margin, permuted_margin[inverse])
    for module in method.source_semantic_feature_extractor.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert module.track_running_stats
            assert not module.training


def test_source_semantic_reference_accepts_zero_variance_bn_channel():
    torch.manual_seed(212)
    source = torch.randn(48, 2, 32)
    source_labels = torch.arange(48).remainder(3)
    source_model = ToyModel().eval()
    metadata = collect_source_semantic_metadata(
        DataLoader(TensorDataset(source, source_labels), batch_size=6),
        source_model,
        num_classes=3,
        reference_samples=48,
        bn_statistics="frozen",
    )
    first_bn = next(iter(metadata["feature_extractor_bn_state"].values()))
    first_bn["running_var"][0] = 0.0
    hparams = common_hparams(
        enable_source_semantic_gate=True,
        enable_confidence_gate=False,
        enable_ssaw=False,
        enable_adaptation=False,
        source_semantic_bn_statistics="frozen",
    )
    method = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        deepcopy(source_model),
        optimizer_factory(hparams),
    )
    method.load_source_semantic_reference(metadata)
    loaded_bn = next(
        module
        for module in method.source_semantic_feature_extractor.modules()
        if isinstance(module, nn.BatchNorm1d)
    )
    assert loaded_bn.running_var[0].item() == 0.0


def test_confidence_and_semantic_metadata_use_distinct_bn_protocols():
    dummy = SimpleNamespace(
        hparams_class=SimpleNamespace(
            alg_hparams={
                "DuSafe": {
                    "bn_statistics": "batch",
                    "source_semantic_bn_statistics": "frozen",
                    "confidence_reference_samples": 128,
                    "source_semantic_reference_samples": 128,
                }
            }
        ),
        hparams={"batch_size": 16},
        src_test_dl=SimpleNamespace(batch_size=16),
        dataset_configs=SimpleNamespace(num_classes=3),
    )
    confidence = TTATrainer._source_confidence_metadata_config(dummy)
    semantic = TTATrainer._source_semantic_metadata_config(dummy)
    assert confidence["bn_statistics"] == "batch"
    assert semantic["bn_statistics"] == "frozen"


def test_source_confidence_quantile_sets_threshold_and_fails_closed():
    method = make_method(
        enable_confidence_gate=True,
        confidence_keep_fraction=0.5,
        enable_adaptation=False,
    )
    method.load_source_confidence_reference(
        {"version": 1, "top1_nll": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    )
    torch.testing.assert_close(method.confidence_nll_threshold, torch.tensor(2.5))

    hparams = common_hparams(
        enable_confidence_gate=True, enable_adaptation=False
    )
    uncalibrated = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        ToyModel(),
        optimizer_factory(hparams),
    )
    uncalibrated.load_source_normalization_reference(
        torch.zeros(2), torch.ones(2)
    )
    with pytest.raises(RuntimeError, match="source confidence"):
        uncalibrated({"data": torch.randn(5, 2, 31)})


def test_source_semantic_gate_fails_closed_without_source_metadata():
    hparams = common_hparams(
        enable_source_semantic_gate=True,
        enable_confidence_gate=False,
        enable_ssaw=False,
        enable_adaptation=False,
    )
    method = DuSafe(
        SimpleNamespace(num_classes=3),
        hparams,
        ToyModel(),
        optimizer_factory(hparams),
    )
    method.load_source_normalization_reference(
        torch.zeros(2), torch.ones(2)
    )
    with pytest.raises(RuntimeError, match="Source semantic"):
        method({"data": torch.randn(5, 2, 31)})


def test_remaining_safety_masks_are_intersected():
    torch.manual_seed(12)
    method = make_method(
        enable_confidence_gate=True,
        confidence_keep_fraction=0.5,
        enable_source_semantic_gate=True,
        enable_adaptation=False,
    )
    semantic_source = torch.randn(96, 2, 64)
    semantic_labels = torch.arange(96).remainder(3)
    method.load_source_semantic_reference(
        collect_source_semantic_metadata(
            DataLoader(
                TensorDataset(semantic_source, semantic_labels),
                batch_size=8,
            ),
            method.model,
            num_classes=3,
            reference_samples=96,
        )
    )
    method({"data": torch.randn(8, 2, 64)})
    gate = method._last_gate_log
    assert torch.equal(
        gate["base_admission_mask"],
        gate["confidence_mask"] & gate["semantic_mask"],
    )
    assert torch.equal(
        gate["admission_mask"],
        gate["base_admission_mask"],
    )
    assert torch.equal(
        gate["ssaw_consistency_mask"],
        gate["admission_mask"] & gate["ssaw_view_selected_mask"],
    )


def test_target_labels_and_metadata_cannot_change_online_decisions():
    torch.manual_seed(21)
    first = make_method()
    second = make_method()
    second.load_state_dict(first.state_dict())
    inputs = torch.randn(6, 2, 31)
    first_logits = first(
        {
            "data": inputs,
            "labels": torch.zeros(6, dtype=torch.long),
            "meta": {"corruption_mask": [False] * 6},
        }
    )
    second_logits = second(
        {
            "data": inputs,
            "labels": torch.full((6,), 2, dtype=torch.long),
            "meta": {"corruption_mask": [True] * 6},
        }
    )
    torch.testing.assert_close(first_logits, second_logits)
    assert torch.equal(
        first._last_gate_log["admission_mask"],
        second._last_gate_log["admission_mask"],
    )
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters()
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_non_finite_update_rolls_back_without_poisoning_optimizer():
    torch.manual_seed(23)
    method = make_method(enable_ssaw=False, learning_rate=1e-2)
    inputs = torch.randn(6, 2, 31)
    logits = method.model.classifier(method._extract_features(method.model, inputs))
    labels = logits.detach().argmax(dim=1)
    loss = F.cross_entropy(logits, labels) * torch.tensor(float("nan"))
    before = {
        name: parameter.detach().clone()
        for name, parameter in method.model.named_parameters()
    }
    log = method._apply_update(
        method.model,
        method.optimizer,
        loss,
        torch.ones(inputs.size(0), dtype=torch.bool),
    )
    assert log == {
        "attempted": True,
        "committed": False,
        "finite": False,
        "update_scale": 1.0,
    }
    for name, parameter in method.model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_optimizer_update_scale_controls_sgd_step_and_restores_lr():
    method = make_method(enable_ssaw=False, learning_rate=0.0)
    full_model = nn.Linear(1, 1, bias=False)
    scaled_model = deepcopy(full_model)
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
    scaled_optimizer = torch.optim.SGD(scaled_model.parameters(), lr=0.1)
    full_before = full_model.weight.detach().clone()
    scaled_before = scaled_model.weight.detach().clone()

    method._apply_update(
        full_model,
        full_optimizer,
        full_model.weight.square().sum(),
        torch.ones(1, dtype=torch.bool),
        update_scale=1.0,
    )
    method._apply_update(
        scaled_model,
        scaled_optimizer,
        scaled_model.weight.square().sum(),
        torch.ones(1, dtype=torch.bool),
        update_scale=0.25,
    )

    full_delta = (full_model.weight - full_before).abs()
    scaled_delta = (scaled_model.weight - scaled_before).abs()
    torch.testing.assert_close(scaled_delta, 0.25 * full_delta)
    assert full_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert scaled_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_conflicting_ssaw_gradient_remains_in_explicit_weighted_objective():
    method = make_method(enable_ssaw=False, learning_rate=0.0)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    raw_loss = model.weight.sum()
    conflicting_auxiliary = -model.weight.sum()

    log = method._apply_update(
        model,
        optimizer,
        raw_loss,
        torch.ones(1, dtype=torch.bool),
        auxiliary_loss=conflicting_auxiliary,
        auxiliary_weight=1.0,
    )

    assert log["auxiliary_gradient_applied"] is True
    # The equal and opposite terms cancel in the declared weighted objective.
    assert model.weight.item() == pytest.approx(1.0)


def test_compatible_ssaw_gradient_is_added_to_raw_anchor():
    method = make_method(enable_ssaw=False, learning_rate=0.0)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    log = method._apply_update(
        model,
        optimizer,
        model.weight.sum(),
        torch.ones(1, dtype=torch.bool),
        auxiliary_loss=model.weight.sum(),
        auxiliary_weight=0.5,
    )

    assert log["auxiliary_gradient_applied"] is True
    assert model.weight.item() == pytest.approx(0.85)


def test_available_auxiliary_is_not_reported_as_applied_on_early_return():
    method = make_method(enable_ssaw=False, learning_rate=0.0)
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    raw_loss = model.weight.sum()
    auxiliary_loss = model.weight.square().sum()

    log = method._apply_update(
        model,
        optimizer,
        raw_loss,
        torch.zeros(1, dtype=torch.bool),
        auxiliary_loss=auxiliary_loss,
        auxiliary_weight=1.0,
    )

    assert log["auxiliary_available"] is True
    assert log["attempted"] is False
    assert log["auxiliary_gradient_applied"] is False


def test_batchnorm_is_stateless_dropout_is_disabled_and_head_is_frozen():
    method = make_method(model=AliasedToyModel(), enable_adaptation=True)
    trainable = {
        name for name, parameter in method.model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "feature_extractor.conv.weight",
        "feature_extractor.conv.bias",
        "feature_extractor.bn.weight",
        "feature_extractor.bn.bias",
    }
    bn = method.model.feature_extractor.bn
    source_mean = bn.running_mean.detach().clone()
    source_variance = bn.running_var.detach().clone()
    source_batches = bn.num_batches_tracked.detach().clone()
    assert bn.training is True
    assert bn.track_running_stats is False
    assert method.model.feature_extractor.dropout.training is False

    method({"data": torch.randn(5, 2, 31) + 20.0})
    torch.testing.assert_close(bn.running_mean, source_mean)
    torch.testing.assert_close(bn.running_var, source_variance)
    torch.testing.assert_close(bn.num_batches_tracked, source_batches)


def test_multiple_inner_steps_retain_every_safety_decision():
    method = make_method(
        enable_ssaw=False,
        steps=3,
    )
    batch = torch.randn(5, 2, 31)
    outputs = method({"data": batch})
    assert outputs.shape == (5, 3)
    assert method._last_gate_log["inner_step_count"] == 3
    assert method._last_gate_log["inner_pseudo_labels"].shape == (3, 5)
    assert method._last_gate_log["inner_admission_masks"].shape == (3, 5)
    assert method._last_gate_log["inner_active_masks"].shape == (3, 5)


def test_step_safety_metrics_use_one_consistent_decision_grain():
    labels = torch.tensor([0, 1])
    pseudo_labels = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
        ]
    )
    admission_masks = torch.tensor(
        [
            [True, True],
            [False, True],
            [True, False],
        ]
    )
    active_masks = torch.tensor(
        [
            [True, True],
            [False, True],
            [False, False],
        ]
    )
    counts = _count_step_safety_decisions(
        pseudo_labels=pseudo_labels,
        admission_masks=admission_masks,
        active_masks=active_masks,
        labels=labels,
        corrupted=torch.tensor([False, True]),
    )
    summary = _summarize_step_safety(counts)

    assert counts == {
        "decision_count": 6,
        "correct_count": 4,
        "wrong_count": 2,
        "admitted_count": 4,
        "admitted_correct_count": 2,
        "admitted_wrong_count": 2,
        "active_count": 3,
        "active_correct_count": 2,
        "active_wrong_count": 1,
        "corrupted_count": 3,
        "admitted_corrupted_count": 2,
        "active_corrupted_count": 2,
        "admitted_unsafe_count": 3,
        "active_unsafe_count": 2,
        "clean_correct_count": 2,
        "admission_rejected_clean_correct_count": 1,
        "inactive_clean_correct_count": 1,
    }
    assert summary["safety_metric_version"] == 2.0
    assert summary["coverage"] == pytest.approx(3 / 6)
    assert summary["accepted_pseudo_label_accuracy"] == pytest.approx(2 / 3)
    assert summary["wrong_update_rate"] == pytest.approx(1 / 3)
    assert summary["wrong_rejection_recall"] == pytest.approx(1 / 2)
    assert summary["correct_false_rejection_rate"] == pytest.approx(2 / 4)
    assert summary["correct_acceptance_rate"] == pytest.approx(2 / 4)
    assert summary["unsafe_update_rate"] == pytest.approx(2 / 3)
    assert summary["admission_coverage"] == pytest.approx(4 / 6)
    assert summary["admitted_pseudo_label_accuracy"] == pytest.approx(2 / 4)
    assert summary["admission_wrong_rejection_recall"] == pytest.approx(0)
    assert summary["admission_correct_false_rejection_rate"] == pytest.approx(
        2 / 4
    )
    assert summary["corruption_rejection_recall"] == pytest.approx(1 / 3)
    assert summary["clean_correct_false_rejection_rate"] == pytest.approx(
        1 / 2
    )


def test_step_safety_metrics_reject_active_samples_outside_admission():
    with pytest.raises(RuntimeError, match="cannot bypass"):
        _count_step_safety_decisions(
            pseudo_labels=torch.tensor([[0]]),
            admission_masks=torch.tensor([[False]]),
            active_masks=torch.tensor([[True]]),
            labels=torch.tensor([0]),
            corrupted=torch.tensor([False]),
        )


def test_post_update_prediction_includes_the_final_optimizer_step():
    torch.manual_seed(31)
    method = make_method(
        enable_ssaw=False,
        steps=1,
        learning_rate=0.1,
    )
    batch = torch.randn(8, 2, 31)
    pre_final = method({"data": batch})
    post_update = _predict_after_adaptation(method, {"data": batch})

    assert pre_final.shape == post_update.shape
    assert not torch.allclose(pre_final, post_update)


def test_calculate_metrics_uses_post_update_logits_and_keeps_legacy_f1():
    trainer = object.__new__(TTAAbstractTrainer)
    trainer.trg_whole_dl = object()
    trainer.ACC = Accuracy(task="multiclass", num_classes=3)
    trainer.F1 = F1Score(
        task="multiclass", num_classes=3, average="macro"
    )
    trainer.AUROC = AUROC(task="multiclass", num_classes=3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    post_update = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
        ]
    )
    pre_final = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]
    )

    def fake_evaluate(loader, tta_model):
        del loader, tta_model
        trainer.full_preds = post_update
        trainer.full_pre_final_update_preds = pre_final
        trainer.full_labels = labels
        trainer.loss = F.cross_entropy(post_update, labels)
        trainer.pre_final_update_loss = F.cross_entropy(pre_final, labels)

    trainer.evaluate = fake_evaluate
    metrics = trainer.calculate_metrics(object())

    assert metrics[0] == pytest.approx(1.0)
    assert metrics[1] == pytest.approx(1.0)
    assert trainer.last_prediction_metric_summary[
        "post_update_macro_f1"
    ] == pytest.approx(1.0)
    assert trainer.last_prediction_metric_summary[
        "pre_final_update_macro_f1"
    ] < 1.0


def test_source_no_update_freezes_every_parameter_and_bn_buffer():
    method = make_method(
        enable_adaptation=False,
        enable_ssaw=False,
        bn_statistics="frozen",
    )
    assert method.optimizer is None
    assert not any(parameter.requires_grad for parameter in method.model.parameters())
    assert method.model.feature_extractor.bn.training is False


def test_invalid_batchnorm_mode_is_rejected():
    with pytest.raises(ValueError, match="must be 'frozen' or 'batch'"):
        make_method(bn_statistics="running")


def test_current_config_is_dataset_level_and_has_no_removed_modules():
    configs = {
        dataset: get_hparams_class(dataset)()
        for dataset in ("EEG", "HAR", "FD")
    }
    assert {
        name: configs[name].alg_hparams["DuSafe"]["learning_rate"]
        for name in configs
    } == {"EEG": 2e-3, "HAR": 3.325e-4, "FD": 3e-6}
    structural_values = {
        "adapt_parameter_scope": "feature_extractor",
        "enable_ssaw": True,
        "enable_confidence_gate": True,
    }
    for config in configs.values():
        dusafe = config.alg_hparams["DuSafe"]
        assert {
            key: dusafe[key] for key in structural_values
        } == structural_values
        assert "ssaw_enable_physical_warp" not in dusafe
        assert "ssaw_require_label_preservation" not in dusafe
    dataset_numeric_keys = {
        "learning_rate",
        "steps",
        "ssaw_auxiliary_weight",
        "confidence_keep_fraction",
    }
    common_keys = set(configs["EEG"].alg_hparams["DuSafe"]) - dataset_numeric_keys
    for name in ("HAR", "FD"):
        assert (
            set(configs[name].alg_hparams["DuSafe"]) - dataset_numeric_keys
        ) == common_keys
    assert {
        name: configs[name].alg_hparams["DuSafe"][
            "confidence_keep_fraction"
        ]
        for name in ("EEG", "HAR")
    } == {"EEG": 1.0, "HAR": 0.995}
    assert {
        name: configs[name].alg_hparams["DuSafe"]["steps"]
        for name in configs
    } == {"EEG": 2, "HAR": 2, "FD": 2}
    assert all(
        configs[name].alg_hparams["DuSafe"]["dusafe_variant"]
        == "spline_residual"
        for name in configs
    )
    assert all(
        configs[name].alg_hparams["DuSafe"]["spline_control_points"] == 10
        and configs[name].alg_hparams["DuSafe"]["spline_num_directions"] == 4
        and configs[name].alg_hparams["DuSafe"]["spline_log_strength"] == 0.2
        and configs[name].alg_hparams["DuSafe"]["spline_radius_levels"]
        == [1.0, 0.5, 0.25]
        for name in configs
    )
    assert {
        name: configs[name].alg_hparams["DuSafe"][
            "ssaw_auxiliary_weight"
        ]
        for name in configs
    } == {"EEG": 0.003, "HAR": 0.1, "FD": 0.05}
    removed_tokens = (
        "fisher",
        "response",
        "history",
        "prior",
        "train_bn",
        "train_classifier",
        "update_loss",
        "pseudo_label_rule",
    )
    for config in configs.values():
        assert set(config.alg_hparams) == {"DuSafe", "NoAdap"}
        dusafe = config.alg_hparams["DuSafe"]
        assert "scenario_overrides" not in dusafe
        assert not {
            "enable_source_semantic_gate",
            "enable_source_semantic_router",
            "source_semantic_reference_samples",
            "source_semantic_bn_statistics",
            "ssaw_sigma",
            "ssaw_control_points",
            "ssaw_strength",
            "ssaw_kl_scale",
            "ssaw_risk_temperature",
            "ssaw_temporal_mode",
            "ssaw_antithetic",
            "ssaw_antithetic_pairs",
            "spline_search_steps",
            "spline_search_step_size",
        } & set(dusafe)
        assert not any(
            token in key for key in dusafe for token in removed_tokens
        )


def test_production_profile_does_not_request_source_semantic_metadata():
    profile = get_hparams_class("HAR")()
    production = SimpleNamespace(
        da_method="DuSafe",
        hparams=profile.alg_hparams["DuSafe"],
    )
    archived = SimpleNamespace(
        da_method="DuSafe",
        hparams={"enable_source_semantic_router": True},
    )

    assert not TTATrainer._requires_source_semantic_metadata(production)
    assert TTATrainer._requires_source_semantic_metadata(archived)


def test_production_registry_builds_only_the_reviewed_spline_state():
    profile = get_hparams_class("HAR")()
    hparams = {
        **profile.alg_hparams["DuSafe"],
        **profile.train_params,
    }
    full_class = get_algorithm_class(
        "DuSafe", variant=hparams["dusafe_variant"]
    )
    full = full_class(
        SimpleNamespace(num_classes=3),
        hparams,
        ToyModel(),
        optimizer_factory(hparams),
    )
    assert isinstance(full.ssaw, UnifiedSplineHardView)
    assert not full.enable_source_semantic_router
    assert not full.enable_source_semantic_gate
    assert full.source_semantic_feature_extractor is None
    assert not any(
        hasattr(full.ssaw, name)
        for name in (
            "sigma",
            "strength",
            "temporal_mode",
            "antithetic",
            "antithetic_pairs",
        )
    )
    assert not any(
        hasattr(full, name)
        for name in (
            "ssaw_risk_temperature",
            "ssaw_kl_scale",
            "ssaw_invariance_weight",
        )
    )

    no_ssaw_class = get_algorithm_class("DuSafe", variant="confidence_raw")
    no_ssaw = no_ssaw_class(
        SimpleNamespace(num_classes=3),
        {**hparams, "enable_ssaw": False},
        ToyModel(),
        optimizer_factory(hparams),
    )
    assert not no_ssaw.enable_ssaw
    assert not no_ssaw.enable_source_semantic_router
    assert isinstance(no_ssaw.ssaw, UnifiedSplineHardView)


def test_production_spline_residual_uses_search_time_mask_end_to_end():
    torch.manual_seed(913)
    source = torch.randn(48, 2, 32)
    labels = torch.arange(48).remainder(3)
    source_loader = DataLoader(
        TensorDataset(source, labels), batch_size=6, shuffle=False
    )
    source_model = ToyModel().eval()
    confidence = collect_source_confidence_metadata(
        source_loader, source_model, reference_samples=48
    )
    profile = get_hparams_class("HAR")()
    hparams = {
        **profile.alg_hparams["DuSafe"],
        **profile.train_params,
        "steps": 1,
        "learning_rate": 0.0,
        "spline_num_directions": 1,
        "dusafe_logging_mode": "evidence",
    }
    method_class = get_algorithm_class(
        "DuSafe", variant=hparams["dusafe_variant"]
    )
    method = method_class(
        SimpleNamespace(num_classes=3),
        hparams,
        deepcopy(source_model),
        optimizer_factory(hparams),
    )
    method.load_source_normalization_reference(
        torch.zeros(2), torch.ones(2)
    )
    method.load_source_confidence_reference(confidence)
    outputs = method({"data": torch.randn(8, 2, 32)})

    assert outputs.shape == (8, 3)
    assert method.ssaw.last_metadata["gathered_forward_applied"] is True
    assert method.ssaw.last_metadata["gathered_training_rule"] == "search_time_mask"
    assert "gathered_recheck_applied" not in method.ssaw.last_metadata
    assert "ssaw_gathered_training_mask" in method._last_gate_log
    assert "ssaw_veto_mask" not in method._last_gate_log
    assert "ssaw_update_weight" not in method._last_gate_log
    assert method.source_semantic_feature_extractor is None


def test_source_training_and_target_stream_profiles_are_independent():
    configs = {
        dataset: get_hparams_class(dataset)()
        for dataset in ("EEG", "HAR", "FD")
    }
    assert {
        name: config.source_train_params["num_epochs"]
        for name, config in configs.items()
    } == {"EEG": 320, "HAR": 100, "FD": 60}
    assert {
        name: config.source_train_params["batch_size"]
        for name, config in configs.items()
    } == {"EEG": 96, "HAR": 16, "FD": 64}
    assert {
        name: config.train_params["batch_size"]
        for name, config in configs.items()
    } == {"EEG": 192, "HAR": 48, "FD": 192}
    assert {
        name: config.alg_hparams["NoAdap"]["pre_learning_rate"]
        for name, config in configs.items()
    } == {"EEG": 5e-4, "HAR": 1e-4, "FD": 1e-2}
    for config in configs.values():
        assert "weight_decay" in config.source_train_params
        assert "weight_decay" in config.train_params


def test_ablation_presets_are_small_and_resolvable():
    assert ablation_names() == (
        "full",
        "source_no_update",
        "ttbn_only",
        "no_ssaw",
        "no_confidence_gate",
        "no_source_semantic_router",
        "no_admission_or_router",
    )
    assert resolve_dusafe_ablation("no-ssaw")["overrides"] == {
        "enable_ssaw": False
    }
    with pytest.raises(ValueError):
        resolve_dusafe_ablation("unknown")


def test_source_normalization_is_reused_by_target_loader():
    source = {
        "samples": torch.tensor([[[0.0, 2.0]], [[2.0, 4.0]]]),
        "labels": torch.tensor([0, 1]),
    }
    target = {
        "samples": torch.tensor([[[10.0, 12.0]]]),
        "labels": torch.tensor([0]),
    }
    configs = SimpleNamespace(input_channels=1, normalize=True)
    source_loaded = Load_Dataset(source, configs)
    target_loaded = Load_Dataset(
        target, configs, normalization_stats=source_loaded.normalization_stats
    )
    mean, std = source_loaded.normalization_stats
    expected = (target["samples"][0] - mean[:, None]) / std[:, None]
    torch.testing.assert_close(target_loaded[0][0], expected)


def test_corruption_rng_is_isolated_from_method_random_draws():
    base = DataLoader(
        TensorDataset(
            torch.zeros(6, 1, 9), torch.zeros(6).long(), torch.arange(6)
        ),
        batch_size=2,
        shuffle=False,
    )

    def random_corruption(inputs, severity):
        del severity
        return inputs + torch.rand_like(inputs)

    first = iter(
        BatchTransformLoader(
            base, random_corruption, "moderate", transform_seed=77
        )
    )
    next(first)
    torch.rand(10_000)
    second_after_draws = next(first)[0]
    replay = iter(
        BatchTransformLoader(
            base, random_corruption, "moderate", transform_seed=77
        )
    )
    next(replay)
    second_without_draws = next(replay)[0]
    torch.testing.assert_close(second_after_draws, second_without_draws)


def test_production_evaluator_skips_only_per_sample_evidence():
    class EvalWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.model.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 0.75]]))
            self._last_batch_log = {"adaptation_loss": 0.25}
            self._last_gate_log = {
                "pseudo_labels": torch.tensor([0, 1, 0]),
                "active_mask": torch.tensor([True, False, True]),
                "admission_mask": torch.tensor([True, True, True]),
            }

        def forward(self, payload):
            return self.model(payload["data"])

    data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    labels = torch.tensor([0, 1, 0])
    indices = torch.arange(3)
    loader = [(data, labels, indices)]

    def run(record_evidence: bool):
        trainer = object.__new__(TTAAbstractTrainer)
        trainer.da_method = "DuSafe"
        trainer.device = torch.device("cpu")
        trainer.hparams = {
            "record_per_sample_evidence": record_evidence,
        }
        trainer.evaluate(loader, EvalWrapper())
        return trainer

    production = run(False)
    evidence = run(True)

    torch.testing.assert_close(
        production.full_preds, evidence.full_preds, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        production.full_labels, evidence.full_labels, rtol=0.0, atol=0.0
    )
    assert production.last_safety_records.empty
    assert production.last_safety_summary == {}
    assert len(evidence.last_safety_records) == 3
    assert evidence.last_safety_summary


def test_evaluator_supplies_label_free_full_batch_graph_workload_hint():
    class WorkloadAwareWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 2, bias=False)
            self.steps = 2
            self.workload_hints = []
            self._last_batch_log = {}
            self._last_gate_log = {
                "pseudo_labels": torch.zeros(3, dtype=torch.long),
                "active_mask": torch.ones(3, dtype=torch.bool),
                "admission_mask": torch.ones(3, dtype=torch.bool),
            }

        def configure_candidate_graph_workload(
            self, *, expected_full_batch_searches
        ):
            self.workload_hints.append(int(expected_full_batch_searches))

        def forward(self, payload):
            return self.model(payload["data"])

    data = torch.randn(7, 2)
    labels = torch.zeros(7, dtype=torch.long)
    indices = torch.arange(7)
    loader = DataLoader(
        TensorDataset(data, labels, indices), batch_size=3, shuffle=False
    )
    trainer = object.__new__(TTAAbstractTrainer)
    trainer.da_method = "DuSafe"
    trainer.device = torch.device("cpu")
    trainer.hparams = {"record_per_sample_evidence": False}
    wrapper = WorkloadAwareWrapper()
    trainer.evaluate(loader, wrapper)
    assert wrapper.workload_hints == [4]
