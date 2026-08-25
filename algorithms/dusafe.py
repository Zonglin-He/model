"""DuSafe: fixed-source safe adaptation with smooth signal hard views.

The production path contains two decisions:

1. Source-calibrated top-1 NLL admits reliable raw pseudo-label anchors.
2. SSAW constructs a bounded smooth sensor-response view. Every admitted
   sample keeps the same raw pseudo-label CE anchor, while eligible physical
   views receive one uniformly weighted residual-consistency objective. A view
   that changes the pseudo-label is excluded from the auxiliary objective; it
   never vetoes the admitted raw update. The no-SSAW ablation removes this
   complete physical-view branch and leaves the raw admission and update path
   unchanged. Predictions always come from the unwarped signal.

There is no stability-based rescue, target-history threshold, target prototype
correction, class-prior alignment, Fisher regularisation, or target-label
feedback in this implementation.
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager, nullcontext
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm


SOURCE_CONFIDENCE_METADATA_VERSION = 1
SOURCE_SEMANTIC_METADATA_VERSION = 2


_BATCH_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _extract_primary_tensor(batch_data):
    """Extract the signal tensor without reading labels or metadata."""
    if isinstance(batch_data, dict):
        data = batch_data.get("data", batch_data)
        return data[0] if isinstance(data, (list, tuple)) else data
    if isinstance(batch_data, (list, tuple)):
        return batch_data[0]
    return batch_data


def _extract_features(model, inputs: torch.Tensor) -> torch.Tensor:
    features = model.feature_extractor(inputs)
    return features[0] if isinstance(features, (tuple, list)) else features


def _normalized_feature_vectors(
    model_or_extractor, inputs: torch.Tensor
) -> torch.Tensor:
    """Return one unit feature vector per sample from a model or extractor."""
    if hasattr(model_or_extractor, "feature_extractor"):
        features = _extract_features(model_or_extractor, inputs)
    else:
        features = model_or_extractor(inputs)
        if isinstance(features, (tuple, list)):
            features = features[0]
    return F.normalize(features.flatten(1), dim=1)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    return -(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)


def _top1_nll(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = logits.argmax(dim=-1)
    values = -logits.log_softmax(dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    return values, labels


def _configure_calibration_copy(
    model,
    bn_statistics: str,
    disable_dropout: bool,
):
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Source calibration requires model parameters") from exc
    calibration_model = copy.deepcopy(model).to(device)
    calibration_model.train()
    for parameter in calibration_model.parameters():
        parameter.requires_grad_(False)
    for module in calibration_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if bn_statistics == "batch":
                module.track_running_stats = False
                module.training = True
            else:
                if module.running_mean is None or module.running_var is None:
                    raise ValueError(
                        "frozen BatchNorm requires source running statistics"
                    )
                module.track_running_stats = True
                module.training = False
        elif disable_dropout and isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ):
            module.training = False
    return calibration_model, device


def _source_semantic_bn_state(feature_extractor) -> Dict[str, Dict[str, torch.Tensor]]:
    """Serialize the running statistics used by the frozen semantic encoder."""
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in feature_extractor.named_modules():
        if not isinstance(module, _BATCH_NORM_TYPES):
            continue
        if module.running_mean is None or module.running_var is None:
            raise ValueError(
                "source semantic BatchNorm calibration did not create running statistics"
            )
        state[name] = {
            "running_mean": module.running_mean.detach().cpu().clone(),
            "running_var": module.running_var.detach().cpu().clone(),
            "num_batches_tracked": (
                torch.zeros((), dtype=torch.long)
                if module.num_batches_tracked is None
                else module.num_batches_tracked.detach().cpu().clone()
            ),
        }
    return state


def _prepare_source_semantic_bn_calibration(model, disable_dropout: bool) -> int:
    """Reset BN buffers so source data, not a target batch, defines semantics."""
    model.train()
    bn_count = 0
    for module in model.modules():
        if isinstance(module, _BATCH_NORM_TYPES):
            bn_count += 1
            device = (
                module.weight.device
                if module.affine
                else next(model.parameters()).device
            )
            dtype = module.weight.dtype if module.affine else torch.float32
            module.track_running_stats = True
            module.running_mean = torch.zeros(
                module.num_features, device=device, dtype=dtype
            )
            module.running_var = torch.ones(
                module.num_features, device=device, dtype=dtype
            )
            module.num_batches_tracked = torch.zeros(
                (), device=device, dtype=torch.long
            )
            # Cumulative averaging makes the calibration independent of an
            # arbitrary momentum inherited from source training.
            module.momentum = None
            module.training = True
        elif disable_dropout and isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ):
            module.training = False
    return bn_count


def _freeze_batch_norm_statistics(model) -> None:
    for module in model.modules():
        if isinstance(module, _BATCH_NORM_TYPES):
            if module.running_mean is None or module.running_var is None:
                raise ValueError("frozen BatchNorm requires calibrated source statistics")
            module.track_running_stats = True
            module.training = False
        elif isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ):
            module.training = False


@torch.no_grad()
def collect_source_confidence_metadata(
    source_loader,
    model,
    reference_samples: int = 4096,
    bn_statistics: str = "batch",
    disable_dropout: bool = True,
) -> Dict[str, object]:
    """Collect source top-1 NLL values without reading source labels."""
    reference_samples = max(1, int(reference_samples))
    bn_statistics = str(bn_statistics).strip().lower()
    if bn_statistics not in {"frozen", "batch"}:
        raise ValueError("bn_statistics must be 'frozen' or 'batch'")
    calibration_model, device = _configure_calibration_copy(
        model, bn_statistics, bool(disable_dropout)
    )
    score_parts = []
    collected = 0
    for batch in source_loader:
        inputs = _extract_primary_tensor(batch)
        if not torch.is_tensor(inputs) or inputs.dim() != 3:
            raise ValueError(
                "Source confidence calibration expects [B, C, T] tensors"
            )
        inputs = inputs.float().to(device)
        logits = calibration_model.classifier(
            _extract_features(calibration_model, inputs)
        )
        scores, _ = _top1_nll(logits)
        remaining = reference_samples - collected
        score_parts.append(scores[:remaining].cpu())
        collected += min(remaining, scores.numel())
        if collected >= reference_samples:
            break
    if not score_parts:
        raise RuntimeError("Confidence calibration received an empty loader")
    return {
        "version": SOURCE_CONFIDENCE_METADATA_VERSION,
        "reference_samples": reference_samples,
        "bn_statistics": bn_statistics,
        "disable_dropout": bool(disable_dropout),
        "source_batch_size": int(getattr(source_loader, "batch_size", 0) or 0),
        "top1_nll": torch.cat(score_parts, dim=0),
    }


@torch.no_grad()
def collect_source_semantic_metadata(
    source_loader,
    model,
    num_classes: int,
    reference_samples: int = 4096,
    bn_statistics: str = "frozen",
    disable_dropout: bool = True,
) -> Dict[str, object]:
    """Fit frozen class-mean feature references from labelled source data."""
    num_classes = int(num_classes)
    reference_samples = max(num_classes, int(reference_samples))
    bn_statistics = str(bn_statistics).strip().lower()
    if num_classes < 2:
        raise ValueError("Source semantic calibration requires at least two classes")
    if bn_statistics not in {"frozen", "batch"}:
        raise ValueError("bn_statistics must be 'frozen' or 'batch'")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Source semantic calibration requires model parameters") from exc
    calibration_model = copy.deepcopy(model).to(device)
    for parameter in calibration_model.parameters():
        parameter.requires_grad_(False)
    bn_calibration_samples = 0
    if bn_statistics == "frozen":
        bn_module_count = _prepare_source_semantic_bn_calibration(
            calibration_model, bool(disable_dropout)
        )
        # First pass: fit source-only BN buffers. Labels are deliberately not
        # read because this pass only defines the feature normalization state.
        for batch in source_loader:
            inputs = _extract_primary_tensor(batch)
            if not torch.is_tensor(inputs) or inputs.dim() != 3:
                raise ValueError(
                    "Source semantic calibration expects [B, C, T] tensors"
                )
            remaining = reference_samples - bn_calibration_samples
            if remaining <= 0:
                break
            inputs = inputs[:remaining].float().to(device)
            _extract_features(calibration_model, inputs)
            bn_calibration_samples += int(inputs.size(0))
        if bn_calibration_samples == 0:
            raise RuntimeError("Source semantic BN calibration received an empty loader")
        _freeze_batch_norm_statistics(calibration_model)
    else:
        calibration_model, device = _configure_calibration_copy(
            model, bn_statistics, bool(disable_dropout)
        )
        bn_module_count = sum(
            isinstance(module, _BATCH_NORM_TYPES)
            for module in calibration_model.feature_extractor.modules()
        )
    feature_sum = None
    class_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    collected = 0
    for batch in source_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Source semantic calibration requires source labels")
        inputs = _extract_primary_tensor(batch)
        labels = torch.as_tensor(batch[1], device=device).view(-1).long()
        if not torch.is_tensor(inputs) or inputs.dim() != 3:
            raise ValueError(
                "Source semantic calibration expects [B, C, T] tensors"
            )
        remaining = reference_samples - collected
        if remaining <= 0:
            break
        inputs = inputs[:remaining].float().to(device)
        labels = labels[:remaining]
        features = _normalized_feature_vectors(calibration_model, inputs)
        if feature_sum is None:
            feature_sum = torch.zeros(
                num_classes,
                features.size(1),
                dtype=features.dtype,
                device=device,
            )
        feature_sum.index_add_(0, labels, features)
        class_counts += torch.bincount(labels, minlength=num_classes)
        collected += labels.numel()
    if feature_sum is None or (class_counts == 0).any():
        missing = torch.where(class_counts == 0)[0].cpu().tolist()
        raise RuntimeError(
            f"Source semantic calibration has missing classes: {missing}"
        )
    prototypes = F.normalize(
        feature_sum / class_counts[:, None].to(feature_sum.dtype), dim=1
    )
    return {
        "version": SOURCE_SEMANTIC_METADATA_VERSION,
        "reference_samples": reference_samples,
        "bn_statistics": bn_statistics,
        "disable_dropout": bool(disable_dropout),
        "source_batch_size": int(getattr(source_loader, "batch_size", 0) or 0),
        "num_classes": num_classes,
        "prototypes": prototypes.cpu(),
        "class_counts": class_counts.cpu(),
        "bn_calibration_samples": int(bn_calibration_samples),
        "bn_module_count": int(bn_module_count),
        "feature_extractor_bn_state": (
            _source_semantic_bn_state(calibration_model.feature_extractor)
            if bn_statistics == "frozen"
            else {}
        ),
    }


class SSAWPhysicalView:
    """Draw a reproducible sensor-calibration view for the SSAW branch.

    A deployment batch receives one physical view that is reused by all inner
    adaptation steps.  This keeps the perturbation fixed while the model is
    updated, instead of injecting a new stochastic gradient direction at every
    step.  The default window-constant mode models a fixed calibration and
    sensor orientation during one acquisition window.  ``smooth`` remains
    available for controlled ablations.
    """

    _spline_geometry_cache: Dict[Tuple[object, ...], Dict[str, torch.Tensor]] = {}

    def __init__(
        self,
        num_control_points: int = 10,
        sigma: float = 0.20,
        sobol_seed: int = 1729,
        strength: float = 10.0,
        temporal_mode: str = "window_constant",
        antithetic: bool = False,
        antithetic_pairs: int = 1,
    ):
        self.num_control_points = max(2, int(num_control_points))
        self.sigma = float(sigma)
        if not 0.0 <= self.sigma < (1.0 / 3.0):
            raise ValueError("sigma must lie in [0, 1/3) to keep SSAW positive")
        self.sobol_seed = int(sobol_seed)
        self.strength = float(strength)
        if not 0.0 <= self.strength <= 90.0:
            raise ValueError("strength must lie in [0, 90] degrees")
        self.temporal_mode = str(temporal_mode).strip().lower()
        if self.temporal_mode not in {"window_constant", "smooth"}:
            raise ValueError(
                "temporal_mode must be 'window_constant' or 'smooth'"
            )
        self.antithetic = bool(antithetic)
        self.antithetic_pairs = int(antithetic_pairs)
        if self.antithetic_pairs < 1:
            raise ValueError("antithetic_pairs must be positive")
        self._sobol = torch.quasirandom.SobolEngine(
            dimension=self.num_control_points,
            scramble=True,
            seed=self.sobol_seed,
        )
        self._physical_call_index = 0
        self.last_warp_curve: Optional[torch.Tensor] = None
        self.last_view_inputs: Optional[torch.Tensor] = None
        self.last_stress_logits: Optional[torch.Tensor] = None
        self.last_stress_features: Optional[torch.Tensor] = None
        self.last_reference_logits: Optional[torch.Tensor] = None
        self.last_reference_features: Optional[torch.Tensor] = None
        self.last_metadata: Dict[str, torch.Tensor | str] = {}
        self._cached_view_inputs: Optional[torch.Tensor] = None
        self._cached_warp_curve: Optional[torch.Tensor] = None
        self._cached_controls: Optional[torch.Tensor] = None
        # Three-axis orientation views keep the sampled SO(3) transform so the
        # antithetic partner can use the exact inverse rotation.  The old
        # implementation reflected the normalized samples around the raw
        # tensor (``2*x-view``), which is only a first-order approximation for
        # a gain perturbation and is not a physically valid inverse rotation.
        self._cached_rotation_matrices: Optional[torch.Tensor] = None
        self._last_rotation_matrix: Optional[torch.Tensor] = None

    def clear_cached_view(self):
        """Start a new deployment batch with a fresh physical perturbation."""
        self._cached_view_inputs = None
        self._cached_warp_curve = None
        self._cached_controls = None
        self._cached_rotation_matrices = None

    @staticmethod
    def _axis_angle_to_matrix(angles: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle vectors to proper SO(3) matrices.

        ``angles[..., :]`` is an axis multiplied by the rotation angle in
        radians.  Rodrigues' formula keeps the perturbation norm explicit and
        avoids the large, order-dependent Euler rotations used previously.
        The returned matrices are orthogonal up to floating-point precision,
        so their transpose is the exact inverse used by the antithetic view.
        """
        if angles.shape[-1] != 3:
            raise ValueError("axis-angle vectors must have a final dimension of 3")
        theta = torch.linalg.vector_norm(angles, dim=-1, keepdim=True)
        theta_sq = theta.square()
        # Skew-symmetric cross-product matrix for each axis-angle vector.
        x, y, z = angles.unbind(dim=-1)
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            (
                zeros,
                -z,
                y,
                z,
                zeros,
                -x,
                -y,
                x,
                zeros,
            ),
            dim=-1,
        ).reshape(*angles.shape[:-1], 3, 3)
        skew_sq = skew @ skew
        # Taylor limits are needed only at the origin; using the first two
        # terms avoids NaNs while preserving gradients for small angles.
        theta_safe = theta.clamp_min(torch.finfo(angles.dtype).eps)
        a = torch.sin(theta_safe) / theta_safe
        b = (1.0 - torch.cos(theta_safe)) / theta_sq.clamp_min(
            torch.finfo(angles.dtype).eps
        )
        eye = torch.eye(3, device=angles.device, dtype=angles.dtype)
        eye = eye.expand(*angles.shape[:-1], 3, 3)
        return eye + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * skew_sq

    @staticmethod
    def _rotate_physical(
        inputs: torch.Tensor,
        rotation: torch.Tensor,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one SO(3) transform to every complete sensor triad."""
        batch_size, channels, target_len = inputs.shape
        triad_channels = (channels // 3) * 3
        if triad_channels == 0:
            return inputs.clone()
        physical = (
            inputs * normalization_std[None, :, None]
            + normalization_mean[None, :, None]
        )
        triads = physical[:, :triad_channels].reshape(
            batch_size,
            triad_channels // 3,
            3,
            target_len,
        ).permute(0, 1, 3, 2)
        rotated = torch.einsum("btij,bktj->bkti", rotation, triads)
        rotated = rotated.permute(0, 1, 3, 2).reshape(
            batch_size,
            triad_channels,
            target_len,
        )
        physical = torch.cat((rotated, physical[:, triad_channels:]), dim=1)
        return (
            physical - normalization_mean[None, :, None]
        ) / normalization_std[None, :, None]

    def _physical_standard_processes(
        self,
        batch_size: int,
        process_count: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw deterministic smooth N(0,1) trajectories for one stream batch."""
        engine = torch.quasirandom.SobolEngine(
            dimension=self.num_control_points,
            scramble=True,
            seed=self.sobol_seed + 1009 * self._physical_call_index,
        )
        self._physical_call_index += 1
        sample_count = batch_size * process_count
        uniforms = engine.draw(sample_count).clamp(1e-7, 1.0 - 1e-7)
        controls = (
            torch.erfinv(2.0 * uniforms - 1.0) * math.sqrt(2.0)
        ).clamp(-3.0, 3.0).to(device=device, dtype=dtype)
        trajectories = self._natural_cubic_spline_upsample(
            controls, target_len
        ).reshape(
            batch_size,
            process_count,
            target_len,
        )
        controls = controls.reshape(
            batch_size,
            process_count,
            self.num_control_points,
        )
        return trajectories, controls

    def _sensor_calibration_view(
        self,
        inputs: torch.Tensor,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate one gain and orientation calibration perturbation.

        Complete three-axis groups use a bounded axis-angle perturbation.  The
        scalar ``strength`` is the maximum *total* orientation error in
        degrees, rather than an independent Euler standard deviation per
        axis.  This gives a direct physical bound and avoids the old
        ``sqrt(3) * 33``-degree typical error (with nearly 100-degree tails).
        The sampled matrix is retained for constructing an exact inverse view
        in ``__call__``.
        """
        batch_size, channels, target_len = inputs.shape
        self._last_rotation_matrix = None
        if self.sigma == 0.0 and self.strength == 0.0:
            curve = torch.ones_like(inputs)
            controls = inputs.new_ones(
                batch_size, channels, self.num_control_points
            )
            # Avoid a denormalize/renormalize round trip: its sub-ULP error is
            # observable after many clipped online updates on EEG.
            return inputs.clone(), curve, controls
        gain, gain_controls = self._sample_curves(
            batch_size,
            channels,
            target_len,
            inputs.device,
            inputs.dtype,
        )
        raw = (
            inputs * normalization_std[None, :, None]
            + normalization_mean[None, :, None]
        )
        physical = raw * gain

        triad_channels = (channels // 3) * 3
        if triad_channels:
            trajectories, _ = self._physical_standard_processes(
                batch_size,
                # Three values determine the rotation axis.  A fourth smooth
                # process supplies a bounded radial magnitude, so all
                # perturbations are inside the configured SO(3) ball instead
                # of clipping three independent Euler angles.
                4,
                target_len if self.temporal_mode == "smooth" else 2,
                inputs.device,
                inputs.dtype,
            )
            if self.temporal_mode == "window_constant":
                trajectories = trajectories[..., :1].expand(
                    -1, -1, target_len
                )
            axis = trajectories[:, :3].permute(0, 2, 1)
            radial = torch.sigmoid(trajectories[:, 3]).unsqueeze(-1)
            angle_radius = radial * math.radians(self.strength)
            axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
            # A zero axis is exceptionally unlikely with Sobol draws; use a
            # fixed x-axis in that case so calibration remains deterministic.
            safe_axis = axis / axis_norm.clamp_min(1e-6)
            fallback = torch.zeros_like(safe_axis)
            fallback[..., 0] = 1.0
            safe_axis = torch.where(
                axis_norm > 1e-6, safe_axis, fallback
            )
            angles = safe_axis * angle_radius
            rotation = self._axis_angle_to_matrix(angles)
            self._last_rotation_matrix = rotation.detach().clone()
            gain_inputs = (
                physical - normalization_mean[None, :, None]
            ) / normalization_std[None, :, None]
            view = self._rotate_physical(
                gain_inputs,
                rotation,
                normalization_mean,
                normalization_std,
            )
            return view, gain, gain_controls

        view = (
            physical - normalization_mean[None, :, None]
        ) / normalization_std[None, :, None]
        return view, gain, gain_controls

    @staticmethod
    def _natural_cubic_spline_upsample(
        controls: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        if controls.dim() != 2:
            raise ValueError(
                f"Expected control tensor [N, M], got {tuple(controls.shape)}"
            )
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        if controls.size(1) == 1:
            return controls.repeat(1, target_len)
        original_dtype = controls.dtype
        work_dtype = (
            torch.float64 if original_dtype == torch.float64 else torch.float32
        )
        values = controls.to(dtype=work_dtype)
        sample_count, control_count = values.shape
        device = values.device
        geometry_key = (
            int(control_count),
            int(target_len),
            device.type,
            device.index,
            work_dtype,
        )
        geometry = SSAWPhysicalView._spline_geometry_cache.get(geometry_key)
        if geometry is None:
            control_x = torch.linspace(
                0.0,
                float(target_len - 1),
                control_count,
                device=device,
                dtype=work_dtype,
            )
            interval = control_x[1:] - control_x[:-1]
            system = torch.empty(0, device=device, dtype=work_dtype)
            if control_count > 2:
                system = torch.zeros(
                    control_count - 2,
                    control_count - 2,
                    device=device,
                    dtype=work_dtype,
                )
                system.diagonal().copy_(
                    2.0 * (interval[:-1] + interval[1:])
                )
                if control_count > 3:
                    system.diagonal(offset=1).copy_(interval[1:-1])
                    system.diagonal(offset=-1).copy_(interval[1:-1])
            evaluation_x = torch.linspace(
                0.0,
                float(target_len - 1),
                target_len,
                device=device,
                dtype=work_dtype,
            )
            indices = torch.bucketize(
                evaluation_x, control_x[1:-1], right=False
            ).clamp(max=control_count - 2)
            x0 = control_x[indices]
            x1 = control_x[indices + 1]
            width = x1 - x0
            geometry = {
                "interval": interval,
                "system": system,
                "indices": indices,
                "width": width,
                "left": x1 - evaluation_x,
                "right": evaluation_x - x0,
            }
            SSAWPhysicalView._spline_geometry_cache[geometry_key] = geometry
        interval = geometry["interval"]
        system = geometry["system"]
        second = torch.zeros(
            sample_count, control_count, device=device, dtype=work_dtype
        )
        if control_count > 2:
            rhs = 6.0 * (
                (values[:, 2:] - values[:, 1:-1]) / interval[1:]
                - (values[:, 1:-1] - values[:, :-2]) / interval[:-1]
            )
            second[:, 1:-1] = torch.linalg.solve(
                system.unsqueeze(0).expand(sample_count, -1, -1), rhs
            )
        indices = geometry["indices"]
        width = geometry["width"]
        left = geometry["left"]
        right = geometry["right"]
        value0 = values[:, indices]
        value1 = values[:, indices + 1]
        second0 = second[:, indices]
        second1 = second[:, indices + 1]
        curves = (
            second0 * left.pow(3) / (6.0 * width)
            + second1 * right.pow(3) / (6.0 * width)
            + (value0 - second0 * width.pow(2) / 6.0) * left / width
            + (value1 - second1 * width.pow(2) / 6.0) * right / width
        )
        return curves.to(dtype=original_dtype)

    @staticmethod
    @contextmanager
    def _preserved_bn_buffers(model):
        # BatchNorm buffers can change only in training mode when running
        # statistics are tracked.  Production batch-statistics BN disables
        # tracking, while frozen-source BN is in eval mode; cloning and
        # restoring buffers in either case is dead work.
        if not DuSafe._bn_buffers_may_update(model):
            yield
            return
        snapshots = DuSafe._snapshot_bn_buffers(model)
        try:
            yield
        finally:
            DuSafe._restore_bn_buffers(snapshots)

    def _sample_curves(
        self,
        batch_size: int,
        channels: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if channels < 1:
            raise ValueError("channels must be positive")
        # Each acquisition channel has its own smooth calibration drift. This
        # is the same transform for scalar sensors and every complete
        # three-axis group; no dataset-specific algorithm branch is required.
        group_count = channels
        if self.sigma == 0.0:
            controls = torch.ones(
                batch_size,
                group_count,
                self.num_control_points,
                device=device,
                dtype=dtype,
            )
            group_curves = torch.ones(
                batch_size,
                group_count,
                target_len,
                device=device,
                dtype=dtype,
            )
            return group_curves, controls

        sample_count = batch_size * group_count
        uniforms = self._sobol.draw(sample_count)
        uniforms = uniforms.clamp(1e-7, 1.0 - 1e-7)
        standard_normal = (
            torch.erfinv(2.0 * uniforms - 1.0) * math.sqrt(2.0)
        ).to(device=device, dtype=dtype)
        if self.temporal_mode == "window_constant":
            standard_normal = standard_normal[:, :1].expand(
                -1, self.num_control_points
            )
        lower = 1.0 - 3.0 * self.sigma
        upper = 1.0 + 3.0 * self.sigma
        # Eq. (2) samples N(1, sigma^2 I) and clips, rather than sampling
        # from the corresponding conditional truncated distribution.
        controls = (1.0 + self.sigma * standard_normal).clamp(lower, upper)
        group_curves = self._natural_cubic_spline_upsample(
            controls, target_len
        ).clamp(lower, upper)
        group_curves = group_curves.reshape(
            batch_size, group_count, target_len
        )
        controls = controls.reshape(
            batch_size,
            group_count,
            self.num_control_points,
        )
        return group_curves, controls

    @torch.no_grad()
    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> Dict[str, torch.Tensor | bool]:
        """Generate the physical views without invoking the deployed model.

        Separating signal generation from model evaluation lets DuSafe reuse
        the same raw/view forwards for both detached safety decisions and the
        differentiable update objective.  The public ``__call__`` path below
        remains available as the legacy evaluate-and-record interface.
        """
        if inputs.dim() != 3:
            raise ValueError(f"Expected input [B, C, T], got {tuple(inputs.shape)}")
        batch_size, channels, target_len = inputs.shape
        if normalization_mean is None or normalization_std is None:
            raise RuntimeError(
                "SSAW requires fixed source normalization mean and std"
            )
        normalization_mean = torch.as_tensor(
            normalization_mean, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        normalization_std = torch.as_tensor(
            normalization_std, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        if (
            normalization_mean.numel() != channels
            or normalization_std.numel() != channels
            or not normalization_std.gt(0.0).all()
        ):
            raise ValueError(
                "source normalization mean/std must contain one valid "
                "value per input channel"
            )
        if reuse_cached_view:
            if (
                self._cached_view_inputs is None
                or self._cached_warp_curve is None
                or self._cached_controls is None
                or (
                    self._cached_rotation_matrices is not None
                    and (
                        self._cached_rotation_matrices.size(1) != batch_size
                        or self._cached_rotation_matrices.size(2) != target_len
                    )
                )
                or (
                    self._cached_view_inputs.dim() == 3
                    and tuple(self._cached_view_inputs.shape)
                    != tuple(inputs.shape)
                )
                or (
                    self._cached_view_inputs.dim() == 4
                    and tuple(self._cached_view_inputs.shape[1:])
                    != tuple(inputs.shape)
                )
            ):
                raise RuntimeError(
                    "SSAW cached view is unavailable or has a different shape"
                )
            positive_views = self._cached_view_inputs.to(
                device=inputs.device, dtype=inputs.dtype
            )
            curves = self._cached_warp_curve.to(
                device=inputs.device, dtype=inputs.dtype
            )
            controls_by_view = self._cached_controls.to(
                device=inputs.device, dtype=inputs.dtype
            )
            rotation_matrices = (
                None
                if self._cached_rotation_matrices is None
                else self._cached_rotation_matrices.to(
                    device=inputs.device, dtype=inputs.dtype
                )
            )
        else:
            pair_count = self.antithetic_pairs if self.antithetic else 1
            generated = []
            rotations = []
            # ``_sensor_calibration_view`` keeps the sampled SO(3) matrix as
            # an implementation detail so the public three-tensor return
            # contract remains compatible with the structural ablation code.
            # A None entry means this is a scalar gain view (EEG/FD), for
            # which the historical centered reflection remains valid.
            for _ in range(pair_count):
                item = self._sensor_calibration_view(
                    inputs, normalization_mean, normalization_std
                )
                generated.append(item)
                last_rotation = getattr(self, "_last_rotation_matrix", None)
                rotations.append(
                    None
                    if last_rotation is None
                    else last_rotation.detach().clone()
                )
            positive_views = torch.stack([item[0] for item in generated])
            curves = torch.stack([item[1] for item in generated])
            controls_by_view = torch.stack([item[2] for item in generated])
            if not self.antithetic:
                positive_views = positive_views.squeeze(0)
                curves = curves.squeeze(0)
                controls_by_view = controls_by_view.squeeze(0)
            rotation_matrices = (
                None
                if not rotations or any(item is None for item in rotations)
                else torch.stack(rotations)
            )
            self._cached_view_inputs = positive_views.detach().clone()
            self._cached_warp_curve = curves.detach().clone()
            self._cached_controls = controls_by_view.detach().clone()
            self._cached_rotation_matrices = (
                None
                if rotation_matrices is None
                else rotation_matrices.detach().clone()
            )
        if positive_views.dim() == 3:
            positive_views = positive_views.unsqueeze(0)
            curves = curves.unsqueeze(0)
            controls_by_view = controls_by_view.unsqueeze(0)
        warped_inputs = positive_views[0]
        if self.antithetic:
            # Reflection around the raw signal is exactly antithetic for gain
            # calibration.  For complete three-axis orientation views, use
            # R^T x instead: it is an
            # exact SO(3) inverse and preserves each physical vector norm.
            inverse_views = []
            use_exact_inverse = (
                rotation_matrices is not None
                and self.sigma == 0.0
                and rotation_matrices.size(0) == positive_views.size(0)
            )
            for pair_index, positive in enumerate(positive_views):
                if use_exact_inverse:
                    inverse_views.append(
                        self._rotate_physical(
                            inputs,
                            rotation_matrices[pair_index].transpose(-1, -2),
                            normalization_mean,
                            normalization_std,
                        )
                    )
                else:
                    inverse_views.append(2.0 * inputs - positive)
            view_inputs = torch.stack(
                tuple(positive_views) + tuple(inverse_views), dim=0
            )
        else:
            view_inputs = positive_views
        return {
            "warped_inputs": warped_inputs,
            "view_inputs": view_inputs,
            "curves": curves,
            "controls_by_view": controls_by_view,
            "reused_view": bool(reuse_cached_view),
        }

    @torch.no_grad()
    def record_evaluation(
        self,
        *,
        reference_logits: torch.Tensor,
        reference_features: torch.Tensor,
        candidate_logits_by_view: torch.Tensor,
        candidate_features_by_view: torch.Tensor,
        prepared_views: Mapping[str, object],
    ) -> None:
        """Record SSAW diagnostics from already-computed raw/view outputs."""
        reference_logits = reference_logits.detach()
        reference_features = reference_features.detach()
        candidate_logits_by_view = candidate_logits_by_view.detach()
        candidate_features_by_view = candidate_features_by_view.detach()
        view_inputs = torch.as_tensor(prepared_views["view_inputs"])
        warped_inputs = torch.as_tensor(prepared_views["warped_inputs"])
        curves = torch.as_tensor(prepared_views["curves"])
        controls_by_view = torch.as_tensor(prepared_views["controls_by_view"])
        reuse_cached_view = bool(prepared_views["reused_view"])
        candidate_features = candidate_features_by_view.mean(dim=0)
        candidate_logits = candidate_logits_by_view.mean(dim=0)
        raw_entropy = _entropy_from_logits(reference_logits)
        _, raw_labels = _top1_nll(reference_logits)
        raw_log_probabilities = reference_logits.log_softmax(dim=1)
        raw_probabilities = raw_log_probabilities.exp()
        candidate_log_probabilities_by_view = (
            candidate_logits_by_view.log_softmax(dim=2)
        )
        candidate_probabilities_by_view = (
            candidate_log_probabilities_by_view.exp()
        )
        candidate_entropy_by_view = -(
            candidate_probabilities_by_view
            * candidate_log_probabilities_by_view
        ).sum(dim=2)
        candidate_entropy = candidate_entropy_by_view.mean(dim=0)
        candidate_labels_by_view = candidate_logits_by_view.argmax(dim=2)
        label_preserving_by_view = candidate_labels_by_view.eq(
            raw_labels.unsqueeze(0)
        )
        model_label_preserving = label_preserving_by_view.all(dim=0)
        candidate_kl_by_view = (
            raw_probabilities.unsqueeze(0)
            * (
                raw_log_probabilities.unsqueeze(0)
                - candidate_log_probabilities_by_view
            )
        ).sum(dim=2)
        candidate_kl = candidate_kl_by_view.mean(dim=0)

        self.last_warp_curve = (
            curves.detach().cpu()
            if self.antithetic
            else curves[0].detach().cpu()
        )
        self.last_view_inputs = (
            view_inputs.detach()
            if self.antithetic
            else warped_inputs.detach()
        )
        self.last_reference_logits = reference_logits.detach()
        self.last_reference_features = reference_features.detach()
        self.last_stress_logits = candidate_logits.detach()
        self.last_stress_features = candidate_features.detach()
        self.last_metadata = {
            "mode": "ssaw_fixed_batch_sensor_calibration",
            "transform_family": "sensor_calibration",
            "temporal_mode": self.temporal_mode,
            "antithetic": self.antithetic,
            "view_count": int(view_inputs.size(0)),
            "reused_view": bool(reuse_cached_view),
            "curve": self.last_warp_curve,
            "control_points": (
                controls_by_view.detach().cpu()
                if self.antithetic
                else controls_by_view[0].detach().cpu()
            ),
            "ssaw_view_selected": model_label_preserving.detach().cpu(),
            "selected_nll": (-candidate_log_probabilities_by_view.gather(
                2,
                raw_labels[None, :, None].expand(view_inputs.size(0), -1, -1),
            ).squeeze(2).mean(dim=0)).detach().cpu(),
            "selected_kl": candidate_kl.detach().cpu(),
            "selected_kl_by_view": candidate_kl_by_view.detach().cpu(),
            "vote_agreement": model_label_preserving.float().cpu(),
            "label_preserving_count": label_preserving_by_view.sum(dim=0).cpu(),
            "entropy_rise": (candidate_entropy - raw_entropy).detach().cpu(),
            "entropy_rise_by_view": (
                candidate_entropy_by_view - raw_entropy.unsqueeze(0)
            ).detach().cpu(),
            "ssaw_label_flip": (~model_label_preserving).cpu(),
            "ssaw_label_flip_by_view": (~label_preserving_by_view).cpu(),
        }

    @torch.no_grad()
    def __call__(
        self,
        inputs: torch.Tensor,
        model,
        reference_logits: Optional[torch.Tensor] = None,
        reference_features: Optional[torch.Tensor] = None,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> torch.Tensor:
        if reference_logits is None:
            with self._preserved_bn_buffers(model):
                reference_features = _extract_features(model, inputs)
                reference_logits = model.classifier(reference_features)
        if reference_features is None:
            raise ValueError("reference_features are required with reference_logits")
        prepared_views = self.prepare_view_inputs(
            inputs,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            reuse_cached_view=reuse_cached_view,
        )
        candidate_feature_views = []
        candidate_logit_views = []
        for current_view in torch.as_tensor(prepared_views["view_inputs"]):
            with self._preserved_bn_buffers(model):
                current_features = _extract_features(model, current_view)
                current_logits = model.classifier(current_features)
            candidate_feature_views.append(current_features)
            candidate_logit_views.append(current_logits)
        self.record_evaluation(
            reference_logits=reference_logits,
            reference_features=reference_features,
            candidate_logits_by_view=torch.stack(candidate_logit_views),
            candidate_features_by_view=torch.stack(candidate_feature_views),
            prepared_views=prepared_views,
        )
        warped_inputs = torch.as_tensor(prepared_views["warped_inputs"])
        return warped_inputs


def evaluate_candidate_pool_sequential(
    model,
    view_inputs: torch.Tensor,
    *,
    require_grad: bool,
    retain_features: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Evaluate each physical candidate as its own ``[B,C,T]`` BN batch.

    Candidate outputs must not depend on how many other views happen to be in
    the pool.  This helper intentionally forbids flattening ``[V,B]`` into a
    single super-batch.
    """
    view_inputs = torch.as_tensor(view_inputs)
    if view_inputs.dim() != 4:
        raise ValueError("candidate inputs must have shape [V, B, C, T]")
    feature_parts = [] if retain_features else None
    logit_parts = []
    # Search-only candidates are re-forwarded after gathering for the actual
    # differentiable objective. Inference mode therefore removes autograd
    # bookkeeping without changing any tensor used by the update graph.
    gradient_context = nullcontext() if require_grad else torch.inference_mode()
    with gradient_context:
        for current_view in view_inputs:
            with SSAWPhysicalView._preserved_bn_buffers(model):
                current_features = _extract_features(model, current_view)
                current_logits = model.classifier(current_features)
            if feature_parts is not None:
                feature_parts.append(current_features)
            logit_parts.append(current_logits)
    return (
        None if feature_parts is None else torch.stack(feature_parts),
        torch.stack(logit_parts),
    )


class _ExactLevelZeroCandidateCudaGraph:
    """Replay the mandatory largest-radius rays without changing BN batches.

    The captured graph contains ``ray_count`` *sequential* model invocations.
    It never flattens rays into a super-batch: every BatchNorm invocation still
    receives exactly one ``[B,C,T]`` candidate.  The first search remains eager
    and supplies the reference logits used by the bitwise self-test.  A graph
    is used only by later searches after that self-test passes.

    This is an execution cache, not model state.  It intentionally is neither
    an ``nn.Module`` nor a registered buffer and therefore never enters a
    checkpoint.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        requested_mode: str,
        max_static_input_mb: float = float("inf"),
    ):
        self.enabled = bool(enabled)
        self.requested_mode = str(requested_mode).strip().lower()
        self.max_static_input_bytes = float(max_static_input_mb) * 1024 * 1024
        self.status = "uninitialized" if self.enabled else "disabled"
        self.setup_ms = 0.0
        self.eager_probe_ms = float("nan")
        self.graph_probe_ms = float("nan")
        self.speedup = float("nan")
        self.capture_count = 0
        self.replay_count = 0
        self.eager_fallback_count = 0
        self.exact_self_test_count = 0
        self.post_update_self_test_passed = False
        self.last_level_zero_used = False
        self.last_event = self.status
        self.expected_full_batch_searches = None
        self._graph = None
        self._capture_stream = None
        self._static_inputs = None
        self._static_logits = None
        self._input_fingerprint = None
        self._model_fingerprint = None
        self._module_flags = None
        self._captured_parameter_versions = None

    @staticmethod
    def _tensor_fingerprint(tensor: torch.Tensor):
        return (
            id(tensor),
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    @classmethod
    def _fingerprint_model(cls, model):
        return (
            tuple(cls._tensor_fingerprint(value) for value in model.parameters()),
            tuple(cls._tensor_fingerprint(value) for value in model.buffers()),
        )

    @staticmethod
    def _parameter_versions(model):
        return tuple(int(parameter._version) for parameter in model.parameters())

    @staticmethod
    def _model_flags(model):
        flags = []
        for module in model.modules():
            flags.append(
                (
                    id(module),
                    bool(module.training),
                    (
                        bool(module.track_running_stats)
                        if isinstance(module, _BATCH_NORM_TYPES)
                        else None
                    ),
                )
            )
        return tuple(flags)

    @staticmethod
    def _has_module_hooks(model) -> bool:
        hook_names = (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
        return any(
            any(bool(getattr(module, name, {})) for name in hook_names)
            for module in model.modules()
        )

    @staticmethod
    def _has_unreviewed_module_buffers(model) -> bool:
        """Reject state whose forward mutation semantics were not audited."""

        batch_norm_buffers = {"running_mean", "running_var", "num_batches_tracked"}
        for module in model.modules():
            local_names = {
                name for name, _ in module.named_buffers(recurse=False)
            }
            if isinstance(module, _BATCH_NORM_TYPES):
                if not local_names.issubset(batch_norm_buffers):
                    return True
            elif local_names:
                return True
        return False

    @staticmethod
    def _autocast_enabled() -> bool:
        enabled = bool(torch.is_autocast_enabled())
        try:
            enabled = enabled or bool(torch.is_autocast_enabled("cuda"))
        except TypeError:
            pass
        return enabled

    @staticmethod
    def _forward_block(model, inputs: torch.Tensor) -> torch.Tensor:
        logits = []
        for current_view in inputs:
            current_features = _extract_features(model, current_view)
            logits.append(model.classifier(current_features))
        return torch.stack(logits)

    def _safe_to_capture(self, model, inputs: torch.Tensor) -> bool:
        if not self.enabled:
            return False
        if not torch.cuda.is_available() or inputs.device.type != "cuda":
            self.status = "disabled_non_cuda"
            return False
        static_input_bytes = int(inputs.numel() * inputs.element_size())
        if static_input_bytes > self.max_static_input_bytes:
            self.status = "disabled_static_input_above_limit"
            self.last_event = self.status
            self.enabled = False
            return False
        if inputs.requires_grad or self._autocast_enabled():
            self.status = "disabled_grad_or_autocast"
            return False
        if torch.cuda.is_current_stream_capturing():
            self.status = "disabled_nested_capture"
            return False
        if DuSafe._bn_buffers_may_update(model):
            self.status = "disabled_stateful_batch_norm"
            return False
        if any(
            isinstance(
                module,
                (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
            )
            and bool(module.training)
            for module in model.modules()
        ):
            self.status = "disabled_training_dropout"
            return False
        if self._has_module_hooks(model):
            self.status = "disabled_module_hooks"
            return False
        if self._has_unreviewed_module_buffers(model):
            self.status = "disabled_unreviewed_module_buffers"
            return False
        return True

    def _storage_is_valid(self, model, inputs: torch.Tensor) -> bool:
        return bool(
            self._graph is not None
            and self._input_fingerprint
            == (
                tuple(inputs.shape),
                tuple(inputs.stride()),
                inputs.dtype,
                inputs.device,
            )
            and self._model_fingerprint == self._fingerprint_model(model)
            and self._module_flags == self._model_flags(model)
        )

    def diagnostics(self) -> Dict[str, object]:
        return {
            "candidate_cuda_graph_requested_mode": self.requested_mode,
            "candidate_cuda_graph_enabled": bool(self.enabled),
            "candidate_cuda_graph_status": self.status,
            "candidate_cuda_graph_last_event": self.last_event,
            "candidate_cuda_graph_setup_ms": float(self.setup_ms),
            "candidate_cuda_graph_eager_probe_ms": float(self.eager_probe_ms),
            "candidate_cuda_graph_graph_probe_ms": float(self.graph_probe_ms),
            "candidate_cuda_graph_speedup": float(self.speedup),
            "candidate_cuda_graph_capture_count": int(self.capture_count),
            "candidate_cuda_graph_replay_count": int(self.replay_count),
            "candidate_cuda_graph_eager_fallback_count": int(
                self.eager_fallback_count
            ),
            "candidate_cuda_graph_exact_self_test_count": int(
                self.exact_self_test_count
            ),
            "candidate_cuda_graph_post_update_self_test_passed": bool(
                self.post_update_self_test_passed
            ),
            "candidate_cuda_graph_used_last_search": bool(
                self.last_level_zero_used
            ),
            "candidate_cuda_graph_expected_full_batch_searches": (
                self.expected_full_batch_searches
            ),
        }

    def _clear_graph(self) -> None:
        graph = self._graph
        self._graph = None
        if graph is not None:
            try:
                graph.reset()
            except Exception:
                pass
        self._capture_stream = None
        self._static_inputs = None
        self._static_logits = None
        self._input_fingerprint = None
        self._model_fingerprint = None
        self._module_flags = None
        self._captured_parameter_versions = None

    def replay(self, model, level_zero_inputs: torch.Tensor):
        """Return static graph logits, or ``None`` for the exact eager path."""

        self.last_level_zero_used = False
        if self._graph is None:
            self.last_event = "eager_graph_not_ready"
            self.eager_fallback_count += 1
            return None
        if not self._safe_to_capture(model, level_zero_inputs):
            self.last_event = self.status
            self.eager_fallback_count += 1
            return None
        input_signature = (
            tuple(level_zero_inputs.shape),
            tuple(level_zero_inputs.stride()),
            level_zero_inputs.dtype,
            level_zero_inputs.device,
        )
        if input_signature != self._input_fingerprint:
            # A short final batch must not destroy the graph for the registered
            # full-batch shape. It follows the historical eager path exactly.
            self.last_event = "eager_shape_or_dtype_mismatch"
            self.eager_fallback_count += 1
            return None
        if not self._storage_is_valid(model, level_zero_inputs):
            self.status = "invalidated_model_storage_or_mode"
            self.last_event = self.status
            self._clear_graph()
            self.eager_fallback_count += 1
            return None
        if (
            not self.post_update_self_test_passed
            and self._captured_parameter_versions
            != self._parameter_versions(model)
        ):
            # The first search after an actual optimizer update remains eager.
            # ``prepare`` replays the graph on the same inputs and requires a
            # bitwise match before all later updated weights may use it.
            self.status = "awaiting_post_update_exact_self_test"
            self.last_event = self.status
            self.eager_fallback_count += 1
            return None
        try:
            with torch.inference_mode():
                self._static_inputs.copy_(level_zero_inputs)
                self._graph.replay()
            self.last_level_zero_used = True
            self.replay_count += 1
            self.status = "replayed_level_zero"
            self.last_event = self.status
            return self._static_logits
        except RuntimeError as error:
            # A normal capture/replay error falls back to the historical exact
            # eager path.  CUDA illegal-access errors are not recoverable and
            # must not be hidden by a broad exception handler.
            message = str(error).lower()
            if "illegal memory access" in message or "access violation" in message:
                raise
            self.status = "disabled_after_replay_failure"
            self.last_event = self.status
            self.enabled = False
            self._clear_graph()
            self.eager_fallback_count += 1
            return None

    @staticmethod
    def _cuda_elapsed_ms(device, operation) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def _validate_post_update(
        self,
        model,
        level_zero_inputs: torch.Tensor,
        eager_logits: torch.Tensor,
    ) -> None:
        with torch.inference_mode():
            self._static_inputs.copy_(level_zero_inputs)
            self._graph.replay()
        torch.cuda.synchronize(level_zero_inputs.device)
        self.exact_self_test_count += 1
        if not torch.equal(eager_logits, self._static_logits):
            self.status = "disabled_post_update_exact_self_test_failed"
            self.enabled = False
            self._clear_graph()
            return
        self.post_update_self_test_passed = True
        self.status = "ready_post_update_exact_self_test_passed"
        self.last_event = self.status

    def prepare(
        self,
        model,
        level_zero_inputs: torch.Tensor,
        eager_logits: torch.Tensor,
    ) -> None:
        """Capture once and enable replay only after a bitwise self-test."""

        if self._graph is not None:
            if (
                not self.post_update_self_test_passed
                and self.status == "awaiting_post_update_exact_self_test"
                and self._storage_is_valid(model, level_zero_inputs)
            ):
                self._validate_post_update(
                    model, level_zero_inputs, eager_logits
                )
            return
        if self.status.startswith("capture_failed") or not self.enabled:
            return
        if not self._safe_to_capture(model, level_zero_inputs):
            return
        start = time.perf_counter()
        graph = None
        try:
            static_inputs = torch.empty_like(level_zero_inputs)
            static_inputs.copy_(level_zero_inputs)
            current_stream = torch.cuda.current_stream(level_zero_inputs.device)
            capture_stream = torch.cuda.Stream(device=level_zero_inputs.device)
            capture_stream.wait_stream(current_stream)
            # PyTorch requires capture workloads to be warmed on a side stream.
            # The warm-up is state-free because capture was already rejected for
            # running-stat BN, training dropout, hooks, and autocast.
            warmup_start = torch.cuda.Event(enable_timing=True)
            warmup_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(capture_stream), torch.inference_mode():
                warmup_start.record()
                warmup_logits = self._forward_block(model, static_inputs)
                warmup_end.record()
            warmup_end.synchronize()
            initial_eager_probe_ms = float(
                warmup_start.elapsed_time(warmup_end)
            )
            if not torch.equal(eager_logits, warmup_logits):
                self.status = "disabled_side_stream_exact_self_test_failed"
                self.enabled = False
                return
            del warmup_logits
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(capture_stream), torch.inference_mode():
                graph.capture_begin()
                static_logits = self._forward_block(model, static_inputs)
                graph.capture_end()
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(level_zero_inputs.device)

            # Test an actual replay, not only the capture-time execution.
            graph_probe_start = torch.cuda.Event(enable_timing=True)
            graph_probe_end = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                graph_probe_start.record()
                static_inputs.copy_(level_zero_inputs)
                graph.replay()
                graph_probe_end.record()
            graph_probe_end.synchronize()
            initial_graph_probe_ms = float(
                graph_probe_start.elapsed_time(graph_probe_end)
            )
            self.exact_self_test_count += 1
            if not torch.equal(eager_logits, static_logits):
                self.status = "disabled_exact_self_test_failed"
                self.enabled = False
                try:
                    graph.reset()
                finally:
                    graph = None
                return

            if self.requested_mode == "auto":
                eager_times = [initial_eager_probe_ms]
                graph_times = [initial_graph_probe_ms]
                preliminary_speedup = initial_eager_probe_ms / max(
                    initial_graph_probe_ms, 1e-12
                )
                # Clear wins need no further setup work. Only an ambiguous
                # first timing receives two more interleaved probes.
                extra_probes = 2 if preliminary_speedup < 1.25 else 0
                for _ in range(extra_probes):
                    eager_holder = []

                    def eager_operation():
                        with torch.inference_mode():
                            eager_holder.append(
                                self._forward_block(model, level_zero_inputs)
                            )

                    eager_times.append(
                        self._cuda_elapsed_ms(
                            level_zero_inputs.device, eager_operation
                        )
                    )
                    if not torch.equal(eager_logits, eager_holder.pop()):
                        self.status = "disabled_eager_probe_exact_test_failed"
                        self.enabled = False
                        graph.reset()
                        graph = None
                        return

                    def graph_operation():
                        with torch.inference_mode():
                            static_inputs.copy_(level_zero_inputs)
                            graph.replay()

                    graph_times.append(
                        self._cuda_elapsed_ms(
                            level_zero_inputs.device, graph_operation
                        )
                    )
                    if not torch.equal(eager_logits, static_logits):
                        self.status = "disabled_graph_probe_exact_test_failed"
                        self.enabled = False
                        graph.reset()
                        graph = None
                        return
                eager_times.sort()
                graph_times.sort()
                self.eager_probe_ms = float(eager_times[len(eager_times) // 2])
                self.graph_probe_ms = float(graph_times[len(graph_times) // 2])
                self.speedup = self.eager_probe_ms / max(
                    self.graph_probe_ms, 1e-12
                )
                # Avoid carrying a graph whose gain is inside timing noise.
                if self.speedup < 1.05:
                    self.status = "disabled_no_material_speedup"
                    self.enabled = False
                    graph.reset()
                    graph = None
                    return

            self._graph = graph
            self._capture_stream = capture_stream
            self._static_inputs = static_inputs
            self._static_logits = static_logits
            self._input_fingerprint = (
                tuple(level_zero_inputs.shape),
                tuple(level_zero_inputs.stride()),
                level_zero_inputs.dtype,
                level_zero_inputs.device,
            )
            self._model_fingerprint = self._fingerprint_model(model)
            self._module_flags = self._model_flags(model)
            self._captured_parameter_versions = self._parameter_versions(model)
            self.capture_count += 1
            self.status = "ready_exact_self_test_passed"
            self.last_event = self.status
        except torch.cuda.OutOfMemoryError:
            self.status = "capture_failed_oom"
            self.enabled = False
            if graph is not None:
                try:
                    graph.reset()
                except Exception:
                    pass
        except RuntimeError as error:
            message = str(error).lower()
            if "illegal memory access" in message or "access violation" in message:
                raise
            self.status = "capture_failed_runtime"
            self.enabled = False
            if graph is not None:
                try:
                    graph.reset()
                except Exception:
                    pass
        finally:
            self.setup_ms = (time.perf_counter() - start) * 1000.0


def evaluate_candidate_pool_exact_backtracking(
    model,
    view_inputs: Optional[torch.Tensor],
    *,
    reference_logits: torch.Tensor,
    ray_count: int,
    level_count: int,
    require_grad: bool,
    retain_features: bool = True,
    level_zero_cuda_graph: Optional[_ExactLevelZeroCandidateCudaGraph] = None,
    candidate_provider=None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Evaluate only candidates that can affect ray backtracking.

    Candidates are laid out as ``[ray0/radius0, ray0/radius1, ...]`` with
    radii ordered from largest to smallest. Once every sample preserves its
    raw pseudo-label at a radius on one ray, smaller radii on that ray cannot
    be selected by the production first-valid-radius rule. Skipping those
    forwards is therefore decision-exact. Unevaluated tensor slots are filled
    from the last evaluated radius solely to retain the audited dense return
    shape; ``evaluated_mask`` identifies real model evaluations.

    Every executed candidate remains an independent ``[B,C,T]`` forward, so
    this optimization does not reintroduce the prohibited ``[V*B,C,T]``
    BatchNorm coupling.
    """

    ray_count = int(ray_count)
    level_count = int(level_count)
    if ray_count < 1 or level_count < 1:
        raise ValueError("ray_count and level_count must be positive")
    if candidate_provider is None:
        view_inputs = torch.as_tensor(view_inputs)
        if view_inputs.dim() != 4:
            raise ValueError("candidate inputs must have shape [V, B, C, T]")
        candidate_count = int(view_inputs.size(0))
        batch_size = int(view_inputs.size(1))
    else:
        if view_inputs is not None:
            raise ValueError("provide either dense inputs or a candidate provider")
        candidate_count = int(candidate_provider.candidate_count)
        level_zero_probe = candidate_provider.materialize_candidate_level(0)
        if level_zero_probe.dim() != 4 or level_zero_probe.size(0) != ray_count:
            raise ValueError("candidate provider returned an invalid level")
        batch_size = int(level_zero_probe.size(1))
    if candidate_count != ray_count * level_count:
        raise ValueError("candidate count does not match ray/level layout")
    if reference_logits.size(0) != batch_size:
        raise ValueError("reference logits and candidates have different batches")

    raw_labels = reference_logits.detach().argmax(dim=1)
    unresolved = torch.ones(
        ray_count,
        batch_size,
        device=raw_labels.device,
        dtype=torch.bool,
    )
    feature_slots = (
        [None] * candidate_count if retain_features else None
    )
    logit_slots = [None] * candidate_count
    evaluated = torch.zeros(
        candidate_count, device="cpu", dtype=torch.bool
    )
    gradient_context = nullcontext() if require_grad else torch.inference_mode()
    with gradient_context:
        for level_index in range(level_count):
            if level_index == 0:
                active_rays = range(ray_count)
            else:
                # One compact device-to-host decision per radius level avoids
                # one synchronization per ray while preserving the same set
                # of candidate forwards.
                active_rays = (
                    torch.nonzero(unresolved.any(dim=1), as_tuple=False)
                    .flatten()
                    .tolist()
                )
                if not active_rays:
                    break
            level_inputs = (
                candidate_provider.materialize_candidate_level(level_index)
                if candidate_provider is not None
                else None
            )
            graph_logits = None
            level_zero_inputs = None
            if (
                level_index == 0
                and level_zero_cuda_graph is not None
                and not require_grad
                and not retain_features
            ):
                level_zero_inputs = (
                    level_inputs
                    if level_inputs is not None
                    else view_inputs[0::level_count]
                )
                graph_logits = level_zero_cuda_graph.replay(
                    model, level_zero_inputs
                )
            for ray_index in active_rays:
                candidate_index = ray_index * level_count + level_index
                if graph_logits is None:
                    with SSAWPhysicalView._preserved_bn_buffers(model):
                        current_features = _extract_features(
                            model,
                            (
                                level_inputs[ray_index]
                                if level_inputs is not None
                                else view_inputs[candidate_index]
                            ),
                        )
                        current_logits = model.classifier(current_features)
                else:
                    current_features = None
                    current_logits = graph_logits[ray_index]
                if feature_slots is not None:
                    feature_slots[candidate_index] = current_features
                logit_slots[candidate_index] = current_logits
                evaluated[candidate_index] = True
                preserves = current_logits.detach().argmax(dim=1).eq(raw_labels)
                unresolved[ray_index] &= ~preserves
            if (
                level_index == 0
                and level_zero_cuda_graph is not None
                and graph_logits is None
                and level_zero_inputs is not None
            ):
                eager_level_zero_logits = torch.stack(
                    [
                        logit_slots[ray_index * level_count]
                        for ray_index in range(ray_count)
                    ]
                )
                level_zero_cuda_graph.prepare(
                    model,
                    level_zero_inputs,
                    eager_level_zero_logits,
                )

    # Dense fillers are never selectable: an earlier evaluated radius already
    # preserved every sample on this ray. They exist only for compatibility
    # with diagnostics that expect [V,B,...] tensors.
    for ray_index in range(ray_count):
        last_feature = None
        last_logit = None
        for level_index in range(level_count):
            candidate_index = ray_index * level_count + level_index
            feature_present = bool(
                feature_slots is not None
                and feature_slots[candidate_index] is not None
            )
            if logit_slots[candidate_index] is not None:
                if feature_present:
                    last_feature = feature_slots[candidate_index]
                last_logit = logit_slots[candidate_index]
            else:
                if last_logit is None or (
                    feature_slots is not None and last_feature is None
                ):
                    raise RuntimeError("backtracking search skipped a ray endpoint")
                if feature_slots is not None:
                    feature_slots[candidate_index] = last_feature
                logit_slots[candidate_index] = last_logit

    return (
        None if feature_slots is None else torch.stack(feature_slots),
        torch.stack(logit_slots),
        evaluated,
    )


class DuSafe(BaseTestTimeAlgorithm):
    """Online fixed-source adaptation with source-only admission calibration."""

    def __init__(self, configs, hparams, model, optimizer):
        self.num_classes = int(configs.num_classes)
        self.bn_statistics = str(
            hparams.get("bn_statistics", "batch")
        ).strip().lower()
        self.adapt_parameter_scope = str(
            hparams.get("adapt_parameter_scope", "batch_norm")
        ).strip().lower()
        if self.adapt_parameter_scope not in {
            "batch_norm", "feature_extractor", "full"
        }:
            raise ValueError(
                "adapt_parameter_scope must be 'batch_norm', "
                "'feature_extractor', or 'full'"
            )
        super().__init__(configs, hparams, model, optimizer)
        self.enable_adaptation = bool(hparams.get("enable_adaptation", True))
        self.record_gradient_diagnostics = bool(
            hparams.get("record_gradient_diagnostics", False)
        )
        self.logging_mode = str(
            hparams.get("dusafe_logging_mode", "evidence")
        ).strip().lower()
        if self.logging_mode not in {"production", "evidence"}:
            raise ValueError(
                "dusafe_logging_mode must be 'production' or 'evidence'"
            )
        self.evidence_logging = self.logging_mode == "evidence"
        self.record_production_batch_diagnostics = bool(
            hparams.get("record_production_batch_diagnostics", True)
        )
        self.record_runtime_stage_markers = bool(
            hparams.get("record_runtime_stage_markers", True)
        )
        # Transaction buffers are ordinary tensors rather than module buffers:
        # they must not enter checkpoints, but can be reused across deployment
        # batches to avoid reallocating the same rollback state every time.
        self._rollback_parameter_ids = ()
        self._rollback_parameter_buffers = []
        self._rollback_optimizer_tensor_buffers = {}
        self.enable_ssaw = bool(hparams.get("enable_ssaw", True))
        self.execution_mode = str(
            hparams.get("dusafe_execution_mode", "fused")
        ).strip().lower()
        if self.execution_mode not in {"legacy", "fused"}:
            raise ValueError(
                "dusafe_execution_mode must be 'legacy' or 'fused'"
            )
        self.update_transaction_scope = str(
            hparams.get("update_transaction_scope", "batch")
        ).strip().lower()
        if self.update_transaction_scope not in {"step", "batch"}:
            raise ValueError(
                "update_transaction_scope must be 'step' or 'batch'"
            )
        self.enable_source_semantic_gate = bool(
            hparams.get("enable_source_semantic_gate", False)
        )
        self.source_semantic_bn_statistics = str(
            hparams.get("source_semantic_bn_statistics", "frozen")
        ).strip().lower()
        if self.source_semantic_bn_statistics != "frozen":
            raise ValueError(
                "source_semantic_bn_statistics must be 'frozen'; target-batch "
                "statistics are not a fixed-source semantic reference"
            )
        # The production method no longer uses a semantic router.  Allocate
        # the frozen source encoder only for explicitly requested archived
        # experiments instead of paying its memory cost in every DuSafe run.
        if self.enable_source_semantic_gate:
            self.source_semantic_feature_extractor = copy.deepcopy(
                self.model.feature_extractor
            )
            for parameter in self.source_semantic_feature_extractor.parameters():
                parameter.requires_grad_(False)
            self._configure_frozen_semantic_extractor()
        else:
            self.source_semantic_feature_extractor = None
        self.register_buffer(
            "source_semantic_prototypes", torch.empty(0), persistent=False
        )
        self.source_semantic_reference_ready = False
        self.register_buffer(
            "source_normalization_mean", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "source_normalization_std", torch.empty(0), persistent=False
        )
        self.ssaw_auxiliary_weight = float(
            hparams.get("ssaw_auxiliary_weight", 0.1)
        )
        if self.ssaw_auxiliary_weight < 0.0:
            raise ValueError("ssaw_auxiliary_weight must be non-negative")
        self.enable_confidence_gate = bool(
            hparams.get("enable_confidence_gate", True)
        )
        self.confidence_keep_fraction = float(
            hparams.get("confidence_keep_fraction", 1.0)
        )
        if not 0.0 < self.confidence_keep_fraction <= 1.0:
            raise ValueError("confidence_keep_fraction must be in (0, 1]")
        self.register_buffer(
            "confidence_nll_threshold",
            torch.tensor(float("inf")),
            persistent=False,
        )
        self.source_confidence_reference_ready = False

        base_sobol_seed = int(hparams.get("ssaw_sobol_seed", 1729))
        test_time_seed = hparams.get("test_time_seed")
        if test_time_seed is None:
            effective_sobol_seed = base_sobol_seed
        else:
            # Keep source training fixed while giving every test-time run an
            # independent, reproducible scrambled Sobol view sequence.
            effective_sobol_seed = (
                base_sobol_seed + 1_000_003 * int(test_time_seed)
            ) % 2_147_483_647
        self.ssaw_base_sobol_seed = base_sobol_seed
        self.test_time_seed = (
            None if test_time_seed is None else int(test_time_seed)
        )
        self.ssaw_effective_sobol_seed = effective_sobol_seed
        self.ssaw = self._build_ssaw(hparams, effective_sobol_seed)
        requested_cuda_graph = hparams.get("ssaw_candidate_cuda_graph", "off")
        if isinstance(requested_cuda_graph, bool):
            requested_cuda_graph = "auto" if requested_cuda_graph else "off"
        requested_cuda_graph = str(requested_cuda_graph).strip().lower()
        if requested_cuda_graph not in {"off", "auto", "force"}:
            raise ValueError(
                "ssaw_candidate_cuda_graph must be 'off', 'auto', or 'force'"
            )
        graph_eligible = bool(
            requested_cuda_graph != "off"
            and self.enable_ssaw
            and not self.evidence_logging
            and not self.record_production_batch_diagnostics
            and self.execution_mode == "fused"
            and getattr(self.ssaw, "decision_only_logging", False)
            and getattr(self.ssaw, "selection_only_candidate_evaluation", False)
            and getattr(self.ssaw, "exact_backtracking_evaluation", False)
        )
        self._candidate_cuda_graph_config_eligible = graph_eligible
        self._candidate_cuda_graph_workload_disabled = False
        self.candidate_cuda_graph_auto_min_expected_searches = int(
            hparams.get(
                "ssaw_candidate_cuda_graph_min_expected_searches", 10
            )
        )
        if self.candidate_cuda_graph_auto_min_expected_searches < 3:
            raise ValueError(
                "ssaw_candidate_cuda_graph_min_expected_searches must be >= 3"
            )
        # Public runtime switch: profilers and protocol diagnostics may force
        # the historical eager path without reconstructing the adapter.
        self.candidate_cuda_graph_runtime_enabled = graph_eligible
        self._candidate_cuda_graph = _ExactLevelZeroCandidateCudaGraph(
            enabled=graph_eligible,
            requested_mode=requested_cuda_graph,
            max_static_input_mb=float(
                hparams.get(
                    "ssaw_candidate_cuda_graph_max_static_input_mb",
                    float("inf"),
                )
            ),
        )
        if requested_cuda_graph != "off" and self.evidence_logging:
            self._candidate_cuda_graph.status = "disabled_evidence_logging"
        self._last_gate_log: Dict[str, object] = {}
        self._last_batch_log: Dict[str, float] = {}
        self._cached_source_semantic_prediction: Optional[torch.Tensor] = None
        self._cached_source_semantic_margin: Optional[torch.Tensor] = None
        self._batch_transaction_active = False
        self._batch_transaction_failed = False
        self._batch_update_snapshot: Optional[Dict[str, object]] = None
        self._batch_gradient_diagnostics: Optional[Dict[str, float]] = None
        # configure_model fixes requires_grad before the optimizer is built;
        # deployment never changes that set, so avoid a full model traversal
        # at every inner update.
        self._adaptation_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def configure_candidate_graph_workload(
        self, *, expected_full_batch_searches: int
    ) -> None:
        """Disable auto capture when a finite stream cannot amortize it.

        The first full-shape search captures the graph and the first search
        after an optimizer update performs the mandatory exact self-test.
        Real-CNN calibration on the deployment GPU measured an approximately
        27 ms setup and 3.5 ms saving per later search, so auto mode requires
        ten full-shape searches by default. The gate is reversible before a
        capture: a wrapper reused for a longer stream can enable the graph
        without being reconstructed. Force mode and evidence mode retain their
        explicit behavior.
        """

        expected = max(0, int(expected_full_batch_searches))
        graph = self._candidate_cuda_graph
        if not self.enable_ssaw:
            graph.expected_full_batch_searches = 0
            self.candidate_cuda_graph_runtime_enabled = False
            graph.status = "disabled_ssaw"
            graph.last_event = graph.status
            return
        graph.expected_full_batch_searches = expected
        if graph.requested_mode != "auto":
            return
        enough_work = bool(
            expected >= self.candidate_cuda_graph_auto_min_expected_searches
        )
        if not enough_work:
            self.candidate_cuda_graph_runtime_enabled = False
            if graph.enabled:
                graph.enabled = False
                self._candidate_cuda_graph_workload_disabled = True
            graph.status = "disabled_insufficient_full_batch_searches"
            graph.last_event = graph.status
            return
        if (
            self._candidate_cuda_graph_workload_disabled
            and self._candidate_cuda_graph_config_eligible
        ):
            graph.enabled = True
            self._candidate_cuda_graph_workload_disabled = False
            graph.status = (
                "uninitialized"
                if graph.capture_count == 0
                else "ready_workload_reenabled"
            )
            graph.last_event = graph.status
        self.candidate_cuda_graph_runtime_enabled = bool(
            self._candidate_cuda_graph_config_eligible and graph.enabled
        )

    def _build_ssaw(self, hparams, effective_sobol_seed: int):
        """Build the view generator used by this algorithm class.

        The reviewed production subclass overrides this factory, so its
        constructor never creates the retired constant-gain/legacy SSAW
        generator.  The fallback remains only for archived diagnostic classes
        that instantiate :class:`DuSafe` directly.
        """

        return SSAWPhysicalView(
            num_control_points=int(hparams.get("ssaw_control_points", 10)),
            sigma=float(hparams.get("ssaw_sigma", 0.20)),
            sobol_seed=effective_sobol_seed,
            strength=float(hparams.get("ssaw_strength", 10.0)),
            temporal_mode=str(
                hparams.get("ssaw_temporal_mode", "window_constant")
            ),
            antithetic=bool(hparams.get("ssaw_antithetic", False)),
            antithetic_pairs=int(hparams.get("ssaw_antithetic_pairs", 1)),
        )

    def _configure_frozen_semantic_extractor(self):
        """Keep source semantics independent of target batch composition."""
        extractor = self.source_semantic_feature_extractor
        if extractor is None:
            if self.enable_source_semantic_gate:
                raise RuntimeError("source semantic encoder was not initialized")
            return
        extractor.eval()
        for module in extractor.modules():
            if isinstance(module, _BATCH_NORM_TYPES):
                if module.running_mean is None or module.running_var is None:
                    raise ValueError(
                        "frozen source-semantic BatchNorm requires calibrated "
                        "source running statistics"
                    )
                module.track_running_stats = True
                module.training = False
            elif isinstance(
                module,
                (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
            ):
                module.training = False

    def load_source_normalization_reference(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        """Load fixed source normalization statistics for physical SSAW."""
        device = next(self.model.parameters()).device
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device).view(-1)
        std = torch.as_tensor(std, dtype=torch.float32, device=device).view(-1)
        if mean.numel() != std.numel() or mean.numel() == 0:
            raise ValueError("source normalization mean/std shapes must match")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("source normalization statistics must be finite")
        if not std.gt(0.0).all():
            raise ValueError("source normalization std must be positive")
        self.source_normalization_mean = mean
        self.source_normalization_std = std

    def load_source_semantic_reference(self, metadata: Mapping[str, object]):
        """Load fixed class means fitted once from labelled source samples."""
        if self.source_semantic_feature_extractor is None:
            raise RuntimeError(
                "source semantic routing is disabled for this method"
            )
        if int(metadata.get("version", -1)) != SOURCE_SEMANTIC_METADATA_VERSION:
            raise ValueError(
                "Unsupported source semantic metadata version: "
                f"{metadata.get('version')!r}"
            )
        if int(metadata.get("num_classes", -1)) != self.num_classes:
            raise ValueError("source semantic metadata class count mismatch")
        if str(metadata.get("bn_statistics", "")).strip().lower() != "frozen":
            raise ValueError(
                "source semantic metadata must use frozen source BN statistics"
            )
        prototypes = torch.as_tensor(
            metadata.get("prototypes"), dtype=torch.float32
        )
        class_counts = torch.as_tensor(metadata.get("class_counts")).flatten()
        if prototypes.dim() != 2 or prototypes.size(0) != self.num_classes:
            raise ValueError("source semantic metadata has invalid prototypes")
        if class_counts.numel() != self.num_classes or not class_counts.gt(0).all():
            raise ValueError("source semantic metadata has invalid class counts")
        if not torch.isfinite(prototypes).all():
            raise ValueError("source semantic prototypes must be finite")
        device = next(self.model.parameters()).device
        bn_state = metadata.get("feature_extractor_bn_state")
        if not isinstance(bn_state, Mapping):
            raise ValueError("source semantic metadata has no calibrated BN state")
        modules = dict(self.source_semantic_feature_extractor.named_modules())
        expected_bn_names = {
            name
            for name, module in modules.items()
            if isinstance(module, _BATCH_NORM_TYPES)
        }
        if set(str(name) for name in bn_state) != expected_bn_names:
            raise ValueError("source semantic metadata BN module set mismatch")
        with torch.no_grad():
            for name in expected_bn_names:
                module = modules[name]
                values = bn_state[name]
                if not isinstance(values, Mapping):
                    raise ValueError("source semantic metadata has invalid BN values")
                running_mean = torch.as_tensor(
                    values.get("running_mean"),
                    device=device,
                    dtype=module.weight.dtype if module.affine else torch.float32,
                ).view(-1)
                running_var = torch.as_tensor(
                    values.get("running_var"),
                    device=device,
                    dtype=module.weight.dtype if module.affine else torch.float32,
                ).view(-1)
                batches = torch.as_tensor(
                    values.get("num_batches_tracked", 0),
                    device=device,
                    dtype=torch.long,
                ).reshape(())
                if (
                    running_mean.numel() != module.num_features
                    or running_var.numel() != module.num_features
                    or not torch.isfinite(running_mean).all()
                    or not torch.isfinite(running_var).all()
                    # A calibrated BatchNorm channel may have exactly zero
                    # source variance (for example, a dead ReLU channel in
                    # FD).  PyTorch's BatchNorm handles it through ``eps``;
                    # only negative or non-finite variance is invalid.
                    or not running_var.ge(0.0).all()
                ):
                    raise ValueError("source semantic metadata has invalid BN buffers")
                module.track_running_stats = True
                module.running_mean = running_mean.clone()
                module.running_var = running_var.clone()
                module.num_batches_tracked = batches.clone()
                module.training = False
        self.source_semantic_prototypes = F.normalize(
            prototypes.to(device), dim=1
        )
        self._configure_frozen_semantic_extractor()
        self.source_semantic_reference_ready = True

    @torch.no_grad()
    def _source_semantic_decision(
        self, inputs: torch.Tensor, pseudo_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_source_semantic_gate:
            mask = torch.ones_like(pseudo_labels, dtype=torch.bool)
            return mask, pseudo_labels.detach(), inputs.new_zeros(
                pseudo_labels.shape
            )
        if not self.source_semantic_reference_ready:
            raise RuntimeError(
                "Source semantic gate is enabled but source semantic metadata "
                "was not loaded"
            )
        self._configure_frozen_semantic_extractor()
        features = _normalized_feature_vectors(
            self.source_semantic_feature_extractor, inputs
        )
        if features.size(1) != self.source_semantic_prototypes.size(1):
            raise RuntimeError(
                "source semantic feature dimension does not match prototypes"
            )
        similarities = features @ self.source_semantic_prototypes.t()
        top_values, top_indices = similarities.topk(k=2, dim=1)
        semantic_predictions = top_indices[:, 0]
        semantic_margin = top_values[:, 0] - top_values[:, 1]
        return (
            semantic_predictions.eq(pseudo_labels),
            semantic_predictions,
            semantic_margin,
        )

    def configure_model(self, model):
        model.train()
        adapt = bool(self.hparams.get("enable_adaptation", True))
        for parameter in model.parameters():
            parameter.requires_grad_(adapt and self.adapt_parameter_scope == "full")
        if adapt and self.adapt_parameter_scope == "feature_extractor":
            for parameter in model.feature_extractor.parameters():
                parameter.requires_grad_(True)

        if self.bn_statistics not in {"frozen", "batch"}:
            raise ValueError("bn_statistics must be 'frozen' or 'batch'")
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if adapt and self.adapt_parameter_scope == "batch_norm":
                    for parameter in module.parameters(recurse=False):
                        parameter.requires_grad_(True)
                if self.bn_statistics == "batch":
                    module.track_running_stats = False
                    module.training = True
                else:
                    if module.running_mean is None or module.running_var is None:
                        raise ValueError(
                            "frozen BatchNorm requires source running statistics"
                        )
                    module.track_running_stats = True
                    module.training = False
            elif isinstance(
                module,
                (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
            ):
                module.training = False
        model._dusafe_bn_buffers_may_update_cache = any(
            isinstance(module, _BATCH_NORM_TYPES)
            and bool(module.training)
            and bool(module.track_running_stats)
            for module in model.modules()
        )
        return model

    @staticmethod
    def _extract_features(model, inputs: torch.Tensor) -> torch.Tensor:
        return _extract_features(model, inputs)

    @staticmethod
    def _bn_buffers_may_update(model) -> bool:
        cached = getattr(model, "_dusafe_bn_buffers_may_update_cache", None)
        if cached is not None:
            return bool(cached)
        return any(
            isinstance(module, _BATCH_NORM_TYPES)
            and bool(module.training)
            and bool(module.track_running_stats)
            for module in model.modules()
        )

    @staticmethod
    def _snapshot_bn_buffers(model):
        snapshots = []
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                snapshots.append(
                    (
                        module,
                        None
                        if module.running_mean is None
                        else module.running_mean.detach().clone(),
                        None
                        if module.running_var is None
                        else module.running_var.detach().clone(),
                        None
                        if module.num_batches_tracked is None
                        else module.num_batches_tracked.detach().clone(),
                    )
                )
        return snapshots

    @staticmethod
    def _restore_bn_buffers(snapshots):
        with torch.no_grad():
            for module, mean, variance, batches in snapshots:
                if mean is not None and module.running_mean is not None:
                    module.running_mean.copy_(mean)
                if variance is not None and module.running_var is not None:
                    module.running_var.copy_(variance)
                if batches is not None and module.num_batches_tracked is not None:
                    module.num_batches_tracked.copy_(batches)

    def load_source_confidence_reference(self, metadata: Mapping[str, object]):
        if int(metadata.get("version", -1)) != SOURCE_CONFIDENCE_METADATA_VERSION:
            raise ValueError(
                "Unsupported source confidence metadata version: "
                f"{metadata.get('version')!r}"
            )
        source_scores = torch.as_tensor(
            metadata.get("top1_nll"), dtype=torch.float32
        ).flatten()
        if source_scores.numel() == 0 or not torch.isfinite(source_scores).all():
            raise ValueError("source confidence metadata has invalid scores")
        threshold = torch.quantile(
            source_scores, self.confidence_keep_fraction
        ).to(self.confidence_nll_threshold.device)
        self.confidence_nll_threshold.copy_(threshold)
        self.source_confidence_reference_ready = True

    def _confidence_admission_mask(
        self,
        raw_top1_nll: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw-anchor confidence admission decision.

        The production implementation is deliberately kept behind this small
        hook so controlled ablations can replace the *selection rule* while
        preserving the admitted count, optimizer steps, and trainable
        parameter set.  The default path is exactly the previous fixed-source
        threshold rule.
        """

        del pseudo_labels
        if not self.enable_confidence_gate:
            return torch.ones_like(raw_top1_nll, dtype=torch.bool)
        if not self.source_confidence_reference_ready:
            raise RuntimeError(
                "Confidence gate is enabled but source confidence metadata "
                "was not loaded"
            )
        return raw_top1_nll.le(self.confidence_nll_threshold)

    def get_ssaw_stress_view(self, inputs: torch.Tensor, model):
        if not self.enable_ssaw:
            return inputs
        return self.ssaw(
            inputs,
            model,
            normalization_mean=self.source_normalization_mean,
            normalization_std=self.source_normalization_std,
        )

    def _physical_view_consistency_loss(
        self,
        model,
        raw_inputs: torch.Tensor,
        raw_target_logits: torch.Tensor,
        view_selection_mask: torch.Tensor,
        raw_admission_mask: torch.Tensor,
        sample_weights: torch.Tensor,
        view_logits_by_view: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match the physical view to the detached raw predictive distribution."""
        raw_log_probabilities = raw_target_logits.detach().log_softmax(dim=1)
        raw_probabilities = raw_log_probabilities.exp()
        if view_logits_by_view is None:
            view_inputs = self.ssaw.last_view_inputs.to(
                device=raw_inputs.device, dtype=raw_inputs.dtype
            )
            if view_inputs.dim() == 3:
                view_inputs = view_inputs.unsqueeze(0)
            computed_logits = []
            for current_view in view_inputs:
                computed_logits.append(
                    model.classifier(_extract_features(model, current_view))
                )
            view_logits_by_view = torch.stack(computed_logits)
        if view_logits_by_view.dim() != 3:
            raise ValueError("view logits must be shaped [V, B, K]")
        view_log_probabilities = view_logits_by_view.log_softmax(dim=2)
        per_sample_loss = (
            raw_probabilities.unsqueeze(0)
            * (
                raw_log_probabilities.unsqueeze(0)
                - view_log_probabilities
            )
        ).sum(dim=2).mean(dim=0)
        # Normalize by the raw admitted population.  KL therefore attenuates
        # only the auxiliary SSAW contribution instead of being cancelled by
        # a second weighted denominator or shrinking the optimizer LR.
        admitted_count = raw_admission_mask.sum().to(
            dtype=per_sample_loss.dtype
        ).clamp_min(1.0)
        return (
            per_sample_loss[view_selection_mask]
            * sample_weights[view_selection_mask]
        ).sum() / admitted_count

    def _prepare_ssaw_auxiliary_training(
        self,
        model,
        raw_inputs: torch.Tensor,
        raw_target_logits: torch.Tensor,
        view_selection_mask: torch.Tensor,
        raw_admission_mask: torch.Tensor,
        sample_weights: torch.Tensor,
        view_logits_by_view: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Allow a view method to revalidate a gathered training batch.

        Production DuSafe has no candidate-gathering stage, so its mask is
        unchanged. Experimental hard-view runners override this hook.
        """
        del (
            model,
            raw_inputs,
            raw_target_logits,
            raw_admission_mask,
            sample_weights,
            view_logits_by_view,
        )
        return view_selection_mask

    def _semantic_admission_mask(
        self, source_semantic_mask: torch.Tensor
    ) -> torch.Tensor:
        """Map frozen-source semantics to the raw-update admission mask.

        Production DuSafe keeps the historical conjunctive admission rule.
        Experimental hard-view runners override this hook so source semantics
        can route only the auxiliary SSAW objective without rejecting raw
        confidence anchors.
        """

        return source_semantic_mask

    def _ssaw_training_router_mask(
        self,
        confidence_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return which admitted anchors may receive the SSAW objective."""

        del confidence_mask, source_semantic_mask
        return torch.ones_like(pseudo_labels, dtype=torch.bool)

    def _runtime_stage(self, name: str):
        """Emit profiler ranges only when the caller requests instrumentation."""

        if self.record_runtime_stage_markers:
            return torch.autograd.profiler.record_function(name)
        return nullcontext()

    @staticmethod
    @torch.no_grad()
    def _tensors_are_finite(tensors) -> bool:
        """Check a tensor collection with one host/device synchronization."""
        values = [tensor for tensor in tensors if tensor is not None]
        if not values:
            return True
        norms = torch._foreach_norm(values, 2.0)
        aggregate = torch.linalg.vector_norm(
            torch.stack([value.float() for value in norms]), 2.0
        )
        return bool(torch.isfinite(aggregate).item())

    @staticmethod
    def _gradient_diagnostics(raw_gradients, auxiliary_gradients, weight):
        raw_sq = None
        auxiliary_sq = None
        dot = None
        for raw_gradient, auxiliary_gradient in zip(
            raw_gradients, auxiliary_gradients
        ):
            if raw_gradient is not None:
                value = raw_gradient.detach().float().square().sum()
                raw_sq = value if raw_sq is None else raw_sq + value
            if auxiliary_gradient is not None:
                value = auxiliary_gradient.detach().float().square().sum()
                auxiliary_sq = (
                    value if auxiliary_sq is None else auxiliary_sq + value
                )
            if raw_gradient is not None and auxiliary_gradient is not None:
                value = (
                    raw_gradient.detach().float()
                    * auxiliary_gradient.detach().float()
                ).sum()
                dot = value if dot is None else dot + value
        device = None
        for value in (raw_sq, auxiliary_sq, dot):
            if value is not None:
                device = value.device
                break
        if device is None:
            device = torch.device("cpu")
        zero = torch.zeros((), device=device)
        raw_norm = (raw_sq if raw_sq is not None else zero).sqrt()
        auxiliary_norm = (
            auxiliary_sq if auxiliary_sq is not None else zero
        ).sqrt()
        cosine = (
            torch.full((), float("nan"), device=device)
            if dot is None
            else dot
            / (raw_norm * auxiliary_norm).clamp_min(1e-12)
        )
        weighted_ratio = (
            auxiliary_norm * float(weight) / raw_norm.clamp_min(1e-12)
        )
        return {
            "raw_gradient_norm": float(raw_norm.item()),
            "ssaw_gradient_norm": float(auxiliary_norm.item()),
            "weighted_ssaw_to_raw_gradient_ratio": float(
                weighted_ratio.item()
            ),
            "raw_ssaw_gradient_cosine": float(cosine.item()),
        }

    @staticmethod
    @torch.no_grad()
    def _copy_tensor_collection_(destinations, sources) -> None:
        groups = {}
        for destination, source in zip(destinations, sources):
            key = (destination.device, destination.dtype)
            group = groups.setdefault(key, ([], []))
            group[0].append(destination)
            group[1].append(source.detach())
        for destination_group, source_group in groups.values():
            torch._foreach_copy_(destination_group, source_group)

    def _capture_parameter_snapshot(self, parameters):
        parameter_ids = tuple(id(parameter) for parameter in parameters)
        compatible = (
            parameter_ids == self._rollback_parameter_ids
            and len(self._rollback_parameter_buffers) == len(parameters)
            and all(
                saved.shape == parameter.shape
                and saved.dtype == parameter.dtype
                and saved.device == parameter.device
                for saved, parameter in zip(
                    self._rollback_parameter_buffers, parameters
                )
            )
        )
        if not compatible:
            self._rollback_parameter_ids = parameter_ids
            self._rollback_parameter_buffers = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            self._copy_tensor_collection_(
                self._rollback_parameter_buffers, parameters
            )
        return list(self._rollback_parameter_buffers)

    def _capture_standard_optimizer_snapshot(self, optimizer):
        base_optimizer = getattr(optimizer, "_optimizer", optimizer)
        if type(base_optimizer) not in {torch.optim.Adam, torch.optim.SGD}:
            return None
        optimizer_parameters = [
            parameter
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        ]
        state_rows = []
        copy_destinations = []
        copy_sources = []
        for parameter in optimizer_parameters:
            current_state = base_optimizer.state.get(parameter)
            if not current_state:
                continue
            saved_state = {}
            for key, value in current_state.items():
                if torch.is_tensor(value):
                    cache_key = (id(parameter), key)
                    saved = self._rollback_optimizer_tensor_buffers.get(
                        cache_key
                    )
                    if (
                        saved is None
                        or saved.shape != value.shape
                        or saved.dtype != value.dtype
                        or saved.device != value.device
                    ):
                        saved = value.detach().clone()
                        self._rollback_optimizer_tensor_buffers[
                            cache_key
                        ] = saved
                    else:
                        copy_destinations.append(saved)
                        copy_sources.append(value)
                    saved_state[key] = saved
                elif isinstance(value, (bool, int, float, str, type(None))):
                    saved_state[key] = value
                else:
                    # Preserve the generic contract for custom/extended state.
                    return None
            state_rows.append((parameter, saved_state))
        if copy_destinations:
            self._copy_tensor_collection_(copy_destinations, copy_sources)
        return {
            "kind": "standard_reusable",
            "optimizer": base_optimizer,
            "parameters": optimizer_parameters,
            "state_rows": state_rows,
        }

    @staticmethod
    @torch.no_grad()
    def _restore_standard_optimizer_snapshot(snapshot) -> None:
        optimizer = snapshot["optimizer"]
        saved_by_parameter = {
            parameter: state for parameter, state in snapshot["state_rows"]
        }
        for parameter in snapshot["parameters"]:
            saved_state = saved_by_parameter.get(parameter)
            if saved_state is None:
                optimizer.state.pop(parameter, None)
                continue
            current_state = optimizer.state.setdefault(parameter, {})
            for key in tuple(current_state):
                if key not in saved_state:
                    del current_state[key]
            for key, saved in saved_state.items():
                if torch.is_tensor(saved):
                    current = current_state.get(key)
                    if (
                        torch.is_tensor(current)
                        and current.shape == saved.shape
                        and current.dtype == saved.dtype
                        and current.device == saved.device
                    ):
                        current.copy_(saved)
                    else:
                        current_state[key] = saved.detach().clone()
                else:
                    current_state[key] = saved

    def _capture_update_snapshot(self, model, optimizer, parameters):
        optimizer_snapshot = self._capture_standard_optimizer_snapshot(
            optimizer
        )
        if optimizer_snapshot is None:
            optimizer_snapshot = {
                "kind": "generic_state_dict",
                "state_dict": copy.deepcopy(optimizer.state_dict()),
            }
        return {
            "parameters": self._capture_parameter_snapshot(parameters),
            "bn": (
                self._snapshot_bn_buffers(model)
                if self._bn_buffers_may_update(model)
                else []
            ),
            "optimizer": optimizer_snapshot,
        }

    def _restore_update_snapshot(
        self, model, optimizer, parameters, snapshot
    ) -> None:
        self._copy_tensor_collection_(parameters, snapshot["parameters"])
        self._restore_bn_buffers(snapshot["bn"])
        optimizer_snapshot = snapshot["optimizer"]
        if optimizer_snapshot["kind"] == "standard_reusable":
            self._restore_standard_optimizer_snapshot(optimizer_snapshot)
        else:
            optimizer.load_state_dict(optimizer_snapshot["state_dict"])

    def _apply_update(
        self,
        model,
        optimizer,
        loss: torch.Tensor,
        admitted_mask: torch.Tensor,
        update_scale: float = 1.0,
        auxiliary_loss: Optional[torch.Tensor] = None,
        auxiliary_weight: float = 0.0,
        admitted_nonempty: Optional[bool] = None,
    ) -> Dict[str, object]:
        """Commit one update only when gradients and parameters are finite."""
        update_scale = float(update_scale)
        if not math.isfinite(update_scale) or not 0.0 <= update_scale <= 1.0:
            raise ValueError("update_scale must be finite and lie in [0, 1]")
        auxiliary_weight = float(auxiliary_weight)
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0.0:
            raise ValueError("auxiliary_weight must be finite and non-negative")
        use_auxiliary = bool(
            auxiliary_loss is not None
            and auxiliary_weight > 0.0
            and auxiliary_loss.requires_grad
        )
        log = {
            "attempted": False,
            "committed": False,
            "finite": True,
            "update_scale": update_scale,
        }
        if use_auxiliary:
            log.update(
                {
                    "auxiliary_available": True,
                    # Availability is not application: an early return before
                    # backward/step must remain observable as not applied.
                    "auxiliary_gradient_applied": False,
                }
            )
        if self._batch_transaction_active and self._batch_transaction_failed:
            log["finite"] = False
            return log
        has_admitted = (
            bool(admitted_mask.any().item())
            if admitted_nonempty is None
            else bool(admitted_nonempty)
        )
        if (
            optimizer is None
            or not self.enable_adaptation
            or not has_admitted
            or not loss.requires_grad
        ):
            return log
        log["attempted"] = True
        parameters = self._adaptation_parameters
        optimizer.zero_grad(set_to_none=True)
        if (
            self.record_gradient_diagnostics
            and self._batch_gradient_diagnostics is None
        ):
            raw_gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            auxiliary_gradients = (
                torch.autograd.grad(
                    auxiliary_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                if use_auxiliary
                else tuple(None for _ in parameters)
            )
            self._batch_gradient_diagnostics = self._gradient_diagnostics(
                raw_gradients,
                auxiliary_gradients,
                auxiliary_weight,
            )
        if use_auxiliary:
            # One explicit objective is easier to audit than a hidden
            # gradient-conflict policy. Label-changing physical views have
            # already been removed by the SSAW mask, and KL supplies the
            # continuous per-sample attenuation before this weighted sum.
            (loss + auxiliary_weight * auxiliary_loss).backward()
            log["auxiliary_gradient_applied"] = True
        else:
            loss.backward()
        objective_tensors = [loss.detach()]
        if use_auxiliary:
            objective_tensors.append(auxiliary_loss.detach())
        prepared_gradient_norm = None
        prepare_gradients = getattr(
            optimizer, "prepare_gradients_for_step", None
        )
        if callable(prepare_gradients):
            prepared_gradient_norm = prepare_gradients()
        finite_check_tensors = list(objective_tensors)
        if prepared_gradient_norm is None:
            finite_check_tensors.extend(
                parameter.grad for parameter in parameters
            )
        else:
            # Norm clipping has already reduced every gradient. Reusing its
            # pre-clip norm preserves the non-finite guard and avoids a second
            # full gradient reduction before the same optimizer step.
            finite_check_tensors.append(prepared_gradient_norm)
        gradients_finite = self._tensors_are_finite(finite_check_tensors)
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            log["finite"] = False
            return log

        use_batch_snapshot = bool(
            self.update_transaction_scope == "batch"
            and self._batch_transaction_active
        )
        if use_batch_snapshot:
            if self._batch_update_snapshot is None:
                self._batch_update_snapshot = self._capture_update_snapshot(
                    model, optimizer, parameters
                )
            update_snapshot = self._batch_update_snapshot
        else:
            update_snapshot = self._capture_update_snapshot(
                model, optimizer, parameters
            )
        if update_scale == 1.0:
            optimizer.step()
        else:
            original_learning_rates = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            try:
                for group, learning_rate in zip(
                    optimizer.param_groups, original_learning_rates
                ):
                    group["lr"] = learning_rate * update_scale
                optimizer.step()
            finally:
                for group, learning_rate in zip(
                    optimizer.param_groups, original_learning_rates
                ):
                    group["lr"] = learning_rate
        parameters_finite = self._tensors_are_finite(parameters)
        log["finite"] = bool(parameters_finite)
        if not parameters_finite:
            self._restore_update_snapshot(
                model, optimizer, parameters, update_snapshot
            )
            if use_batch_snapshot:
                self._batch_transaction_failed = True
            return log
        log["committed"] = True
        return log

    def forward_and_adapt(
        self,
        batch_data,
        model,
        optimizer,
        trg_idx=None,
        reuse_ssaw_view: bool = False,
    ):
        del trg_idx
        production_minimal = bool(
            not self.evidence_logging
            and not self.record_production_batch_diagnostics
        )
        raw_inputs = _extract_primary_tensor(batch_data)
        if not torch.is_tensor(raw_inputs) or raw_inputs.dim() != 3:
            raise ValueError("DuSafe expects a tensor shaped [B, C, T]")
        if not reuse_ssaw_view:
            self._cached_source_semantic_prediction = None
            self._cached_source_semantic_margin = None
        raw_train_logits: Optional[torch.Tensor] = None
        view_train_logits_by_view: Optional[torch.Tensor] = None
        if self.execution_mode == "legacy":
            bn_snapshot = self._snapshot_bn_buffers(model)
            with torch.no_grad():
                if self.enable_ssaw:
                    if (
                        self.source_normalization_mean.numel()
                        != raw_inputs.size(1)
                        or self.source_normalization_std.numel()
                        != raw_inputs.size(1)
                    ):
                        raise RuntimeError(
                            "physical SSAW requires fixed source normalization "
                            "mean and std"
                        )
                    reference_features = _extract_features(model, raw_inputs)
                    reference_logits = model.classifier(reference_features)
                    self.ssaw(
                        raw_inputs,
                        model,
                        reference_logits=reference_logits,
                        reference_features=reference_features,
                        normalization_mean=self.source_normalization_mean,
                        normalization_std=self.source_normalization_std,
                        reuse_cached_view=reuse_ssaw_view,
                    )
                    raw_logits = self.ssaw.last_reference_logits
                    raw_features = self.ssaw.last_reference_features
                    stress_logits = self.ssaw.last_stress_logits
                    stress_features = self.ssaw.last_stress_features
                else:
                    raw_features = _extract_features(model, raw_inputs)
                    raw_logits = model.classifier(raw_features)
                    stress_logits = raw_logits
                    stress_features = raw_features
            self._restore_bn_buffers(bn_snapshot)
        else:
            if self.enable_ssaw:
                if (
                    self.source_normalization_mean.numel()
                    != raw_inputs.size(1)
                    or self.source_normalization_std.numel()
                    != raw_inputs.size(1)
                ):
                    raise RuntimeError(
                        "physical SSAW requires fixed source normalization "
                        "mean and std"
                    )
                with self._runtime_stage("dusafe.raw_forward"):
                    raw_train_features = _extract_features(model, raw_inputs)
                    raw_train_logits = model.classifier(raw_train_features)
                with self._runtime_stage("dusafe.view_generation"):
                    prepared_views = self.ssaw.prepare_view_inputs(
                        raw_inputs,
                        normalization_mean=self.source_normalization_mean,
                        normalization_std=self.source_normalization_std,
                        reuse_cached_view=reuse_ssaw_view,
                    )
                candidate_provider = prepared_views.get("candidate_provider")
                prepared_view_inputs = (
                    None
                    if candidate_provider is not None
                    else torch.as_tensor(prepared_views["view_inputs"])
                )
                # Never flatten candidates into [V*B,C,T]. BatchNorm would
                # then couple one view to every other candidate in the pool.
                # Search-only pools are detached and the selected mixed batch
                # is re-forwarded by the experimental runner for training.
                selection_only = bool(
                    getattr(
                        self.ssaw,
                        "selection_only_candidate_evaluation",
                        False,
                    )
                )
                lazy_backtracking = bool(
                    selection_only
                    and getattr(
                        self.ssaw,
                        "exact_backtracking_evaluation",
                        False,
                    )
                )
                retain_candidate_features = not bool(
                    getattr(self.ssaw, "decision_only_logging", False)
                )
                candidate_cuda_graph = (
                    self._candidate_cuda_graph
                    if (
                        self.candidate_cuda_graph_runtime_enabled
                        and production_minimal
                        and lazy_backtracking
                        and not retain_candidate_features
                    )
                    else None
                )
                with self._runtime_stage("dusafe.candidate_search"):
                    if lazy_backtracking:
                        (
                            view_train_features_by_view,
                            view_train_logits_by_view,
                            candidate_evaluated_mask,
                        ) = evaluate_candidate_pool_exact_backtracking(
                            model,
                            prepared_view_inputs,
                            reference_logits=raw_train_logits,
                            ray_count=int(self.ssaw.ray_count),
                            level_count=len(self.ssaw.radius_levels),
                            require_grad=False,
                            retain_features=retain_candidate_features,
                            level_zero_cuda_graph=candidate_cuda_graph,
                            candidate_provider=candidate_provider,
                        )
                        prepared_views = dict(prepared_views)
                        prepared_views["candidate_evaluated_mask"] = (
                            candidate_evaluated_mask
                        )
                        prepared_views["candidate_search_execution"] = (
                            "exact_lazy_backtracking_cuda_graph_level_zero"
                            if (
                                candidate_cuda_graph is not None
                                and candidate_cuda_graph.last_level_zero_used
                            )
                            else "exact_lazy_backtracking"
                        )
                    else:
                        (
                            view_train_features_by_view,
                            view_train_logits_by_view,
                        ) = evaluate_candidate_pool_sequential(
                            model,
                            prepared_view_inputs,
                            require_grad=not selection_only,
                            retain_features=retain_candidate_features,
                        )
                with self._runtime_stage("dusafe.candidate_selection"):
                    self.ssaw.record_evaluation(
                        reference_logits=raw_train_logits,
                        reference_features=raw_train_features,
                        candidate_logits_by_view=view_train_logits_by_view,
                        candidate_features_by_view=view_train_features_by_view,
                        prepared_views=prepared_views,
                    )
                    if candidate_cuda_graph is not None:
                        self.ssaw.last_metadata.update(
                            candidate_cuda_graph.diagnostics()
                        )
                raw_logits = self.ssaw.last_reference_logits
                if self.evidence_logging:
                    raw_features = self.ssaw.last_reference_features
                    stress_logits = self.ssaw.last_stress_logits
                    stress_features = self.ssaw.last_stress_features
                else:
                    # Candidate feature stacks and stress-view summaries do
                    # not participate in production selection or updates.
                    raw_features = None
                    stress_logits = raw_logits
                    stress_features = None
            else:
                raw_train_features = _extract_features(model, raw_inputs)
                raw_train_logits = model.classifier(raw_train_features)
                raw_features = raw_train_features.detach()
                raw_logits = raw_train_logits.detach()
                stress_logits = raw_logits
                stress_features = raw_features
        prediction_logits = raw_logits.detach()
        with torch.no_grad():
            raw_entropy = (
                _entropy_from_logits(raw_logits)
                if self.evidence_logging
                else None
            )
            raw_top1_nll, pseudo_labels = _top1_nll(raw_logits)
            stress_entropy = (
                _entropy_from_logits(stress_logits)
                if self.evidence_logging
                else None
            )
            if (
                self.enable_source_semantic_gate
                and reuse_ssaw_view
                and self._cached_source_semantic_prediction is not None
                and self._cached_source_semantic_margin is not None
            ):
                semantic_predictions = (
                    self._cached_source_semantic_prediction
                )
                semantic_margin = self._cached_source_semantic_margin
                source_semantic_mask = semantic_predictions.eq(pseudo_labels)
            else:
                (
                    source_semantic_mask,
                    semantic_predictions,
                    semantic_margin,
                ) = self._source_semantic_decision(raw_inputs, pseudo_labels)
                if self.enable_source_semantic_gate:
                    self._cached_source_semantic_prediction = (
                        semantic_predictions.detach()
                    )
                    self._cached_source_semantic_margin = (
                        semantic_margin.detach()
                    )

        semantic_mask = torch.as_tensor(
            self._semantic_admission_mask(source_semantic_mask),
            device=pseudo_labels.device,
            dtype=torch.bool,
        )
        if semantic_mask.shape != pseudo_labels.shape:
            raise RuntimeError("semantic admission mask shape mismatch")

        confidence_mask = torch.as_tensor(
            self._confidence_admission_mask(raw_top1_nll, pseudo_labels),
            device=pseudo_labels.device,
            dtype=torch.bool,
        )
        if confidence_mask.shape != pseudo_labels.shape:
            raise RuntimeError("confidence admission mask shape mismatch")
        if self.enable_ssaw:
            ssaw_label_flip = torch.as_tensor(
                self.ssaw.last_metadata["ssaw_label_flip"],
                device=pseudo_labels.device,
                dtype=torch.bool,
            )
            ssaw_kl = (
                torch.as_tensor(
                    self.ssaw.last_metadata["selected_kl"],
                    device=pseudo_labels.device,
                    dtype=raw_top1_nll.dtype,
                )
                if self.evidence_logging
                else (
                    None
                    if production_minimal
                    else torch.zeros_like(raw_top1_nll)
                )
            )
            ssaw_view_selected_mask = ~ssaw_label_flip
            if not production_minimal:
                self.ssaw.last_metadata["ssaw_view_selected"] = (
                    ssaw_view_selected_mask.detach().cpu()
                    if self.evidence_logging
                    else ssaw_view_selected_mask.detach()
                )
        else:
            ssaw_label_flip = torch.zeros_like(pseudo_labels, dtype=torch.bool)
            ssaw_kl = torch.zeros_like(raw_top1_nll)
            ssaw_view_selected_mask = torch.zeros_like(
                pseudo_labels, dtype=torch.bool
            )
        base_admission_mask = confidence_mask & semantic_mask
        admission_mask = base_admission_mask
        ssaw_router_mask = torch.as_tensor(
            self._ssaw_training_router_mask(
                confidence_mask,
                source_semantic_mask,
                pseudo_labels,
            ),
            device=pseudo_labels.device,
            dtype=torch.bool,
        )
        if ssaw_router_mask.shape != pseudo_labels.shape:
            raise RuntimeError("SSAW router mask shape mismatch")
        ssaw_consistency_mask = (
            admission_mask
            & ssaw_router_mask
            & ssaw_view_selected_mask
        )
        update_weights = torch.ones_like(raw_top1_nll)

        if self.enable_ssaw:
            with self._runtime_stage("dusafe.gathered_forward"):
                ssaw_consistency_mask = torch.as_tensor(
                    self._prepare_ssaw_auxiliary_training(
                        model,
                        raw_inputs,
                        raw_logits,
                        ssaw_consistency_mask,
                        admission_mask,
                        update_weights,
                        view_train_logits_by_view,
                    ),
                    device=pseudo_labels.device,
                    dtype=torch.bool,
                )
            if ssaw_consistency_mask.shape != pseudo_labels.shape:
                raise RuntimeError("prepared SSAW training mask shape mismatch")

        # One synchronization supplies both control-flow decisions.  Reusing
        # them below avoids repeated mask.any() device-to-host barriers.
        mask_presence = torch.stack(
            (admission_mask.any(), ssaw_consistency_mask.any())
        ).detach().cpu()
        has_admitted = bool(mask_presence[0].item())
        has_ssaw_training = bool(mask_presence[1].item())

        raw_ce_loss = raw_inputs.new_zeros(())
        consistency_loss = raw_inputs.new_zeros(())
        weighted_consistency_loss = raw_inputs.new_zeros(())
        realized_consistency_ratio = raw_inputs.new_zeros(())
        selected_view_fraction = raw_inputs.new_zeros(())
        optimizer_update_scale = 1.0
        auxiliary_loss_for_update = None
        if has_admitted and self.enable_adaptation:
            admitted_count = admission_mask.sum().to(
                dtype=raw_top1_nll.dtype
            ).clamp_min(1.0)
            if raw_train_logits is None:
                raw_train_logits = model.classifier(
                    _extract_features(model, raw_inputs)
                )
            per_sample_raw_ce = F.cross_entropy(
                raw_train_logits,
                pseudo_labels.detach(),
                reduction="none",
            )
            raw_ce_loss = per_sample_raw_ce[admission_mask].mean()
            adaptation_loss = raw_ce_loss
            if (
                self.enable_ssaw
                and self.ssaw_auxiliary_weight > 0.0
                and has_ssaw_training
            ):
                consistency_loss = self._physical_view_consistency_loss(
                    model,
                    raw_inputs,
                    raw_logits,
                    ssaw_consistency_mask,
                    admission_mask,
                    update_weights,
                    view_logits_by_view=view_train_logits_by_view,
                )
                auxiliary_loss_for_update = consistency_loss
                if not production_minimal:
                    selected_weight_sum = update_weights[
                        ssaw_consistency_mask
                    ].sum()
                    selected_view_fraction = selected_weight_sum / admitted_count
                    weighted_consistency_loss = (
                        self.ssaw_auxiliary_weight * consistency_loss
                    )
                    adaptation_loss = raw_ce_loss + weighted_consistency_loss
                    realized_consistency_ratio = (
                        weighted_consistency_loss.detach()
                        / raw_ce_loss.detach().clamp_min(1e-8)
                    )
        else:
            adaptation_loss = raw_inputs.new_zeros(())

        with self._runtime_stage("dusafe.backward_and_update"):
            update_log = self._apply_update(
                model,
                optimizer,
                raw_ce_loss,
                admission_mask,
                update_scale=optimizer_update_scale,
                auxiliary_loss=auxiliary_loss_for_update,
                auxiliary_weight=(
                    self.ssaw_auxiliary_weight if self.enable_ssaw else 0.0
                ),
                admitted_nonempty=has_admitted,
            )
        active_mask = (
            None
            if production_minimal
            else (
                admission_mask
                if bool(update_log["committed"])
                else torch.zeros_like(admission_mask)
            )
        )

        if (
            production_minimal
        ):
            self._last_gate_log = {
                "update_attempted": bool(update_log["attempted"]),
                "update_committed": bool(update_log["committed"]),
            }
            self._last_batch_log = {
                "dusafe_logging_mode": self.logging_mode,
                "production_output_profile": "minimal",
                "update_attempted": float(bool(update_log["attempted"])),
                "update_committed": float(bool(update_log["committed"])),
                "update_finite": float(bool(update_log["finite"])),
            }
            return prediction_logits

        # Production evaluation only needs the online masks and a compact set
        # of batch scalars. The tensors below are post-update diagnostics:
        # materializing every per-sample vector on the host cannot change an
        # adaptation decision, but it introduces repeated CUDA synchronizations.
        # Evidence runs retain the complete historical schema.
        if not self.evidence_logging and self.record_production_batch_diagnostics:
            compact_labels = torch.stack((pseudo_labels, semantic_predictions))
            compact_masks = torch.stack(
                (
                    confidence_mask,
                    semantic_mask,
                    source_semantic_mask,
                    ssaw_router_mask,
                    ssaw_consistency_mask,
                    base_admission_mask,
                    admission_mask,
                    active_mask,
                    ssaw_view_selected_mask,
                )
            )
            self._last_gate_log = {
                "pseudo_labels": compact_labels[0].detach(),
                "source_semantic_prediction": compact_labels[1].detach(),
                "confidence_mask": compact_masks[0].detach(),
                "semantic_mask": compact_masks[1].detach(),
                "source_semantic_router_mask": compact_masks[2].detach(),
                "ssaw_router_mask": compact_masks[3].detach(),
                "ssaw_consistency_mask": compact_masks[4].detach(),
                "base_admission_mask": compact_masks[5].detach(),
                "admission_mask": compact_masks[6].detach(),
                "active_mask": compact_masks[7].detach(),
                "ssaw_view_selected_mask": compact_masks[8].detach(),
                "selected_mask": compact_masks[7].detach(),
                "update_attempted": bool(update_log["attempted"]),
                "update_committed": bool(update_log["committed"]),
            }
            admitted_count_tensor = admission_mask.float().sum()
            compact_scalar_values = torch.stack(
                (
                    confidence_mask.float().mean(),
                    admission_mask.float().mean(),
                    active_mask.float().mean(),
                    ssaw_consistency_mask.float().mean(),
                    admitted_count_tensor,
                    adaptation_loss.detach(),
                    raw_ce_loss.detach(),
                    consistency_loss.detach(),
                    weighted_consistency_loss.detach(),
                    selected_view_fraction.detach(),
                )
            ).float()
            compact_scalar_names = (
                "confidence_pass_rate",
                "admission_rate",
                "active_rate",
                "ssaw_training_participation_rate",
                "admitted_count",
                "adaptation_loss",
                "raw_ce_loss",
                "ssaw_consistency_loss",
                "ssaw_weighted_consistency_loss",
                "ssaw_selected_view_fraction",
            )
            self._last_batch_log = {
                "_production_scalar_values": compact_scalar_values.detach(),
                "_production_scalar_names": compact_scalar_names,
            }
            self._last_batch_log.update(
                {
                    "dusafe_logging_mode": self.logging_mode,
                    "ssaw_consistency_weight": float(
                        self.ssaw_auxiliary_weight if self.enable_ssaw else 0.0
                    ),
                    "ssaw_candidate_count": float(
                        self.ssaw.last_metadata.get("view_count", 0)
                        if self.enable_ssaw
                        else 0
                    ),
                    "ssaw_candidate_forward_count": float(
                        self.ssaw.last_metadata.get("candidate_forward_count", 0)
                        if self.enable_ssaw
                        else 0
                    ),
                    "update_attempted": float(bool(update_log["attempted"])),
                    "update_committed": float(bool(update_log["committed"])),
                    "update_finite": float(bool(update_log["finite"])),
                }
            )
            return prediction_logits

        with torch.no_grad():
            raw_probabilities = raw_logits.softmax(dim=1)
            stress_probabilities = stress_logits.softmax(dim=1)
            kl_divergence = (
                raw_probabilities
                * (
                    raw_probabilities.clamp_min(1e-8).log()
                    - stress_probabilities.clamp_min(1e-8).log()
                )
            ).sum(dim=1)
            feature_distance = (
                stress_features - raw_features
            ).flatten(1).norm(dim=1)
            if self.enable_ssaw:
                metadata = self.ssaw.last_metadata
                vote_agreement = torch.as_tensor(
                    metadata["vote_agreement"], dtype=torch.float32
                )
                preserving_count = torch.as_tensor(
                    metadata["label_preserving_count"], dtype=torch.float32
                )
                selected_nll = torch.as_tensor(
                    metadata["selected_nll"], dtype=torch.float32
                )
                entropy_rise = torch.as_tensor(
                    metadata["entropy_rise"], dtype=torch.float32
                )
                label_flip = torch.as_tensor(
                    metadata["ssaw_label_flip"], dtype=torch.bool
                )
                stress_top2 = stress_logits.topk(
                    k=min(2, stress_logits.size(1)), dim=1
                ).values
                if stress_top2.size(1) == 1:
                    selected_margin = stress_top2[:, 0]
                else:
                    selected_margin = (
                        stress_top2[:, 0] - stress_top2[:, 1]
                    )
            else:
                size = raw_inputs.size(0)
                vote_agreement = torch.ones(size)
                preserving_count = torch.ones(size)
                selected_nll = raw_top1_nll.detach().cpu().float()
                entropy_rise = torch.zeros(size)
                label_flip = torch.zeros(size, dtype=torch.bool)
                selected_margin = raw_inputs.new_zeros(size)

        gate_float_names = (
            "raw_entropy",
            "ssaw_entropy",
            "ssaw_entropy_shift",
            "kl_divergence",
            "ssaw_feature_distance",
            "raw_top1_nll",
            "source_semantic_margin",
            "ssaw_selected_margin",
        )
        gate_float_values = torch.stack(
            (
                raw_entropy,
                stress_entropy,
                stress_entropy - raw_entropy,
                kl_divergence,
                feature_distance,
                raw_top1_nll,
                semantic_margin,
                selected_margin,
            )
        ).detach().cpu()
        gate_floats = {
            name: gate_float_values[index]
            for index, name in enumerate(gate_float_names)
        }
        gate_bool_names = (
            "ssaw_view_selected_mask",
            "confidence_mask",
            "semantic_mask",
            "source_semantic_router_mask",
            "ssaw_router_mask",
            "ssaw_consistency_mask",
            "base_admission_mask",
            "admission_mask",
            "active_mask",
            "ssaw_label_flip",
        )
        gate_bool_values = torch.stack(
            (
                ssaw_view_selected_mask,
                confidence_mask,
                semantic_mask,
                source_semantic_mask,
                ssaw_router_mask,
                ssaw_consistency_mask,
                base_admission_mask,
                admission_mask,
                active_mask,
                ssaw_label_flip,
            )
        ).detach().cpu()
        gate_bools = {
            name: gate_bool_values[index]
            for index, name in enumerate(gate_bool_names)
        }
        gate_label_values = torch.stack(
            (pseudo_labels, semantic_predictions)
        ).detach().cpu()

        self._last_gate_log = {
            "pseudo_labels": gate_label_values[0],
            **gate_floats,
            "ssaw_vote_agreement": vote_agreement,
            "ssaw_label_preserving_count": preserving_count,
            "ssaw_selected_nll": selected_nll,
            "ssaw_entropy_rise": entropy_rise,
            **gate_bools,
            "source_semantic_prediction": gate_label_values[1],
            "selected_mask": gate_bools["active_mask"],
            "update_attempted": bool(update_log["attempted"]),
            "update_committed": bool(update_log["committed"]),
        }

        admitted_count_tensor = admission_mask.float().sum()
        admitted_denominator = admitted_count_tensor.clamp_min(1.0)
        batch_scalar_names = (
            "raw_entropy_mean",
            "ssaw_entropy_mean",
            "ssaw_view_selection_rate",
            "ssaw_prediction_kl_mean",
            "ssaw_label_flip_rate",
            "ssaw_training_participation_rate",
            "ssaw_admitted_participation_rate",
            "confidence_nll_threshold",
            "confidence_pass_rate",
            "semantic_pass_rate",
            "source_semantic_router_pass_rate",
            "ssaw_router_pass_rate",
            "source_semantic_margin_mean",
            "admission_rate",
            "active_rate",
            "admitted_count",
            "adaptation_loss",
            "raw_ce_loss",
            "ssaw_consistency_loss",
            "ssaw_weighted_consistency_loss",
            "ssaw_realized_consistency_ratio",
            "ssaw_selected_view_fraction",
            "base_admission_rate",
        )
        batch_scalar_values = torch.stack(
            (
                raw_entropy.mean(),
                stress_entropy.mean(),
                ssaw_view_selected_mask.float().mean(),
                ssaw_kl.mean(),
                ssaw_label_flip.float().mean(),
                ssaw_consistency_mask.float().mean(),
                ssaw_consistency_mask.float().sum()
                / admitted_denominator,
                self.confidence_nll_threshold.detach(),
                confidence_mask.float().mean(),
                semantic_mask.float().mean(),
                source_semantic_mask.float().mean(),
                ssaw_router_mask.float().mean(),
                semantic_margin.mean(),
                admission_mask.float().mean(),
                active_mask.float().mean(),
                admitted_count_tensor,
                adaptation_loss.detach(),
                raw_ce_loss.detach(),
                consistency_loss.detach(),
                weighted_consistency_loss.detach(),
                realized_consistency_ratio.detach(),
                selected_view_fraction.detach(),
                base_admission_mask.float().mean(),
            )
        ).float().cpu()
        batch_scalars = {
            name: float(batch_scalar_values[index])
            for index, name in enumerate(batch_scalar_names)
        }
        self._last_batch_log = {
            **batch_scalars,
            "ssaw_selected_nll_mean": float(selected_nll.mean().item()),
            "ssaw_hard_view_loss": batch_scalars[
                "ssaw_consistency_loss"
            ],
            "ssaw_consistency_weight": float(
                self.ssaw_auxiliary_weight if self.enable_ssaw else 0.0
            ),
            "ssaw_hard_view_weight": float(
                self.ssaw_auxiliary_weight if self.enable_ssaw else 0.0
            ),
            "ssaw_effective_training_mass": float(
                batch_scalars["ssaw_selected_view_fraction"]
                * (self.ssaw_auxiliary_weight if self.enable_ssaw else 0.0)
            ),
            "optimizer_update_scale": float(
                update_log["update_scale"]
            ),
            "ssaw_gradient_available": float(
                bool(update_log.get("auxiliary_available", False))
            ),
            "ssaw_gradient_applied": float(
                bool(update_log.get("auxiliary_gradient_applied", False))
            ),
            "ssaw_view_reused": float(
                bool(
                    self.enable_ssaw
                    and self.ssaw.last_metadata.get("reused_view", False)
                )
            ),
            "update_attempted": float(bool(update_log["attempted"])),
            "update_committed": float(bool(update_log["committed"])),
            "update_finite": float(bool(update_log["finite"])),
            "ssaw_view_count": float(
                self.ssaw.last_metadata.get("view_count", 0)
                if self.enable_ssaw
                else 0
            ),
            **(
                self._batch_gradient_diagnostics
                if self.record_gradient_diagnostics
                and self._batch_gradient_diagnostics is not None
                else {}
            ),
        }
        return prediction_logits

    def forward(self, inputs, trg_idx=None):
        """Run inner updates and retain safety decisions from every step."""
        outputs = None
        pseudo_labels = []
        admission_masks = []
        active_masks = []
        confidence_masks = []
        semantic_masks = []
        base_admission_masks = []
        batch_logs = []
        production_minimal = bool(
            not self.evidence_logging
            and not self.record_production_batch_diagnostics
        )
        if self.enable_ssaw:
            self.ssaw.clear_cached_view()
        self._batch_transaction_active = (
            self.update_transaction_scope == "batch"
        )
        self._batch_transaction_failed = False
        self._batch_update_snapshot = None
        self._batch_gradient_diagnostics = None
        try:
            for step_index in range(self.steps):
                outputs = self.forward_and_adapt(
                    inputs,
                    self.model,
                    self.optimizer,
                    trg_idx,
                    reuse_ssaw_view=bool(step_index),
                )
                if self.evidence_logging:
                    pseudo_labels.append(self._last_gate_log["pseudo_labels"])
                    admission_masks.append(self._last_gate_log["admission_mask"])
                    active_masks.append(self._last_gate_log["active_mask"])
                    confidence_masks.append(
                        self._last_gate_log["confidence_mask"]
                    )
                    semantic_masks.append(self._last_gate_log["semantic_mask"])
                    base_admission_masks.append(
                        self._last_gate_log["base_admission_mask"]
                    )
                batch_logs.append(
                    self._last_batch_log
                    if production_minimal
                    else dict(self._last_batch_log)
                )
        finally:
            self._batch_transaction_active = False
            self._batch_transaction_failed = False
            self._batch_update_snapshot = None

        if not self.evidence_logging and self.record_production_batch_diagnostics:
            # Only the final-step per-sample decisions are exposed in
            # production. Pack labels and masks into one host transfer after
            # all inner updates instead of synchronizing every step.
            label_keys = (
                "pseudo_labels",
                "source_semantic_prediction",
            )
            mask_keys = (
                "confidence_mask",
                "semantic_mask",
                "source_semantic_router_mask",
                "ssaw_router_mask",
                "ssaw_consistency_mask",
                "base_admission_mask",
                "admission_mask",
                "active_mask",
                "ssaw_view_selected_mask",
            )
            packed_gate_log = torch.cat(
                (
                    torch.stack(
                        [self._last_gate_log[key] for key in label_keys]
                    ),
                    torch.stack(
                        [self._last_gate_log[key] for key in mask_keys]
                    ).to(dtype=torch.long),
                ),
                dim=0,
            ).detach().cpu()
            update_attempted = bool(
                self._last_gate_log["update_attempted"]
            )
            update_committed = bool(
                self._last_gate_log["update_committed"]
            )
            final_gate_log = {
                key: packed_gate_log[index]
                for index, key in enumerate(label_keys)
            }
            mask_offset = len(label_keys)
            final_gate_log.update(
                {
                    key: packed_gate_log[mask_offset + index].bool()
                    for index, key in enumerate(mask_keys)
                }
            )
            final_gate_log.update(
                {
                    "selected_mask": final_gate_log["active_mask"],
                    "update_attempted": update_attempted,
                    "update_committed": update_committed,
                }
            )
            self._last_gate_log = final_gate_log

            scalar_names = batch_logs[-1]["_production_scalar_names"]
            scalar_values_cpu = torch.stack(
                [log["_production_scalar_values"] for log in batch_logs]
            ).cpu()
            final_batch_log = dict(batch_logs[-1])
            final_batch_log.pop("_production_scalar_values", None)
            final_batch_log.pop("_production_scalar_names", None)
            final_batch_log.update(
                {
                    name: float(
                        sum(
                            float(scalar_values_cpu[step_index, value_index])
                            for step_index in range(self.steps)
                        )
                        / self.steps
                    )
                    for value_index, name in enumerate(scalar_names)
                }
            )
            self._last_batch_log = final_batch_log
        elif not self.evidence_logging:
            self._last_batch_log = dict(batch_logs[-1])

        self._last_gate_log["inner_step_count"] = self.steps
        if self.evidence_logging:
            self._last_gate_log.update(
                {
                    "inner_pseudo_labels": torch.stack(pseudo_labels),
                    "inner_admission_masks": torch.stack(admission_masks),
                    "inner_active_masks": torch.stack(active_masks),
                    # Evidence mode keeps the complete step-by-sample path for
                    # safety audits. Production mode intentionally omits it.
                    "inner_confidence_masks": torch.stack(confidence_masks),
                    "inner_semantic_masks": torch.stack(semantic_masks),
                    "inner_base_admission_masks": torch.stack(
                        base_admission_masks
                    ),
                }
            )
        if self.steps > 1:
            numeric_keys = set.intersection(
                *(set(log) for log in batch_logs)
            )
            self._last_batch_log.update(
                {
                    key: float(
                        sum(float(log[key]) for log in batch_logs)
                        / self.steps
                    )
                    for key in numeric_keys
                    if all(
                        isinstance(log[key], (int, float))
                        for log in batch_logs
                    )
                }
            )
        return outputs


__all__ = [
    "DuSafe",
    "SSAWPhysicalView",
    "collect_source_confidence_metadata",
    "collect_source_semantic_metadata",
    "SOURCE_CONFIDENCE_METADATA_VERSION",
    "SOURCE_SEMANTIC_METADATA_VERSION",
]
