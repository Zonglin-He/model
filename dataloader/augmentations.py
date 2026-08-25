"""Reference perturbations used only to compare SSAW with standard warps."""

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def _as_numpy_nlc(inputs):
    if torch.is_tensor(inputs):
        inputs = inputs.detach().cpu().numpy()
    return np.asarray(inputs).transpose(0, 2, 1)


def magnitude_warp(inputs, sigma=0.1, knot=8):
    values = _as_numpy_nlc(inputs)
    steps = np.arange(values.shape[1])
    knots = np.linspace(0, values.shape[1] - 1, int(knot) + 2)
    controls = np.random.normal(
        1.0,
        sigma,
        size=(values.shape[0], len(knots), values.shape[2]),
    )
    output = np.empty_like(values)
    for sample_index, sample in enumerate(values):
        curves = np.stack(
            [
                CubicSpline(knots, controls[sample_index, :, channel])(
                    steps
                )
                for channel in range(values.shape[2])
            ],
            axis=1,
        )
        output[sample_index] = sample * curves
    return output.transpose(0, 2, 1)


def time_warp(inputs, sigma=0.1, knot=8):
    values = _as_numpy_nlc(inputs)
    steps = np.arange(values.shape[1])
    knots = np.linspace(0, values.shape[1] - 1, int(knot) + 2)
    controls = np.random.normal(
        1.0,
        sigma,
        size=(values.shape[0], len(knots), values.shape[2]),
    )
    output = np.empty_like(values)
    for sample_index, sample in enumerate(values):
        for channel in range(values.shape[2]):
            warped_steps = CubicSpline(
                knots, knots * controls[sample_index, :, channel]
            )(steps)
            warped_steps *= (values.shape[1] - 1) / max(
                warped_steps[-1], 1e-8
            )
            output[sample_index, :, channel] = np.interp(
                steps,
                np.clip(warped_steps, 0, values.shape[1] - 1),
                sample[:, channel],
            )
    return output.transpose(0, 2, 1)


__all__ = ["magnitude_warp", "time_warp"]
