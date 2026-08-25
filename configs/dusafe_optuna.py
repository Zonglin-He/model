"""Wide, one-coordinate-at-a-time search spaces for DuSafe.

The production algorithm remains identical across datasets.  Only numeric
dataset-level values are searched.  Lists are deliberately explicit so an
Optuna GridSampler evaluates every value once instead of hiding the actual
budget behind a continuous sampler.
"""

from __future__ import annotations

SOURCE_PARAMETER_ORDER = (
    "pre_learning_rate",
    "batch_size",
    "num_epochs",
    "weight_decay",
)

TTA_PARAMETER_ORDER = (
    "batch_size",
    "learning_rate",
    "steps",
    "ssaw_auxiliary_weight",
    "weight_decay",
    "grad_clip",
    "spline_log_strength",
    "spline_control_points",
    "spline_num_directions",
    "confidence_keep_fraction",
)


_SOURCE_LR = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]

_TTA_LR = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]

_WEIGHT_DECAY = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

_SOURCE_BATCH = {
    "EEG": [16, 24, 32, 48, 64, 80, 96, 128, 160, 192],
    "HAR": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    "FD": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192],
    # Reuse the conservative HAR grid as a search scaffold.  HHAR values are
    # not a tuned profile; the integration queue remains dry-run by default.
    "HHAR": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
}

_TTA_BATCH = {
    "EEG": [8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256],
    "HAR": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    "FD": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256],
    "HHAR": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
}


def source_search_space(dataset: str) -> dict[str, list[int | float]]:
    """Return the exhaustive source-training coordinate grids."""
    dataset = str(dataset).upper()
    return {
        "pre_learning_rate": list(_SOURCE_LR),
        "batch_size": list(_SOURCE_BATCH[dataset]),
        "num_epochs": [10, 20, 30, 40, 60, 80, 100, 140, 180, 240, 320],
        "weight_decay": list(_WEIGHT_DECAY),
    }


def tta_search_space(dataset: str) -> dict[str, list[int | float]]:
    """Return the exhaustive TTA and SSAW coordinate grids."""
    dataset = str(dataset).upper()
    return {
        "batch_size": list(_TTA_BATCH[dataset]),
        "learning_rate": list(_TTA_LR),
        "steps": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "ssaw_auxiliary_weight": [
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            8.0,
            12.0,
        ],
        "weight_decay": list(_WEIGHT_DECAY),
        "grad_clip": [0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0],
        "spline_log_strength": [
            0.02,
            0.05,
            0.08,
            0.12,
            0.16,
            0.20,
            0.25,
            0.30,
            0.40,
        ],
        "spline_control_points": [3, 4, 6, 8, 10, 12, 16],
        "spline_num_directions": [1, 2, 4, 6, 8],
        "confidence_keep_fraction": [
            0.80,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
            1.0,
        ],
    }


__all__ = [
    "SOURCE_PARAMETER_ORDER",
    "TTA_PARAMETER_ORDER",
    "source_search_space",
    "tta_search_space",
]
