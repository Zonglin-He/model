"""Frozen HAR DuSafe profile used by formal reviewer experiments.

This snapshot is deliberately duplicated from the composed HAR configuration.
Formal runners validate it before launching so a later config edit cannot
silently turn the Full-vs-no-SSAW panel into another hyperparameter search.
"""

from __future__ import annotations

from configs.tta_hparams_new import get_hparams_class


PROFILE_ID = (
    "har_dusafe_confidence_admitted_spline_residual_steps2_lambda_0p1_v6"
)

FROZEN_HAR_TTA_PARAMS = {
    "adapt_parameter_scope": "feature_extractor",
    "batch_size": 48,
    "bn_statistics": "batch",
    "confidence_keep_fraction": 0.995,
    "confidence_reference_samples": 4096,
    "dusafe_variant": "spline_residual",
    "enable_adaptation": True,
    "enable_confidence_gate": True,
    "enable_ssaw": True,
    "dusafe_execution_mode": "fused",
    "grad_clip": 0.01,
    "grad_clip_value": None,
    "learning_rate": 3.325e-4,
    "normalization_reference": "source",
    "optim_method": "adam",
    "record_optimizer_diagnostics": False,
    "spline_control_points": 10,
    "spline_log_strength": 0.2,
    "spline_num_directions": 4,
    "spline_radius_levels": [1.0, 0.5, 0.25],
    "ssaw_auxiliary_weight": 0.1,
    "ssaw_sobol_seed": 1729,
    "steps": 2,
    "update_transaction_scope": "batch",
    "weight_decay": 1e-6,
}

# Development-panel evidence used when this profile was frozen.  These are
# paired Full-minus-no-SSAW means, not significance claims.
DEVELOPMENT_EFFECT = {
    "clean_f1_delta": 0.0,
    "screening_flows": ["2->11", "6->23", "7->13", "9->18", "12->16"],
    "flows": 5,
    "source_seeds": [1],
    "stream_seed": 42,
    "selection_rule": (
        "remove coefficient-gradient refinement after the five-flow HAR/EEG "
        "single-seed component audit"
    ),
    "confirmatory": False,
    "target_labels_used_for_profile_selection": True,
    "target_labels_used_online": False,
}


def composed_har_tta_params() -> dict:
    """Return the configuration that a HAR DuSafe trainer will receive."""

    hparams = get_hparams_class("HAR")()
    return {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
    }


def validate_frozen_har_profile() -> dict:
    """Fail if the checked-in HAR configuration drifted after selection."""

    observed = composed_har_tta_params()
    missing = sorted(set(FROZEN_HAR_TTA_PARAMS) - set(observed))
    mismatched = {
        key: {
            "expected": expected,
            "observed": observed.get(key),
        }
        for key, expected in FROZEN_HAR_TTA_PARAMS.items()
        if observed.get(key) != expected
    }
    if missing or mismatched:
        raise RuntimeError(
            "Frozen HAR profile drifted; "
            f"missing={missing}, mismatched={mismatched}"
        )
    return {key: observed[key] for key in FROZEN_HAR_TTA_PARAMS}
