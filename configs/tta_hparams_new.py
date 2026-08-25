"""Repository defaults for fixed-source DuSafe experiments.

These values are the dataset-level fallback.  Registered paper experiments
may overlay TTA-only per-flow values from ``paper_flow_profiles_v2.json``;
source-training settings remain tied to the fixed source checkpoint.
"""

from copy import deepcopy


def get_hparams_class(dataset_name):
    try:
        return globals()[dataset_name]
    except KeyError as exc:
        raise NotImplementedError(f"Dataset not found: {dataset_name}") from exc


_COMMON_DUSAFE = {
    "optim_method": "adam",
    # Every dataset uses the same sampled-spline residual-KL algorithm. The
    # paired no-SSAW runner selects ``confidence_raw`` at runtime.
    "dusafe_variant": "spline_residual",
    "enable_adaptation": True,
    # Execution-only optimizations. ``fused`` reuses the differentiable
    # raw/view forwards for detached gates and diagnostics; it does not change
    # the objective. A single rollback snapshot covers all finite inner steps
    # of one deployment batch. Expensive per-parameter clip statistics remain
    # available for audits but are disabled in production timing.
    "dusafe_execution_mode": "fused",
    # Main-table/deployment runs keep only online masks and compact scalars.
    # Safety, mechanism, and replay runners explicitly override this to
    # ``evidence`` when they require per-sample or per-inner-step diagnostics.
    "dusafe_logging_mode": "production",
    # Main-table/deployment runs do not materialize target-label-dependent
    # per-sample safety rows. Evidence runners explicitly enable them.
    "record_per_sample_evidence": False,
    # Compact online masks remain available as an opt-in monitoring profile.
    # F1-only production and overhead runs avoid their device-to-host transfer.
    "record_production_batch_diagnostics": False,
    # Runtime profiler ranges are enabled transiently by the overhead
    # profiler, but omitted from ordinary production timing.
    "record_runtime_stage_markers": False,
    # Production may cache the eight mandatory largest-radius candidate
    # forwards as one sequential CUDA Graph. Each candidate remains its own
    # [B,C,T] BatchNorm batch. ``auto`` requires bitwise eager/graph checks and
    # keeps the graph only when a setup-time timing probe shows >=5% speedup.
    # Evidence runs always use the historical eager path.
    "ssaw_candidate_cuda_graph": "auto",
    # Capture plus the mandatory post-update exact self-test amortizes only on
    # streams with roughly ten or more full-shape candidate searches.
    "ssaw_candidate_cuda_graph_min_expected_searches": 10,
    # Very large level-zero tensors are bandwidth/compute bound rather than
    # launch bound; capturing them adds setup time and persistent graph memory.
    "ssaw_candidate_cuda_graph_max_static_input_mb": 24.0,
    # A bitwise-equivalent level-wise materializer remains available for
    # memory-constrained deployments. Formal profiles keep the faster dense
    # construction; measured EEG/FD latency and FLOPs did not improve lazily.
    "ssaw_lazy_candidate_materialization": False,
    # Small HAR/HHAR banks are faster as one vectorized allocation; large
    # EEG/FD banks use level-wise materialization to reduce memory traffic.
    "ssaw_lazy_candidate_min_bank_mb": 8.0,
    # Do not scan target labels before adaptation merely to print a histogram.
    # Labels remain available only to the post-hoc evaluator.
    "record_target_label_histogram": False,
    # Production retains only candidate decisions; evidence runners override
    # logging mode and keep the complete feature/probability diagnostics.
    "ssaw_production_decision_only": True,
    "update_transaction_scope": "batch",
    "record_optimizer_diagnostics": False,
    "normalization_reference": "source",
    "adapt_parameter_scope": "feature_extractor",
    # Current-batch normalization is explicit and stateless: source running
    # buffers are retained for guard evaluation but never updated on target.
    "bn_statistics": "batch",
    # The feature extractor is adapted while the classifier remains frozen.
    "enable_ssaw": True,
    "ssaw_sobol_seed": 1729,
    # SSAW is an auxiliary objective; raw pseudo-label CE remains the
    # optimization anchor for every confidence-admitted sample.
    # Confidence admission is calibrated once from fixed source data.
    "enable_confidence_gate": True,
    "confidence_reference_samples": 4096,
    # Unified sampled-spline search. Candidate directions are drawn once per
    # deployment batch and reused across inner steps; there is no coefficient
    # gradient refinement.
    "spline_control_points": 10,
    "spline_num_directions": 4,
    "spline_log_strength": 0.20,
    "spline_radius_levels": [1.0, 0.5, 0.25],
}


# Dataset-level fallback values.  The paper runner overlays its registered
# per-flow TTA profile while preserving these values for unspecified fields.
_DATASET_DUSAFE = {
    "EEG": {
        "learning_rate": 2e-3,
        "steps": 2,
        "ssaw_auxiliary_weight": 0.003,
        "confidence_keep_fraction": 1.0,
    },
    "HAR": {
        # Raw updates and SSAW eligibility share confidence admission.
        "learning_rate": 3.325e-4,
        "steps": 2,
        "ssaw_auxiliary_weight": 0.1,
        "confidence_keep_fraction": 0.995,
    },
    "FD": {
        "learning_rate": 3e-6,
        "steps": 2,
        "ssaw_auxiliary_weight": 0.05,
        # Source-only FD calibration (clean + fixed 50% signal-freeze panel)
        # selected the highest candidate that preserved both F1 values within
        # 0.2 percentage points while minimizing clean-correct false rejection:
        # q=.95 vs the previous q=.90 baseline.
        "confidence_keep_fraction": 0.95,
    },
    "HHAR": {
        # These are conservative protocol-safe starting values for the new
        # HHAR integration.  They are not tuned or selected using HHAR target
        # labels. It uses the same sampled spline mechanism as every dataset.
        "learning_rate": 1e-4,
        "steps": 1,
        "ssaw_auxiliary_weight": 1.0,
        "confidence_keep_fraction": 1.0,
    },
}


# Supervised source training is independent of the deployment stream.  These
# dataset-level recipes are fixed before adaptation; every compared TTA method
# must start from the same source-domain checkpoint for a paired run.
_SOURCE_TRAIN_PARAMS = {
    "EEG": {
        "num_epochs": 320,
        "batch_size": 96,
        "weight_decay": 1e-7,
    },
    "HAR": {
        "num_epochs": 100,
        "batch_size": 16,
        "weight_decay": 1e-4,
    },
    "FD": {
        "num_epochs": 60,
        "batch_size": 64,
        "weight_decay": 1e-4,
    },
    "HHAR": {
        "num_epochs": 100,
        "batch_size": 16,
        "weight_decay": 1e-4,
    },
}


# Deployment defaults may be overridden per registered flow without changing
# the corresponding fixed source checkpoint.
_TARGET_RUNTIME_PARAMS = {
    "EEG": {
        "batch_size": 192,
        "weight_decay": 0.0,
        "grad_clip": 0.03,
        "grad_clip_value": None,
    },
    "HAR": {
        "batch_size": 48,
        "weight_decay": 1e-6,
        "grad_clip": 0.01,
        "grad_clip_value": None,
    },
    "FD": {
        "batch_size": 192,
        "weight_decay": 1e-4,
        "grad_clip": 0.03,
        "grad_clip_value": None,
    },
    "HHAR": {
        "batch_size": 48,
        "weight_decay": 1e-6,
        "grad_clip": 0.01,
        "grad_clip_value": None,
    },
}


_SOURCE_LEARNING_RATES = {
    "EEG": 5e-4,
    "HAR": 1e-4,
    "FD": 1e-2,
    "HHAR": 1e-4,
}


class _DatasetHParams:
    dataset_name = None

    def __init__(self):
        dataset_name = self.dataset_name
        self.source_train_params = deepcopy(
            _SOURCE_TRAIN_PARAMS[dataset_name]
        )
        self.train_params = deepcopy(_TARGET_RUNTIME_PARAMS[dataset_name])
        self.alg_hparams = {
            "DuSafe": {
                **deepcopy(_COMMON_DUSAFE),
                **deepcopy(_DATASET_DUSAFE[dataset_name]),
            },
            "NoAdap": {
                "pre_learning_rate": _SOURCE_LEARNING_RATES[dataset_name],
                "normalization_reference": "source",
            },
        }


class EEG(_DatasetHParams):
    dataset_name = "EEG"


class HAR(_DatasetHParams):
    dataset_name = "HAR"


class FD(_DatasetHParams):
    dataset_name = "FD"


class HHAR(_DatasetHParams):
    dataset_name = "HHAR"
