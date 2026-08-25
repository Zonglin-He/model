"""Configuration and provenance for isolated reviewer baseline adapters.

The benchmark registry is deliberately separate from the production DuSafe
configuration.  Source-training settings are copied from the current
dataset-level profile so every benchmark method consumes the same fixed-source
checkpoint; adaptation defaults are the published/default values recorded by
the local official source audit and the historical pre-cleanup ports.
"""

from copy import deepcopy

from configs.tta_hparams_new import get_hparams_class as get_current_hparams_class


BENCHMARK_METHODS = (
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
)

# Keep these identifiers in one auditable place.  The clone commit and license
# are also recorded in third_party/official_baselines/BASELINE_SOURCES.md.
PROVENANCE = {
    "Tent": {
        "repository": "https://github.com/DequanWang/tent",
        "local_path": "third_party/official_baselines/TENT",
        "commit": "e9e926a668d85244c66a6d5c006efbd2b82e83e8",
        "license": "MIT",
        "historical_port": "4de8bad8:algorithms/tent_tta.py",
        "port_class": "adapted_historical_port",
        "time_series_substitution": None,
        "historical_differences": "Base API accepts trg_idx; no behavioral change intended.",
    },
    "EATA": {
        "repository": "https://github.com/mr-eggplant/EATA",
        "local_path": "third_party/official_baselines/EATA",
        "commit": "f739b3668cc7617e9b9f1979c1a358497a3472c3",
        "license": "MIT",
        "historical_port": "4de8bad8:algorithms/eata_accup.py",
        "port_class": "adapted_historical_port",
        "time_series_substitution": None,
        "historical_differences": "Fisher diagonal is computed from the fixed source train split by the overhead runner; no plain-L2 fallback is used.",
    },
    "SAR": {
        "repository": "https://github.com/mr-eggplant/SAR",
        "local_path": "third_party/official_baselines/SAR",
        "commit": "20f6e24b17525f34503510afccedc0629b67b7c4",
        "license": "BSD-3-Clause",
        "historical_port": "4de8bad8:algorithms/sar_tta.py",
        "port_class": "adapted_historical_port",
        "time_series_substitution": None,
        "historical_differences": "Base API accepts trg_idx; SAM wrapper is local and keeps the historical two-step update.",
    },
    "ACCUPOfficial": {
        "repository": "https://github.com/Tokenmw/ACCUP-main",
        "local_path": "third_party/official_baselines/ACCUPOfficial",
        "commit": "920c43c092c6aa96a7950d2e3c0df5c2e4216f99",
        "license": "MIT",
        "historical_port": "4de8bad8:algorithms/accup_official.py",
        "port_class": "adapted_historical_port",
        "time_series_substitution": None,
        "historical_differences": "Optional second time-series view is accepted; target labels remain unused.",
    },
    "CoTTA": {
        "repository": "https://github.com/qinenergy/cotta",
        "local_path": "third_party/official_baselines/CoTTA",
        "commit": "c212a204b32be4005092e4323105a24a29ad2952",
        "license": "MIT",
        "historical_port": "4de8bad8:algorithms/cotta_tta.py",
        "port_class": "time_series_substitution",
        "time_series_substitution": "smooth_amplitude_warp_for_image_augmentation",
        "historical_differences": "Image augmentation is replaced by the local smooth amplitude warp; Base API adds trg_idx.",
    },
    "SoTTA": {
        "repository": "https://github.com/taeckyung/SoTTA",
        "local_path": "third_party/official_baselines/SoTTA",
        "commit": "09d568f467cd0343d1af2d751fb7186d839817ae",
        "license": "MIT",
        "historical_port": "4de8bad8:algorithms/sotta_tta.py",
        "port_class": "time_series_substitution",
        "time_series_substitution": "smooth_amplitude_warp_for_image_augmentation",
        "historical_differences": "Time-series BN/memory port from 4de8bad8; official image data pipeline is not used.",
    },
    "RoTTA": {
        "repository": "https://github.com/BIT-DA/RoTTA",
        "local_path": "third_party/official_baselines/RoTTA",
        "commit": "67e34c900cdd355fc07e55edd4c577ea7b8ebcc9",
        "license": "MIT",
        "historical_port": "4de8bad8:algorithms/rotta_tta.py",
        "port_class": "time_series_substitution",
        "time_series_substitution": "smooth_amplitude_warp_for_image_augmentation",
        "historical_differences": "Time-series robust BN and smooth amplitude warp from 4de8bad8; official image transforms are not used.",
    },
    "COME": {
        "repository": "https://github.com/BlueWhaleLab/COME",
        "local_path": "third_party/official_baselines/COME",
        "commit": "409a19b71f62c765b1a5be62347a9455524ec176",
        "license": "unresolved",
        "historical_port": "4de8bad8:algorithms/come_tta.py",
        "port_class": "time_series_substitution",
        "time_series_substitution": "smooth_amplitude_warp_not_used; norm_scope_is_time_series",
        "historical_differences": "COME evidence objective is retained; normalization scope is the time-series norm adapter from 4de8bad8.",
    },
    "NOTE": {
        "repository": "https://github.com/TaesikGong/NOTE",
        "local_path": "third_party/official_baselines/NOTE",
        "commit": "a714a2a2a9406903ba787b0bc240a95dd0342de5",
        "license": "MIT",
        "historical_port": None,
        "port_class": "time_series_substitution",
        "time_series_substitution": "official_image_NOTE_BN_and_memory_rules_on_[B,C,T]_streams",
        "selection_semantics": "PBRS_pseudo_label_memory_acceptance",
    },
    "DuSafe": {
        "repository": "local production implementation",
        "local_path": "algorithms/dusafe.py",
        "commit": "working production registry reference",
        "license": "repository license",
        "historical_port": None,
        "port_class": "production_reference",
        "time_series_substitution": None,
        "selection_semantics": (
            "fixed_source_confidence_admission_with_unified_spline_"
            "hard_view_residual_consistency"
        ),
    },
}


_BASELINE_COMMON = {
    "optim_method": "adam",
    "weight_decay": 0.0,
    "grad_clip": 0.0,
    "grad_clip_value": None,
    "steps": 1,
    "normalization_reference": "source",
}


def _baseline_defaults(dataset_name: str) -> dict[str, dict]:
    current = get_current_hparams_class(dataset_name)()
    source_lr = float(current.alg_hparams["NoAdap"].get("pre_learning_rate", 1e-3))
    dataset_values = {
        "EEG": {
            "Tent": {"learning_rate": 2.5e-4, "grad_clip": 0.5},
            "EATA": {
                "learning_rate": 3e-4,
                "e_margin": 0.6437751649736402,
                "d_margin": 0.05,
                "fisher_alpha": 2000.0,
                "fisher_enabled": True,
                "fisher_samples": 2000,
                "adapt_keywords": ("classifier", "adapter"),
                "grad_clip": 0.5,
            },
            "SAR": {
                "learning_rate": 2.5e-4,
                "sar_margin_e0": -1.0,
                "sar_reset_constant_em": 0.2,
                "sar_rho": 0.05,
                "sar_adaptive": False,
                "sar_base_optimizer": "sgd",
                "grad_clip": 0.5,
            },
            "ACCUPOfficial": {
                "learning_rate": 1e-5,
                "filter_K": 50,
                "tau": 50.0,
                "temperature": 0.3,
            },
        },
        "HAR": {
            "Tent": {"learning_rate": 2.5e-4, "grad_clip": 1.0},
            "EATA": {
                "learning_rate": 3e-5,
                "e_margin": 0.716703787691222,
                "d_margin": 0.05,
                "fisher_alpha": 2000.0,
                "fisher_enabled": True,
                "fisher_samples": 2000,
                "adapt_keywords": ("classifier", "adapter"),
                "grad_clip": 1.0,
            },
            "SAR": {
                "learning_rate": 2.5e-4,
                "sar_margin_e0": -1.0,
                "sar_reset_constant_em": 0.2,
                "sar_rho": 0.05,
                "sar_adaptive": False,
                "sar_base_optimizer": "sgd",
                "grad_clip": 1.0,
            },
            "ACCUPOfficial": {
                "learning_rate": 3e-4,
                "filter_K": 10,
                "tau": 20.0,
                "temperature": 0.7,
            },
        },
        # HHAR benchmark defaults are a protocol-safe port of the existing
        # six-class phone-sensor baseline recipe.  They are not tuned on HHAR.
        "HHAR": {
            "Tent": {"learning_rate": 2.5e-4, "grad_clip": 1.0},
            "EATA": {
                "learning_rate": 3e-5,
                "e_margin": 0.716703787691222,
                "d_margin": 0.05,
                "fisher_alpha": 2000.0,
                "fisher_enabled": True,
                "fisher_samples": 2000,
                "adapt_keywords": ("classifier", "adapter"),
                "grad_clip": 1.0,
            },
            "SAR": {
                "learning_rate": 2.5e-4,
                "sar_margin_e0": -1.0,
                "sar_reset_constant_em": 0.2,
                "sar_rho": 0.05,
                "sar_adaptive": False,
                "sar_base_optimizer": "sgd",
                "grad_clip": 1.0,
            },
            "ACCUPOfficial": {
                "learning_rate": 3e-4,
                "filter_K": 10,
                "tau": 20.0,
                "temperature": 0.7,
            },
        },
        "FD": {
            "Tent": {"learning_rate": 2.5e-4, "grad_clip": 0.5},
            "EATA": {
                "learning_rate": 1e-4,
                "e_margin": 0.43944491546724396,
                "d_margin": 0.05,
                "fisher_alpha": 2000.0,
                "fisher_enabled": True,
                "fisher_samples": 2000,
                "adapt_keywords": ("classifier", "adapter"),
                "grad_clip": 0.5,
            },
            "SAR": {
                "learning_rate": 2.5e-4,
                "sar_margin_e0": -1.0,
                "sar_reset_constant_em": 0.2,
                "sar_rho": 0.05,
                "sar_adaptive": False,
                "sar_base_optimizer": "sgd",
                "grad_clip": 0.5,
            },
            "ACCUPOfficial": {
                "learning_rate": 3e-4,
                "filter_K": 100,
                "tau": 1.0,
                "temperature": 0.6,
            },
        },
    }[dataset_name]
    robust = {
        "CoTTA": {
            "learning_rate": 1e-3,
            "cotta_mt_alpha": 0.999,
            "cotta_restore_probability": 0.01,
            "cotta_anchor_threshold": 0.9,
            "cotta_num_augmentations": 32,
            "cotta_augmentation_sigma": 0.1,
            "cotta_control_points": 8,
            "cotta_train_all": True,
        },
        "SoTTA": {
            "learning_rate": 1e-3,
            "sotta_memory_size": 64,
            "sotta_update_frequency": 64,
            "sotta_confidence_threshold": 0.99,
            "sotta_temperature": 1.0,
            "sotta_rho": 0.05,
            "sotta_bn_momentum": 0.2,
        },
        "RoTTA": {
            "learning_rate": 1e-3,
            "rotta_memory_size": 64,
            "rotta_update_frequency": 64,
            "rotta_bn_alpha": 0.05,
            "rotta_nu": 0.001,
            "rotta_lambda_t": 1.0,
            "rotta_lambda_u": 1.0,
            "rotta_augmentation_sigma": 0.1,
        },
        "COME": {
            "learning_rate": 2.5e-4,
            "come_evidence_prior": "num_classes",
        },
        "NOTE": {
            # Defaults follow the official NOTE temporal-stream script.  The
            # memory and BN rules are adapted to this repository's batch API.
            "learning_rate": 1e-4,
            "note_memory_size": 64,
            "note_update_frequency": 64,
            "note_bn_momentum": 0.01,
            "note_memory_type": "PBRS",
            "note_epoch": 1,
            "note_temperature": 1.0,
            "note_use_learned_stats": True,
        },
    }
    values = {name: {**_BASELINE_COMMON, **settings} for name, settings in dataset_values.items()}
    values.update({name: {**_BASELINE_COMMON, **settings} for name, settings in robust.items()})
    for settings in values.values():
        settings["pre_learning_rate"] = source_lr
    return values


def get_benchmark_hparams_class(dataset_name):
    """Return a trainer-compatible hparams class for the benchmark registry."""
    current = get_current_hparams_class(dataset_name)
    benchmark_values = _baseline_defaults(dataset_name)

    class BenchmarkHParams:
        def __init__(self):
            current_instance = current()
            self.source_train_params = deepcopy(current_instance.source_train_params)
            self.train_params = deepcopy(current_instance.train_params)
            # Keep DuSafe in the benchmark profile because trainer source-stage
            # calibration metadata is defined against the production gates.
            self.alg_hparams = {
                "NoAdap": deepcopy(current_instance.alg_hparams["NoAdap"]),
                "DuSafe": deepcopy(current_instance.alg_hparams["DuSafe"]),
                **deepcopy(benchmark_values),
            }

    return BenchmarkHParams


__all__ = [
    "BENCHMARK_METHODS",
    "PROVENANCE",
    "get_benchmark_hparams_class",
]
