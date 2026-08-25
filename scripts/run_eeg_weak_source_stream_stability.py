"""Run the weak-source stability audit on EEG 7->18 at alpha=0.20."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_fd_weak_source_stream_stability as runner  # noqa: E402
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    load_paper_flow_profiles,
    profile_for_flow,
)


DATASET = "EEG"
SCENARIO = "7->18"
TARGET_SAMPLES = 566
NUM_CLASSES = 5
PROTOCOL = "eeg_7_to_18_weak_source_stability_v1_alpha020"


def _runtime_hparams(profile_path: Path) -> dict[str, Any]:
    hparams = get_hparams_class(DATASET)()
    runtime = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    profiles = load_paper_flow_profiles(profile_path, datasets=[DATASET])
    runtime.update(profile_for_flow(profiles, DATASET, SCENARIO))
    runtime.update(
        {
            "spline_log_strength": 0.20,
            "enable_source_semantic_router": False,
            "dusafe_logging_mode": "production",
            "record_per_sample_evidence": False,
            "record_production_batch_diagnostics": True,
            "ssaw_candidate_cuda_graph": "off",
            "ssaw_production_decision_only": True,
        }
    )
    expected = {
        "batch_size": 192,
        "steps": 1,
        "learning_rate": 7.5e-4,
        "ssaw_auxiliary_weight": 0.1,
        "spline_log_strength": 0.20,
    }
    for key, value in expected.items():
        actual = runtime[key]
        if isinstance(value, float):
            if not np.isclose(float(actual), value):
                raise RuntimeError(f"registered {key} changed: {actual} != {value}")
        elif int(actual) != value:
            raise RuntimeError(f"registered {key} changed: {actual} != {value}")
    return runtime


def main(argv: list[str] | None = None) -> int:
    runner.DATASET = DATASET
    runner.SCENARIO = SCENARIO
    runner.TARGET_SAMPLES = TARGET_SAMPLES
    runner.NUM_CLASSES = NUM_CLASSES
    runner.PROTOCOL = PROTOCOL
    runner._runtime_hparams = _runtime_hparams
    effective = list(sys.argv[1:] if argv is None else argv)
    if "--output-dir" not in effective:
        effective.extend(
            [
                "--output-dir",
                str(
                    ROOT
                    / "results"
                    / "paper_evidence_v5"
                    / "eeg_7_to_18_weak_source_stability"
                ),
            ]
        )
    return runner.main(effective)


if __name__ == "__main__":
    raise SystemExit(main())
