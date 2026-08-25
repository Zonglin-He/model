"""Run the CPU-only held-out SSAW mechanism panel on an NPZ tensor bundle.

The bundle must contain six arrays:

``clean_signal``, ``held_out_signal``
    ``[batch, channels, time]`` tensors.
``clean_logits``, ``held_out_logits``
    ``[batch, classes]`` model outputs.
``clean_features``, ``held_out_features``
    ``[batch, ...]`` representations from the same algorithm adapter.

Queue artifacts may additionally contain six label-free per-sample arrays
(``heldout_confidence_admitted_mask``, ``heldout_eligible_mask``,
``heldout_margin_ratio``, ``heldout_flip_rate``, ``heldout_worst_margin``,
and ``heldout_consistency``).  When present, the runner reports eligible
coverage among confidence-admitted anchors and the held-out Sobol direction
diagnostics; partial optional bundles are rejected.

This runner is intentionally a separate offline entry point.  It does not
instantiate DuSafe, does not train a model, and rejects ``--device cuda``.
Metadata and split identifiers are written into the manifest so held-out
trajectory/operator/seed separation is auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import numpy as np
import torch

# ``python scripts/run_heldout_ssaw_mechanism.py`` places ``scripts/`` first on
# sys.path; add the repository root explicitly so the runner works both as a
# module and as a direct CLI without relying on the caller's PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssaw_evaluation.heldout_mechanism import (
    HeldOutCase,
    build_manifest,
    compute_mechanism_metrics,
    summarize_heldout_direction_diagnostics,
    write_json,
)


REQUIRED_KEYS = (
    "clean_signal",
    "held_out_signal",
    "clean_logits",
    "held_out_logits",
    "clean_features",
    "held_out_features",
)
OPTIONAL_DIRECTION_KEYS = (
    "heldout_confidence_admitted_mask",
    "heldout_eligible_mask",
    "heldout_margin_ratio",
    "heldout_flip_rate",
    "heldout_worst_margin",
    "heldout_consistency",
)


def _load_metadata(raw: Optional[str]) -> Mapping[str, Any]:
    if raw is None:
        raise ValueError(
            "--metadata-json is required; sampling and operator metadata cannot be inferred"
        )
    path = Path(raw)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("metadata JSON must encode an object")
    return payload


def load_npz_bundle(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a CPU tensor bundle and fail closed on missing or object arrays."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as bundle:
        missing = sorted(set(REQUIRED_KEYS) - set(bundle.files))
        if missing:
            raise ValueError(f"NPZ bundle is missing required arrays: {missing}")
        tensors = {}
        for key in REQUIRED_KEYS:
            array = np.asarray(bundle[key])
            if array.dtype.kind in {"O", "U", "S"}:
                raise TypeError(f"NPZ array {key!r} must be numeric, not {array.dtype}")
            tensors[key] = torch.from_numpy(array).cpu()
        optional_present = [key in bundle.files for key in OPTIONAL_DIRECTION_KEYS]
        if any(optional_present) and not all(optional_present):
            missing_optional = [
                key for key, present in zip(OPTIONAL_DIRECTION_KEYS, optional_present)
                if not present
            ]
            raise ValueError(
                "NPZ bundle has partial held-out direction diagnostics; missing "
                f"arrays: {missing_optional}"
            )
        for key in OPTIONAL_DIRECTION_KEYS:
            if key in bundle.files:
                array = np.asarray(bundle[key])
                if array.dtype.kind in {"O", "U", "S"}:
                    raise TypeError(
                        f"NPZ array {key!r} must be numeric, not {array.dtype}"
                    )
                tensors[key] = torch.from_numpy(array).cpu()
    return tensors


def run_bundle(
    input_path: str | Path,
    *,
    case: HeldOutCase | Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a bundle and optionally write its JSON result."""

    tensors = load_npz_bundle(input_path)
    normalized_case = case
    metrics = compute_mechanism_metrics(
        normalized_case.dataset if isinstance(normalized_case, HeldOutCase) else normalized_case["dataset"],
        tensors["clean_signal"],
        tensors["held_out_signal"],
        clean_logits=tensors["clean_logits"],
        held_out_logits=tensors["held_out_logits"],
        clean_features=tensors["clean_features"],
        held_out_features=tensors["held_out_features"],
        metadata=(
            normalized_case.metadata
            if isinstance(normalized_case, HeldOutCase)
            else normalized_case["metadata"]
        ),
    )
    if all(key in tensors for key in OPTIONAL_DIRECTION_KEYS):
        metrics.update(
            summarize_heldout_direction_diagnostics(
                {
                    "confidence_admitted_mask": tensors[
                        "heldout_confidence_admitted_mask"
                    ],
                    "eligible_mask": tensors["heldout_eligible_mask"],
                    "margin_ratio": tensors["heldout_margin_ratio"],
                    "heldout_flip_rate": tensors["heldout_flip_rate"],
                    "heldout_worst_margin": tensors["heldout_worst_margin"],
                    "heldout_consistency": tensors["heldout_consistency"],
                }
            )
        )
    else:
        metrics.update(
            {
                "eligible_coverage": None,
                "margin_ratio": None,
                "heldout_flip_rate": None,
                "heldout_worst_margin": None,
                "heldout_consistency": None,
                "confidence_admitted_count": None,
                "eligible_count": None,
            }
        )
    manifest = build_manifest(
        normalized_case,
        metrics=metrics,
        input_shape=tensors["clean_signal"].shape,
    )
    result = {"manifest": manifest, "metrics": metrics}
    if output_path is not None:
        write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only held-out SSAW physical mechanism evaluation",
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, help="NPZ tensor bundle")
    parser.add_argument("--output", required=True, help="JSON result/manifest path")
    parser.add_argument("--dataset", required=True, choices=("EEG", "HAR", "FD", "HHAR"))
    parser.add_argument("--training-view-family", required=True)
    parser.add_argument("--held-out-view-family", required=True)
    parser.add_argument("--held-out-trajectory", required=True)
    parser.add_argument("--held-out-operator", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--test-seed", required=True, type=int)
    parser.add_argument("--algorithm", default="offline_bundle")
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="JSON object or path containing sampling/operator metadata",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Must remain cpu; this flag exists to fail closed if a GPU is requested",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if str(args.device).strip().lower() != "cpu":
        parser.error("held-out mechanism runner is CPU-only; GPU execution is disabled")
    metadata = _load_metadata(args.metadata_json)
    case = HeldOutCase(
        dataset=args.dataset,
        training_view_family=args.training_view_family,
        held_out_view_family=args.held_out_view_family,
        held_out_trajectory=args.held_out_trajectory,
        held_out_operator=args.held_out_operator,
        training_seed=args.training_seed,
        test_seed=args.test_seed,
        metadata=metadata,
        algorithm=args.algorithm,
    )
    result = run_bundle(args.input, case=case, output_path=args.output)
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "load_npz_bundle", "main", "run_bundle"]
