"""CPU-only audit for HHAR processed windows and normalization provenance.

The audit is safe to run before data conversion.  Missing ``train_i.pt`` /
``test_i.pt`` files are reported as ``missing`` rather than fabricated; this
lets protocol configuration tests run in environments without the dataset.
No target-domain statistics are computed or fitted by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data_model_configs import HHAR  # noqa: E402
from scripts.hhar_protocol import (  # noqa: E402
    HHAR_DOMAIN_IDS,
    HHAR_LABEL_MAP,
    HHAR_SAMPLE_LENGTH,
    source_normalization_manifest,
    validate_source_normalization_manifest,
)
from utils.utils import safe_torch_load  # noqa: E402


PROVENANCE_MANIFEST_NAME = "source_normalization_manifest.json"


def _as_samples(payload: Mapping, path: Path) -> torch.Tensor:
    if "samples" not in payload:
        raise ValueError(f"{path}: missing samples")
    samples = torch.as_tensor(payload["samples"])
    if samples.ndim != 3:
        raise ValueError(f"{path}: samples must be rank-3, got {tuple(samples.shape)}")
    valid = {(3, HHAR_SAMPLE_LENGTH), (HHAR_SAMPLE_LENGTH, 3)}
    if tuple(samples.shape[1:]) not in valid:
        raise ValueError(
            f"{path}: expected [N,3,128] or [N,128,3], got {tuple(samples.shape)}"
        )
    if not torch.isfinite(samples.float()).all().item():
        raise ValueError(f"{path}: samples contain NaN or infinity")
    return samples


def _as_labels(payload: Mapping, sample_count: int, path: Path) -> torch.Tensor:
    if "labels" not in payload:
        raise ValueError(f"{path}: missing labels")
    labels = torch.as_tensor(payload["labels"])
    if labels.ndim != 1 or labels.numel() != sample_count:
        raise ValueError(
            f"{path}: labels must be [N] aligned to samples; got {tuple(labels.shape)}"
        )
    if not torch.isfinite(labels.float()).all().item():
        raise ValueError(f"{path}: labels contain NaN or infinity")
    if not torch.equal(labels, labels.long()):
        raise ValueError(f"{path}: labels must be integer-valued")
    if labels.numel() and (
        labels.min().item() < 0 or labels.max().item() >= len(HHAR_LABEL_MAP)
    ):
        raise ValueError(f"{path}: labels must be in [0, 5]")
    return labels.long()


def audit_payload(payload: Mapping, *, domain: str, split: str, path: Path) -> dict:
    """Validate one processed tensor payload without moving data to GPU."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a mapping payload")
    samples = _as_samples(payload, path)
    labels = _as_labels(payload, samples.shape[0], path)
    declared_domain = payload.get("domain")
    if declared_domain is not None and str(declared_domain) != str(domain):
        raise ValueError(
            f"{path}: declared domain {declared_domain!r} != {domain!r}"
        )
    declared_split = payload.get("split")
    if declared_split is not None and str(declared_split) != split:
        raise ValueError(f"{path}: declared split {declared_split!r} != {split!r}")
    if payload.get("normalization_applied") is True:
        raise ValueError(f"{path}: converter output must remain unstandardized raw windows")
    return {
        "domain": str(domain),
        "split": split,
        "path": str(path),
        "samples": int(samples.shape[0]),
        "shape": [int(value) for value in samples.shape],
        "labels_present": sorted(int(value) for value in labels.unique().tolist()),
        "normalization_applied": bool(payload.get("normalization_applied", False)),
    }


def audit_hhar_dataset(
    data_dir: str | Path,
    *,
    require_all_files: bool = True,
    provenance_path: str | Path | None = None,
) -> dict:
    """Return a JSON-serializable schema/provenance audit report."""

    data_dir = Path(data_dir)
    report = {
        "dataset": "HHAR",
        "data_dir": str(data_dir.resolve()),
        "expected_domains": list(HHAR_DOMAIN_IDS),
        "expected_num_classes": HHAR.num_classes,
        "expected_input_channels": HHAR.input_channels,
        "expected_sequence_len": HHAR.sequence_len,
        "normalization_reference": "source",
        "files": [],
        "missing_files": [],
        "errors": [],
    }
    for domain in HHAR_DOMAIN_IDS:
        for split in ("train", "test"):
            path = data_dir / f"{split}_{domain}.pt"
            if not path.is_file():
                report["missing_files"].append(str(path))
                continue
            try:
                payload = safe_torch_load(path, map_location="cpu")
                report["files"].append(
                    audit_payload(payload, domain=domain, split=split, path=path)
                )
            except Exception as exc:  # report all files in one audit pass
                report["errors"].append({"path": str(path), "error": str(exc)})

    manifest_path = (
        Path(provenance_path)
        if provenance_path is not None
        else data_dir / PROVENANCE_MANIFEST_NAME
    )
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_source_normalization_manifest(manifest)
            report["provenance"] = manifest
        except Exception as exc:
            report["errors"].append(
                {"path": str(manifest_path), "error": str(exc)}
            )
    else:
        report["provenance"] = source_normalization_manifest()
        report["provenance_status"] = "expected_manifest_missing"

    report["status"] = (
        "ok"
        if not report["errors"]
        and (not require_all_files or not report["missing_files"])
        else "missing"
        if not report["errors"] and report["missing_files"]
        else "invalid"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--provenance", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success for a missing processed dataset; useful before conversion.",
    )
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_hhar_dataset(
        args.data_dir,
        require_all_files=not args.allow_missing,
        provenance_path=args.provenance,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_hhar_dataset", "audit_payload", "build_parser", "main"]
