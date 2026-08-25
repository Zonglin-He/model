"""Build the redistributable processed-dataset release archive.

The archive deliberately excludes checkpoints, results, raw HHAR CSV files,
and duplicate HHAR extraction directories. PyTorch ``.pt`` files are already
ZIP-based containers, so ZIP_STORED avoids wasting CPU time on ineffective
recompression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
import zipfile


EXPECTED_DOMAINS = {
    "EEG": tuple(range(20)),
    "HAR": tuple(range(1, 31)),
    "FD": tuple(range(4)),
    "HHAR": tuple(range(9)),
}

HHAR_METADATA = (
    "HHAR_manifest.json",
    "schema_audit.json",
    "source_normalization_manifest.json",
)

LICENSE_NOTICE = """# Dataset sources and licenses

This archive contains processed tensors used by the DuSafe experiments. It
does not contain source checkpoints or experiment outputs. The source-code MIT
license does not apply to the dataset payloads.

| Directory | Source | License |
|---|---|---|
| EEG | Subject-wise Sleep Stage Data, https://doi.org/10.21979/N9/UD1IM9 | CC BY-NC 4.0 |
| HAR | UCI HAR Dataset Processed, https://doi.org/10.21979/N9/0SYHTZ | CC BY-NC 4.0 |
| FD | Machine Fault Diagnosis, https://doi.org/10.21979/N9/PU85XN | CC BY-NC 4.0 |
| HHAR | UCI Heterogeneity Activity Recognition, https://doi.org/10.24432/C5689X | CC BY 4.0 |

Retain attribution and comply with each upstream license. EEG, HAR, and FD are
restricted to non-commercial use by CC BY-NC 4.0.
"""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files(dataset_root: Path) -> list[Path]:
    files: list[Path] = []
    for dataset, domain_ids in EXPECTED_DOMAINS.items():
        directory = dataset_root / dataset
        if not directory.is_dir():
            raise FileNotFoundError(f"missing dataset directory: {directory}")
        expected_names = {
            f"{split}_{domain}.pt"
            for split in ("train", "test")
            for domain in domain_ids
        }
        actual_names = {path.name for path in directory.glob("*.pt")}
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            raise RuntimeError(
                f"{dataset} tensor set mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )
        files.extend(directory / name for name in sorted(expected_names))
    hhar_root = dataset_root / "HHAR"
    for name in HHAR_METADATA:
        path = hhar_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing HHAR provenance file: {path}")
        files.append(path)
    return files


def manifest_bytes(dataset_root: Path, files: list[Path]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=("relative_path", "bytes", "sha256")
    )
    writer.writeheader()
    for path in files:
        writer.writerow(
            {
                "relative_path": path.relative_to(dataset_root.parent.parent)
                .as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return stream.getvalue().encode("utf-8")


def build_archive(dataset_root: Path, output_path: Path) -> tuple[int, str]:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    files = payload_files(dataset_root)
    manifest = manifest_bytes(dataset_root, files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    prefix = "DuSafe_processed_datasets_20260825"
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        archive.writestr(f"{prefix}/DATASET_LICENSES.md", LICENSE_NOTICE)
        archive.writestr(f"{prefix}/DATASET_MANIFEST.csv", manifest)
        for path in files:
            relative = path.relative_to(dataset_root.parent.parent).as_posix()
            archive.write(path, arcname=f"{prefix}/{relative}")
    with zipfile.ZipFile(temporary, mode="r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"archive CRC validation failed: {bad_member}")
    size = temporary.stat().st_size
    if size >= 2 * 1024**3:
        raise RuntimeError(
            f"archive is {size} bytes and cannot be one GitHub Release asset"
        )
    temporary.replace(output_path)
    digest = sha256_file(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{digest}  {output_path.name}\n", encoding="ascii"
    )
    return size, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/Dataset")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/DuSafe_processed_datasets_20260825.zip"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size, digest = build_archive(args.dataset_root, args.output)
    print(f"archive={args.output.resolve()}")
    print(f"bytes={size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
