"""Replace selected fixed-table cells with an explicitly diagnostic overlay."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_main_table import JOB_KEY_COLUMNS, _write_rows, analyze


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(JOB_KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing key columns: {missing}")
    return frame


def _key_set(frame: pd.DataFrame) -> set[tuple]:
    return {
        tuple(row[column] for column in JOB_KEY_COLUMNS)
        for row in frame.to_dict("records")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overlay-dataset", default="HAR")
    parser.add_argument("--overlay-method", default="DuSafe")
    parser.add_argument("--reference-method", default="DuSafe")
    parser.add_argument(
        "--selection-provenance",
        default="target-label-selected oracle diagnostic",
    )
    args = parser.parse_args(argv)
    base_dir = Path(args.base_dir)
    overlay_dir = Path(args.overlay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "per_source_seed_results.csv"
    base = _load(base_dir / filename)
    overlay_all = _load(overlay_dir / filename)
    overlay = overlay_all[
        overlay_all["dataset"].astype(str).eq(args.overlay_dataset)
        & overlay_all["method"].astype(str).eq(args.overlay_method)
    ].copy()
    if overlay.empty:
        raise ValueError("The requested overlay contains no matching rows")
    unexpected = overlay_all.drop(index=overlay.index)
    if not unexpected.empty:
        raise ValueError("Overlay directory contains cells outside its declared scope")
    base_scope = base[
        base["dataset"].astype(str).eq(args.overlay_dataset)
        & base["method"].astype(str).eq(args.overlay_method)
    ]
    if _key_set(base_scope) != _key_set(overlay):
        raise ValueError("Overlay keys do not exactly match the fixed-table scope")
    overlay_keys = _key_set(overlay)
    retained = base[
        ~base.apply(
            lambda row: tuple(row[column] for column in JOB_KEY_COLUMNS)
            in overlay_keys,
            axis=1,
        )
    ]
    merged = pd.concat([retained, overlay], ignore_index=True)
    if len(_key_set(merged)) != len(merged):
        raise ValueError("Merged table contains duplicate job keys")
    _write_rows(merged.to_dict("records"), output_dir / filename)
    analyze(merged, output_dir, reference_method=args.reference_method)
    manifest = {
        "protocol": "main-table diagnostic overlay v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir.resolve()),
        "overlay_dir": str(overlay_dir.resolve()),
        "overlay_dataset": args.overlay_dataset,
        "overlay_method": args.overlay_method,
        "selection_provenance": args.selection_provenance,
        "target_labels_used_for_overlay_selection": True,
        "valid_use": "oracle diagnostic and failure analysis only",
        "invalid_use": "unbiased fixed-source main-table claim",
        "rows": int(len(merged)),
    }
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
