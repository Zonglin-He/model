"""Build a HAR safety diagnostic by overlaying tuned DuSafe cells."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SUMMARY_KEYS = (
    "dataset",
    "scenario",
    "method",
    "variant",
    "corruption",
    "severity",
    "source_seed",
    "stream_seed",
    "corruption_seed",
)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _replace(base: pd.DataFrame, overlay: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    missing = sorted(set(keys) - set(base.columns))
    missing += sorted(set(keys) - set(overlay.columns))
    if missing:
        raise ValueError(f"Missing merge key columns: {sorted(set(missing))}")
    overlay_index = pd.MultiIndex.from_frame(overlay[list(keys)])
    base_index = pd.MultiIndex.from_frame(base[list(keys)])
    retained = base[~base_index.isin(overlay_index)]
    merged = pd.concat([retained, overlay], ignore_index=True)
    if merged.duplicated(list(keys)).any():
        raise ValueError("Overlay merge produced duplicate cells")
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="HAR")
    parser.add_argument("--method", default="DuSafe")
    args = parser.parse_args(argv)
    base_dir = Path(args.base_dir)
    overlay_dir = Path(args.overlay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_summary = _read(base_dir / "summary_raw.csv")
    overlay_summary_all = _read(overlay_dir / "summary_raw.csv")
    base_summary = base_summary[
        base_summary["dataset"].astype(str).eq(args.dataset)
    ].copy()
    overlay_summary = overlay_summary_all[
        overlay_summary_all["dataset"].astype(str).eq(args.dataset)
        & overlay_summary_all["method"].astype(str).eq(args.method)
    ].copy()
    if overlay_summary.empty or len(overlay_summary) != len(overlay_summary_all):
        raise ValueError("Safety overlay contains missing or out-of-scope rows")
    merged_summary = _replace(base_summary, overlay_summary, SUMMARY_KEYS)
    merged_summary.to_csv(output_dir / "summary_raw.csv", index=False)

    base_aurc = _read(base_dir / "aurc_per_source_seed.csv")
    overlay_aurc_all = _read(overlay_dir / "aurc_per_source_seed.csv")
    base_aurc = base_aurc[base_aurc["dataset"].astype(str).eq(args.dataset)]
    overlay_aurc = overlay_aurc_all[
        overlay_aurc_all["dataset"].astype(str).eq(args.dataset)
        & overlay_aurc_all["method"].astype(str).eq(args.method)
    ]
    merged_aurc = _replace(base_aurc, overlay_aurc, SUMMARY_KEYS)
    merged_aurc.to_csv(output_dir / "aurc_per_source_seed.csv", index=False)

    manifest = {
        "protocol": "controlled-safety diagnostic overlay v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir.resolve()),
        "overlay_dir": str(overlay_dir.resolve()),
        "dataset": args.dataset,
        "method": args.method,
        "target_labels_used_for_overlay_selection": True,
        "valid_use": "oracle diagnostic only",
        "invalid_use": "unbiased fixed-source safety claim",
        "summary_rows": int(len(merged_summary)),
        "aurc_rows": int(len(merged_aurc)),
    }
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
