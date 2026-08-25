"""Rebuild common predictive risk--coverage/AURC from saved safety records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_controlled_safety_benchmark import (  # noqa: E402
    read_csv_records,
    try_safety_job_key,
    write_common_predictive_risk_artifacts,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    records_dir = input_dir / "sample_records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"Missing sample-record directory: {records_dir}")
    safety_manifest_path = input_dir / "manifest.json"
    if not safety_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing completed safety manifest: {safety_manifest_path}"
        )
    safety_manifest = json.loads(
        safety_manifest_path.read_text(encoding="utf-8")
    )
    if not bool(safety_manifest.get("finalize_only")) or int(
        safety_manifest.get("requested_missing_job_count", -1)
    ) != 0:
        raise RuntimeError(
            "Safety benchmark is not a complete finalize-only panel"
        )
    summary_rows = read_csv_records(input_dir / "summary_raw.csv")
    expected_signatures = {}
    for row in summary_rows:
        key = try_safety_job_key(row)
        signature = str(row.get("protocol_signature", ""))
        if key is not None and signature:
            expected_signatures[key] = signature
    expected_count = int(safety_manifest.get("requested_job_count", -1))
    if len(expected_signatures) != expected_count:
        raise RuntimeError(
            "Signed summary/sample protocol is incomplete: "
            f"expected {expected_count}, found {len(expected_signatures)}"
        )
    curves, aurc = write_common_predictive_risk_artifacts(
        records_dir,
        input_dir,
        expected_signatures=expected_signatures,
    )
    payload = {
        "protocol": "controlled safety common predictive risk backfill v2",
        "input_dir": str(input_dir),
        "risk_policies": [
            "common_pre_update_top1_nll",
            "common_post_update_top1_nll",
        ],
        "cross_method_primary_policy": "common_post_update_top1_nll",
        "posthoc_only": True,
        "online_admission_policy": False,
        "signed_summary_jobs": int(len(expected_signatures)),
        "curve_rows": int(len(curves)),
        "aurc_rows": int(len(aurc)),
    }
    temporary = input_dir / ".predictive_risk_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(input_dir / "predictive_risk_manifest.json")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
