"""Fail-closed finalization for the logging-split paper-evidence v4 rerun.

The numerical aggregation is intentionally reused from the audited v3
finalizer.  This wrapper changes only the frozen protocol/result roots and
adds a logging-mode contract check before any table is emitted.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.finalize_paper_evidence_v3 as v3  # noqa: E402


RESULT_ROOT = ROOT / "results" / "paper_evidence_v4"
PROTOCOL_PATH = ROOT / "configs" / "paper_evidence_protocol_v4.json"


def _worker_specs(directory: Path) -> list[dict]:
    specs = []
    for path in sorted(directory.rglob("worker_spec.json")):
        specs.append(json.loads(path.read_text(encoding="utf-8")))
    return specs


def _require_logging_mode(directory: Path, expected: str, count: int) -> None:
    specs = _worker_specs(directory)
    if len(specs) != count:
        raise v3.EvidenceError(
            f"{directory.name} worker-spec count {len(specs)} != {count}"
        )
    observed = {
        str(spec.get("tta_config", {}).get("dusafe_logging_mode", ""))
        for spec in specs
    }
    if observed != {expected}:
        raise v3.EvidenceError(
            f"{directory.name} logging modes {sorted(observed)} != [{expected}]"
        )


def main() -> int:
    _require_logging_mode(RESULT_ROOT / "main_full_no_ssaw", "production", 120)
    _require_logging_mode(
        RESULT_ROOT / "core_ablation_har_hhar", "production", 120
    )
    v3.PROTOCOL_PATH = PROTOCOL_PATH
    v3.RESULT_ROOT = RESULT_ROOT
    v3.OUTPUT_DIR = RESULT_ROOT / "final"
    v3.MAIN_DIR = RESULT_ROOT / "main_full_no_ssaw"
    v3.CORE_DIR = RESULT_ROOT / "core_ablation_har_hhar"
    v3.SAFETY_DIR = RESULT_ROOT / "safety_har_12_to_16_physical_s3_s6"
    return v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
