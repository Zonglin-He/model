"""Run HAR 12->16 under the Sleep-EDF cross-dataset safety protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import har_cross_dataset_corruptions as har_corruptions  # noqa: E402
from scripts import run_eeg_cross_dataset_safety_replication as template  # noqa: E402


PROTOCOL = "har_12_to_16_cross_dataset_safety_v1"
OUTPUT_DIR = (
    ROOT
    / "results"
    / "paper_evidence_v5"
    / "har_12_to_16_cross_dataset_safety"
)


def _install_protocol() -> None:
    original_payload = template._protocol_payload

    def protocol_payload(args):
        payload = original_payload(args)
        wrapper = Path(__file__).resolve()
        corruption_code = (
            ROOT / "dataloader" / "har_cross_dataset_corruptions.py"
        )
        payload["input_files"]["corruption_code"] = {
            "path": str(corruption_code),
            "sha256": template._sha256(corruption_code),
        }
        payload["input_files"]["wrapper_code"] = {
            "path": str(wrapper),
            "sha256": template._sha256(wrapper),
        }
        signature_source = dict(payload)
        signature_source.pop("registered_at_utc")
        signature_source.pop("protocol_sha256", None)
        encoded = json.dumps(
            signature_source,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    template.PROTOCOL = PROTOCOL
    template.DATASET = "HAR"
    template.SCENARIO = "12->16"
    template.corruptions = har_corruptions
    template._protocol_payload = protocol_payload


def main(argv: list[str] | None = None) -> int:
    _install_protocol()
    defaults = [
        "--output-dir",
        str(OUTPUT_DIR),
        "--flow-profile-json",
        str(ROOT / "configs" / "paper_flow_profiles_v1.json"),
        "--source-profile-json",
        str(
            ROOT
            / "results"
            / "optuna"
            / "flowwise_ssaw_deadline_v3"
            / "selected_profiles.json"
        ),
        "--source-reference-csv",
        str(
            ROOT
            / "results"
            / "optuna"
            / "flowwise_ssaw_deadline_v3"
            / "validation_seeds_0_1_2"
            / "paired_raw.csv"
        ),
    ]
    return template.main(defaults + list(argv or []))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
