"""Load the signed per-flow TTA overrides used by the final paper panel.

The JSON file contains TTA-only values.  Source-training settings remain in
the dataset hparams/checkpoint and are deliberately not read or overridden by
this helper.  Keeping the loader small and dependency-free lets CPU dry-run
planners validate the exact flow profile before any trainer is constructed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from configs.formal_evaluation_protocol import formal_scenario_pairs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_FLOW_PROFILE_JSON = ROOT / "configs" / "paper_flow_profiles_v1.json"


def _canonical_dataset(dataset: str) -> str:
    value = str(dataset).strip().upper().replace("MFD", "FD")
    return value


def load_paper_flow_profiles(
    path: str | Path | None = DEFAULT_PAPER_FLOW_PROFILE_JSON,
    datasets: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read exact ``DATASET:source->target`` TTA profiles.

    Unknown datasets in a shared JSON are ignored, while a malformed or
    non-formal profile for a selected dataset fails closed.  The current paper
    protocol requires a strictly positive SSAW auxiliary weight in every
    selected Full profile; checking it here catches stale/partial profile
    files during CPU planning rather than after a costly trainer starts.
    """

    if path is None or not str(path).strip():
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"paper flow profile JSON is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("paper flow profile JSON must encode an object")
    raw_profiles = payload.get("profiles", payload)
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("paper flow profile JSON 'profiles' must encode an object")

    selected = None
    if datasets is not None:
        selected = {_canonical_dataset(dataset) for dataset in datasets}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_key, raw_config in raw_profiles.items():
        dataset_text, separator, scenario = str(raw_key).partition(":")
        dataset = _canonical_dataset(dataset_text)
        scenario = scenario.strip()
        if not separator or (selected is not None and dataset not in selected):
            continue
        formal = {
            f"{source_id}->{target_id}"
            for source_id, target_id in formal_scenario_pairs(dataset)
        }
        if scenario not in formal:
            raise ValueError(f"non-formal paper flow profile: {raw_key!r}")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"paper flow profile {raw_key!r} must be an object")
        values = dict(raw_config)
        if "ssaw_auxiliary_weight" not in values:
            raise ValueError(
                f"paper flow profile {raw_key!r} lacks ssaw_auxiliary_weight"
            )
        try:
            weight = float(values["ssaw_auxiliary_weight"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"paper flow profile {raw_key!r} has a non-numeric ssaw_auxiliary_weight"
            ) from exc
        if weight <= 0.0:
            raise ValueError(
                f"paper flow profile {raw_key!r} must have positive ssaw_auxiliary_weight"
            )
        key = (dataset, scenario)
        if key in result:
            raise ValueError(f"duplicate paper flow profile: {raw_key!r}")
        result[key] = values

    if selected is not None:
        missing = []
        for dataset in sorted(selected):
            for source_id, target_id in formal_scenario_pairs(dataset):
                key = (dataset, f"{source_id}->{target_id}")
                if key not in result:
                    missing.append(f"{dataset}:{key[1]}")
        if missing:
            raise ValueError(
                "paper flow profile JSON lacks selected formal flows: "
                + ", ".join(missing)
            )
    return result


def profile_for_flow(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    dataset: str,
    scenario: str,
) -> dict[str, Any]:
    """Return a defensive copy for one flow, failing on a missing profile."""

    key = (_canonical_dataset(dataset), str(scenario).strip())
    try:
        values = profiles[key]
    except KeyError as exc:
        raise KeyError(f"missing paper flow profile for {key[0]}:{key[1]}") from exc
    return dict(values)


__all__ = [
    "DEFAULT_PAPER_FLOW_PROFILE_JSON",
    "load_paper_flow_profiles",
    "profile_for_flow",
]
