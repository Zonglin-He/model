"""Flow registry for the formal, currently reported evaluation panels.

The data-model configuration intentionally keeps the complete HHAR ten-flow
registry because it is also consumed by the tuner and source/checkpoint
protocol.  Formal A--F evidence panels use the pre-registered five-flow
subset below.  These flows are the same flows used for HHAR parameter
selection, so their results are descriptive/target-selected rather than
confirmatory.

This module is deliberately lightweight: importing it must not import a
trainer, dataset, torch model, or initialize CUDA.
"""

from __future__ import annotations

from typing import Any, Mapping


HHAR_RAW_DOMAIN_COUNT = 9

# The formal five-flow HHAR protocol.  Keep this as an ordered tuple because
# queue keys and manifests must be deterministic.
HHAR_DEVELOPMENT_FLOWS = (
    "0->6",
    "1->6",
    "2->7",
    "3->8",
    "4->5",
)

# Public name used by formal evaluation code.  It intentionally aliases the
# development subset: the current protocol reports the same five flows used
# to select the HHAR profile.  The remaining five data-model flows are not
# part of A--F.
HHAR_REPORTED_FLOWS = HHAR_DEVELOPMENT_FLOWS
HHAR_REPORTED_PARTITION = "target_selected_evaluation"
HHAR_PARAMETER_SELECTION_DATA_OVERLAP = True
HHAR_CONFIRMATORY = False


def _canonical_dataset(dataset: str) -> str:
    value = str(dataset).strip().upper()
    if value == "MFD":
        value = "FD"
    return value


def formal_scenario_pairs(dataset: str) -> tuple[tuple[str, str], ...]:
    """Return scenarios used by formal A--F panels.

    Non-HHAR datasets retain their complete configured scenario registry.
    HHAR is explicitly restricted to the five reported flows above; callers
    must not use ``configs.data_model_configs.HHAR.scenarios`` directly for
    formal panel planning.
    """

    dataset_name = _canonical_dataset(dataset)
    if dataset_name == "HHAR":
        return tuple(tuple(flow.split("->", 1)) for flow in HHAR_REPORTED_FLOWS)
    from configs.data_model_configs import scenario_pairs

    return tuple(
        (str(source), str(target))
        for source, target in scenario_pairs(dataset_name)
    )


def evaluation_partition_metadata(dataset: str, scenario: str) -> Mapping[str, Any]:
    """Return auditable partition metadata for one formal flow.

    The helper returns a mapping instead of a positional tuple so serialized
    manifests remain self-documenting and future protocol fields can be added
    without changing call-site unpacking.
    """

    dataset_name = _canonical_dataset(dataset)
    scenario_name = str(scenario)
    registered = {
        f"{source}->{target}" for source, target in formal_scenario_pairs(dataset_name)
    }
    # Non-HHAR synthetic/unit-test cells may use an arbitrary scenario label;
    # their production registry remains unchanged and has the same
    # target-selected metadata.  HHAR is restricted here because silently
    # accepting one of its excluded data-model flows would contaminate A--F.
    if dataset_name == "HHAR" and scenario_name not in registered:
        raise ValueError(
            f"unregistered formal evaluation flow for {dataset_name}: {scenario_name}"
        )
    if dataset_name == "HHAR":
        return {
            "evaluation_partition": HHAR_REPORTED_PARTITION,
            "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
            "selection_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
            "confirmatory": HHAR_CONFIRMATORY,
            "raw_domain_count": HHAR_RAW_DOMAIN_COUNT,
        }
    return {
        "evaluation_partition": "target_selected_evaluation",
        "parameter_selection_data_overlap": True,
        "selection_overlap": True,
        "confirmatory": False,
    }


def formal_flow_metadata(dataset: str) -> Mapping[str, Any]:
    """Return a serializable summary of the formal flow registry."""

    dataset_name = _canonical_dataset(dataset)
    pairs = formal_scenario_pairs(dataset_name)
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "flow_count": len(pairs),
        "flows": [f"{source}->{target}" for source, target in pairs],
    }
    if dataset_name == "HHAR":
        result.update(
            {
                "raw_domain_count": HHAR_RAW_DOMAIN_COUNT,
                "development_flows": list(HHAR_DEVELOPMENT_FLOWS),
                "reported_flows": list(HHAR_REPORTED_FLOWS),
                "evaluation_partition": HHAR_REPORTED_PARTITION,
                "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
                "confirmatory": HHAR_CONFIRMATORY,
                "excluded_data_model_flows": [
                    "5->0",
                    "6->1",
                    "7->4",
                    "8->3",
                    "0->2",
                ],
            }
        )
    return result


__all__ = [
    "HHAR_RAW_DOMAIN_COUNT",
    "HHAR_DEVELOPMENT_FLOWS",
    "HHAR_REPORTED_FLOWS",
    "HHAR_REPORTED_PARTITION",
    "HHAR_PARAMETER_SELECTION_DATA_OVERLAP",
    "HHAR_CONFIRMATORY",
    "formal_scenario_pairs",
    "evaluation_partition_metadata",
    "formal_flow_metadata",
]
