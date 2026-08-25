"""Minimal component ablations for the production DuSafe path."""

from copy import deepcopy


_PRESETS = {
    "full": {},
    "source_no_update": {
        "enable_adaptation": False,
        "bn_statistics": "frozen",
        "enable_ssaw": False,
        "enable_confidence_gate": False,
        "enable_source_semantic_router": False,
    },
    "ttbn_only": {
        "enable_adaptation": False,
        "enable_ssaw": False,
        "enable_confidence_gate": False,
        "enable_source_semantic_router": False,
    },
    # Remove physical view generation, pseudo-label-preserving view selection,
    # and the SSAW consistency objective as one atomic branch.
    "no_ssaw": {"enable_ssaw": False},
    "no_confidence_gate": {"enable_confidence_gate": False},
    "no_source_semantic_router": {"enable_source_semantic_router": False},
    "no_admission_or_router": {
        "enable_confidence_gate": False,
        "enable_source_semantic_router": False,
    },
}


# Read-only aliases for archived command lines. They resolve to the new names;
# neither alias restores a semantic raw-admission gate.
_ALIASES = {
    "no_source_semantic_gate": "no_source_semantic_router",
    "no_safety_gates": "no_admission_or_router",
}


_SSAW_CUMULATIVE_STAGES = (
    ("no_ssaw", "entire_ssaw_branch_removed"),
    ("full", "complete_ssaw_branch"),
)


def ablation_names():
    return tuple(_PRESETS)


def ssaw_cumulative_ablation_stages():
    """Return the paired whole-branch SSAW ablation."""
    return tuple(
        {
            "stage_index": index,
            "name": name,
            "added_module": added_module,
        }
        for index, (name, added_module) in enumerate(
            _SSAW_CUMULATIVE_STAGES
        )
    )


def resolve_dusafe_ablation(name):
    normalized = str(name).strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    try:
        overrides = deepcopy(_PRESETS[normalized])
    except KeyError as exc:
        choices = ", ".join(ablation_names())
        raise ValueError(
            f"Unknown DuSafe ablation '{name}'. Expected one of: {choices}."
        ) from exc
    return {"name": normalized, "overrides": overrides}


__all__ = [
    "ablation_names",
    "resolve_dusafe_ablation",
    "ssaw_cumulative_ablation_stages",
]
