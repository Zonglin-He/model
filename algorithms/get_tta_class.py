"""Production TTA method registry.

The public ``DuSafe`` name remains the stable registry entry.  Dataset-level
profiles may select a reviewed DuSafe implementation variant without exposing
experiment-only class names through the command line.
"""

_DUSAFE_VARIANT_ALIASES = {
    None: "spline_residual",
    "": "spline_residual",
    "spline_residual": "spline_residual",
    "confidence_raw": "confidence_raw",
    # Archived experiment manifests use the former screen names. They resolve
    # to the reviewed production implementations but are not emitted by any
    # current configuration.
    "fixed_kl_b4": "spline_residual",
    "confidence_raw_n2": "confidence_raw",
}


def get_algorithm_class(algorithm_name, *, variant=None):
    if algorithm_name != "DuSafe":
        raise NotImplementedError(
            f"Unknown adaptation method: {algorithm_name}"
        )
    try:
        resolved = _DUSAFE_VARIANT_ALIASES[
            None if variant is None else str(variant).strip().lower()
        ]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unknown DuSafe implementation variant: {variant}"
        ) from exc
    from algorithms.dusafe_spline_hard_view import (
        ConfidenceAdmittedSplineResidualKL,
        ConfidenceRawOnly,
    )

    if resolved == "spline_residual":
        return ConfidenceAdmittedSplineResidualKL
    if resolved == "confidence_raw":
        return ConfidenceRawOnly
    raise AssertionError(f"Unhandled DuSafe variant: {resolved}")


__all__ = ["get_algorithm_class"]
