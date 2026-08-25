"""Isolated benchmark-only TTA adapters.

Nothing in this package is imported by the production DuSafe registry unless
the caller explicitly selects ``algorithm_registry=benchmark``.
"""

from benchmark_baselines.registry import (
    get_algorithm_class,
    list_methods,
    provenance,
)

__all__ = ["get_algorithm_class", "list_methods", "provenance"]
