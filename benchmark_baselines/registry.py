"""Explicit benchmark-only method registry."""

from benchmark_baselines.adapters import (
    ACCUPOfficial,
    COME,
    CoTTA,
    EATA,
    NOTE,
    RoTTA,
    SAR,
    SoTTA,
    Tent,
)
from algorithms.dusafe import DuSafe
from configs.benchmark_baselines import BENCHMARK_METHODS, PROVENANCE


_REGISTRY = {
    "Tent": Tent,
    "EATA": EATA,
    "SAR": SAR,
    "ACCUPOfficial": ACCUPOfficial,
    "CoTTA": CoTTA,
    "SoTTA": SoTTA,
    "RoTTA": RoTTA,
    "COME": COME,
    "NOTE": NOTE,
    # Explicit production reference for paired safety/overhead commands.  The
    # production registry itself is untouched; this alias only exists when a
    # caller opts into the benchmark registry.
    "DuSafe": DuSafe,
}


def get_algorithm_class(name):
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unknown benchmark adaptation method: {name}. "
            f"Available: {', '.join(BENCHMARK_METHODS)}"
        ) from exc


def list_methods():
    return tuple(BENCHMARK_METHODS)


def provenance(name=None):
    if name is None:
        return {key: dict(value) for key, value in PROVENANCE.items()}
    if name not in PROVENANCE:
        raise KeyError(name)
    return dict(PROVENANCE[name])


__all__ = ["get_algorithm_class", "list_methods", "provenance"]
