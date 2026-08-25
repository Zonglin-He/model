"""Dedicated production-path runners for DuSafe factorial ablations.

The runners share the exact production :class:`algorithms.dusafe.DuSafe`
implementation and numeric hyperparameters.  A runner changes only the
presence of three structural components:

``W``
    The complete SSAW physical-view branch, including label-preserving
    qualification, continuous KL risk weighting, and the auxiliary objective.
``C``
    The source-calibrated confidence admission rule.
``S``
    The frozen source-semantic admission rule.

All eight cells are represented by distinct classes so an ablation cannot
silently acquire cell-specific hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from algorithms.dusafe import DuSafe


class DuSafeFactorialRunner(DuSafe):
    """Production DuSafe with class-fixed factorial components."""

    runner_name = "factorial_base"
    factor_ssaw = False
    factor_confidence = False
    factor_semantic = False

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(configs, hparams, model, optimizer)
        # Component presence belongs to the runner class, not to a numeric
        # weight supplied by the experiment launcher.
        self.enable_ssaw = bool(self.factor_ssaw)
        self.enable_confidence_gate = bool(self.factor_confidence)
        self.enable_source_semantic_gate = bool(self.factor_semantic)
        self.factorial_runner_name = self.runner_name


class RawOnlyRunner(DuSafeFactorialRunner):
    """W=0, C=0, S=0: raw pseudo-label entropy minimization."""

    runner_name = "raw_only"


class ConfidenceOnlyRunner(DuSafeFactorialRunner):
    """W=0, C=1, S=0."""

    runner_name = "confidence_only"
    factor_confidence = True


class SemanticOnlyRunner(DuSafeFactorialRunner):
    """W=0, C=0, S=1."""

    runner_name = "semantic_only"
    factor_semantic = True


class DualGateOnlyRunner(DuSafeFactorialRunner):
    """W=0, C=1, S=1: both raw-view gates without SSAW."""

    runner_name = "dual_gate_only"
    factor_confidence = True
    factor_semantic = True


class SSAWOnlyRunner(DuSafeFactorialRunner):
    """W=1, C=0, S=0: SSAW without either raw-view gate."""

    runner_name = "ssaw_only"
    factor_ssaw = True


class SSAWConfidenceRunner(DuSafeFactorialRunner):
    """W=1, C=1, S=0."""

    runner_name = "ssaw_confidence"
    factor_ssaw = True
    factor_confidence = True


class SSAWSemanticRunner(DuSafeFactorialRunner):
    """W=1, C=0, S=1."""

    runner_name = "ssaw_semantic"
    factor_ssaw = True
    factor_semantic = True


class FullFactorialRunner(DuSafeFactorialRunner):
    """W=1, C=1, S=1: the unchanged production component set."""

    runner_name = "full"
    factor_ssaw = True
    factor_confidence = True
    factor_semantic = True


@dataclass(frozen=True)
class FactorialRunnerSpec:
    runner_class: type[DuSafeFactorialRunner]
    ssaw: bool
    confidence: bool
    semantic: bool

    @property
    def bits(self) -> tuple[int, int, int]:
        return int(self.ssaw), int(self.confidence), int(self.semantic)


FACTORIAL_RUNNER_SPECS = {
    runner.runner_name: FactorialRunnerSpec(
        runner_class=runner,
        ssaw=runner.factor_ssaw,
        confidence=runner.factor_confidence,
        semantic=runner.factor_semantic,
    )
    for runner in (
        RawOnlyRunner,
        ConfidenceOnlyRunner,
        SemanticOnlyRunner,
        DualGateOnlyRunner,
        SSAWOnlyRunner,
        SSAWConfidenceRunner,
        SSAWSemanticRunner,
        FullFactorialRunner,
    )
}

RUNNER_BY_BITS = {
    spec.bits: name for name, spec in FACTORIAL_RUNNER_SPECS.items()
}


def get_factorial_runner(name: str) -> type[DuSafeFactorialRunner]:
    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return FACTORIAL_RUNNER_SPECS[normalized].runner_class
    except KeyError as exc:
        choices = ", ".join(FACTORIAL_RUNNER_SPECS)
        raise ValueError(
            f"Unknown DuSafe factorial runner '{name}'. Expected: {choices}."
        ) from exc


__all__ = [
    "DuSafeFactorialRunner",
    "FACTORIAL_RUNNER_SPECS",
    "FactorialRunnerSpec",
    "RUNNER_BY_BITS",
    "get_factorial_runner",
]
