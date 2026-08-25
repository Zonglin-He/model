"""Standard probability, calibration, and selective-prediction metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


def _validated(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, K]")
    if labels.size != probabilities.shape[0]:
        raise ValueError("labels and probabilities must contain the same samples")
    if labels.size == 0:
        raise ValueError("at least one sample is required")
    if probabilities.shape[1] < 2:
        raise ValueError("multiclass metrics require at least two classes")
    if labels.min() < 0 or labels.max() >= probabilities.shape[1]:
        raise ValueError("label index lies outside the probability columns")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("probabilities must lie in [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("each probability row must sum to one")
    return labels, probabilities


def multiclass_nll(labels, probabilities, *, epsilon: float = 1e-12) -> float:
    labels, probabilities = _validated(labels, probabilities)
    true_probability = probabilities[np.arange(labels.size), labels]
    return float(-np.log(np.clip(true_probability, epsilon, 1.0)).mean())


def multiclass_brier(labels, probabilities) -> float:
    """Mean summed multiclass Brier score; lower is better."""

    labels, probabilities = _validated(labels, probabilities)
    targets = np.zeros_like(probabilities)
    targets[np.arange(labels.size), labels] = 1.0
    return float(np.square(probabilities - targets).sum(axis=1).mean())


def expected_calibration_error(
    labels, probabilities, *, bins: int = 15
) -> float:
    """Top-label ECE with fixed equal-width confidence bins."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    labels, probabilities = _validated(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.size
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence > lower) & (confidence <= upper)
        count = int(selected.sum())
        if not count:
            continue
        accuracy = float(correctness[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        ece += count / total * abs(accuracy - mean_confidence)
    return float(ece)


def classwise_expected_calibration_error(
    labels, probabilities, *, bins: int = 15
) -> float:
    """Macro one-vs-rest ECE across all probability columns."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    labels, probabilities = _validated(labels, probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    per_class = []
    for class_index in range(probabilities.shape[1]):
        confidence = probabilities[:, class_index]
        outcomes = labels == class_index
        class_ece = 0.0
        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            if index == 0:
                selected = (confidence >= lower) & (confidence <= upper)
            else:
                selected = (confidence > lower) & (confidence <= upper)
            count = int(selected.sum())
            if not count:
                continue
            class_ece += count / labels.size * abs(
                float(outcomes[selected].mean())
                - float(confidence[selected].mean())
            )
        per_class.append(class_ece)
    return float(np.mean(per_class))


def predictive_risk_coverage(labels, probabilities) -> Dict[str, np.ndarray]:
    """Exact samplewise 0/1 risk--coverage curve sorted by confidence."""

    labels, probabilities = _validated(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    errors = (predictions != labels).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    ordered_errors = errors[order]
    counts = np.arange(1, labels.size + 1, dtype=np.float64)
    return {
        "coverage": counts / labels.size,
        "risk": np.cumsum(ordered_errors) / counts,
        "order": order,
    }


def aurc_eaurc(labels, probabilities) -> Mapping[str, float]:
    """Discrete AURC and excess AURC relative to an oracle ordering."""

    labels, probabilities = _validated(labels, probabilities)
    curve = predictive_risk_coverage(labels, probabilities)
    aurc = float(np.mean(curve["risk"]))
    predictions = probabilities.argmax(axis=1)
    errors = (predictions != labels).astype(np.float64)
    # The oracle accepts every correct sample before any error.  Using the same
    # discrete coverages avoids an approximation-dependent E-AURC baseline.
    oracle_errors = np.sort(errors, kind="stable")
    counts = np.arange(1, labels.size + 1, dtype=np.float64)
    oracle_aurc = float(np.mean(np.cumsum(oracle_errors) / counts))
    return {
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "eaurc": max(0.0, aurc - oracle_aurc),
    }


def selective_macro_f1_curve(
    labels,
    probabilities,
    *,
    coverages: Iterable[float] = tuple(np.linspace(0.1, 1.0, 10)),
) -> Dict[str, np.ndarray]:
    """Macro-F1 at declared coverage points; this is not an AURC risk."""

    labels, probabilities = _validated(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    requested = np.asarray(tuple(coverages), dtype=np.float64)
    if requested.size == 0 or (requested <= 0.0).any() or (requested > 1.0).any():
        raise ValueError("coverages must lie in (0, 1]")
    class_count = probabilities.shape[1]
    values = []
    realized = []
    for coverage in requested:
        count = max(1, int(round(float(coverage) * labels.size)))
        selected = order[:count]
        y_true = labels[selected]
        y_pred = predictions[selected]
        class_f1 = []
        for class_index in range(class_count):
            true_positive = int(((y_true == class_index) & (y_pred == class_index)).sum())
            false_positive = int(((y_true != class_index) & (y_pred == class_index)).sum())
            false_negative = int(((y_true == class_index) & (y_pred != class_index)).sum())
            denominator = 2 * true_positive + false_positive + false_negative
            class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        realized.append(count / labels.size)
        values.append(float(np.mean(class_f1)))
    return {
        "coverage": np.asarray(realized, dtype=np.float64),
        "macro_f1": np.asarray(values, dtype=np.float64),
    }


def summarize_probability_metrics(
    labels, probabilities, *, calibration_bins: int = 15
) -> Dict[str, float]:
    labels, probabilities = _validated(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    aurc_values = aurc_eaurc(labels, probabilities)
    return {
        "accuracy": float((predictions == labels).mean()),
        "nll": multiclass_nll(labels, probabilities),
        "brier": multiclass_brier(labels, probabilities),
        "ece": expected_calibration_error(
            labels, probabilities, bins=calibration_bins
        ),
        "classwise_ece": classwise_expected_calibration_error(
            labels, probabilities, bins=calibration_bins
        ),
        **aurc_values,
    }
