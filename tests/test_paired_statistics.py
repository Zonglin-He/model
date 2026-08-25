import numpy as np
import pandas as pd

from scripts.run_significance_test import (
    exact_paired_sign_flip,
    hierarchical_paired_ci,
    holm_adjust,
)


def test_exact_sign_flip_known_extremes():
    assert exact_paired_sign_flip([0.0, 0.0]) == 1.0
    # With four equally positive pairs, only the two all-equal sign patterns
    # attain the observed absolute mean: 2 / 16.
    assert exact_paired_sign_flip([1.0, 1.0, 1.0, 1.0]) == 0.125


def test_holm_adjustment_is_bounded_and_monotonic_in_rank():
    raw = np.asarray([0.01, 0.04, 0.03])
    adjusted = holm_adjust(raw)
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0.0)
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_hierarchical_ci_uses_whole_scenario_rows():
    frame = pd.DataFrame(
        {
            "source_seed": [1, 1, 2, 2],
            "scenario": ["a", "b", "a", "b"],
            "reference": [0.8, 0.7, 0.9, 0.8],
            "baseline": [0.7, 0.6, 0.8, 0.7],
        }
    )
    low, high = hierarchical_paired_ci(
        frame, "reference", "baseline", n_bootstrap=500, seed=3
    )
    assert abs(low - 0.1) < 1e-12
    assert abs(high - 0.1) < 1e-12
