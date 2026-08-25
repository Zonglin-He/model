from __future__ import annotations

import pandas as pd
import pytest

from scripts.finalize_minimal_safety_full_no_ssaw import finalize


def _frame() -> pd.DataFrame:
    rows = []
    for corruption in ("blackout", "signal_freeze"):
        for severity in ("moderate", "severe"):
            for seed in (0, 1, 2):
                for variant, delta in (("no_ssaw", 0.0), ("full", 0.01)):
                    rows.append(
                        {
                            "dataset": "HAR",
                            "scenario": "12->16",
                            "method": "DuSafe",
                            "variant": variant,
                            "corruption": corruption,
                            "severity": severity,
                            "source_seed": seed,
                            "stream_seed": 42,
                            "corruption_seed": 314159,
                            "source_model_sha256": f"hash-{seed}",
                            "f1": 0.7 + delta,
                            "corrupted_f1": 0.6 + delta,
                            "admitted_accuracy": 0.8 + delta,
                            "coverage": 0.5 + delta,
                        }
                    )
    return pd.DataFrame(rows)


def test_compact_safety_finalizer_requires_and_pairs_24_cells():
    aggregate, effects, manifest = finalize(_frame())
    assert len(aggregate) == 8
    assert len(effects) == 12
    assert manifest["rows"] == 24
    assert effects["full_minus_no_ssaw_f1"].mean() == pytest.approx(0.01)


def test_compact_safety_finalizer_rejects_missing_cell():
    with pytest.raises(RuntimeError, match="expected 24"):
        finalize(_frame().iloc[:-1])
