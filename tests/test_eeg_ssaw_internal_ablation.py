from types import SimpleNamespace

from algorithms.dusafe_spline_mechanism_matrix import (
    SSAW_INTERNAL_ABLATION_RUNNERS,
)
from scripts.run_eeg_ssaw_internal_ablation import (
    DATASET,
    FLOWS,
    RUNNERS,
    _build_specs,
)


def _args(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path / "out",
        data_path=tmp_path / "data",
        device="cpu",
        backbone="CNN",
        pretrain_cache_dir=tmp_path / "cache",
        gpu_lock_path=tmp_path / "gpu.lock",
        max_batches=1,
    )


def test_eeg_seed1_internal_ablation_has_35_cells(tmp_path):
    specs = _build_specs(_args(tmp_path))
    assert DATASET == "EEG"
    assert FLOWS == (
        ("0", "11"),
        ("12", "5"),
        ("7", "18"),
        ("16", "1"),
        ("9", "14"),
    )
    assert RUNNERS == SSAW_INTERNAL_ABLATION_RUNNERS
    assert len(specs) == 5 * 7
    assert {spec["dataset"] for spec in specs} == {"EEG"}
    assert {spec["source_seed"] for spec in specs} == {1}
    assert {spec["stream_seed"] for spec in specs} == {42}


def test_eeg_internal_ablation_freezes_current_profile(tmp_path):
    specs = _build_specs(_args(tmp_path))
    assert {spec["tta_config"]["steps"] for spec in specs} == {2}
    assert {spec["tta_config"]["learning_rate"] for spec in specs} == {2e-3}
    assert {spec["tta_config"]["ssaw_auxiliary_weight"] for spec in specs} == {
        0.003
    }
    assert {spec["tta_config"]["batch_size"] for spec in specs} == {192}
    assert len({repr(spec["source_config"]) for spec in specs}) == 1
    assert len({repr(spec["tta_config"]) for spec in specs}) == 1
