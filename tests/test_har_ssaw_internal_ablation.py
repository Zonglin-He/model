from types import SimpleNamespace

from algorithms.dusafe_spline_mechanism_matrix import (
    B4NoCoefficientSearch,
    B4NoGatheredRecheck,
    B4NoMarginFilter,
    B4NoRadiusBacktracking,
    B4NoSemanticRouter,
    SSAW_INTERNAL_ABLATION_RUNNERS,
    get_mechanism_runner,
)
from scripts.run_har_ssaw_internal_ablation import (
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


def test_internal_ablation_registry_removes_one_component_at_a_time():
    assert RUNNERS == SSAW_INTERNAL_ABLATION_RUNNERS
    assert get_mechanism_runner("A2_no_semantic_router") is B4NoSemanticRouter
    assert B4NoSemanticRouter.router_mode == "all"
    assert B4NoCoefficientSearch.view_kind == "random"
    assert B4NoCoefficientSearch.require_margin_reduction is True
    assert B4NoMarginFilter.require_margin_reduction is False
    assert B4NoRadiusBacktracking.spline_radius_levels_override == (1.0,)
    assert B4NoGatheredRecheck.recheck_gathered_training is False


def test_seed1_five_flow_matrix_has_35_independent_cells(tmp_path):
    specs = _build_specs(_args(tmp_path))
    assert len(FLOWS) == 5
    assert len(RUNNERS) == 7
    assert len(specs) == 35
    assert {spec["source_seed"] for spec in specs} == {1}
    assert {spec["stream_seed"] for spec in specs} == {42}
    assert {spec["tta_config"]["steps"] for spec in specs} == {2}
    assert {spec["tta_config"]["ssaw_auxiliary_weight"] for spec in specs} == {0.1}
    assert {spec["tta_config"]["learning_rate"] for spec in specs} == {3.325e-4}


def test_every_flow_uses_the_same_source_and_tta_profiles(tmp_path):
    specs = _build_specs(_args(tmp_path))
    assert len({repr(spec["source_config"]) for spec in specs}) == 1
    assert len({repr(spec["tta_config"]) for spec in specs}) == 1
