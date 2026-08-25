from pathlib import Path
from types import SimpleNamespace

from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
    ConfidenceRawOnly,
)
from algorithms.get_tta_class import get_algorithm_class
from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.run_final_ssaw_full_no_ssaw_five_flow import (
    DATASETS,
    PROTOCOL,
    SOURCE_SEEDS,
    VARIANTS,
    _parse_source_seeds,
    _signature,
    _parse_flow_keys,
    _parse_profile_overrides,
    _load_flow_profile_overrides,
    _profile_manifest_metadata,
    build_specs,
    production_code_sha256,
)


def test_explicit_source_seed_set_can_include_zero():
    assert _parse_source_seeds("0,1,2") == (0, 1, 2)


def test_source_seed_set_rejects_negative_or_duplicate_values():
    for value in ("-1,0,1", "0,1,1"):
        try:
            _parse_source_seeds(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid source seed set: {value}")


def _args(tmp_path: Path):
    return SimpleNamespace(
        output_dir=tmp_path / "out",
        data_path=tmp_path / "data",
        device="cpu",
        backbone="CNN",
        stream_seed=42,
        gpu_lock_path=tmp_path / "gpu.lock",
        max_batches=None,
    )


def test_four_dataset_five_flow_three_seed_pair_plan(tmp_path):
    specs = build_specs(_args(tmp_path), DATASETS, SOURCE_SEEDS)
    assert len(specs) == 4 * 5 * 3 * 2
    assert {spec["dataset"] for spec in specs} == set(DATASETS)
    assert {spec["source_seed"] for spec in specs} == {1, 2, 3}
    assert {spec["variant"] for spec in specs} == set(VARIANTS)
    assert all(spec["stream_seed"] == 42 for spec in specs)
    for dataset in DATASETS:
        assert len({tuple(spec["flow"]) for spec in specs if spec["dataset"] == dataset}) == 5


def test_full_and_no_ssaw_differ_only_by_atomic_branch(tmp_path):
    specs = build_specs(_args(tmp_path), DATASETS, (1,))
    for dataset in DATASETS:
        flow = next(
            tuple(spec["flow"]) for spec in specs if spec["dataset"] == dataset
        )
        pair = {
            spec["variant"]: spec
            for spec in specs
            if spec["dataset"] == dataset and tuple(spec["flow"]) == flow
        }
        full = dict(pair["full"]["tta_config"])
        no_ssaw = dict(pair["no_ssaw"]["tta_config"])
        assert full["dusafe_variant"] == "spline_residual"
        assert no_ssaw["dusafe_variant"] == "confidence_raw"
        assert full["enable_ssaw"] is True
        assert no_ssaw["enable_ssaw"] is False
        assert full["enable_source_semantic_router"] is False
        assert no_ssaw["enable_source_semantic_router"] is False
        for key in {
            "dusafe_variant",
            "enable_ssaw",
            "enable_source_semantic_router",
        }:
            full.pop(key)
            no_ssaw.pop(key)
        assert full == no_ssaw
        assert pair["full"]["source_config"] == pair["no_ssaw"]["source_config"]


def test_registry_and_resume_signature_use_reviewed_implementations(tmp_path):
    assert (
        get_algorithm_class("DuSafe", variant="spline_residual")
        is ConfidenceAdmittedSplineResidualKL
    )
    assert get_algorithm_class("DuSafe", variant="confidence_raw") is ConfidenceRawOnly
    digest = production_code_sha256()
    assert len(digest) == 64
    spec = build_specs(_args(tmp_path), ("HAR",), (1,))[0]
    signature = _signature(spec)
    assert signature["protocol"] == PROTOCOL
    assert signature["production_code_sha256"] == digest
    assert signature["target_labels_used_for_online_decision"] is False


def test_representative_flow_and_dataset_override_are_signed(tmp_path):
    flows = _parse_flow_keys(
        "EEG:12->5,HAR:12->16,HHAR:2->7", ("EEG", "HAR", "HHAR")
    )
    overrides = _parse_profile_overrides(
        [
            "EEG:ssaw_auxiliary_weight=0.1",
            "HAR:learning_rate=0.001",
            "HHAR:steps=2",
        ],
        ("EEG", "HAR", "HHAR"),
    )
    specs = build_specs(
        _args(tmp_path),
        ("EEG", "HAR", "HHAR"),
        (1,),
        flow_overrides=flows,
        profile_overrides=overrides,
    )
    assert len(specs) == 3 * 2
    assert {tuple(spec["flow"]) for spec in specs} == {
        ("12", "5"),
        ("12", "16"),
        ("2", "7"),
    }
    for spec in specs:
        if spec["dataset"] == "EEG":
            assert spec["tta_config"]["ssaw_auxiliary_weight"] == 0.1
        if spec["dataset"] == "HAR":
            assert spec["tta_config"]["learning_rate"] == 0.001
        if spec["dataset"] == "HHAR":
            assert spec["tta_config"]["steps"] == 2
        assert _signature(spec)["tta_config"] == spec["tta_config"]


def test_flow_profile_overrides_are_exact_and_nonzero(tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        '{"profiles":{"HAR:12->16":{"learning_rate":0.001,'
        '"ssaw_auxiliary_weight":4.0},"HAR:6->23":'
        '{"learning_rate":0.001,"ssaw_auxiliary_weight":0.1}}}',
        encoding="utf-8",
    )
    profiles = _load_flow_profile_overrides(profile_path, ("HAR",))
    flows = _parse_flow_keys("HAR:12->16,HAR:6->23", ("HAR",))
    specs = build_specs(
        _args(tmp_path),
        ("HAR",),
        (1,),
        flow_overrides=flows,
        flow_profile_overrides=profiles,
    )
    assert len(specs) == 4
    by_flow = {
        tuple(spec["flow"]): spec["tta_config"]
        for spec in specs
        if spec["variant"] == "full"
    }
    assert by_flow[("12", "16")]["ssaw_auxiliary_weight"] == 4.0
    assert by_flow[("6", "23")]["ssaw_auxiliary_weight"] == 0.1
    assert by_flow[("12", "16")]["learning_rate"] == 0.001


def test_manifest_profile_scope_is_flow_specific_when_effective_flows_differ(tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        '{"profiles":{"HAR:12->16":{"learning_rate":0.001},'
        '"HAR:6->23":{"learning_rate":0.002}}}',
        encoding="utf-8",
    )
    profiles = _load_flow_profile_overrides(profile_path, ("HAR",))
    specs = build_specs(
        _args(tmp_path),
        ("HAR",),
        (1, 2, 3),
        flow_profile_overrides=profiles,
    )
    metadata = _profile_manifest_metadata(specs, ("HAR",))

    assert metadata["flow_specific_tta_profiles"] is True
    assert metadata["dataset_level_profiles"] is False
    assert metadata["same_profile_for_selected_flows"] is False
    assert metadata["same_profile_for_all_five_flows"] is False
    assert metadata["flow_specific_tta_profiles_by_dataset"] == {"HAR": True}
    assert set(metadata["effective_profiles_by_flow"]) == {
        f"HAR:{source}->{target}"
        for source, target in formal_scenario_pairs("HAR")
    }


def test_manifest_same_profile_is_conditional_for_representative_subset(tmp_path):
    flows = _parse_flow_keys("HAR:12->16,HAR:6->23", ("HAR",))
    specs = build_specs(
        _args(tmp_path),
        ("HAR",),
        (1,),
        flow_overrides=flows,
    )
    metadata = _profile_manifest_metadata(specs, ("HAR",))

    assert metadata["formal_five_flow_panel"] is False
    assert metadata["same_profile_for_selected_flows"] is True
    assert metadata["same_profile_for_all_five_flows"] is None
