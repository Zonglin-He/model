import pandas as pd

from scripts.run_controlled_safety_benchmark import PROBABILITY_RECORD_SCHEMA
from scripts.run_baseline_physical_reference_queue import (
    METHODS,
    SOURCE_SEEDS,
    Group,
    _manifest_payload,
    _scenario_scope,
    _select_scenarios,
    _select_methods,
    _status,
    group_command,
    group_completed,
    expected_group_protocol_signature,
    groups,
)


def test_plan_has_all_ten_baselines_flows_corruptions_and_two_severities():
    plan = groups()
    assert len(plan) == 20 * len(METHODS) * 6 * 2
    assert len(plan) * len(SOURCE_SEEDS) == 7200
    assert {group.dataset for group in plan} == {"EEG", "HAR", "FD", "HHAR"}
    assert {group.scenario for group in plan if group.dataset == "HHAR"} == {
        "0->6", "1->6", "2->7", "3->8", "4->5"
    }


def test_eata_worker_uses_benchmark_registry_and_fisher(tmp_path):
    group = Group("HHAR", "0", "6", "EATA", "blackout", "s3")
    command = group_command(
        group,
        data_path=tmp_path / "data",
        device="cuda",
        backbone="CNN",
        raw_output_dir=tmp_path / "raw",
        cache_root=tmp_path / "cache",
    )
    joined = " ".join(command)
    assert "--registry benchmark" in joined
    assert "--fisher_cache_dir" in command
    assert "hhar_formal" in joined
    assert "HHAR:0->6" in joined


def test_resume_requires_exact_three_seed_probability_rows():
    group = Group("EEG", "0", "11", "Tent", "signal_freeze", "s6")
    rows = []
    for seed in SOURCE_SEEDS:
        rows.append(
            {
                "dataset": group.dataset,
                "scenario": group.scenario,
                "method": group.method,
                "variant": "full",
                "corruption": group.corruption,
                "severity": group.severity,
                "source_seed": seed,
                "stream_seed": 42,
                "corruption_seed": 1,
                "probability_record_schema": PROBABILITY_RECORD_SCHEMA,
            }
        )
    frame = pd.DataFrame(rows)
    assert group_completed(frame, group)
    assert not group_completed(frame.iloc[:-1], group)


def test_resume_rejects_stale_protocol_signature(tmp_path):
    group = Group("EEG", "0", "11", "Tent", "signal_freeze", "s6")
    rows = []
    for seed in SOURCE_SEEDS:
        rows.append(
            {
                "dataset": group.dataset,
                "scenario": group.scenario,
                "method": group.method,
                "variant": "full",
                "corruption": group.corruption,
                "severity": group.severity,
                "source_seed": seed,
                "stream_seed": 42,
                "corruption_seed": 1,
                "probability_record_schema": "full_multiclass_logits_probabilities_v1",
                "protocol_signature": "stale-signature",
            }
        )
    frame = pd.DataFrame(rows)
    current = expected_group_protocol_signature(
        group,
        data_path=tmp_path / "data",
        backbone="CNN",
        cache_root=tmp_path / "cache",
    )
    assert group_completed(frame, group)
    assert not group_completed(
        frame, group, expected_protocol_signature=current
    )
    for row in rows:
        row["protocol_signature"] = current
    assert group_completed(
        pd.DataFrame(rows), group, expected_protocol_signature=current
    )


def test_representative_har_scope_is_exactly_sixty_cells(tmp_path):
    selected_methods = METHODS
    selected_corruptions = ("signal_freeze", "packet_loss")
    selected_severities = ("s6",)
    selected_scenarios = ("12->16",)
    plan = groups(
        datasets=("HAR",),
        methods=selected_methods,
        scenarios=selected_scenarios,
        corruptions=selected_corruptions,
        severities=selected_severities,
    )
    assert len(plan) == 20
    assert len(plan) * len(SOURCE_SEEDS) == 60
    assert {group.dataset for group in plan} == {"HAR"}
    assert {group.scenario for group in plan} == {"12->16"}
    assert {group.corruption for group in plan} == set(selected_corruptions)
    assert {group.severity for group in plan} == {"s6"}
    assert _scenario_scope(
        datasets=("HAR",),
        methods=selected_methods,
        scenarios=selected_scenarios,
        corruptions=selected_corruptions,
        severities=selected_severities,
        source_seeds=SOURCE_SEEDS,
    ) == "registered_representative_subset"
    manifest = _manifest_payload(
        all_groups=plan,
        datasets=("HAR",),
        methods=selected_methods,
        scenarios=selected_scenarios,
        corruptions=selected_corruptions,
        severities=selected_severities,
        source_seeds=SOURCE_SEEDS,
        scenario_scope="registered_representative_subset",
        data_path=tmp_path / "data",
        device="cpu",
        backbone="CNN",
        cache_root=tmp_path / "cache",
    )
    assert manifest["scenario_scope"] == "registered_representative_subset"
    assert manifest["expected_cells"] == 60
    assert manifest["scenarios"] == {"HAR": ["12->16"]}


def test_scenario_filter_requires_one_registered_dataset():
    assert _select_scenarios("HAR:12->16", ("HAR",)) == ("12->16",)
    try:
        _select_scenarios("12->16", ("HAR", "EEG"))
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multi-dataset scenario filter must be rejected")

    try:
        _select_scenarios("12->99", ("HAR",))
    except ValueError as error:
        assert "unregistered" in str(error)
    else:
        raise AssertionError("unregistered scenario must be rejected")


def test_group_command_uses_selected_source_seed_batch(tmp_path):
    group = Group("HAR", "12", "16", "EATA", "packet_loss", "s6")
    command = group_command(
        group,
        data_path=tmp_path / "data",
        device="cpu",
        backbone="CNN",
        raw_output_dir=tmp_path / "raw",
        cache_root=tmp_path / "cache",
        source_seeds=(2, 3),
    )
    assert command[command.index("--source_seeds") + 1] == "2,3"


def test_representative_scope_can_include_dusafe_reference_cells():
    methods = _select_methods(",".join((*METHODS, "DuSafe")))
    plan = groups(
        datasets=("HAR",),
        methods=methods,
        scenarios=("12->16",),
        corruptions=("signal_freeze", "packet_loss"),
        severities=("s3", "s6"),
    )
    status = _status(
        phase="baseline_physical_panel",
        all_groups=plan,
        completed=(),
        current=None,
        failures=(),
        datasets=("HAR",),
        methods=methods,
        corruptions=("signal_freeze", "packet_loss"),
        severities=("s3", "s6"),
        source_seeds=(1, 2, 3),
        scenario_scope="registered_representative_subset",
    )
    assert status["expected_cells"] == 132
    assert status["baseline_cells"] == 120
    assert status["dusafe_reference_cells"] == 12
