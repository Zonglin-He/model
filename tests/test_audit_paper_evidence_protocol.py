from scripts.audit_paper_evidence_protocol import audit


def test_current_evidence_audit_fails_closed_on_incompatible_protocol():
    protocol = {
        "protocol": "test",
        "production_code_sha256": (
            "ed87901551f26f7ab5f9c456342463ff24a6f589cc01f571d775600871adf6ac"
        ),
        "source_seeds": [0, 1, 2],
        "stream_seed": 42,
        "tta_profile_json": "configs/paper_flow_profiles_v2.json",
        "method_contract": {"source_semantic_router": False},
        "safety_protocol": {
            "source_seeds": [0, 1, 2],
            "corruption_seed": 314159,
            "physical_protocol": True,
            "severity_policy": (
                "pre_registered_s0_to_s6_normalized_with_physical_parameters"
            ),
        },
    }
    frame, report = audit(protocol)
    assert report["status"] == "rerun_required"
    assert not frame.loc[
        (frame["artifact"] == "old_main_full_no_ssaw")
        & (frame["field"] == "production_code_sha256"),
        "compatible",
    ].item()
    assert not frame.loc[
        (frame["artifact"] == "new_core_ablation")
        & (frame["field"] == "tta_profile_json"),
        "compatible",
    ].item()
    assert not frame.loc[
        (frame["artifact"] == "new_safety_table")
        & (frame["field"] == "physical_protocol"),
        "compatible",
    ].item()
