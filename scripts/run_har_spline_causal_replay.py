"""One-step batch-start causal replay for the HAR spline mechanism matrix."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe import SSAWPhysicalView, _extract_features  # noqa: E402
from algorithms.dusafe_spline_mechanism_matrix import (  # noqa: E402
    MECHANISM_RUNNERS,
    get_mechanism_runner,
)
from optim.optimizer import build_optimizer  # noqa: E402
from scripts.dusafe_factorial_runner_common import (  # noqa: E402
    current_profiles,
    tensor_state_sha256,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_har_spline_mechanism_matrix import (  # noqa: E402
    FLOWS,
    SOURCE_SEED,
    SPLINE_PROFILE,
    STREAM_SEED,
)
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


PROTOCOL = "har_spline_causal_replay_v2_per_view_bn_seed_hash_contract"
BRANCH_RUNNERS = tuple(MECHANISM_RUNNERS)
MATCHED_RUNNER = "Bmatch_raw_update_norm_to_B1"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "har_spline_causal_replay_seed1_v2_corrected"
)
DEFAULT_ONLINE_MATRIX_DIR = (
    ROOT
    / "results"
    / "ablation"
    / "har_spline_mechanism_matrix_seed1_v2_corrected"
)
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"
EXPECTED_EFFECTIVE_SSAW_SEED = 42_001_855


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _capture_rng() -> tuple[torch.Tensor, list[torch.Tensor]]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return cpu, cuda


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[1])


def _move_data(data, device):
    if isinstance(data, list):
        return [value.float().to(device) for value in data]
    return data.float().to(device)


def _predict_read_only(adapter, data: torch.Tensor) -> torch.Tensor:
    with SSAWPhysicalView._preserved_bn_buffers(adapter.model), torch.no_grad():
        return adapter.model.classifier(_extract_features(adapter.model, data))


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    predictions = logits.detach().argmax(dim=1).cpu().numpy()
    targets = labels.detach().cpu().numpy()
    return (
        float(f1_score(targets, predictions, average="macro", zero_division=0)),
        float(F.cross_entropy(logits.detach(), labels).item()),
    )


def _trainable_snapshot(adapter) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in adapter.model.named_parameters()
        if parameter.requires_grad
    }


def _parameter_delta(adapter, before: Mapping[str, torch.Tensor]):
    vectors = []
    layers = []
    for name, parameter in adapter.model.named_parameters():
        if name not in before:
            continue
        delta = (parameter.detach() - before[name]).float()
        vectors.append(delta.reshape(-1).cpu())
        layers.append((name, float(delta.square().sum().sqrt().item())))
    vector = torch.cat(vectors) if vectors else torch.zeros(0)
    return vector, layers


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.numel() or left.numel() != right.numel():
        return math.nan
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return math.nan
    return float(torch.dot(left, right) / denominator)


def _causal_branch_hparams(r1_adapter, hparams: Mapping[str, object]) -> dict:
    """Derive a one-step branch without dropping the formal online seed."""
    return {
        **dict(r1_adapter.hparams),
        **dict(hparams),
        "steps": 1,
        "test_time_seed": r1_adapter.test_time_seed,
    }


def _make_branch(
    *,
    runner_name: str,
    r1_adapter,
    trainer,
    hparams: Mapping[str, object],
    batch_index: int,
    learning_rate_scale: float = 1.0,
):
    branch_hparams = _causal_branch_hparams(r1_adapter, hparams)
    runner_class = get_mechanism_runner(runner_name)
    model_copy = copy.deepcopy(r1_adapter.model)
    branch = runner_class(
        trainer.dataset_configs,
        branch_hparams,
        model_copy,
        build_optimizer(branch_hparams),
    )
    if branch.test_time_seed != r1_adapter.test_time_seed:
        raise RuntimeError("causal branch test-time seed differs from online adapter")
    if branch.ssaw_effective_sobol_seed != r1_adapter.ssaw_effective_sobol_seed:
        raise RuntimeError("causal branch effective SSAW seed differs from online adapter")
    branch.source_semantic_feature_extractor.load_state_dict(
        r1_adapter.source_semantic_feature_extractor.state_dict()
    )
    branch.load_source_normalization_reference(
        r1_adapter.source_normalization_mean.detach().cpu(),
        r1_adapter.source_normalization_std.detach().cpu(),
    )
    branch.load_source_confidence_reference(trainer.source_confidence_metadata)
    branch.load_source_semantic_reference(trainer.source_semantic_metadata)
    branch = branch.to(trainer.device)
    if branch.optimizer is not None and r1_adapter.optimizer is not None:
        branch.optimizer.load_state_dict(
            copy.deepcopy(r1_adapter.optimizer.state_dict())
        )
        base_lr = float(branch_hparams["learning_rate"])
        for group in branch.optimizer.param_groups:
            group["lr"] = base_lr * float(learning_rate_scale)
    if branch.enable_ssaw and hasattr(branch.ssaw, "_spline_call_index"):
        branch.ssaw._spline_call_index = int(batch_index)
    return branch


def _run_branch(
    *,
    runner_name: str,
    reported_name: str,
    r1_adapter,
    trainer,
    hparams,
    batch_index: int,
    data,
    labels,
    next_data,
    learning_rate_scale: float,
    rng_state,
):
    _restore_rng(rng_state)
    branch = _make_branch(
        runner_name=runner_name,
        r1_adapter=r1_adapter,
        trainer=trainer,
        hparams=hparams,
        batch_index=batch_index,
        learning_rate_scale=learning_rate_scale,
    )
    try:
        before = _trainable_snapshot(branch)
        branch({"data": data})
        delta, layers = _parameter_delta(branch, before)
        current_logits = _predict_read_only(branch, data)
        current_f1, current_nll = _metrics(current_logits, labels)
        if next_data is None:
            next_f1 = next_nll = math.nan
        else:
            next_logits = _predict_read_only(branch, next_data[0])
            next_f1, next_nll = _metrics(next_logits, next_data[1])
        gate_log = branch._last_gate_log
        pseudo_labels = torch.as_tensor(
            gate_log["inner_pseudo_labels"][0], dtype=torch.long
        ).cpu()
        admission_mask = torch.as_tensor(
            gate_log["inner_admission_masks"][0], dtype=torch.bool
        ).cpu()
        batch_log = dict(branch._last_batch_log)
        row = {
            "runner": reported_name,
            "implementation_runner": runner_name,
            "learning_rate_scale": float(learning_rate_scale),
            "current_post_f1": current_f1,
            "current_post_nll": current_nll,
            "next_post_f1": next_f1,
            "next_post_nll": next_nll,
            "parameter_delta_norm": float(delta.norm()),
            **{
                f"diag_{key}": float(value)
                for key, value in batch_log.items()
                if isinstance(value, (int, float))
                and not key.startswith("parameter_delta_norm__")
            },
        }
        contract = copy.deepcopy(
            getattr(branch, "_last_auxiliary_contract", {})
        )
        candidate_hash = (
            None
            if not branch.enable_ssaw
            else branch.ssaw.last_metadata.get("candidate_sha256")
        )
        row.update(
            {
                "test_time_seed": branch.test_time_seed,
                "ssaw_effective_sobol_seed": branch.ssaw_effective_sobol_seed,
                "candidate_sha256": candidate_hash,
            }
        )
        return (
            row,
            delta,
            layers,
            pseudo_labels,
            admission_mask,
            contract,
            candidate_hash,
        )
    finally:
        if branch.optimizer is not None:
            branch.optimizer.state.clear()
        branch.to("cpu")
        del branch
        release_cuda()
        gc.collect()


def _run_flow(spec: Mapping[str, object]):
    flow = tuple(str(value) for value in spec["flow"])
    source_config = dict(spec["source_config"])
    tta_config = dict(spec["tta_config"])
    trainer = build_trainer(
        data_path=str(spec["data_path"]),
        device=str(spec["device"]),
        dataset="HAR",
        da_method="DuSafe",
        backbone=str(spec["backbone"]),
        exp_name="spline_causal_replay",
        seed=STREAM_SEED,
        source_seed=SOURCE_SEED,
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    r1_adapter = source_model = None
    rows, layer_rows = [], []
    try:
        trainer.get_tta_model_class = lambda: get_mechanism_runner("B0_raw_only")
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        r1_adapter, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=STREAM_SEED
        )
        if r1_adapter.ssaw_effective_sobol_seed != EXPECTED_EFFECTIVE_SSAW_SEED:
            raise RuntimeError(
                "formal online adapter effective SSAW seed mismatch: "
                f"{r1_adapter.ssaw_effective_sobol_seed}"
            )
        source_hash = tensor_state_sha256(source_model)
        batches = list(trainer.trg_whole_dl)
        for batch_index, (raw_data, raw_labels, _) in enumerate(batches):
            data = _move_data(raw_data, trainer.device)
            if isinstance(data, list):
                raise RuntimeError("HAR causal replay expects one tensor input")
            labels = raw_labels.view(-1).long().to(trainer.device)
            if batch_index + 1 < len(batches):
                next_raw_data, next_raw_labels, _ = batches[batch_index + 1]
                next_data = (
                    _move_data(next_raw_data, trainer.device),
                    next_raw_labels.view(-1).long().to(trainer.device),
                )
            else:
                next_data = None
            pre_logits = _predict_read_only(r1_adapter, data)
            pre_f1, pre_nll = _metrics(pre_logits, labels)
            if next_data is None:
                pre_next_f1 = pre_next_nll = math.nan
            else:
                pre_next_logits = _predict_read_only(r1_adapter, next_data[0])
                pre_next_f1, pre_next_nll = _metrics(
                    pre_next_logits, next_data[1]
                )
            rng_state = _capture_rng()
            branch_results = {}
            for runner_name in BRANCH_RUNNERS:
                (
                    row,
                    delta,
                    layers,
                    pseudo,
                    admission,
                    contract,
                    candidate_hash,
                ) = _run_branch(
                    runner_name=runner_name,
                    reported_name=runner_name,
                    r1_adapter=r1_adapter,
                    trainer=trainer,
                    hparams=tta_config,
                    batch_index=batch_index,
                    data=data,
                    labels=labels,
                    next_data=next_data,
                    learning_rate_scale=1.0,
                    rng_state=rng_state,
                )
                branch_results[runner_name] = {
                    "row": row,
                    "delta": delta,
                    "layers": layers,
                    "pseudo": pseudo,
                    "admission": admission,
                    "contract": contract,
                    "candidate_hash": candidate_hash,
                }
            b0_norm = float(branch_results["B0_raw_only"]["delta"].norm())
            b1_norm = float(
                branch_results["B1_random_spline_view_ce"]["delta"].norm()
            )
            matched_scale = (
                min(5.0, max(0.1, b1_norm / max(b0_norm, 1e-12)))
                if b0_norm > 0
                else 1.0
            )
            (
                row,
                delta,
                layers,
                pseudo,
                admission,
                contract,
                candidate_hash,
            ) = _run_branch(
                runner_name="B0_raw_only",
                reported_name=MATCHED_RUNNER,
                r1_adapter=r1_adapter,
                trainer=trainer,
                hparams=tta_config,
                batch_index=batch_index,
                data=data,
                labels=labels,
                next_data=next_data,
                learning_rate_scale=matched_scale,
                rng_state=rng_state,
            )
            branch_results[MATCHED_RUNNER] = {
                "row": row,
                "delta": delta,
                "layers": layers,
                "pseudo": pseudo,
                "admission": admission,
                "contract": contract,
                "candidate_hash": candidate_hash,
            }

            random_names = (
                "Bdup_raw_duplicate",
                "B1_random_spline_view_ce",
                "B3_random_spline_residual_kl",
            )
            random_hashes = {
                branch_results[name]["candidate_hash"] for name in random_names
            }
            if len(random_hashes) != 1 or None in random_hashes:
                raise RuntimeError(
                    f"{flow} batch {batch_index}: random-view candidate hash mismatch"
                )
            boundary_hashes = {
                branch_results[name]["candidate_hash"]
                for name in (
                    "B2_boundary_spline_view_ce",
                    "B4_boundary_spline_residual_kl",
                )
            }
            if len(boundary_hashes) != 1 or None in boundary_hashes:
                raise RuntimeError(
                    f"{flow} batch {batch_index}: boundary-view candidate hash mismatch"
                )
            bdup_contract = branch_results["Bdup_raw_duplicate"]["contract"]
            b1_contract = branch_results["B1_random_spline_view_ce"]["contract"]
            for key in (
                "pseudo_labels",
                "raw_admission_mask",
                "eligibility_mask",
                "sample_weights",
            ):
                if not torch.equal(
                    torch.as_tensor(bdup_contract[key]),
                    torch.as_tensor(b1_contract[key]),
                ):
                    raise RuntimeError(
                        f"{flow} batch {batch_index}: Bdup/B1 {key} mismatch"
                    )
            if (
                bdup_contract["denominator"] != b1_contract["denominator"]
                or bdup_contract["candidate_sha256"]
                != b1_contract["candidate_sha256"]
            ):
                raise RuntimeError(
                    f"{flow} batch {batch_index}: Bdup/B1 scalar contract mismatch"
                )

            reference_pseudo = branch_results["B0_raw_only"]["pseudo"]
            reference_admission = branch_results["B0_raw_only"]["admission"]
            reference_delta = branch_results["B0_raw_only"]["delta"]
            b1_delta = branch_results["B1_random_spline_view_ce"]["delta"]
            for runner_name, payload in branch_results.items():
                pseudo_match = torch.equal(payload["pseudo"], reference_pseudo)
                admission_match = torch.equal(
                    payload["admission"], reference_admission
                )
                if not pseudo_match or not admission_match:
                    raise RuntimeError(
                        f"{flow} batch {batch_index} {runner_name}: replay mask mismatch"
                    )
                result_row = {
                    "protocol": PROTOCOL,
                    "dataset": "HAR",
                    "scenario": _flow_label(flow),
                    "source_seed": SOURCE_SEED,
                    "stream_seed": STREAM_SEED,
                    "batch_index": batch_index,
                    "batch_samples": int(labels.numel()),
                    "source_model_sha256": source_hash,
                    "pre_current_f1": pre_f1,
                    "pre_current_nll": pre_nll,
                    "pre_next_f1": pre_next_f1,
                    "pre_next_nll": pre_next_nll,
                    **payload["row"],
                    "delta_current_f1": payload["row"]["current_post_f1"] - pre_f1,
                    "delta_current_nll": payload["row"]["current_post_nll"] - pre_nll,
                    "delta_next_f1": payload["row"]["next_post_f1"] - pre_next_f1,
                    "delta_next_nll": payload["row"]["next_post_nll"] - pre_next_nll,
                    "update_cosine_to_B0": _cosine(
                        payload["delta"], reference_delta
                    ),
                    "update_cosine_to_B1": _cosine(payload["delta"], b1_delta),
                    "pseudo_labels_match_B0": pseudo_match,
                    "admission_mask_match_B0": admission_match,
                    "candidate_hash_verified": (
                        payload["candidate_hash"] is not None
                        or runner_name in {"B0_raw_only", MATCHED_RUNNER}
                    ),
                    "bdup_b1_contract_verified": True,
                    "target_labels_used_for_online_decision": False,
                }
                rows.append(result_row)
                for layer_name, layer_norm in payload["layers"]:
                    layer_rows.append(
                        {
                            "scenario": _flow_label(flow),
                            "batch_index": batch_index,
                            "runner": runner_name,
                            "layer": layer_name,
                            "update_norm": layer_norm,
                        }
                    )

            _restore_rng(rng_state)
            r1_adapter({"data": data})
            del data, labels
            if next_data is not None:
                del next_data
            release_cuda()
        row_frame = pd.DataFrame(rows)
        online_path = (
            Path(spec["online_matrix_dir"])
            / "cells"
            / f"flow_{flow[0]}_to_{flow[1]}"
            / "B1_random_spline_view_ce"
            / "batch_diagnostics.csv"
        )
        if not online_path.is_file():
            raise RuntimeError(
                f"online B1 candidate-hash reference is missing: {online_path}"
            )
        online = pd.read_csv(
            online_path,
            dtype={"candidate_sha256": str},
        )
        replay = row_frame[
            row_frame["runner"].eq("B1_random_spline_view_ce")
        ].sort_values("batch_index")
        online = online.sort_values("batch_index")
        if (
            len(online) != len(replay)
            or online["batch_index"].tolist() != replay["batch_index"].tolist()
            or online["candidate_sha256"].tolist()
            != replay["candidate_sha256"].tolist()
            or not online["effective_ssaw_seed"]
            .eq(EXPECTED_EFFECTIVE_SSAW_SEED)
            .all()
        ):
            raise RuntimeError(
                f"{flow}: online/replay candidate hash or effective seed mismatch"
            )
        row_frame["online_replay_candidate_hash_verified"] = True
        return row_frame, pd.DataFrame(layer_rows), source_hash
    finally:
        cleanup_trainer(trainer, r1_adapter, source_model, close_summary=True)
        r1_adapter = source_model = None
        release_cuda()
        gc.collect()


def _signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "flow": list(spec["flow"]),
        "source_config": spec["source_config"],
        "tta_config": spec["tta_config"],
        "online_matrix_dir": spec["online_matrix_dir"],
        "branches": [*BRANCH_RUNNERS, MATCHED_RUNNER],
        "target_labels_used_for_online_decision": False,
    }


def _worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["cell_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_hash = _hash_json(_signature(spec))
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") == "ok"
            and summary.get("signature_hash") == signature_hash
        ):
            return 0
    try:
        lock = (
            wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
            if str(spec["device"]).lower().startswith("cuda")
            else None
        )
        if lock is None:
            rows, layers, source_hash = _run_flow(spec)
        else:
            with lock:
                rows, layers, source_hash = _run_flow(spec)
        atomic_write_csv(rows, output_dir / "causal_rows.csv", index=False)
        atomic_write_csv(layers, output_dir / "layer_rows.csv", index=False)
        atomic_write_json(
            {
                "status": "ok",
                "protocol": PROTOCOL,
                "scenario": _flow_label(spec["flow"]),
                "signature_hash": signature_hash,
                "source_model_sha256": source_hash,
                "rows": int(len(rows)),
                "batches": int(rows["batch_index"].nunique()),
            },
            summary_path,
        )
        return 0
    except BaseException as exc:
        atomic_write_json(
            {
                "status": "failed",
                "protocol": PROTOCOL,
                "scenario": _flow_label(spec["flow"]),
                "signature_hash": signature_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exc).lower(),
            },
            summary_path,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def _aggregate(output_dir: Path) -> None:
    rows, layers = [], []
    for flow in FLOWS:
        cell_dir = output_dir / "cells" / f"flow_{flow[0]}_to_{flow[1]}"
        row_path = cell_dir / "causal_rows.csv"
        layer_path = cell_dir / "layer_rows.csv"
        if row_path.is_file():
            rows.append(pd.read_csv(row_path))
        if layer_path.is_file():
            layers.append(pd.read_csv(layer_path))
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    layer_frame = pd.concat(layers, ignore_index=True) if layers else pd.DataFrame()
    atomic_write_csv(frame, output_dir / "causal_rows.csv", index=False)
    atomic_write_csv(layer_frame, output_dir / "layer_rows.csv", index=False)
    if frame.empty:
        atomic_write_csv(pd.DataFrame(), output_dir / "causal_summary.csv", index=False)
        return
    metrics = [
        "delta_current_f1",
        "delta_current_nll",
        "delta_next_f1",
        "delta_next_nll",
        "parameter_delta_norm",
        "update_cosine_to_B0",
        "update_cosine_to_B1",
        "diag_pre_clip_gradient_norm_mean",
        "diag_post_clip_gradient_norm_mean",
        "diag_clip_trigger_rate",
    ]
    summary = frame.groupby("runner", as_index=False)[metrics].mean(numeric_only=True)
    atomic_write_csv(summary, output_dir / "causal_summary.csv", index=False)


def _run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_config, tta_config = current_profiles("HAR")
    tta_config = {**tta_config, **SPLINE_PROFILE}
    specs = [
        {
            "flow": list(flow),
            "cell_dir": str(
                (args.output_dir / "cells" / f"flow_{flow[0]}_to_{flow[1]}").resolve()
            ),
            "source_config": source_config,
            "tta_config": tta_config,
            "data_path": str(args.data_path.resolve()),
            "device": args.device,
            "backbone": args.backbone,
            "pretrain_cache_dir": str(args.pretrain_cache_dir.resolve()),
            "gpu_lock_path": str(args.gpu_lock_path.resolve()),
            "online_matrix_dir": str(args.online_matrix_dir.resolve()),
        }
        for flow in FLOWS
    ]
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "flows": [_flow_label(flow) for flow in FLOWS],
        "source_seed": SOURCE_SEED,
        "stream_seed": STREAM_SEED,
        "effective_ssaw_seed": EXPECTED_EFFECTIVE_SSAW_SEED,
        "branches": [*BRANCH_RUNNERS, MATCHED_RUNNER],
        "main_trajectory": "B0 raw-only with checked-in 23 inner steps",
        "counterfactual_steps": 1,
        "same_batch_start_model_optimizer_bn_state": True,
        "mask_and_pseudo_label_equality_asserted": True,
        "effective_seed_and_candidate_hash_asserted": True,
        "online_matrix_dir": str(args.online_matrix_dir.resolve()),
        "bdup_reuses_b1_contract": True,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_offline_evaluation": True,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_new_spline_parameter_selection": False,
        "effective_tta_config": tta_config,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    completed, failures = 0, []
    for spec in specs:
        cell_dir = Path(spec["cell_dir"])
        cell_dir.mkdir(parents=True, exist_ok=True)
        spec_path = cell_dir / "worker_spec.json"
        atomic_write_json(spec, spec_path)
        log_path = cell_dir / "worker.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            failures.append(
                {
                    "scenario": _flow_label(spec["flow"]),
                    "returncode": int(process.returncode),
                    "log": str(log_path),
                }
            )
            if args.fail_fast:
                break
        else:
            completed += 1
        _aggregate(args.output_dir)
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_flows": completed,
                "failures": failures,
            },
            args.output_dir / "status.json",
        )
    _aggregate(args.output_dir)
    status = "complete" if completed == len(specs) and not failures else "failed"
    final = {
        **manifest,
        "status": status,
        "completed_flows": completed,
        "failures": failures,
    }
    atomic_write_json(final, args.output_dir / "manifest.json")
    atomic_write_json(final, args.output_dir / "status.json")
    return 0 if status == "complete" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "Dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument(
        "--online-matrix-dir", type=Path, default=DEFAULT_ONLINE_MATRIX_DIR
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        return _worker(args.worker_spec)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
