"""Process-isolated representative causal panels for the current DuSafe.

This runner is intentionally separate from the production algorithm and from
the formal five-flow main table.  A cell is one dataset/flow/source-seed and
is executed in a fresh subprocess.  Within a cell every variant starts from
the same fixed-source checkpoint and every batch fork starts from an exact
model+buffer+optimizer+RNG snapshot.

Panels
-------
``A`` compares accept-all raw updates with confidence-admitted raw updates.
``B`` compares confidence-only, matched raw duplication, random eligible
spline, and the current hard SSAW branch.  Each row also carries the same
unseen-Sobol direction-bank diagnostics (flip, worst margin, consistency,
eligible coverage, and margin ratio) plus the update norm.
``C`` reports the hard-SSAW effect overall and conditional on active batches
(``eligible_coverage >= 0.25``).  The active-batch selector uses only the
admission/active masks; labels are consumed only for the conditional F1 and
admission endpoints.

The parent command is plan-only unless ``--execute`` is supplied.  The child
``--cell`` command is the only path that constructs a trainer, so importing
this module or building a plan cannot start a GPU job.  Within a cell,
``confidence_only`` is the committed reference trajectory: every variant is
restored to that batch-start state, and only the reference post-update state
advances the stream.  This is a causal fork, not four independent online
adaptation trajectories.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.representative_causal_ablation import (  # noqa: E402
    PANEL_A_VARIANTS,
    PANEL_B_VARIANTS,
    REPRESENTATIVE_VARIANTS as CORE_REPRESENTATIVE_VARIANTS,
    RepresentativeRandomEligibleSpline,
)
from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from dataloader.corruption_transforms import CORRUPTION_REGISTRY  # noqa: E402
from scripts.counterfactual_horizon_common import (  # noqa: E402
    BatchView,
    future_metrics,
    normalize_batch,
    restore_state,
    snapshot_state,
    state_hash,
    _nested_equal,
    _restore_rng_state,
    _restore_training_state,
)
from scripts.paper_flow_profiles import (  # noqa: E402
    DEFAULT_PAPER_FLOW_PROFILE_JSON,
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.run_dusafe_replacement_ablation import (  # noqa: E402
    ABLATION_CODE_FILES,
    DEFAULT_EEG_PROFILE_ROOT,
    DEFAULT_PROFILE_ROOT,
    ablation_code_sha256,
    _load_profiles as _load_flowwise_source_profiles,
    _load_source_references,
)
from scripts.run_final_ssaw_full_no_ssaw_five_flow import production_code_sha256  # noqa: E402
from scripts.run_controlled_safety_benchmark import deterministic_mask_fn  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    BatchTransformLoader,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)
from optim.optimizer import build_optimizer  # noqa: E402


PROTOCOL = "paper_representative_causal_ablation_v5_stable_radius"
DATASETS = ("EEG", "HAR", "HHAR")
# The prospective evidence bundle uses the same zero-based source-seed
# convention as the v5 finalizer and all other formal panels.
SOURCE_SEEDS = (0, 1, 2)
STREAM_SEED = 42
HORIZON = 5
ACTIVE_THRESHOLD = 0.25
FAILED_ARCHIVE_DIRNAME = "failed_attempts"
DEFAULT_OUTPUT = ROOT / "results" / "paper_evidence_v1" / "representative_causal_ablation"
DEFAULT_SELECTION = ROOT / "results" / "paper_evidence_v1" / "representative_flows" / "selected_flows.json"
DEFAULT_PROFILE = ROOT / "configs" / "paper_flow_profiles_v1.json"
DEFAULT_HELDOUT_BANK_TAG = "test_v1"


class EvidenceMatchedRandomEligibleSpline(RepresentativeRandomEligibleSpline):
    """Evidence-only random control with Full's hard-candidate budget.

    The formal core ablation keeps its original label-preserving random view.
    The causal mechanism table instead requires the matched eligible/hard
    candidate pool so Random and Hard differ only in ranking.  Keeping this
    override in the causal runner prevents evidence-only matching changes from
    invalidating the formal core-ablation implementation signature.
    """

    spline_selection_mode = "random_hard_candidate"


REPRESENTATIVE_VARIANTS = dict(CORE_REPRESENTATIVE_VARIANTS)
REPRESENTATIVE_VARIANTS["random_eligible_spline"] = (
    EvidenceMatchedRandomEligibleSpline
)


def get_representative_variant(name: str):
    try:
        return REPRESENTATIVE_VARIANTS[str(name).strip()]
    except KeyError as exc:
        raise ValueError(
            f"unknown representative causal variant: {name!r}; "
            f"expected one of {tuple(REPRESENTATIVE_VARIANTS)}"
        ) from exc

# Causal mechanism evidence has its own implementation digest.  It is kept
# separate from both the production digest and replacement-ablation digest so
# a rerun of the causal fork cannot silently reuse an old state-replay or
# held-out-direction implementation.
CAUSAL_EVIDENCE_CODE_FILES = (
    ROOT / "algorithms" / "representative_causal_ablation.py",
    ROOT / "scripts" / "counterfactual_horizon_common.py",
    ROOT / "scripts" / "run_representative_causal_ablation.py",
)


def causal_evidence_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in CAUSAL_EVIDENCE_CODE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"missing causal evidence code file: {path}")
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_heldout_bank_tag(value: str) -> str:
    """Validate a stable, filesystem-safe direction-bank identifier."""

    tag = str(value).strip()
    if not tag or re.fullmatch(r"[A-Za-z0-9_.-]+", tag) is None:
        raise ValueError(
            "heldout_bank_tag must contain only letters, digits, '.', '_', or '-'"
        )
    return tag


def heldout_bank_seed(
    *,
    dataset: str,
    scenario: str,
    source_seed: int,
    stream_seed: int,
    heldout_bank_tag: str,
    training_seed: int,
) -> int:
    """Derive one reproducible Sobol seed, disjoint from training/stream seeds."""

    tag = normalize_heldout_bank_tag(heldout_bank_tag)
    material = (
        f"{dataset}|{scenario}|{int(source_seed)}|{int(stream_seed)}|{tag}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    seed %= 2_147_483_647
    if seed < 1:
        seed = 1
    while seed in {int(training_seed), int(stream_seed)}:
        seed = (seed % 2_147_483_646) + 1
    return seed


def summarize_discrete_stable_radius(
    raw_logits: torch.Tensor,
    candidate_logits_by_view: torch.Tensor,
    *,
    confidence_mask: torch.Tensor,
    ray_count: int,
    radius_levels: Sequence[float],
    log_strength: float,
) -> dict[str, float | None]:
    """Summarize the contiguous label-stable radius on unseen spline rays.

    The candidate layout is ``[ray, radius, batch, class]`` with radii in the
    generator's declared order.  We sort radii from small to large and count
    only the stable prefix from zero; a prediction that flips at a smaller
    radius and flips back later therefore cannot inflate the measured radius.
    True target labels are never read.
    """

    raw = torch.as_tensor(raw_logits, dtype=torch.float64, device="cpu")
    candidates = torch.as_tensor(
        candidate_logits_by_view, dtype=torch.float64, device="cpu"
    )
    admitted = torch.as_tensor(
        confidence_mask, dtype=torch.bool, device="cpu"
    ).reshape(-1)
    levels = torch.as_tensor(
        tuple(float(value) for value in radius_levels), dtype=torch.float64
    )
    if raw.ndim != 2 or candidates.ndim != 3:
        raise ValueError("stable-radius logits must be [B,K] and [V,B,K]")
    if raw.size(0) != candidates.size(1) or raw.size(1) != candidates.size(2):
        raise ValueError("stable-radius raw/candidate shapes disagree")
    if admitted.numel() != raw.size(0):
        raise ValueError("stable-radius admission mask has the wrong length")
    if int(ray_count) < 1 or levels.numel() < 1:
        raise ValueError("stable-radius rays and radius levels must be non-empty")
    if candidates.size(0) != int(ray_count) * int(levels.numel()):
        raise ValueError("stable-radius candidate count disagrees with ray grid")
    if not torch.isfinite(raw).all() or not torch.isfinite(candidates).all():
        raise ValueError("stable-radius logits must be finite")
    if not torch.isfinite(levels).all() or bool((levels <= 0.0).any()):
        raise ValueError("stable-radius levels must be finite and positive")
    strength = float(log_strength)
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError("stable-radius log_strength must be finite and positive")

    predictions = raw.argmax(dim=1)
    preserving = candidates.argmax(dim=2).eq(predictions[None, :]).reshape(
        int(ray_count), int(levels.numel()), raw.size(0)
    )
    order = torch.argsort(levels)
    ordered_levels = levels[order] * strength
    ordered_preserving = preserving[:, order, :]
    contiguous = ordered_preserving.to(torch.int64).cumprod(dim=1).bool()
    radii = ordered_levels.view(1, -1, 1)
    per_ray = torch.where(contiguous, radii, torch.zeros_like(radii)).amax(dim=1)
    per_sample = per_ray.mean(dim=0)
    normalized = per_sample / strength
    cap_stable = contiguous[:, -1, :]
    admitted_count = int(admitted.sum().item())
    if admitted_count:
        selected_radius = per_sample[admitted]
        selected_normalized = normalized[admitted]
        cap_successes = int(cap_stable[:, admitted].sum().item())
        cap_total = int(int(ray_count) * admitted_count)
        return {
            "heldout_stable_radius": float(selected_radius.mean().item()),
            "heldout_stable_radius_normalized": float(
                selected_normalized.mean().item()
            ),
            "heldout_stable_radius_sum": float(selected_radius.sum().item()),
            "heldout_stable_radius_normalized_sum": float(
                selected_normalized.sum().item()
            ),
            "heldout_stable_radius_admitted_count": float(admitted_count),
            "heldout_cap_stable_ray_fraction": float(cap_successes / cap_total),
            "heldout_cap_stable_ray_successes": float(cap_successes),
            "heldout_cap_stable_ray_total": float(cap_total),
            "heldout_stable_radius_q10": float(
                torch.quantile(selected_radius, 0.10).item()
            ),
            "heldout_stable_radius_q50": float(
                torch.quantile(selected_radius, 0.50).item()
            ),
            "heldout_stable_radius_q90": float(
                torch.quantile(selected_radius, 0.90).item()
            ),
        }
    return {
        "heldout_stable_radius": None,
        "heldout_stable_radius_normalized": None,
        "heldout_stable_radius_sum": 0.0,
        "heldout_stable_radius_normalized_sum": 0.0,
        "heldout_stable_radius_admitted_count": 0.0,
        "heldout_cap_stable_ray_fraction": None,
        "heldout_cap_stable_ray_successes": 0.0,
        "heldout_cap_stable_ray_total": 0.0,
        "heldout_stable_radius_q10": None,
        "heldout_stable_radius_q50": None,
        "heldout_stable_radius_q90": None,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _artifact_sha256(path: Path) -> str:
    """Hash a completed artifact after its atomic write has landed."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_scenario(text: str) -> str:
    value = str(text).strip().replace(",", "->")
    if "->" not in value:
        raise ValueError(f"scenario must be source->target: {text!r}")
    source, target = (part.strip() for part in value.split("->", 1))
    if not source or not target:
        raise ValueError(f"scenario must be source->target: {text!r}")
    return f"{source}->{target}"


def parse_conditions(text: str | Sequence[str] | None) -> tuple[tuple[str, str | None], ...]:
    """Parse ``clean`` or ``corruption:severity`` condition names."""

    severity_aliases = {"s3": "moderate", "s6": "severe"}
    values = (text if isinstance(text, (list, tuple)) else str(text or "clean").split(","))
    result: list[tuple[str, str | None]] = []
    for raw in values:
        item = str(raw).strip()
        if not item:
            continue
        if item.lower() == "clean":
            key = ("clean", None)
        else:
            if ":" not in item:
                raise ValueError(
                    f"condition {item!r} must be clean or corruption:severity"
                )
            corruption, severity = (part.strip() for part in item.split(":", 1))
            if corruption not in CORRUPTION_REGISTRY or not severity:
                raise ValueError(f"unknown corruption condition: {item!r}")
            severity = severity_aliases.get(severity.lower(), severity.lower())
            if severity not in {"mild", "moderate", "severe"}:
                raise ValueError(f"unknown corruption severity: {item!r}")
            key = (corruption, severity)
        if key not in result:
            result.append(key)
    if not result:
        raise ValueError("at least one condition is required")
    return tuple(result)


def condition_label(condition: tuple[str, str | None]) -> str:
    corruption, severity = condition
    return "clean" if corruption == "clean" else f"{corruption}:{severity}"


def condition_slug(condition: tuple[str, str | None]) -> str:
    """Return a filesystem-safe condition directory name on all platforms."""

    return condition_label(condition).replace(":", "_")


def load_selected_flows(path: str | Path, *, datasets: Sequence[str]) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"representative selected_flows.json is missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "paper_representative_flow_selection_v1":
        raise ValueError("selected flow protocol mismatch")
    if payload.get("selection_uses_target_labels") is not False or payload.get(
        "selection_uses_f1"
    ) is not False:
        raise ValueError("representative flow selection must be label-free")
    selected = payload.get("selected_flows")
    if not isinstance(selected, Mapping):
        raise ValueError("selected_flows.json lacks selected_flows")
    result: dict[str, str] = {}
    for raw_dataset in datasets:
        dataset = str(raw_dataset).strip().upper()
        scenario = parse_scenario(selected.get(dataset, ""))
        formal = {f"{src}->{trg}" for src, trg in formal_scenario_pairs(dataset)}
        if scenario not in formal:
            raise ValueError(f"selected representative flow is not formal: {dataset}:{scenario}")
        result[dataset] = scenario
    return result


def build_plan(
    *,
    datasets: Sequence[str] = DATASETS,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
    conditions: Sequence[tuple[str, str | None]] = (("clean", None),),
    selected_flows: Mapping[str, str],
    output_dir: str | Path = DEFAULT_OUTPUT,
    data_path: str | Path = ROOT / "data" / "Dataset",
    device: str = "cuda:0",
    backbone: str = "CNN",
    pretrain_cache_dir: str | Path = ROOT / "results" / "pretrain_cache" / "optuna_stepwise",
    profile_json: str | Path = DEFAULT_PROFILE,
    source_profile_root: str | Path | None = None,
    eeg_source_profile_root: str | Path | None = None,
    source_reference_csv: str | Path | None = None,
    horizons: Sequence[int] = (HORIZON,),
    heldout_bank_tag: str = DEFAULT_HELDOUT_BANK_TAG,
) -> dict[str, Any]:
    datasets = tuple(str(value).upper() for value in datasets)
    if not datasets or any(value not in DATASETS for value in datasets):
        raise ValueError(f"datasets must be a non-empty subset of {DATASETS}")
    seeds = tuple(int(value) for value in source_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("source_seeds must be non-empty and unique")
    profiles = load_paper_flow_profiles(profile_json, datasets=datasets)
    normalized_horizons = tuple(sorted({int(value) for value in horizons}))
    if not normalized_horizons or any(value < 1 for value in normalized_horizons):
        raise ValueError("horizons must contain positive integers")
    heldout_bank_tag = normalize_heldout_bank_tag(heldout_bank_tag)
    current_causal_digest = causal_evidence_code_sha256()
    current_production_digest = production_code_sha256()
    current_ablation_digest = ablation_code_sha256()
    cells: list[dict[str, Any]] = []
    for dataset in datasets:
        scenario = parse_scenario(selected_flows[dataset])
        profile_for_flow(profiles, dataset, scenario)
        source, target = scenario.split("->", 1)
        for seed in seeds:
            for condition in conditions:
                key = f"{dataset}:{source}->{target}:source{seed}:{condition_label(condition)}"
                output = (
                    Path(output_dir)
                    / dataset
                    / f"flow_{source}_to_{target}"
                    / f"source_seed_{seed}"
                    / condition_slug(condition)
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--cell",
                    "--dataset",
                    dataset,
                    "--scenario",
                    scenario,
                    "--source-seed",
                    str(seed),
                    "--stream-seed",
                    str(STREAM_SEED),
                    "--condition",
                    condition_label(condition),
                    "--data-path",
                    str(data_path),
                    "--device",
                    str(device),
                    "--backbone",
                    str(backbone),
                    "--pretrain-cache-dir",
                    str(pretrain_cache_dir),
                    "--profile-json",
                    str(profile_json),
                    "--output-dir",
                    str(output),
                    "--horizons",
                    ",".join(str(value) for value in normalized_horizons),
                    "--heldout-bank-tag",
                    heldout_bank_tag,
                ]
                if source_profile_root is not None:
                    command.extend(
                        ["--source-profile-root", str(source_profile_root)]
                    )
                if eeg_source_profile_root is not None:
                    command.extend(
                        [
                            "--eeg-source-profile-root",
                            str(eeg_source_profile_root),
                        ]
                    )
                if source_reference_csv is not None:
                    command.extend(
                        ["--source-reference-csv", str(source_reference_csv)]
                    )
                cells.append(
                    {
                        "key": key,
                        "dataset": dataset,
                        "scenario": scenario,
                        "source_seed": seed,
                        "stream_seed": STREAM_SEED,
                        "condition": condition_label(condition),
                        "horizons": list(normalized_horizons),
                        "command": command,
                        "output_dir": str(output),
                        "status": "planned",
                        "causal_evidence_code_sha256": current_causal_digest,
                        "production_code_sha256": current_production_digest,
                        "ablation_code_sha256": current_ablation_digest,
                        "dusafe_logging_mode": "evidence",
                        "logging_mode": "evidence",
                        "candidate_cuda_graph_requested_mode": "auto",
                        "candidate_cuda_graph_enabled": False,
                        "candidate_cuda_graph_status": "disabled_evidence_logging",
                        "candidate_cuda_graph_mode": "disabled",
                    }
                )
    return {
        "protocol": PROTOCOL,
        "production_code_sha256": current_production_digest,
        "ablation_code_sha256": current_ablation_digest,
        "ablation_code_files": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in ABLATION_CODE_FILES
        ],
        "causal_evidence_code_sha256": current_causal_digest,
        "causal_evidence_code_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CAUSAL_EVIDENCE_CODE_FILES],
        "status": "planned",
        "datasets": list(datasets),
        "selected_flows": dict(selected_flows),
        "source_seeds": list(seeds),
        "stream_seed": STREAM_SEED,
        "conditions": [condition_label(value) for value in conditions],
        "horizons": list(normalized_horizons),
        "heldout_bank_tag": heldout_bank_tag,
        "active_batch_threshold": ACTIVE_THRESHOLD,
        "variants": list(REPRESENTATIVE_VARIANTS),
        "panels": {
            "A": list(PANEL_A_VARIANTS),
            "B": list(PANEL_B_VARIANTS),
            "C": (
                "hard_ssaw eligible-coverage active batches versus "
                "confidence_only; threshold >=0.25; overall coverage retained"
            ),
        },
        "target_labels_used_for_online_updates": False,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_selected_descriptive": True,
        "confirmatory": False,
        "dusafe_logging_mode": "evidence",
        "logging_mode": "evidence",
        "candidate_cuda_graph_requested_mode": "auto",
        "candidate_cuda_graph_enabled": False,
        "candidate_cuda_graph_status": "disabled_evidence_logging",
        "candidate_cuda_graph_mode": "disabled",
        "evaluation_partition": "target_selected_evaluation",
        "parameter_selection_data_overlap": True,
        "selection_uses_target_labels": False,
        "flow_profile_json": str(Path(profile_json).resolve()),
        "source_profile_root": (
            None
            if source_profile_root is None
            else str(Path(source_profile_root).resolve())
        ),
        "eeg_source_profile_root": (
            None
            if eeg_source_profile_root is None
            else str(Path(eeg_source_profile_root).resolve())
        ),
        "source_reference_csv": (
            None
            if source_reference_csv is None
            else str(Path(source_reference_csv).resolve())
        ),
        "cells": cells,
        "expected_cells": len(cells),
        "process_isolation": "one fresh subprocess per representative flow/seed/condition cell",
        "joint_causal_stream": {
            "reference_variant": "confidence_only",
            "reference_only_commit": True,
            "shared_optimizer_state_replayed": True,
        },
    }


def _adapter_model(adapter: Any) -> torch.nn.Module:
    return getattr(adapter, "model", adapter)


def _adapter_device(adapter: Any) -> torch.device:
    """Return the device that owns the adapter's trainable model."""

    model = _adapter_model(adapter)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _refresh_adaptation_parameter_cache(adapter: Any) -> None:
    """Rebind DuSafe's cached trainable parameters after state replay.

    ``snapshot_state`` captures ordinary adapter attributes by detached clone.
    That is correct for serializable runtime state, but it would replace
    DuSafe's ``_adaptation_parameters`` tuple with non-Parameter tensors when
    ``restore_state`` replays it.  The production adapter expects this cache to
    contain the live model ``Parameter`` objects used by autograd.  Evidence
    branches therefore rebuild only this runtime cache after replay; model,
    optimizer, RNG, and all numerical state remain restored by ``restore_state``.
    """

    if not hasattr(adapter, "_adaptation_parameters"):
        return
    model = _adapter_model(adapter)
    adapter._adaptation_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )


def _restore_evidence_state(adapter: Any, state: Any) -> None:
    """Restore a branch state and repair non-serializable parameter handles."""

    restore_state(adapter, state)
    _refresh_adaptation_parameter_cache(adapter)


def _evidence_nested_equal(left: Any, right: Any) -> bool:
    """Compare replay diagnostics while treating stable NaNs as equal.

    The candidate CUDA-graph diagnostic object is intentionally disabled for
    evidence logging and carries ``nan`` timing fields.  Generic state
    equality uses Python/torch equality, where ``nan != nan``; that made an
    untouched CPU future evaluation report a false mutation.  This helper is
    local to the evidence runner and only changes the audit comparison, not
    the restored adapter state or any production computation.
    """

    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device == right.device
            and left.dtype == right.dtype
            and left.shape == right.shape
        ):
            return False
        if left.is_floating_point() or left.is_complex():
            equal = (left == right) | (torch.isnan(left) & torch.isnan(right))
            return bool(equal.all())
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
        ):
            return False
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(
                _evidence_nested_equal(left[key], right[key]) for key in left
            )
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _evidence_nested_equal(a, b) for a, b in zip(left, right)
            )
        )
    if hasattr(left, "__dict__") or hasattr(right, "__dict__"):
        return (
            type(left) is type(right)
            and hasattr(left, "__dict__")
            and _evidence_nested_equal(vars(left), vars(right))
        )
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        try:
            if math.isnan(float(left)) and math.isnan(float(right)):
                return True
        except (TypeError, ValueError):
            pass
    return _nested_equal(left, right)


def _evidence_states_equal(left: Any, right: Any) -> bool:
    """Return replay-state equality with NaN-safe diagnostic comparison."""

    return all(
        _evidence_nested_equal(getattr(left, name), getattr(right, name))
        for name in (
            "model_state",
            "optimizer_state",
            "adapter_module_state",
            "adapter_buffer_state",
            "runtime_state",
            "training_state",
        )
    )


def _execute_adapter_update(adapter: Any, batch: BatchView) -> Any:
    """Execute the production update path, including configured inner steps.

    Production TTA adapters are ``nn.Module`` instances whose ``forward``
    method owns the inner-step loop.  Lightweight test doubles may expose only
    ``forward_and_adapt`` and therefore use the single-step fallback.
    """

    inputs = {"data": batch.data}
    if isinstance(adapter, torch.nn.Module):
        return adapter(inputs, batch.indices)
    return adapter.forward_and_adapt(
        inputs,
        _adapter_model(adapter),
        getattr(adapter, "optimizer", None),
        batch.indices,
    )


def _move_tensor_tree(value: Any, device: torch.device) -> Any:
    """Move tensor-valued batch fields without changing non-tensor metadata."""

    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, Mapping):
        return {
            key: _move_tensor_tree(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensor_tree(item, device) for item in value]
    return value


def _route_batch_to_adapter_device(
    adapter: Any,
    batch: BatchView,
    *,
    move_labels: bool = False,
) -> BatchView:
    """Route online inputs and tensor metadata to the model device.

    Labels remain on CPU by default because they are offline-only metrics and
    are never passed to ``forward_and_adapt``.  Future-evaluation callers may
    request device labels explicitly; ``future_metrics`` also accepts either
    placement and performs its own final alignment.
    """

    device = _adapter_device(adapter)
    return BatchView(
        data=_move_tensor_tree(batch.data, device),
        labels=(
            _move_tensor_tree(batch.labels, device)
            if move_labels
            else batch.labels
        ),
        indices=_move_tensor_tree(batch.indices, device),
    )


def _metric_from_gate(adapter: Any, batch: BatchView) -> dict[str, Any]:
    gate = getattr(adapter, "_last_gate_log", {}) or {}
    pseudo = torch.as_tensor(gate.get("pseudo_labels", []), dtype=torch.long).view(-1)
    admission = torch.as_tensor(gate.get("admission_mask", []), dtype=torch.bool).view(-1)
    active = torch.as_tensor(gate.get("active_mask", admission), dtype=torch.bool).view(-1)
    labels = batch.labels.detach().cpu().view(-1).long()
    if len(pseudo) != len(labels) or len(admission) != len(labels):
        return {
            "coverage": math.nan,
            "eligible_coverage": math.nan,
            "admitted_count": math.nan,
            "admission_mask_sha256": "",
            "admitted_accuracy": math.nan,
            "incorrect_admission_rate": math.nan,
            "wrong_accept_recall": math.nan,
            "correct_false_rejection_rate": math.nan,
            "unsafe_update_rate": math.nan,
        }
    correct = pseudo.eq(labels)
    admitted = admission
    accepted = int(admitted.sum())
    correct_count = int(correct.sum())
    wrong_count = int((~correct).sum())
    active_count = int(active.sum())
    mask_digest = hashlib.sha256(admitted.numpy().tobytes()).hexdigest()
    return {
        "coverage": float(admitted.float().mean()),
        # This is the label-free eligible/update coverage among confidence-
        # admitted anchors.  It is the selector for Panel C; keep the overall
        # confidence coverage above so conditional rows cannot hide rejected
        # batches or change their denominator post hoc.
        "eligible_coverage": (
            float(active_count / accepted) if accepted else math.nan
        ),
        "admitted_count": accepted,
        "admission_mask_sha256": mask_digest,
        "admitted_accuracy": float(correct[admitted].float().mean()) if accepted else math.nan,
        # Kept for compatibility with already-generated v2 cells.  This is
        # exactly 1 - admitted_accuracy and is not used as a separate headline
        # metric in the new causal panel.
        "incorrect_admission_rate": float((~correct[admitted]).float().mean()) if accepted else math.nan,
        # Unlike incorrect_admission_rate, this measures how many of all wrong
        # pseudo-labels passed admission.  It therefore separates filtering
        # power from the precision of the admitted subset.
        "wrong_accept_recall": (
            float(((~correct) & admitted).float().sum() / wrong_count)
            if wrong_count
            else math.nan
        ),
        "correct_false_rejection_rate": float((correct & ~admitted).float().sum() / max(correct_count, 1)),
        "unsafe_update_rate": float((~correct[active]).float().mean()) if active_count else math.nan,
    }


def _diagnostics_from_adapter(adapter: Any) -> dict[str, Any]:
    log = dict(getattr(adapter, "_last_batch_log", {}) or {})
    gate = dict(getattr(adapter, "_last_gate_log", {}) or {})
    result: dict[str, Any] = {}
    for name in (
        "ssaw_training_participation_rate",
        "ssaw_admitted_participation_rate",
        "ssaw_weighted_consistency_loss",
        "raw_ce_loss",
        "ssaw_selected_view_fraction",
        "ssaw_selected_margin_drop_mean",
        "ssaw_selected_normalized_margin_ratio_mean",
    ):
        if name in log:
            result[name] = float(log[name])
    candidate_hash = gate.get("ssaw_candidate_sha256") or gate.get("candidate_sha256")
    if candidate_hash is not None:
        result["candidate_pool_sha256"] = str(candidate_hash)
    return result


def _evidence_candidate_graph_metadata(adapter: Any) -> dict[str, Any]:
    """Return and validate the candidate-graph contract for one branch.

    Causal/evidence logging must never execute the production CUDA graph.  The
    request is intentionally kept visible (``auto`` in the prospective
    profile), while the adapter must report the explicit
    ``disabled_evidence_logging`` state.  Checking the actual adapter object,
    rather than only the requested hyperparameter, prevents a later refactor
    from silently re-enabling graph replay in a mechanism panel.
    """

    graph = getattr(adapter, "_candidate_cuda_graph", None)
    diagnostics = getattr(graph, "diagnostics", None)
    if not callable(diagnostics):
        raise RuntimeError(
            "causal evidence adapter lacks candidate CUDA graph diagnostics"
        )
    observed = dict(diagnostics())
    requested = str(
        observed.get("candidate_cuda_graph_requested_mode", "")
    ).strip().lower()
    enabled = observed.get("candidate_cuda_graph_enabled", None)
    status = str(observed.get("candidate_cuda_graph_status", "")).strip()
    if requested not in {"off", "auto", "force"}:
        raise RuntimeError(
            "causal evidence candidate graph request is invalid: "
            f"{requested!r}"
        )
    if bool(enabled):
        raise RuntimeError(
            "causal evidence candidate CUDA graph unexpectedly enabled"
        )
    if status != "disabled_evidence_logging":
        raise RuntimeError(
            "causal evidence candidate CUDA graph status is not "
            f"disabled_evidence_logging: {status!r}"
        )
    return {
        "candidate_cuda_graph_requested_mode": requested,
        "candidate_cuda_graph_enabled": False,
        "candidate_cuda_graph_status": status,
        "candidate_cuda_graph_mode": "disabled",
    }


def _state_update_norm(pre_state: Any, post_state: Any) -> float:
    """Return the model-state L2 update norm for one batch.

    The state snapshots include parameters and persistent buffers.  Reporting
    this as a separate diagnostic avoids treating optimizer/update magnitude
    as another F1 endpoint, while keeping the quantity identical across the
    role-matched variants.
    """

    squared = 0.0
    for name, before in pre_state.model_state.items():
        after = post_state.model_state.get(name)
        if after is None or not torch.is_tensor(before) or not torch.is_tensor(after):
            continue
        delta = after.to(dtype=torch.float64) - before.to(dtype=torch.float64)
        squared += float(delta.square().sum().item())
    return float(math.sqrt(max(squared, 0.0)))


def _shared_model_buffer_hash(state: Any) -> str:
    """Hash only model/BN and adapter buffer state used by a branch start.

    Adapter runtime state is intentionally excluded.  The representative
    variants have different immutable roles (for example, ``enable_ssaw``),
    while their model parameters, persistent buffers, and optimizer slots
    must be identical at the causal fork.  This hash is therefore the
    row-level audit key for the shared ``S_t`` contract.
    """

    digest = hashlib.sha256()

    def add(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8"))
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif value is None:
            digest.update(b"<none>")
        else:
            digest.update(repr(value).encode("utf-8"))

    add(state.model_state)
    add(state.adapter_buffer_state)
    add(state.training_state)
    return digest.hexdigest()


def _shared_optimizer_hash(state: Any) -> str:
    """Hash optimizer state separately for the shared causal-fork audit."""

    digest = hashlib.sha256()

    def add(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8"))
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif value is None:
            digest.update(b"<none>")
        else:
            digest.update(repr(value).encode("utf-8"))

    add(state.optimizer_state)
    return digest.hexdigest()


def _restore_shared_branch_state(adapter: Any, state: Any) -> None:
    """Restore common model/optimizer/RNG state while preserving branch role.

    ``restore_state`` also restores mutable adapter runtime attributes.  That
    is correct for replaying one adapter, but incorrect for a causal fork:
    restoring the confidence-only runtime object onto the SSAW branch would
    disable or replace its branch-specific candidate generator.  This helper
    restores the common model, persistent adapter buffers, optimizer slots,
    training mode, and process RNG only; branch-specific runtime state remains
    owned by the branch.
    """

    model = _adapter_model(adapter)
    model.load_state_dict(copy.deepcopy(state.model_state), strict=True)

    buffers = getattr(adapter, "_buffers", None)
    if isinstance(buffers, Mapping) and state.adapter_buffer_state is not None:
        for name, value in state.adapter_buffer_state.items():
            if name not in buffers:
                continue
            cloned = copy.deepcopy(value)
            target = buffers[name]
            if torch.is_tensor(cloned) and torch.is_tensor(target):
                cloned = cloned.to(device=target.device, dtype=target.dtype)
            buffers[name] = cloned

    optimizer = getattr(adapter, "optimizer", None)
    if optimizer is not None:
        if state.optimizer_state is None:
            raise RuntimeError("shared causal state has no optimizer state")
        optimizer.load_state_dict(copy.deepcopy(state.optimizer_state))
        optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = None

    _restore_training_state(adapter, state.training_state)
    if state.rng_state is not None:
        _restore_rng_state(state.rng_state)


def _assert_shared_start_state(
    adapter: Any,
    reference_state: Any,
    *,
    expected_model_buffer_hash: str,
    expected_optimizer_hash: str,
) -> tuple[str, str]:
    """Restore and verify one branch's exact causal start state."""

    _restore_shared_branch_state(adapter, reference_state)
    observed = snapshot_state(adapter, cpu=True)
    model_buffer_hash = _shared_model_buffer_hash(observed)
    optimizer_hash = _shared_optimizer_hash(observed)
    if model_buffer_hash != expected_model_buffer_hash:
        raise RuntimeError("causal branch model/buffer start state diverged")
    if optimizer_hash != expected_optimizer_hash:
        raise RuntimeError("causal branch optimizer start state diverged")
    return model_buffer_hash, optimizer_hash


def _heldout_direction_callback(
    *,
    dataset: str,
    scenario: str,
    source_seed: int,
    stream_seed: int,
    profile: Mapping[str, Any],
    normalization_stats: tuple[Any, Any],
    heldout_bank_tag: str = DEFAULT_HELDOUT_BANK_TAG,
):
    """Build a label-free unseen-Sobol diagnostic callback for one variant.

    Each variant receives a fresh generator with the same held-out seed and
    profile, so the direction bank and candidate budget are paired while the
    model state remains variant-specific.  The callback never reads the
    ``BatchView.labels`` field.
    """

    from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView
    from ssaw_evaluation.heldout_mechanism import (
        heldout_direction_diagnostics,
        summarize_heldout_direction_diagnostics,
    )

    training_seed = int(profile.get("ssaw_sobol_seed", 1729))
    heldout_bank_tag = normalize_heldout_bank_tag(heldout_bank_tag)
    heldout_seed = heldout_bank_seed(
        dataset=dataset,
        scenario=scenario,
        source_seed=source_seed,
        stream_seed=stream_seed,
        heldout_bank_tag=heldout_bank_tag,
        training_seed=training_seed,
    )
    generator = UnifiedSplineHardView(
        num_control_points=int(profile.get("spline_control_points", 10)),
        num_directions=int(profile.get("spline_num_directions", 4)),
        log_strength=float(profile.get("spline_log_strength", 0.2)),
        radius_levels=tuple(
            float(value)
            for value in profile.get("spline_radius_levels", (1.0, 0.5, 0.25))
        ),
        sobol_seed=heldout_seed,
    )
    mean, std = normalization_stats

    def callback(current_adapter: Any, batch: BatchView, _batch_index: int) -> dict[str, Any]:
        raw_data = batch.data.float()
        device = next(_adapter_model(current_adapter).parameters()).device
        inputs = raw_data.to(device)
        prepared = generator.prepare_view_inputs(
            inputs,
            normalization_mean=mean,
            normalization_std=std,
        )
        candidate_inputs = torch.as_tensor(prepared["view_inputs"])
        view_count, batch_size = candidate_inputs.shape[:2]
        # Use the read-only model path directly for margin diagnostics.  It
        # preserves module modes, buffers, and process RNG state.
        from scripts.counterfactual_horizon_common import model_logits

        raw_logits = model_logits(current_adapter, raw_data, device=device).cpu()
        candidate_logits = model_logits(
            current_adapter,
            candidate_inputs.reshape(view_count * batch_size, *candidate_inputs.shape[2:]),
            device=device,
        ).reshape(view_count, batch_size, -1).cpu()
        gate = getattr(current_adapter, "_last_gate_log", {}) or {}
        admission = torch.as_tensor(
            gate.get("admission_mask", torch.ones(batch_size, dtype=torch.bool)),
            dtype=torch.bool,
        ).reshape(-1)
        direction_summary = summarize_heldout_direction_diagnostics(
            heldout_direction_diagnostics(
                raw_logits,
                candidate_logits,
                confidence_mask=admission,
            )
        )
        radius_summary = summarize_discrete_stable_radius(
            raw_logits,
            candidate_logits,
            confidence_mask=admission,
            ray_count=generator.ray_count,
            radius_levels=generator.radius_levels,
            log_strength=generator.log_strength,
        )
        summary = {
            "heldout_eligible_coverage": direction_summary["eligible_coverage"],
            "heldout_margin_ratio": direction_summary["margin_ratio"],
            "heldout_flip_rate": direction_summary["heldout_flip_rate"],
            "heldout_worst_margin": direction_summary["heldout_worst_margin"],
            "heldout_consistency": direction_summary["heldout_consistency"],
            "heldout_confidence_admitted_count": direction_summary[
                "confidence_admitted_count"
            ],
            "heldout_eligible_count": direction_summary["eligible_count"],
            "heldout_sample_count": float(batch_size),
            **radius_summary,
        }
        summary["heldout_candidate_pool_sha256"] = UnifiedSplineHardView.candidate_sha256(
            candidate_inputs
        )
        summary["heldout_candidate_count"] = int(view_count)
        summary["heldout_sobol_seed"] = int(heldout_seed)
        summary["heldout_bank_tag"] = heldout_bank_tag
        return summary

    return callback


def run_variant_horizon(
    adapter: Any,
    batches: Iterable[Any],
    *,
    variant: str,
    condition: str,
    horizons: Sequence[int] = (HORIZON,),
    device: str | torch.device | None = None,
    num_classes: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    heldout_diagnostics_fn: Any | None = None,
) -> pd.DataFrame:
    """Run one variant with exact batch-start snapshots and read-only futures."""

    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    stream = iter(batches)
    buffer: deque[BatchView] = deque()
    for _ in range(max(horizons) + 1):
        try:
            buffer.append(normalize_batch(next(stream)))
        except StopIteration:
            break
    rows: list[dict[str, Any]] = []
    batch_index = 0
    while buffer:
        current = buffer[0]
        routed_current = _route_batch_to_adapter_device(adapter, current)
        future_batches = list(buffer)[1:]
        pre = snapshot_state(adapter, cpu=True)
        pre_hash = state_hash(pre)
        _restore_evidence_state(adapter, pre)
        _ = _execute_adapter_update(adapter, routed_current)
        post = snapshot_state(adapter, cpu=True)
        post_hash = state_hash(post)
        update_norm = _state_update_norm(pre, post)
        safety = _metric_from_gate(adapter, current)
        diagnostics = _diagnostics_from_adapter(adapter)
        heldout_diagnostics = (
            dict(heldout_diagnostics_fn(adapter, current, batch_index))
            if heldout_diagnostics_fn is not None
            else {}
        )
        for horizon in horizons:
            if horizon > len(future_batches):
                continue
            _restore_evidence_state(adapter, post)
            routed_future = [
                _route_batch_to_adapter_device(adapter, batch, move_labels=True)
                for batch in future_batches[:horizon]
            ]
            metrics = future_metrics(
                adapter,
                routed_future,
                device=device,
                num_classes=num_classes,
            )
            after_eval = snapshot_state(adapter, cpu=True)
            eval_untouched = _evidence_states_equal(after_eval, post)
            rng_untouched = _rng_states_equal(after_eval.rng_state, post.rng_state)
            row = {
                **dict(metadata or {}),
                "variant": variant,
                "condition": condition,
                "batch_index": batch_index,
                "horizon": horizon,
                "future_macro_f1": float(metrics["macro_f1"]),
                "future_true_label_nll": float(metrics["true_label_nll"]),
                "future_samples": int(metrics["samples"]),
                "pre_batch_state_hash": pre_hash,
                "post_update_state_hash": post_hash,
                "update_norm": update_norm,
                "future_eval_untouched": bool(eval_untouched),
                "future_eval_rng_untouched": bool(rng_untouched),
                **safety,
                **diagnostics,
                **heldout_diagnostics,
            }
            rows.append(row)
        _restore_evidence_state(adapter, post)
        buffer.popleft()
        try:
            buffer.append(normalize_batch(next(stream)))
        except StopIteration:
            pass
        batch_index += 1
    if not rows:
        raise ValueError("no complete future horizon windows were evaluated")
    return pd.DataFrame(rows)


def run_joint_variant_horizon(
    variants: Mapping[str, Any],
    batches: Iterable[Any],
    *,
    reference_variant: str = "confidence_only",
    condition: str,
    horizons: Sequence[int] = (HORIZON,),
    device: str | torch.device | None = None,
    num_classes: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    heldout_diagnostics_fns: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run all variants as causal forks from one committed reference path.

    At batch ``t`` the confidence-only adapter is the canonical trajectory
    ``S_t``.  Every variant is restored to the same model/BN/optimizer/RNG
    state before its update.  Future windows are evaluated from that branch's
    post-update state without updates.  Only the confidence-only post-update
    state is committed to advance to ``S_{t+1}``; all other branch states are
    discarded at the fork.  This prevents the common but invalid pattern in
    which each variant independently adapts the whole stream and is later
    compared after different online histories.
    """

    if reference_variant not in variants:
        raise ValueError(f"reference variant is missing: {reference_variant}")
    if not variants:
        raise ValueError("at least one variant is required")
    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons must contain positive integers")

    stream = iter(batches)
    buffer: deque[BatchView] = deque()
    for _ in range(max(horizons) + 1):
        try:
            buffer.append(normalize_batch(next(stream)))
        except StopIteration:
            break

    rows: list[dict[str, Any]] = []
    batch_index = 0
    reference = variants[reference_variant]
    callback_map = dict(heldout_diagnostics_fns or {})
    while buffer:
        current = buffer[0]
        future_batches = list(buffer)[1:]
        reference_state = snapshot_state(reference, cpu=True)
        expected_model_buffer_hash = _shared_model_buffer_hash(reference_state)
        expected_optimizer_hash = _shared_optimizer_hash(reference_state)
        reference_pre_hash = state_hash(reference_state)
        reference_post_state = None

        branch_results: dict[str, dict[str, Any]] = {}
        for name, adapter in variants.items():
            observed_model_hash, observed_optimizer_hash = _assert_shared_start_state(
                adapter,
                reference_state,
                expected_model_buffer_hash=expected_model_buffer_hash,
                expected_optimizer_hash=expected_optimizer_hash,
            )
            routed_current = _route_batch_to_adapter_device(adapter, current)
            _ = _execute_adapter_update(adapter, routed_current)
            post_state = snapshot_state(adapter, cpu=True)
            if name == reference_variant:
                reference_post_state = post_state
            diagnostics = _diagnostics_from_adapter(adapter)
            safety = _metric_from_gate(adapter, current)
            heldout_callback = callback_map.get(name)
            heldout_diagnostics = (
                dict(heldout_callback(adapter, current, batch_index))
                if heldout_callback is not None
                else {}
            )
            # The held-out callback and future evaluation are read-only by
            # protocol, but restore the exact post-update state before every
            # evaluation so diagnostics cannot leak BN/RNG/runtime changes.
            _restore_evidence_state(adapter, post_state)
            branch_results[name] = {
                "adapter": adapter,
                "post_state": post_state,
                "update_norm": _state_update_norm(reference_state, post_state),
                "safety": safety,
                "diagnostics": diagnostics,
                "heldout_diagnostics": heldout_diagnostics,
                "pre_batch_model_buffer_hash": observed_model_hash,
                "pre_batch_optimizer_hash": observed_optimizer_hash,
            }

        if reference_post_state is None:
            raise RuntimeError("reference variant did not produce a post-update state")

        for name, result in branch_results.items():
            adapter = result["adapter"]
            post_state = result["post_state"]
            _restore_evidence_state(adapter, post_state)
            for horizon in horizons:
                if horizon > len(future_batches):
                    continue
                _restore_evidence_state(adapter, post_state)
                routed_future = [
                    _route_batch_to_adapter_device(
                        adapter, batch, move_labels=True
                    )
                    for batch in future_batches[:horizon]
                ]
                metrics = future_metrics(
                    adapter,
                    routed_future,
                    device=device,
                    num_classes=num_classes,
                )
                after_eval = snapshot_state(adapter, cpu=True)
                eval_untouched = _evidence_states_equal(after_eval, post_state)
                rng_untouched = _rng_states_equal(after_eval.rng_state, post_state.rng_state)
                rows.append(
                    {
                        **dict(metadata or {}),
                        "variant": name,
                        "condition": condition,
                        "batch_index": batch_index,
                        "horizon": horizon,
                        "future_macro_f1": float(metrics["macro_f1"]),
                        "future_true_label_nll": float(metrics["true_label_nll"]),
                        "future_samples": int(metrics["samples"]),
                        "pre_batch_state_hash": reference_pre_hash,
                        "pre_batch_model_buffer_hash": result[
                            "pre_batch_model_buffer_hash"
                        ],
                        "pre_batch_optimizer_hash": result[
                            "pre_batch_optimizer_hash"
                        ],
                        "shared_reference_variant": reference_variant,
                        "joint_causal_start_state": True,
                        "post_update_state_hash": state_hash(post_state),
                        "update_norm": result["update_norm"],
                        "future_eval_untouched": bool(eval_untouched),
                        "future_eval_rng_untouched": bool(rng_untouched),
                        **result["safety"],
                        **result["diagnostics"],
                        **result["heldout_diagnostics"],
                    }
                )
                _restore_evidence_state(adapter, post_state)

        # Only the predeclared reference path advances the online history.
        _restore_evidence_state(reference, reference_post_state)
        buffer.popleft()
        try:
            buffer.append(normalize_batch(next(stream)))
        except StopIteration:
            pass
        batch_index += 1

    if not rows:
        raise ValueError("no complete future horizon windows were evaluated")
    result = pd.DataFrame(rows)
    # This check catches accidental future refactors that restore a different
    # branch as the canonical history or omit a fork row.
    expected_hashes = result.groupby("batch_index")["pre_batch_model_buffer_hash"].nunique()
    if not expected_hashes.eq(1).all():
        raise RuntimeError("joint causal branches do not share model/buffer start hashes")
    expected_opt_hashes = result.groupby("batch_index")["pre_batch_optimizer_hash"].nunique()
    if not expected_opt_hashes.eq(1).all():
        raise RuntimeError("joint causal branches do not share optimizer start hashes")
    return result


def _rng_states_equal(left: Any, right: Any) -> bool:
    """Compare process RNG snapshots without depending on a private helper."""
    return _nested_equal(left, right)


def _make_condition_loader(base_loader: Iterable[Any], condition: tuple[str, str | None]):
    corruption, severity = condition
    if corruption == "clean":
        return base_loader
    return BatchTransformLoader(
        base_loader,
        CORRUPTION_REGISTRY[corruption],
        severity,
        sample_mask_fn=deterministic_mask_fn(0.5, 1),
        meta={"corruption_type": corruption, "severity": severity},
        transform_seed=20_001,
    )


def _instantiate_variant(trainer, source_model, variant: str):
    variant_class = get_representative_variant(variant)
    model = copy.deepcopy(source_model)
    # BaseTestTimeAlgorithm expects an optimizer *factory* and instantiates it
    # after selecting trainable parameters.  Passing an already-created
    # optimizer here raises ``TypeError: object is not callable`` before the
    # first experiment step and also bypasses the production parameter scope.
    optimizer_factory = build_optimizer(trainer.hparams)
    adapter = variant_class(
        trainer.dataset_configs,
        trainer.hparams,
        model,
        optimizer_factory,
    ).to(trainer.device)
    normalization_stats = getattr(
        trainer.src_train_dl.dataset, "normalization_stats", None
    )
    if hasattr(adapter, "load_source_normalization_reference"):
        if normalization_stats is None:
            raise RuntimeError("representative SSAW requires source normalization stats")
        adapter.load_source_normalization_reference(*normalization_stats)
    if getattr(adapter, "enable_confidence_gate", False):
        adapter.load_source_confidence_reference(trainer.source_confidence_metadata)
    if getattr(adapter, "enable_source_semantic_gate", False):
        adapter.load_source_semantic_reference(trainer.source_semantic_metadata)
    return adapter


def _load_cell_profile(
    args: argparse.Namespace, dataset: str, scenario: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either the paper profile or a preregistered unseen-flow profile."""

    runtime_profile_path = getattr(args, "runtime_profile_json", None)
    evaluation_role = str(
        getattr(args, "evaluation_role", "descriptive")
    ).strip().lower()
    if runtime_profile_path is None:
        if evaluation_role != "descriptive":
            raise ValueError(
                "confirmatory cells require --runtime-profile-json"
            )
        profiles = load_paper_flow_profiles(args.profile_json, datasets=[dataset])
        return profile_for_flow(profiles, dataset, scenario), {
            "evaluation_role": "descriptive",
            "evaluation_partition": "target_selected_evaluation",
            "parameter_selection_data_overlap": True,
            "target_labels_used_for_parameter_selection": True,
            "confirmatory": False,
            "profile_source": str(Path(args.profile_json).resolve()),
            "calibration_flow": None,
        }

    if evaluation_role != "confirmatory":
        raise ValueError(
            "--runtime-profile-json is reserved for confirmatory cells"
        )
    path = Path(runtime_profile_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "ssaw_unseen_flow_confirmation_profile_v1":
        raise ValueError("unseen-flow confirmation profile protocol mismatch")
    if str(payload.get("dataset", "")).upper() != dataset:
        raise ValueError("confirmation profile dataset mismatch")
    allowed = {parse_scenario(value) for value in payload.get("evaluation_flows", [])}
    if scenario not in allowed:
        raise ValueError("scenario is not preregistered for unseen-flow confirmation")
    calibration_flow = str(payload.get("calibration_flow", "")).strip().upper()
    if calibration_flow == f"{dataset}:{scenario}":
        raise ValueError("calibration flow cannot enter unseen-flow confirmation")
    if payload.get("frozen_before_confirmation") is not True:
        raise ValueError("confirmation profile was not frozen before evaluation")
    if payload.get("target_labels_used_for_evaluation_flow_selection") is not False:
        raise ValueError("confirmation flows must not be target-label selected")
    profile = payload.get("runtime_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("confirmation profile lacks runtime_profile")
    profile = dict(profile)
    required = {
        "batch_size",
        "learning_rate",
        "steps",
        "ssaw_auxiliary_weight",
        "spline_log_strength",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise ValueError(f"confirmation runtime profile lacks {missing}")
    if float(profile["ssaw_auxiliary_weight"]) <= 0.0:
        raise ValueError("confirmation SSAW weight must be positive")
    return profile, {
        "evaluation_role": "confirmatory",
        "evaluation_partition": "unseen_hhar_flow_and_direction_bank",
        "parameter_selection_data_overlap": False,
        "target_labels_used_for_parameter_selection": False,
        "confirmatory": True,
        "profile_source": str(path),
        "calibration_flow": calibration_flow,
        "preregistered_evaluation_flows": sorted(allowed),
    }


def run_cell(args: argparse.Namespace) -> pd.DataFrame:
    dataset = str(args.dataset).strip().upper()
    scenario = parse_scenario(args.scenario)
    source_id, target_id = scenario.split("->", 1)
    profile, evidence_role = _load_cell_profile(args, dataset, scenario)
    current_causal_digest = causal_evidence_code_sha256()
    current_production_digest = production_code_sha256()
    current_ablation_digest = ablation_code_sha256()
    profile = {
        **profile,
        "dusafe_logging_mode": "evidence",
        # Keep the request visible for audit, but DuSafe must turn it off
        # because evidence logging is deliberately eager and state-safe.
        "ssaw_candidate_cuda_graph": "auto",
        # Panel B proves that the matched branches received the exact same
        # candidate pool.  Production disables this byte-level hash because
        # it is not used by the online decision, but causal evidence requires
        # it and fails closed when any branch omits it.
        "record_ssaw_candidate_hash": True,
    }
    heldout_bank_tag = normalize_heldout_bank_tag(args.heldout_bank_tag)
    if float(profile["ssaw_auxiliary_weight"]) <= 0:
        raise ValueError("representative causal panel requires positive SSAW weight")
    condition = parse_conditions([args.condition])[0]
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="representative_causal_ablation",
        seed=int(args.stream_seed),
        source_seed=int(args.source_seed),
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    expected_source_hash = None
    if args.source_profile_root is not None:
        eeg_root = (
            args.eeg_source_profile_root
            if args.eeg_source_profile_root is not None
            else DEFAULT_EEG_PROFILE_ROOT
        )
        source_profiles = _load_flowwise_source_profiles(
            Path(args.source_profile_root), Path(eeg_root)
        )
        source_key = f"{dataset}:{scenario}"
        try:
            source_profile = source_profiles[source_key]
        except KeyError as exc:
            raise RuntimeError(
                f"flow-specific source profile is missing: {source_key}"
            ) from exc
        trainer.source_hparams.update(dict(source_profile["source_config"]))
        if args.source_reference_csv is None:
            raise ValueError(
                "--source-reference-csv is required with --source-profile-root"
            )
        source_references = _load_source_references(
            Path(args.source_reference_csv)
        )
        reference_key = (dataset, scenario, int(args.source_seed))
        try:
            expected_source_hash = source_references[reference_key]
        except KeyError as exc:
            raise RuntimeError(
                f"source checkpoint reference is missing: {reference_key}"
            ) from exc
    trainer.set_runtime_hparams(profile)
    hard_adapter = source_model = None
    variants: dict[str, Any] = {}
    try:
        hard_adapter, source_model = create_tta_model(
            trainer, source_id, target_id, run_seed=int(args.stream_seed)
        )
        if not bool(getattr(hard_adapter, "enable_ssaw", False)):
            raise RuntimeError("production adapter did not construct SSAW")
        variants["hard_ssaw"] = hard_adapter
        for name in REPRESENTATIVE_VARIANTS:
            if name == "hard_ssaw":
                continue
            variants[name] = _instantiate_variant(trainer, source_model, name)
        graph_metadata_by_variant = {
            name: _evidence_candidate_graph_metadata(adapter)
            for name, adapter in variants.items()
        }
        graph_metadata = graph_metadata_by_variant["hard_ssaw"]
        for name, observed in graph_metadata_by_variant.items():
            if observed != graph_metadata:
                raise RuntimeError(
                    "causal evidence candidate graph metadata diverged for "
                    f"variant {name!r}"
                )
        source_hash = _sha256_model(source_model)
        if (
            expected_source_hash is not None
            and source_hash != str(expected_source_hash)
        ):
            raise RuntimeError(
                "representative causal source checkpoint mismatch: "
                f"{source_hash} != {expected_source_hash}"
            )
        normalization_stats = getattr(
            trainer.src_train_dl.dataset, "normalization_stats", None
        )
        if normalization_stats is None:
            raise RuntimeError(
                "representative causal panel requires source normalization stats"
            )
        base_loader = trainer.trg_whole_dl
        partition = str(evidence_role["evaluation_partition"])
        # Persist the effective deployment configuration after dataset/base
        # defaults and flow overrides have been merged.  The compact override
        # alone omits identity fields such as confidence_keep_fraction, which
        # makes the fixed-source metadata context impossible to reconstruct.
        runtime_profile = dict(trainer.hparams)
        runtime_profile.update(graph_metadata)
        runtime_profile["dusafe_logging_mode"] = "evidence"
        metadata = {
            "status": "ok",
            "dataset": dataset,
            "scenario": scenario,
            "source_seed": int(args.source_seed),
            "stream_seed": int(args.stream_seed),
            "source_model_sha256": source_hash,
            "expected_source_model_sha256": expected_source_hash,
            "target_labels_used_for_online_updates": False,
            "target_labels_used_for_online_decision": False,
            "target_labels_used_for_parameter_selection": bool(
                evidence_role["target_labels_used_for_parameter_selection"]
            ),
            "parameter_selection_data_overlap": bool(
                evidence_role["parameter_selection_data_overlap"]
            ),
            "evaluation_partition": partition,
            "dusafe_logging_mode": "evidence",
            "logging_mode": "evidence",
            **graph_metadata,
            "confirmatory": bool(evidence_role["confirmatory"]),
            "target_selected_descriptive": not bool(evidence_role["confirmatory"]),
            "evaluation_role": str(evidence_role["evaluation_role"]),
            "target_labels_used_for_metrics": True,
            "profile": runtime_profile,
            "runtime_hparams": json.dumps(runtime_profile, sort_keys=True),
            "heldout_bank_tag": heldout_bank_tag,
            "production_code_sha256": current_production_digest,
            "ablation_code_sha256": current_ablation_digest,
            "causal_evidence_code_sha256": current_causal_digest,
        }
        heldout_callbacks = {
            name: _heldout_direction_callback(
                dataset=dataset,
                scenario=scenario,
                source_seed=int(args.source_seed),
                stream_seed=int(args.stream_seed),
                profile=dict(trainer.hparams),
                normalization_stats=normalization_stats,
                heldout_bank_tag=heldout_bank_tag,
            )
            for name in variants
        }
        result = run_joint_variant_horizon(
            variants,
            _make_condition_loader(base_loader, condition),
            reference_variant="confidence_only",
            condition=condition_label(condition),
            horizons=tuple(int(v) for v in args.horizons),
            device=args.device,
            num_classes=int(trainer.dataset_configs.num_classes),
            metadata=metadata,
            heldout_diagnostics_fns=heldout_callbacks,
        )
        # Metadata is copied into every row by the joint runner.  Verify the
        # persisted row-level contract before writing raw.csv, so a future
        # runner change cannot silently drop the evidence logging guard.
        for field, expected in {
            "dusafe_logging_mode": "evidence",
            "logging_mode": "evidence",
            "candidate_cuda_graph_requested_mode": graph_metadata[
                "candidate_cuda_graph_requested_mode"
            ],
            "candidate_cuda_graph_enabled": False,
            "candidate_cuda_graph_status": "disabled_evidence_logging",
            "candidate_cuda_graph_mode": "disabled",
            "target_labels_used_for_online_decision": False,
        }.items():
            if field not in result.columns:
                raise RuntimeError(f"causal raw rows missing metadata field: {field}")
            if field == "candidate_cuda_graph_enabled":
                values = result[field].map(bool)
            else:
                values = result[field].astype(str)
                expected = str(expected)
            if not values.eq(expected).all():
                raise RuntimeError(
                    f"causal raw rows violate metadata field {field}: "
                    f"expected {expected!r}"
                )
        # Panel B is a causal control: confidence admission and the physical
        # candidate pool must be identical across its four variants.  Fail
        # closed instead of reporting a comparison with different samples.
        panel_b = result[result["variant"].isin(PANEL_B_VARIANTS)]
        key_columns = ["condition", "batch_index", "horizon"]
        for _, group in panel_b.groupby(key_columns, sort=False):
            if group["variant"].nunique() != len(PANEL_B_VARIANTS):
                raise RuntimeError("Panel B variant grid is incomplete")
            if group["admission_mask_sha256"].nunique(dropna=False) != 1:
                raise RuntimeError("Panel B confidence admission masks diverged")
            if group["admitted_count"].nunique(dropna=False) != 1:
                raise RuntimeError("Panel B admission denominators diverged")
            if group["pre_batch_model_buffer_hash"].nunique(dropna=False) != 1:
                raise RuntimeError("Panel B model/buffer fork states diverged")
            if group["pre_batch_optimizer_hash"].nunique(dropna=False) != 1:
                raise RuntimeError("Panel B optimizer fork states diverged")
            candidate_hashes = group["candidate_pool_sha256"].dropna().astype(str)
            expected_candidate_variants = {
                "matched_raw_duplicate",
                "random_eligible_spline",
                "hard_ssaw",
            }
            observed_candidate_variants = set(
                group.loc[group["candidate_pool_sha256"].notna(), "variant"]
            )
            if observed_candidate_variants != expected_candidate_variants:
                raise RuntimeError("Panel B candidate-pool hashes are incomplete")
            if candidate_hashes.nunique() != 1:
                raise RuntimeError("Panel B candidate pools diverged")
            heldout_hashes = group["heldout_candidate_pool_sha256"].dropna().astype(str)
            if heldout_hashes.empty:
                raise RuntimeError("Panel B held-out candidate pools are missing")
            if heldout_hashes.nunique() != 1:
                raise RuntimeError("Panel B held-out candidate pools diverged")
            if group["heldout_candidate_count"].nunique(dropna=False) != 1:
                raise RuntimeError("Panel B held-out candidate budgets diverged")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        raw_path = output / "raw.csv"
        _atomic_csv(result, raw_path)
        raw_sha256 = _artifact_sha256(raw_path)
        _atomic_json(
            {
                "protocol": PROTOCOL,
                "production_code_sha256": current_production_digest,
                "ablation_code_sha256": current_ablation_digest,
                "ablation_code_files": [
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in ABLATION_CODE_FILES
                ],
                "causal_evidence_code_sha256": current_causal_digest,
                "causal_evidence_code_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CAUSAL_EVIDENCE_CODE_FILES],
                "status": "complete",
                "raw_sha256": raw_sha256,
                "dataset": dataset,
                "scenario": scenario,
                "source_seed": int(args.source_seed),
                "stream_seed": int(args.stream_seed),
                "condition": condition_label(condition),
                "horizons": [int(v) for v in args.horizons],
                "heldout_bank_tag": heldout_bank_tag,
                "variants": list(variants),
                "source_model_sha256": source_hash,
                "target_labels_used_for_online_updates": False,
                "target_labels_used_for_online_decision": False,
                "target_labels_used_for_parameter_selection": bool(
                    evidence_role["target_labels_used_for_parameter_selection"]
                ),
                "target_selected_descriptive": not bool(evidence_role["confirmatory"]),
                "evaluation_partition": partition,
                "parameter_selection_data_overlap": bool(
                    evidence_role["parameter_selection_data_overlap"]
                ),
                "confirmatory": bool(evidence_role["confirmatory"]),
                "dusafe_logging_mode": "evidence",
                "logging_mode": "evidence",
                **graph_metadata,
                "evaluation_role": str(evidence_role["evaluation_role"]),
                "profile_source": str(evidence_role["profile_source"]),
                "calibration_flow": evidence_role.get("calibration_flow"),
                "same_batch_start_state": True,
                "joint_reference_variant": "confidence_only",
                "reference_only_commit": True,
                "shared_optimizer_state_replayed": True,
                "shared_fork_audit_columns": [
                    "pre_batch_model_buffer_hash",
                    "pre_batch_optimizer_hash",
                ],
                "candidate_pool_and_denominator_shared": True,
                "heldout_direction_metrics": {
                    "label_free": True,
                    "direction_bank": "unseen_spline_sobol_direction",
                    "candidate_pool_and_budget_shared": True,
                    "metrics": [
                        "heldout_eligible_coverage",
                        "heldout_margin_ratio",
                        "heldout_flip_rate",
                        "heldout_worst_margin",
                        "heldout_consistency",
                        "heldout_stable_radius",
                        "heldout_stable_radius_normalized",
                        "heldout_cap_stable_ray_fraction",
                    ],
                },
                "update_norm": (
                    "L2 norm of the post-update minus pre-update model state; "
                    "parameters and persistent buffers, CPU snapshot"
                ),
                "protocol_passed": bool(
                    result["future_eval_untouched"].all()
                    and result["future_eval_rng_untouched"].all()
                    and result["joint_causal_start_state"].all()
                    and not bool(graph_metadata["candidate_cuda_graph_enabled"])
                    and graph_metadata["candidate_cuda_graph_status"]
                    == "disabled_evidence_logging"
                ),
            },
            output / "manifest.json",
        )
        return result
    finally:
        cleanup_trainer(trainer, *variants.values(), source_model, close_summary=True)


def _pairwise_panel(frame: pd.DataFrame, variants: Sequence[str], reference: str) -> pd.DataFrame:
    required = set(variants)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    # Keep aggregation backward-compatible with pre-eligibility raw cells;
    # current cells always populate this label-free selector.
    value_columns = (
        "future_macro_f1",
        "future_true_label_nll",
        "coverage",
        "eligible_coverage",
        "admitted_accuracy",
        "incorrect_admission_rate",
        "wrong_accept_recall",
        "correct_false_rejection_rate",
        "unsafe_update_rate",
        "update_norm",
        "ssaw_training_participation_rate",
        "ssaw_selected_normalized_margin_ratio_mean",
        "heldout_eligible_coverage",
        "heldout_margin_ratio",
        "heldout_flip_rate",
        "heldout_worst_margin",
        "heldout_consistency",
        "heldout_stable_radius",
        "heldout_stable_radius_normalized",
        "heldout_cap_stable_ray_fraction",
        "heldout_stable_radius_sum",
        "heldout_stable_radius_normalized_sum",
        "heldout_stable_radius_admitted_count",
        "heldout_cap_stable_ray_successes",
        "heldout_cap_stable_ray_total",
        "heldout_sample_count",
        "heldout_candidate_count",
    )
    for column in value_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    keys = ["dataset", "scenario", "source_seed", "stream_seed", "condition", "batch_index", "horizon"]
    pivot = frame[frame["variant"].isin(required)].pivot_table(
        index=keys,
        columns="variant",
        values=list(value_columns),
        aggfunc="first",
    )
    if pivot.empty or reference not in set(pivot.columns.get_level_values(1)):
        return pd.DataFrame()
    pivot.columns = [f"{metric}__{variant}" for metric, variant in pivot.columns]
    pivot = pivot.reset_index()
    for variant in variants:
        if variant == reference:
            continue
        for metric in (
            "future_macro_f1",
            "future_true_label_nll",
            "coverage",
            "eligible_coverage",
            "admitted_accuracy",
            "incorrect_admission_rate",
            "wrong_accept_recall",
            "correct_false_rejection_rate",
            "unsafe_update_rate",
            "update_norm",
            "heldout_eligible_coverage",
            "heldout_margin_ratio",
            "heldout_flip_rate",
            "heldout_worst_margin",
            "heldout_consistency",
            "heldout_stable_radius",
            "heldout_stable_radius_normalized",
            "heldout_cap_stable_ray_fraction",
        ):
            left = f"{metric}__{variant}"
            right = f"{metric}__{reference}"
            if left in pivot and right in pivot:
                sign = -1.0 if metric == "future_true_label_nll" else 1.0
                pivot[f"delta_{metric}_{variant}_vs_{reference}"] = sign * (pivot[left] - pivot[right])
    return pivot


def aggregate_panels(frame: pd.DataFrame, *, active_threshold: float = ACTIVE_THRESHOLD) -> dict[str, pd.DataFrame]:
    frame = frame.copy()
    for optional_column in (
        "eligible_coverage",
        "wrong_accept_recall",
        "correct_false_rejection_rate",
        "heldout_flip_rate",
        "heldout_worst_margin",
        "heldout_stable_radius",
        "heldout_stable_radius_normalized",
        "heldout_cap_stable_ray_fraction",
    ):
        if optional_column not in frame.columns:
            frame[optional_column] = np.nan
    panel_a = _pairwise_panel(frame, PANEL_A_VARIANTS, "accept_all_raw")
    panel_b = _pairwise_panel(frame, PANEL_B_VARIANTS, "confidence_only")
    hard = frame[frame["variant"].eq("hard_ssaw")].copy()
    confidence = frame[frame["variant"].eq("confidence_only")].copy()
    keys = ["dataset", "scenario", "source_seed", "stream_seed", "condition", "batch_index", "horizon"]
    joined = hard.merge(
        confidence[
            keys
            + [
                "future_macro_f1",
                "future_true_label_nll",
                "coverage",
                "eligible_coverage",
                "admitted_accuracy",
                "incorrect_admission_rate",
                "wrong_accept_recall",
                "correct_false_rejection_rate",
                "heldout_flip_rate",
                "heldout_worst_margin",
                "heldout_stable_radius",
                "heldout_stable_radius_normalized",
                "heldout_cap_stable_ray_fraction",
            ]
        ],
        on=keys,
        how="inner",
        suffixes=("_hard_ssaw", "_confidence_only"),
        validate="one_to_one",
    )
    if not joined.empty:
        joined["active_batch"] = pd.to_numeric(
            joined["eligible_coverage_hard_ssaw"], errors="coerce"
        ).ge(float(active_threshold))
        f1_delta = (
            joined["future_macro_f1_hard_ssaw"]
            - joined["future_macro_f1_confidence_only"]
        )
        nll_delta = (
            joined["future_true_label_nll_confidence_only"]
            - joined["future_true_label_nll_hard_ssaw"]
        )
        # Preserve the population effect and expose active/inactive conditional
        # effects in separate columns.  Overwriting the overall effect with an
        # active-only value would make the conditional panel easy to misread.
        joined["delta_future_macro_f1_overall"] = f1_delta
        joined["delta_future_macro_f1_active"] = f1_delta.where(
            joined["active_batch"]
        )
        joined["delta_future_macro_f1_inactive"] = f1_delta.where(
            ~joined["active_batch"]
        )
        joined["delta_future_true_label_nll_overall"] = nll_delta
        joined["delta_future_true_label_nll_active"] = nll_delta.where(
            joined["active_batch"]
        )
        joined["delta_future_true_label_nll_inactive"] = nll_delta.where(
            ~joined["active_batch"]
        )
        # Backward-compatible name now truthfully denotes the overall effect.
        joined["delta_future_macro_f1_hard_ssaw_vs_confidence_only"] = f1_delta
        joined["delta_future_true_label_nll_hard_ssaw_vs_confidence_only"] = nll_delta
        joined["beneficial_update"] = f1_delta.gt(0.0)
        joined["beneficial_update_active"] = joined["beneficial_update"].where(
            joined["active_batch"]
        )
        joined["beneficial_update_inactive"] = joined["beneficial_update"].where(
            ~joined["active_batch"]
        )
        hard_flip = "heldout_flip_rate_hard_ssaw"
        confidence_flip = "heldout_flip_rate_confidence_only"
        if hard_flip in joined and confidence_flip in joined:
            flip_reduction = joined[confidence_flip] - joined[hard_flip]
            joined["heldout_flip_reduction_overall"] = flip_reduction
            joined["heldout_flip_reduction_active"] = flip_reduction.where(
                joined["active_batch"]
            )
            joined["heldout_flip_reduction_inactive"] = flip_reduction.where(
                ~joined["active_batch"]
            )
    panel_c = joined
    summary_rows = []
    for panel_name, panel in (("A", panel_a), ("B", panel_b), ("C", panel_c)):
        if panel.empty:
            continue
        numeric = panel.select_dtypes(include=[np.number])
        for column in numeric.columns:
            if column in {"source_seed", "stream_seed", "batch_index", "horizon"}:
                continue
            summary_rows.append({
                "panel": panel_name,
                "metric": column,
                "mean": float(pd.to_numeric(panel[column], errors="coerce").mean()),
                "std": float(pd.to_numeric(panel[column], errors="coerce").std(ddof=0)),
                "rows": int(panel[column].notna().sum()),
            })
        if panel_name == "C":
            for column, metric_name in (
                ("active_batch", "active_batch_coverage"),
                ("beneficial_update", "beneficial_update_rate_overall"),
                ("beneficial_update_active", "beneficial_update_rate_active"),
                ("beneficial_update_inactive", "beneficial_update_rate_inactive"),
            ):
                if column not in panel:
                    continue
                values = pd.to_numeric(panel[column], errors="coerce")
                summary_rows.append(
                    {
                        "panel": panel_name,
                        "metric": metric_name,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "rows": int(values.notna().sum()),
                    }
                )
    return {"panel_a": panel_a, "panel_b": panel_b, "panel_c": panel_c, "summary": pd.DataFrame(summary_rows)}


def _cell_is_complete(output: str | Path) -> bool:
    """Return True only for a current, protocol-passed cell artifact."""

    output = Path(output)
    raw_path = output / "raw.csv"
    manifest_path = output / "manifest.json"
    if not raw_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        manifest.get("protocol") == PROTOCOL
        and manifest.get("status") == "complete"
        and manifest.get("protocol_passed") is True
        and manifest.get("production_code_sha256") == production_code_sha256()
        and manifest.get("ablation_code_sha256") == ablation_code_sha256()
        and manifest.get("causal_evidence_code_sha256") == causal_evidence_code_sha256()
    )


def _archive_stale_cell_artifacts(output: str | Path) -> Path | None:
    """Move stale/failed cell artifacts aside before an isolated retry.

    Logs and arbitrary files are left untouched.  Only the root ``raw.csv``
    and ``manifest.json`` that could otherwise be mistaken for a completed
    cell are moved below ``failed_attempts/``.  The aggregate scanner ignores
    that archive, preserving failure evidence without contaminating a retry.
    """

    output = Path(output)
    candidates = [
        path for path in (output / "raw.csv", output / "manifest.json") if path.is_file()
    ]
    if not candidates:
        return None
    archive = output / FAILED_ARCHIVE_DIRNAME / f"attempt_{time.time_ns()}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.move(str(path), str(archive / path.name))
    _atomic_json(
        {
            "protocol": PROTOCOL,
            "status": "archived_stale_cell",
            "source_cell": str(output),
            "reason": "resume_requires_current_complete_protocol_passed_manifest",
        },
        archive / "resume_action.json",
    )
    return archive


def aggregate_directory(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    current_production_digest = production_code_sha256()
    current_ablation_digest = ablation_code_sha256()
    current_causal_digest = causal_evidence_code_sha256()
    frames = []
    manifests = []
    for path in input_dir.rglob("raw.csv"):
        if path.parent == input_dir:
            # Ignore the aggregate generated by a previous invocation.
            continue
        if FAILED_ARCHIVE_DIRNAME in path.parts:
            # Failed attempts are retained for audit but never enter the
            # completed panel aggregate.
            continue
        manifest = path.with_name("manifest.json")
        if not manifest.is_file():
            continue
        try:
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not (
            manifest_payload.get("protocol") == PROTOCOL
            and manifest_payload.get("status") == "complete"
            and manifest_payload.get("protocol_passed") is True
            and manifest_payload.get("production_code_sha256") == current_production_digest
            and manifest_payload.get("ablation_code_sha256") == current_ablation_digest
            and manifest_payload.get("causal_evidence_code_sha256") == current_causal_digest
        ):
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
        manifests.append(manifest_payload)
    if not frames:
        raise ValueError(f"no completed causal cells found under {input_dir}")
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "dataset", "scenario", "source_seed", "variant", "condition",
        "batch_index", "horizon", "future_macro_f1",
        "production_code_sha256", "ablation_code_sha256",
        "causal_evidence_code_sha256",
        "dusafe_logging_mode", "logging_mode",
        "candidate_cuda_graph_requested_mode",
        "candidate_cuda_graph_enabled",
        "candidate_cuda_graph_status", "candidate_cuda_graph_mode",
        "target_labels_used_for_online_decision",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"causal raw rows missing columns: {sorted(missing)}")
    if set(frame["production_code_sha256"].dropna().astype(str)) != {current_production_digest}:
        raise ValueError("causal raw rows disagree with current production code digest")
    if set(frame["ablation_code_sha256"].dropna().astype(str)) != {current_ablation_digest}:
        raise ValueError("causal raw rows disagree with current ablation code digest")
    if set(frame["causal_evidence_code_sha256"].dropna().astype(str)) != {current_causal_digest}:
        raise ValueError("causal raw rows disagree with current causal evidence code digest")
    if set(frame["dusafe_logging_mode"].dropna().astype(str)) != {"evidence"}:
        raise ValueError("causal raw rows are not evidence logging")
    if set(frame["logging_mode"].dropna().astype(str)) != {"evidence"}:
        raise ValueError("causal raw rows logging mode mismatch")
    observed_graph_requests = sorted(
        set(frame["candidate_cuda_graph_requested_mode"].dropna().astype(str))
    )
    if set(observed_graph_requests) - {"off", "auto", "force"}:
        raise ValueError("causal raw rows contain an invalid graph request")
    if len(observed_graph_requests) != 1:
        raise ValueError("causal raw rows disagree on graph request")
    graph_enabled = frame["candidate_cuda_graph_enabled"].map(
        lambda value: bool(value)
        if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes", "on"}
    )
    if bool(graph_enabled.any()):
        raise ValueError("causal raw rows contain enabled candidate graphs")
    if set(frame["candidate_cuda_graph_status"].dropna().astype(str)) != {
        "disabled_evidence_logging"
    }:
        raise ValueError("causal raw rows graph status is not evidence-disabled")
    if set(frame["candidate_cuda_graph_mode"].dropna().astype(str)) != {"disabled"}:
        raise ValueError("causal raw rows graph mode is not disabled")
    online_flags = frame["target_labels_used_for_online_decision"].map(
        lambda value: bool(value)
        if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes", "on"}
    )
    if bool(online_flags.any()):
        raise ValueError("causal raw rows use target labels online")
    observed_source_seeds = sorted({int(value) for value in frame["source_seed"].dropna()})
    if observed_source_seeds != list(SOURCE_SEEDS):
        raise ValueError(
            "causal source seeds mismatch: "
            f"{observed_source_seeds} != {list(SOURCE_SEEDS)}"
        )
    observed_stream_seeds = sorted({int(value) for value in frame["stream_seed"].dropna()})
    if observed_stream_seeds != [STREAM_SEED]:
        raise ValueError("causal stream seed mismatch")
    keys = ["dataset", "scenario", "source_seed", "stream_seed", "condition", "batch_index", "horizon", "variant"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate causal raw row key")
    bank_tags = {
        normalize_heldout_bank_tag(manifest["heldout_bank_tag"])
        for manifest in manifests
    }
    if len(bank_tags) != 1:
        raise ValueError("causal aggregate mixes held-out direction banks")
    heldout_bank_tag = next(iter(bank_tags))
    if "heldout_bank_tag" not in frame.columns:
        raise ValueError("causal raw rows lack heldout_bank_tag")
    if set(frame["heldout_bank_tag"].dropna().astype(str)) != {heldout_bank_tag}:
        raise ValueError("causal raw rows disagree with held-out bank manifest")
    evaluation_roles = {str(value.get("evaluation_role", "descriptive")) for value in manifests}
    if len(evaluation_roles) != 1:
        raise ValueError("causal aggregate mixes descriptive and confirmatory cells")
    evaluation_role = next(iter(evaluation_roles))
    confirmatory = evaluation_role == "confirmatory"
    if any(bool(value.get("confirmatory", False)) != confirmatory for value in manifests):
        raise ValueError("causal aggregate confirmatory flags disagree")
    selection_overlap = any(
        bool(value.get("parameter_selection_data_overlap", True))
        for value in manifests
    )
    target_label_selected = any(
        bool(value.get("target_labels_used_for_parameter_selection", True))
        for value in manifests
    )
    if confirmatory and (selection_overlap or target_label_selected):
        raise ValueError("confirmatory aggregate contains parameter-selection overlap")
    if confirmatory:
        raise ValueError("causal evidence aggregate must be descriptive")
    if not target_label_selected or not selection_overlap:
        raise ValueError(
            "causal descriptive aggregate lacks target-selected parameter provenance"
        )
    panels = aggregate_panels(frame)
    observed_horizons = sorted(
        {int(value) for value in frame["horizon"].dropna().tolist()}
    )
    if not observed_horizons:
        raise ValueError("causal raw rows contain no observed horizons")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, output_dir / "raw.csv")
    raw_sha256 = _artifact_sha256(output_dir / "raw.csv")
    for name, panel in panels.items():
        _atomic_csv(panel, output_dir / f"{name}.csv")
    manifest = {
        "protocol": PROTOCOL,
        "production_code_sha256": current_production_digest,
        "ablation_code_sha256": current_ablation_digest,
        "ablation_code_files": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in ABLATION_CODE_FILES
        ],
        "causal_evidence_code_sha256": current_causal_digest,
        "causal_evidence_code_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CAUSAL_EVIDENCE_CODE_FILES],
        "status": "complete",
        "input_cells": len(manifests),
        "raw_rows": len(frame),
        "raw_sha256": raw_sha256,
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "heldout_bank_tag": heldout_bank_tag,
        "panels": {name: len(panel) for name, panel in panels.items()},
        "active_batch_threshold": ACTIVE_THRESHOLD,
        "active_batch_selector": (
            "eligible_coverage = active_count / confidence_admitted_count; "
            "selector is label-free and overall coverage is retained"
        ),
        "evidence_roles": {
            "A": {
                "variants": list(PANEL_A_VARIANTS),
                "future_horizon": (
                    observed_horizons[0]
                    if len(observed_horizons) == 1
                    else None
                ),
                "future_horizons": observed_horizons,
                "metrics": [
                    "future_macro_f1",
                    "coverage",
                    "admitted_accuracy",
                    "incorrect_admission_rate",
                ],
            },
            "B": {
                "variants": list(PANEL_B_VARIANTS),
                "metrics": [
                    "future_macro_f1",
                    "heldout_eligible_coverage",
                    "heldout_margin_ratio",
                    "heldout_flip_rate",
                    "heldout_worst_margin",
                    "heldout_consistency",
                    "heldout_stable_radius",
                    "heldout_stable_radius_normalized",
                    "heldout_cap_stable_ray_fraction",
                    "update_norm",
                ],
                "candidate_pool_and_budget_shared": True,
            },
            "C": {
                "selector": "eligible_coverage >= 0.25",
                "overall_coverage_also_reported": True,
                "conditional_effect_only": True,
            },
            "two_by_two": "audit_only",
        },
        "same_batch_start_state_required": True,
        "candidate_pool_and_denominator_shared_required": True,
        "target_labels_used_for_online_updates": False,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": target_label_selected,
        "parameter_selection_data_overlap": selection_overlap,
        "evaluation_role": evaluation_role,
        "confirmatory": confirmatory,
        "descriptive_target_selected_evaluation": not confirmatory,
        "target_selected_descriptive": not confirmatory,
        "logging_mode": "evidence",
        "dusafe_logging_mode": "evidence",
        "candidate_cuda_graph_requested_mode": observed_graph_requests[0],
        "candidate_cuda_graph_enabled": False,
        "candidate_cuda_graph_status": "disabled_evidence_logging",
        "candidate_cuda_graph_mode": "disabled",
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"))
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--source-profile-root",
        type=Path,
        default=None,
        help="Optional flow-specific source-training profile root.",
    )
    parser.add_argument(
        "--eeg-source-profile-root",
        type=Path,
        default=None,
        help="Optional EEG source-profile root paired with --source-profile-root.",
    )
    parser.add_argument(
        "--source-reference-csv",
        type=Path,
        default=None,
        help="Expected flow/source-seed checkpoint hashes.",
    )
    parser.add_argument(
        "--runtime-profile-json",
        type=Path,
        default=None,
        help=(
            "preregistered fixed runtime profile for an unseen-flow "
            "confirmatory cell"
        ),
    )
    parser.add_argument(
        "--evaluation-role",
        choices=("descriptive", "confirmatory"),
        default="descriptive",
    )
    parser.add_argument("--selected-flows-json", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--source-seeds", default=",".join(str(value) for value in SOURCE_SEEDS))
    parser.add_argument("--conditions", default="clean")
    parser.add_argument("--horizons", default=str(HORIZON))
    parser.add_argument(
        "--heldout-bank-tag",
        default=DEFAULT_HELDOUT_BANK_TAG,
        help="stable identifier used to derive the unseen Sobol direction bank",
    )
    parser.add_argument("--execute", action="store_true", help="execute the planned isolated cells")
    parser.add_argument("--gpu-lock-path", type=Path, default=ROOT / "results" / ".current_experiment_gpu.lock")
    parser.add_argument("--aggregate", action="store_true", help="aggregate completed cell outputs")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--source-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED, help=argparse.SUPPRESS)
    parser.add_argument("--condition", default="clean", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.aggregate:
        aggregate_directory(args.input_dir or args.output_dir, args.output_dir)
        return 0
    if args.cell:
        if not args.dataset or not args.scenario or args.source_seed is None:
            raise ValueError("--cell requires dataset, scenario, and source-seed")
        args.horizons = tuple(int(value.strip()) for value in str(args.horizons).split(",") if value.strip())
        run_cell(args)
        return 0
    datasets = tuple(value.strip().upper() for value in str(args.datasets).split(",") if value.strip())
    seeds = tuple(int(value.strip()) for value in str(args.source_seeds).split(",") if value.strip())
    conditions = parse_conditions(args.conditions)
    horizons = tuple(int(value.strip()) for value in str(args.horizons).split(",") if value.strip())
    selected = load_selected_flows(args.selected_flows_json, datasets=datasets)
    plan = build_plan(
        datasets=datasets,
        source_seeds=seeds,
        conditions=conditions,
        selected_flows=selected,
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
        backbone=args.backbone,
        pretrain_cache_dir=args.pretrain_cache_dir,
        profile_json=args.profile_json,
        source_profile_root=args.source_profile_root,
        eeg_source_profile_root=args.eeg_source_profile_root,
        source_reference_csv=args.source_reference_csv,
        horizons=horizons,
        heldout_bank_tag=args.heldout_bank_tag,
    )
    _atomic_json(plan, Path(args.output_dir) / "plan.json")
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    for cell in plan["cells"]:
        output = Path(cell["output_dir"])
        if _cell_is_complete(output):
            continue
        _archive_stale_cell_artifacts(output)
        lock_context = (
            wait_for_gpu_experiment_lock(args.gpu_lock_path)
            if str(args.device).lower().startswith("cuda")
            else contextlib.nullcontext()
        )
        with lock_context:
            completed = subprocess.run(cell["command"], cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"representative causal cell failed: {cell['key']}")
    aggregate_directory(args.output_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_THRESHOLD",
    "PANEL_A_VARIANTS",
    "PANEL_B_VARIANTS",
    "PROTOCOL",
    "ABLATION_CODE_FILES",
    "CAUSAL_EVIDENCE_CODE_FILES",
    "ablation_code_sha256",
    "causal_evidence_code_sha256",
    "production_code_sha256",
    "DEFAULT_HELDOUT_BANK_TAG",
    "aggregate_directory",
    "aggregate_panels",
    "build_plan",
    "condition_label",
    "load_selected_flows",
    "heldout_bank_seed",
    "normalize_heldout_bank_tag",
    "summarize_discrete_stable_radius",
    "parse_conditions",
    "parse_scenario",
    "run_joint_variant_horizon",
    "run_variant_horizon",
    "_archive_stale_cell_artifacts",
    "_cell_is_complete",
    "_instantiate_variant",
    "_execute_adapter_update",
]
