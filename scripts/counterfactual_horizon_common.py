"""Strict multi-horizon Full/no-SSAW counterfactual audit primitives.

The functions in this module are independent of the experiment runners.  They
operate on an already prepared TTA adapter and a finite stream of batches.  A
batch is always represented as ``(data, labels, indices)`` (or an equivalent
mapping), but labels are stripped before an online update is called.  Labels
are consumed only by :func:`future_metrics` after all branch states have been
created.

At every batch ``t`` the Full adapter is the canonical online history.  Three
branches start from an exact snapshot of that history:

* ``no_update`` keeps the snapshot;
* ``no_ssaw`` performs one configured update with ``enable_ssaw=False``; and
* ``full`` performs one configured production update.

Each branch is evaluated on the same future batches without further updates.
The Full post-update state is restored as the canonical history before moving
to ``t+1``.  Model parameters, buffers (including BatchNorm buffers), and
optimizer state are copied/restored exactly; branch state hashes and explicit
equivalence booleans are emitted for auditability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


DEFAULT_HORIZONS = (1, 3, 5)
BRANCHES = ("no_update", "no_ssaw", "full")
IMPACT_LABELS = ("beneficial", "harmful", "tied")


@dataclass
class AdapterState:
    """A deep snapshot of all state that can affect a counterfactual.

    ``model.state_dict()`` is not sufficient for a TTA adapter.  DuSafe keeps
    a Sobol sampler and cached physical views outside the model, and a
    ``Module`` can also own non-persistent buffers.  The original audit only
    copied parameters, model buffers, and the optimizer, which allowed those
    objects to leak from one branch into the next.  The extra fields are kept
    optional for compatibility with callers that construct ``AdapterState``
    directly.
    """

    model_state: dict[str, Any]
    optimizer_state: Any
    adapter_module_state: Any = None
    adapter_buffer_state: Any = None
    runtime_state: Any = None
    training_state: Any = None
    rng_state: Any = None


@dataclass
class BatchView:
    """A normalized stream batch with labels retained only for evaluation."""

    data: Any
    labels: torch.Tensor
    indices: Any


def _primary(data: Any) -> Any:
    if isinstance(data, Mapping):
        data = data.get("data", data)
    if isinstance(data, (tuple, list)):
        return data[0]
    return data


def normalize_batch(batch: Any) -> BatchView:
    """Normalize tuple/dict batches without changing the label tensor."""

    if isinstance(batch, Mapping):
        data = batch.get("data")
        labels = batch.get("labels")
        indices = batch.get("indices", batch.get("idx"))
    else:
        values = tuple(batch)
        if len(values) < 2:
            raise ValueError("A counterfactual batch needs data and labels")
        data, labels = values[:2]
        indices = values[2] if len(values) > 2 else None
    if labels is None:
        raise ValueError("True labels are required for offline horizon metrics")
    labels = torch.as_tensor(labels).view(-1).long()
    return BatchView(data=data, labels=labels, indices=indices)


def _module(adapter: Any) -> torch.nn.Module:
    model = getattr(adapter, "model", adapter)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("adapter must expose a torch.nn.Module as .model")
    return model


def _optimizer(adapter: Any):
    return getattr(adapter, "optimizer", None)


def _clone_value(value: Any, *, cpu: bool = False) -> Any:
    if torch.is_tensor(value):
        cloned = value.detach().clone()
        return cloned.cpu() if cpu else cloned
    if isinstance(value, Mapping):
        return type(value)(
            (key, _clone_value(item, cpu=cpu)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_clone_value(item, cpu=cpu) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item, cpu=cpu) for item in value)
    return copy.deepcopy(value)


_RUNTIME_STATE_EXCLUDED = {
    # These are restored independently or are immutable configuration.  In
    # particular ``enable_ssaw`` must remain False on the no-SSAW branch when
    # a Full snapshot is restored into it.
    "model",
    "optimizer",
    "_modules",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "enable_ssaw",
    "seen_label_keys",
    "_last_gate_log",
    "_last_batch_log",
}


def _runtime_state(adapter: Any, *, cpu: bool = False) -> dict[str, Any]:
    """Capture mutable adapter-side state without copying diagnostics.

    The production DuSafe adapter stores the stateful SSAW Sobol sampler as a
    normal Python attribute rather than as a ``torch.nn.Module``.  Capturing
    that object's ``__dict__`` is therefore necessary for exact branch
    replay.  Unknown scalar/list attributes are captured too, while the
    existing tests' observation-only ``seen_label_keys`` and gate logs are
    deliberately excluded because they do not affect an update.
    """

    values: dict[str, Any] = {}
    for name, value in getattr(adapter, "__dict__", {}).items():
        if name in _RUNTIME_STATE_EXCLUDED or name.startswith("_last_"):
            continue
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
            continue
        # Static configuration is harmless to copy but can contain objects
        # that are not deepcopy-able (for example a logger or a data loader).
        # Keep only state that is plausibly mutable and update-relevant.
        if name in {"configs", "hparams", "config", "dataset_configs"}:
            continue
        try:
            values[name] = _clone_value(value, cpu=cpu)
        except (TypeError, RuntimeError, ValueError):
            continue
    return values


def _adapter_module_state(adapter: Any, *, cpu: bool = False) -> Any:
    if not isinstance(adapter, torch.nn.Module):
        return None
    try:
        return _clone_value(adapter.state_dict(), cpu=cpu)
    except (TypeError, RuntimeError, ValueError):
        return None


def _adapter_buffer_state(adapter: Any, *, cpu: bool = False) -> Any:
    buffers = getattr(adapter, "_buffers", None)
    if not isinstance(buffers, Mapping):
        return None
    return {
        name: _clone_value(value, cpu=cpu)
        for name, value in buffers.items()
        if value is not None
    }


def _training_state(adapter: Any) -> dict[str, bool]:
    model = _module(adapter)
    return {
        name: bool(module.training)
        for name, module in model.named_modules()
    }


def _restore_training_state(adapter: Any, state: Mapping[str, bool] | None) -> None:
    if not state:
        return
    model = _module(adapter)
    modules = dict(model.named_modules())
    for name, training in state.items():
        module = modules.get(name)
        if module is not None:
            module.training = bool(training)


def snapshot_state(adapter: Any, *, cpu: bool = False) -> AdapterState:
    """Clone model, adapter, optimizer, mode, and RNG state exactly.

    ``cpu=True`` is used by the low-memory queue.  It releases each
    post-branch snapshot from a GPU immediately while retaining the complete
    optimizer/BN/RNG state needed for an exact restore.
    """

    model = _module(adapter)
    optimizer = _optimizer(adapter)
    model_state = {
        name: (value.detach().clone().cpu() if cpu else value.detach().clone())
        for name, value in model.state_dict().items()
    }
    optimizer_state = (
        None
        if optimizer is None
        else _clone_value(optimizer.state_dict(), cpu=cpu)
    )
    return AdapterState(
        model_state=model_state,
        optimizer_state=optimizer_state,
        adapter_module_state=_adapter_module_state(adapter, cpu=cpu),
        adapter_buffer_state=_adapter_buffer_state(adapter, cpu=cpu),
        runtime_state=_runtime_state(adapter, cpu=cpu),
        training_state=_training_state(adapter),
        rng_state=_clone_value(
            _capture_rng_state(
                include_cuda=next(model.parameters(), torch.empty(0)).device.type
                == "cuda"
            ),
            cpu=cpu,
        ),
    )


def restore_state(adapter: Any, state: AdapterState) -> None:
    """Restore every captured state component and clear stale gradients."""

    model = _module(adapter)
    model.load_state_dict(state.model_state, strict=True)
    if state.adapter_module_state is not None and isinstance(
        adapter, torch.nn.Module
    ):
        # ``strict=False`` is intentional: the model state above is the
        # canonical part and some adapters have non-persistent buffers or
        # optional source modules whose keys vary across implementations.
        adapter.load_state_dict(
            _clone_value(state.adapter_module_state), strict=False
        )
    if state.adapter_buffer_state is not None:
        buffers = getattr(adapter, "_buffers", None)
        if isinstance(buffers, Mapping):
            for name, value in state.adapter_buffer_state.items():
                if name in buffers:
                    cloned = _clone_value(value)
                    target = buffers[name]
                    if torch.is_tensor(cloned) and torch.is_tensor(target):
                        cloned = cloned.to(device=target.device, dtype=target.dtype)
                    buffers[name] = cloned
    optimizer = _optimizer(adapter)
    if optimizer is not None:
        if state.optimizer_state is None:
            raise ValueError("snapshot has no optimizer state")
        optimizer.load_state_dict(_clone_value(state.optimizer_state))
        optimizer.zero_grad(set_to_none=True)
    # Gradients are transient execution state and are not part of a
    # ``state_dict``.  Clear parameters outside the optimizer's scope too so
    # a failed/partial update cannot leak stale gradients across branches.
    for parameter in model.parameters():
        parameter.grad = None
    if state.runtime_state is not None:
        for name, value in state.runtime_state.items():
            if name in _RUNTIME_STATE_EXCLUDED:
                continue
            current = getattr(adapter, name, None)
            if isinstance(current, (torch.nn.Module, torch.optim.Optimizer)):
                continue
            cloned = _clone_value(value)
            if torch.is_tensor(cloned) and torch.is_tensor(current):
                cloned = cloned.to(device=current.device, dtype=current.dtype)
            setattr(adapter, name, cloned)
    _restore_training_state(adapter, state.training_state)
    if state.rng_state is not None:
        _restore_rng_state(state.rng_state)


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Generator) or isinstance(right, torch.Generator):
        return (
            isinstance(left, torch.Generator)
            and isinstance(right, torch.Generator)
            and torch.equal(left.get_state(), right.get_state())
        )
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device == right.device
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return (
            set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    if hasattr(left, "__dict__") or hasattr(right, "__dict__"):
        return (
            type(left) is type(right)
            and hasattr(left, "__dict__")
            and _nested_equal(vars(left), vars(right))
        )
    result = left == right
    return bool(result) if not isinstance(result, np.ndarray) else bool(result.all())


def states_equal(left: AdapterState, right: AdapterState) -> bool:
    """Return equality of adapter state (RNG is checked separately).

    RNG is process-global rather than adapter-owned.  Keeping it out of this
    historical helper preserves callers that restore only model/optimizer
    state; the horizon audit records explicit ``*_rng_*`` checks alongside
    every state check.
    """

    return all(
        (
            _nested_equal(left.model_state, right.model_state),
            _nested_equal(left.optimizer_state, right.optimizer_state),
            _nested_equal(left.adapter_module_state, right.adapter_module_state),
            _nested_equal(left.adapter_buffer_state, right.adapter_buffer_state),
            _nested_equal(left.runtime_state, right.runtime_state),
            _nested_equal(left.training_state, right.training_state),
        )
    )


def batchnorm_state(adapter: Any) -> dict[str, torch.Tensor]:
    """Extract BatchNorm running statistics and counters for explicit checks."""

    model = _module(adapter)
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            names.update(
                f"{prefix}{suffix}"
                for suffix in ("running_mean", "running_var", "num_batches_tracked")
                if suffix in module._buffers and module._buffers[suffix] is not None
            )
    state = model.state_dict()
    return {
        name: state[name].detach().clone()
        for name in sorted(names)
        if name in state
    }


def batchnorm_states_equal(left: Any, right: Any) -> bool:
    left_state = left if isinstance(left, Mapping) else batchnorm_state(left)
    right_state = right if isinstance(right, Mapping) else batchnorm_state(right)
    return _nested_equal(left_state, right_state)


def state_hash(state_or_adapter: AdapterState | Any) -> str:
    """Hash adapter state for manifest/row provenance (RNG is audited apart)."""

    state = (
        state_or_adapter
        if isinstance(state_or_adapter, AdapterState)
        else snapshot_state(state_or_adapter)
    )
    digest = hashlib.sha256()

    def add(value: Any) -> None:
        if isinstance(value, torch.Generator):
            digest.update(b"<torch.Generator>")
            add(value.get_state())
        elif torch.is_tensor(value):
            cpu = value.detach().cpu().contiguous()
            digest.update(str(cpu.dtype).encode("utf-8"))
            digest.update(str(tuple(cpu.shape)).encode("utf-8"))
            digest.update(cpu.numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8"))
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif hasattr(value, "__dict__"):
            digest.update(f"<{type(value).__module__}.{type(value).__qualname__}>".encode("utf-8"))
            add(vars(value))
        else:
            digest.update(repr(value).encode("utf-8"))

    add(state.model_state)
    add(state.optimizer_state)
    add(state.adapter_module_state)
    add(state.adapter_buffer_state)
    add(state.runtime_state)
    add(state.training_state)
    return digest.hexdigest()


def adapter_state_hash(adapter: Any) -> str:
    return state_hash(adapter)


def clone_branch_adapters(full_adapter: Any) -> dict[str, Any]:
    """Create isolated no-update/no-SSAW adapters from one Full adapter.

    The caller owns the returned objects and should release them after the
    audit.  ``copy.deepcopy`` is intentional: model parameters, buffers,
    optimizer slots, source references, and SSAW state must not alias.
    """

    no_update = copy.deepcopy(full_adapter)
    no_ssaw = copy.deepcopy(full_adapter)
    no_ssaw.enable_ssaw = False
    return {
        "no_update": no_update,
        "no_ssaw": no_ssaw,
        "full": full_adapter,
    }


def _capture_rng_state(*, include_cuda: bool | None = None) -> dict[str, Any]:
    state = {"torch": torch.random.get_rng_state()}
    if include_cuda is None:
        include_cuda = bool(torch.cuda.is_available())
    if include_cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    state["python"] = random.getstate()
    state["numpy"] = np.random.get_state()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _device_for(adapter: Any, fallback: torch.device | str | None = None) -> torch.device:
    if fallback is not None:
        return torch.device(fallback)
    try:
        return next(_module(adapter).parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_primary(data: Any, device: torch.device) -> torch.Tensor:
    primary = _primary(data)
    if not torch.is_tensor(primary):
        raise TypeError("batch data must contain a tensor")
    return primary.float().to(device)


def online_update(
    adapter: Any,
    batch: BatchView,
    branch: str,
    *,
    device: torch.device | str | None = None,
) -> Any:
    """Apply exactly one configured update with labels excluded from input."""

    if branch == "no_update":
        return {"committed": False, "online_update": False}
    if branch not in {"no_ssaw", "full"}:
        raise ValueError(f"Unknown counterfactual branch: {branch}")
    model = _module(adapter)
    optimizer = _optimizer(adapter)
    if optimizer is None:
        raise RuntimeError("counterfactual update requires an optimizer")
    data = _move_primary(batch.data, _device_for(adapter, device))
    # DuSafe extracts only ``data``.  No true labels are placed in this dict.
    model_input = {"data": data}
    indices = batch.indices
    return adapter.forward_and_adapt(model_input, model, optimizer, indices)


def model_logits(adapter: Any, data: Any, *, device=None) -> torch.Tensor:
    """Read-only raw prediction path used for future horizon metrics.

    ``inference_mode`` disables autograd but does not disable BatchNorm
    running-stat updates or dropout.  Future-horizon evaluation is an offline
    read and must not change either buffers or the random stream, so preserve
    every submodule's training flag and evaluate with ``model.eval()``.
    """

    model = _module(adapter)
    inputs = _move_primary(data, _device_for(adapter, device))
    training = {
        name: bool(module.training) for name, module in model.named_modules()
    }
    rng_state = _capture_rng_state(
        include_cuda=next(model.parameters(), torch.empty(0)).device.type
        == "cuda"
    )
    try:
        model.eval()
        with torch.inference_mode():
            features = model.feature_extractor(inputs)
            if isinstance(features, (tuple, list)):
                features = features[0]
            return model.classifier(features)
    finally:
        for name, module in model.named_modules():
            if name in training:
                module.training = training[name]
        _restore_rng_state(rng_state)


def future_metrics(
    adapter: Any,
    batches: Sequence[BatchView],
    *,
    device: torch.device | str | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Compute macro-F1 and true-label NLL on a fixed future window."""

    if not batches:
        raise ValueError("future horizon must contain at least one batch")
    logits = []
    labels = []
    with torch.inference_mode():
        for batch in batches:
            values = model_logits(adapter, batch.data, device=device)
            target = batch.labels.to(values.device)
            logits.append(values.detach().cpu())
            labels.append(target.detach().cpu())
    values = torch.cat(logits, dim=0)
    target = torch.cat(labels, dim=0).long()
    predictions = values.argmax(dim=1)
    class_labels = (
        list(range(int(num_classes)))
        if num_classes is not None
        else list(range(int(values.size(1))))
    )
    return {
        "samples": int(target.numel()),
        "macro_f1": float(
            f1_score(
                target.numpy(),
                predictions.numpy(),
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "true_label_nll": float(F.cross_entropy(values, target).item()),
        "predictions": predictions,
        "labels": target,
    }


def classify_impact(improvement: float, tolerance: float = 1e-9) -> str:
    """Classify a higher-is-better improvement as beneficial/harmful/tied."""

    improvement = float(improvement)
    tolerance = abs(float(tolerance))
    if improvement > tolerance:
        return "beneficial"
    if improvement < -tolerance:
        return "harmful"
    return "tied"


def _impact_fields(left: dict, right: dict, prefix: str, tolerance: float) -> dict:
    f1_improvement = float(left["macro_f1"] - right["macro_f1"])
    # NLL is lower-is-better, so improvement is right - left.
    nll_improvement = float(right["true_label_nll"] - left["true_label_nll"])
    return {
        f"{prefix}_f1_delta": f1_improvement,
        f"{prefix}_true_label_nll_improvement": nll_improvement,
        f"{prefix}_f1_impact": classify_impact(f1_improvement, tolerance),
        f"{prefix}_nll_impact": classify_impact(nll_improvement, tolerance),
    }


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in horizons}))
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons must contain positive integers")
    return values


def run_horizon_audit(
    full_adapter: Any,
    batches: Iterable[Any],
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    no_ssaw_adapter: Any | None = None,
    no_update_adapter: Any | None = None,
    device: torch.device | str | None = None,
    num_classes: int | None = None,
    impact_tolerance: float = 1e-9,
    metadata: Mapping[str, Any] | None = None,
    update_fn: Callable[[Any, BatchView, str], Any] | None = None,
    low_memory: bool | None = None,
    snapshot_cpu: bool | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run strict branch forks and return per-window rows plus audit summary.

    With ``low_memory=True`` the Full adapter is the only live adapter.  Each
    no-update/no-SSAW/Full branch is represented by a CPU ``AdapterState``;
    the adapter is restored and reused sequentially.  The reference mode
    (``low_memory=False``) keeps independently cloned adapters for callers
    that need branch-local diagnostics.
    The stream is consumed through a bounded look-ahead buffer of
    ``max(horizons)+1`` batches; a full target stream is never retained in
    memory.
    """

    horizons = _validate_horizons(horizons)
    metadata_values = dict(metadata or {})
    if bool(metadata_values.get("target_labels_used_for_updates", False)):
        raise ValueError("target labels are forbidden in online horizon updates")
    target_selected_parameters = bool(
        metadata_values.get("target_labels_used_for_parameter_selection", False)
    )
    evaluation_partition = str(
        metadata_values.get("evaluation_partition", "")
    ).strip()
    if target_selected_parameters and evaluation_partition not in {
        "target_selected_evaluation",
    }:
        raise ValueError(
            "target-selected parameters require an explicit evaluation_partition"
        )
    if target_selected_parameters and metadata_values.get(
        "parameter_selection_data_overlap"
    ) is not True:
        raise ValueError(
            "target-selected evaluation requires parameter_selection_data_overlap=true"
        )
    if bool(metadata_values.get("target_labels_used", False)):
        raise ValueError(
            "target labels may only be used for explicitly offline metrics"
        )
    stream = iter(batches)
    buffer: deque[BatchView] = deque()
    for _ in range(max(horizons) + 1):
        try:
            buffer.append(normalize_batch(next(stream)))
        except StopIteration:
            break
    supplied_branches = no_ssaw_adapter is not None or no_update_adapter is not None
    if low_memory is None:
        # Existing callers that explicitly supply clones retain the reference
        # behavior; a lone Full adapter defaults to the safe low-memory mode.
        low_memory = not supplied_branches
    low_memory = bool(low_memory)
    if snapshot_cpu is None:
        snapshot_cpu = low_memory
    snapshot_cpu = bool(snapshot_cpu)
    branches = {
        "full": full_adapter,
        "no_ssaw": no_ssaw_adapter,
        "no_update": no_update_adapter,
    }
    if low_memory:
        # State snapshots, not model copies, represent the two counterfactual
        # branches.  This keeps one model/optimizer live even on CUDA.
        branches["no_ssaw"] = full_adapter
        branches["no_update"] = full_adapter
    elif branches["no_ssaw"] is None or branches["no_update"] is None:
        clones = clone_branch_adapters(full_adapter)
        if branches["no_ssaw"] is None:
            branches["no_ssaw"] = clones["no_ssaw"]
        if branches["no_update"] is None:
            branches["no_update"] = clones["no_update"]
    if update_fn is None:
        update_fn = lambda adapter, batch, branch: online_update(
            adapter, batch, branch, device=device
        )

    # The canonical Full branch is the only state carried to the next t.
    rows: list[dict[str, Any]] = []
    equivalence_failures = 0
    rng_equivalence_failures = 0
    state_checks = 0
    original_enable_ssaw = getattr(full_adapter, "enable_ssaw", None)

    def _set_branch_mode(branch: str) -> None:
        if not low_memory or not hasattr(full_adapter, "enable_ssaw"):
            return
        if branch == "no_ssaw":
            full_adapter.enable_ssaw = False
        elif original_enable_ssaw is not None:
            full_adapter.enable_ssaw = bool(original_enable_ssaw)

    batch_index = 0
    while buffer:
        current_batch = buffer[0]
        future_batches = list(buffer)[1:]
        _set_branch_mode("full")
        pre_state = snapshot_state(branches["full"], cpu=snapshot_cpu)
        pre_hash = state_hash(pre_state)
        pre_bn = batchnorm_state(branches["full"])
        rng_state = _capture_rng_state(
            include_cuda=next(
                _module(branches["full"]).parameters(), torch.empty(0)
            ).device.type
            == "cuda"
        )
        branch_before: dict[str, AdapterState] = {}
        branch_after: dict[str, AdapterState] = {}
        branch_checks: dict[str, bool] = {}

        # Every branch must be a distinct object, optimizer, and model.  A
        # caller accidentally passing the Full adapter as a control branch
        # would otherwise produce apparently valid but non-counterfactual
        # rows.
        branch_objects_independent = low_memory or all(
            branches[left] is not branches[right]
            and _module(branches[left]) is not _module(branches[right])
            and _optimizer(branches[left]) is not _optimizer(branches[right])
            for left in BRANCHES
            for right in BRANCHES
            if left < right
        )
        branch_checks["branch_objects_independent"] = branch_objects_independent
        state_checks += 1
        if not branch_objects_independent:
            equivalence_failures += 1

        for branch in BRANCHES:
            _set_branch_mode(branch)
            restore_state(branches[branch], pre_state)
            branch_before[branch] = snapshot_state(
                branches[branch], cpu=snapshot_cpu
            )
            state_checks += 1
            equivalent = states_equal(branch_before[branch], pre_state)
            branch_checks[f"{branch}_starts_equivalent"] = equivalent
            if not equivalent:
                equivalence_failures += 1
            branch_checks[f"{branch}_rng_starts_equivalent"] = _nested_equal(
                branch_before[branch].rng_state, pre_state.rng_state
            )
            state_checks += 1
            if not branch_checks[f"{branch}_rng_starts_equivalent"]:
                rng_equivalence_failures += 1
            branch_checks[f"{branch}_bn_pre_equivalent"] = batchnorm_states_equal(
                pre_bn, batchnorm_state(branches[branch])
            )
            state_checks += 1
            if not branch_checks[f"{branch}_bn_pre_equivalent"]:
                equivalence_failures += 1

        # no-update is an explicit branch, even though it has no call.
        _set_branch_mode("no_update")
        branch_after["no_update"] = snapshot_state(
            branches["no_update"], cpu=snapshot_cpu
        )
        branch_checks["no_update_untouched"] = states_equal(
            branch_before["no_update"], branch_after["no_update"]
        )
        state_checks += 1
        if not branch_checks["no_update_untouched"]:
            equivalence_failures += 1

        for branch in ("no_ssaw", "full"):
            _set_branch_mode(branch)
            restore_state(branches[branch], pre_state)
            _restore_rng_state(rng_state)
            update_fn(branches[branch], current_batch, branch)
            branch_after[branch] = snapshot_state(
                branches[branch], cpu=snapshot_cpu
            )

        # ``no_ssaw`` and Full must not share a post-update object/state.
        branch_checks["branch_state_objects_independent"] = (
            low_memory
            or branch_checks["branch_objects_independent"]
            and branches["no_ssaw"] is not branches["full"]
            and _module(branches["no_ssaw"]) is not _module(branches["full"])
            and _optimizer(branches["no_ssaw"]) is not _optimizer(branches["full"])
        )
        state_checks += 1
        if not branch_checks["branch_state_objects_independent"]:
            equivalence_failures += 1

        for horizon in horizons:
            future_start = batch_index + 1
            future_end = future_start + horizon
            if horizon > len(future_batches):
                continue
            future = future_batches[:horizon]
            metrics: dict[str, dict[str, Any]] = {}
            for branch in BRANCHES:
                _set_branch_mode(branch)
                restore_state(branches[branch], branch_after[branch])
                before_eval = snapshot_state(
                    branches[branch], cpu=snapshot_cpu
                )
                metrics[branch] = future_metrics(
                    branches[branch],
                    future,
                    device=device,
                    num_classes=num_classes,
                )
                # Evaluation is read-only by protocol.  Restore the exact
                # post-update snapshot even when a BN implementation tracks
                # running buffers during prediction.
                restore_state(branches[branch], before_eval)
                after_eval = snapshot_state(
                    branches[branch], cpu=snapshot_cpu
                )
                key = f"{branch}_future_eval_untouched"
                branch_checks[key] = states_equal(before_eval, after_eval)
                state_checks += 1
                if not branch_checks[key]:
                    equivalence_failures += 1
                eval_rng_equal = _nested_equal(
                    before_eval.rng_state, after_eval.rng_state
                )
                branch_checks[f"{branch}_future_eval_rng_untouched"] = (
                    eval_rng_equal
                )
                state_checks += 1
                if not eval_rng_equal:
                    rng_equivalence_failures += 1

            row = {
                **dict(metadata or {}),
                "batch_index": int(batch_index),
                "horizon": int(horizon),
                "future_start_batch": int(future_start),
                "future_end_batch_exclusive": int(future_end),
                "current_sample_count": int(current_batch.labels.numel()),
                "target_labels_used_for_updates": bool(
                    (metadata or {}).get("target_labels_used_for_updates", False)
                ),
                "target_labels_used_for_parameter_selection": bool(
                    (metadata or {}).get(
                        "target_labels_used_for_parameter_selection", False
                    )
                ),
                "target_labels_used_for_metrics": bool(
                    (metadata or {}).get("target_labels_used_for_metrics", True)
                ),
                "pre_batch_state_hash": pre_hash,
                "no_update_post_state_hash": state_hash(branch_after["no_update"]),
                "no_ssaw_post_state_hash": state_hash(branch_after["no_ssaw"]),
                "full_post_state_hash": state_hash(branch_after["full"]),
                "state_equivalence_failures_total": int(equivalence_failures),
                "state_checks_total": int(state_checks),
                **{
                    f"state_{name}": bool(value)
                    for name, value in branch_checks.items()
                },
            }
            for branch in BRANCHES:
                row[f"{branch}_samples"] = metrics[branch]["samples"]
                row[f"{branch}_macro_f1"] = metrics[branch]["macro_f1"]
                row[f"{branch}_true_label_nll"] = metrics[branch]["true_label_nll"]
            row.update(
                _impact_fields(
                    metrics["full"],
                    metrics["no_ssaw"],
                    "full_vs_no_ssaw",
                    impact_tolerance,
                )
            )
            row.update(
                _impact_fields(
                    metrics["full"],
                    metrics["no_update"],
                    "full_vs_no_update",
                    impact_tolerance,
                )
            )
            row.update(
                _impact_fields(
                    metrics["no_ssaw"],
                    metrics["no_update"],
                    "no_ssaw_vs_no_update",
                    impact_tolerance,
                )
            )
            rows.append(row)

        # Restore the Full post-update state as the only online history.  A
        # future read must never alter the state carried into t+1.
        _set_branch_mode("full")
        restore_state(branches["full"], branch_after["full"])
        buffer.popleft()
        try:
            buffer.append(normalize_batch(next(stream)))
        except StopIteration:
            pass
        batch_index += 1

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no complete future horizon windows were evaluated")
    summary_rows = []
    for horizon, group in frame.groupby("horizon", sort=True):
        summary = {
            **dict(metadata or {}),
            "horizon": int(horizon),
            "windows": int(len(group)),
            "state_equivalence_failures": int(equivalence_failures),
            "state_checks": int(state_checks),
        }
        for prefix in (
            "full_vs_no_ssaw",
            "full_vs_no_update",
            "no_ssaw_vs_no_update",
        ):
            for metric in ("f1_delta", "true_label_nll_improvement"):
                column = f"{prefix}_{metric}"
                summary[f"{column}_mean"] = float(group[column].mean())
            impact_column = f"{prefix}_nll_impact"
            for label in IMPACT_LABELS:
                summary[f"{prefix}_nll_{label}_fraction"] = float(
                    group[impact_column].eq(label).mean()
                )
            f1_impact_column = f"{prefix}_f1_impact"
            for label in IMPACT_LABELS:
                summary[f"{prefix}_f1_{label}_fraction"] = float(
                    group[f1_impact_column].eq(label).mean()
                )
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    audit = {
        **dict(metadata or {}),
        "horizons": list(horizons),
        "branch_history": "canonical Full post-update state only",
        "low_memory": bool(low_memory),
        "state_snapshot_device": "cpu" if snapshot_cpu else "adapter device",
        "branch_start_state": "exact model+buffers+optimizer snapshot per batch",
        "future_evaluation": "same future batches, read-only, true labels offline only",
        "stream_buffer_max_batches": int(max(horizons) + 1),
        "state_equivalence_checks": int(state_checks),
        "state_equivalence_failures": int(equivalence_failures),
        "state_equivalence_passed": equivalence_failures == 0,
        "rng_equivalence_failures": int(rng_equivalence_failures),
        "rng_equivalence_passed": rng_equivalence_failures == 0,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_parameter_selection": target_selected_parameters,
        "evaluation_partition": evaluation_partition or "source_only_evaluation",
        "summary": summary_frame.to_dict(orient="records"),
    }
    return frame, audit


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{id(payload)}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{id(frame)}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


__all__ = [
    "AdapterState",
    "BRANCHES",
    "BatchView",
    "DEFAULT_HORIZONS",
    "IMPACT_LABELS",
    "atomic_write_frame",
    "atomic_write_json",
    "batchnorm_state",
    "batchnorm_states_equal",
    "classify_impact",
    "clone_branch_adapters",
    "future_metrics",
    "model_logits",
    "normalize_batch",
    "online_update",
    "restore_state",
    "run_horizon_audit",
    "snapshot_state",
    "state_hash",
    "states_equal",
]
