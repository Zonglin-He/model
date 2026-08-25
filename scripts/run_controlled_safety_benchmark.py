"""Controlled corruption benchmark with independent sample-level annotations."""

import argparse
import ast
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.corruption_transforms import CORRUPTION_REGISTRY
from dataloader.physical_corruption_transforms import (
    PHYSICAL_CORRUPTION_REGISTRY,
    physical_corruption_metadata,
    resolve_severity,
)
from benchmark_baselines.fisher import FISHER_CACHE_VERSION, ensure_source_fisher
from configs.benchmark_baselines import get_benchmark_hparams_class
from configs.tta_hparams_new import get_hparams_class
from scripts.supplementary_utils import (
    BatchTransformLoader,
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    enforce_common_batch_size,
    ensure_dir,
)
from scripts.paper_flow_profiles import (
    DEFAULT_PAPER_FLOW_PROFILE_JSON,
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.run_final_ssaw_full_no_ssaw_five_flow import production_code_sha256
from utils.probability_metrics import summarize_probability_metrics


DEFAULT_SCENARIOS = {"EEG": ("16", "1"), "HAR": ("12", "16"), "FD": ("2", "3")}
CONTROLLED_VARIANTS = {
    "full": {},
    "no_ssaw": {
        "enable_ssaw": False,
    },
    "no_confidence_gate": {
        "enable_confidence_gate": False,
    },
}

# These are hparams, not post-construction attributes.  In particular,
# ``dusafe_variant`` selects the reviewed production class through
# ``algorithms.get_tta_class`` and therefore must be applied before the TTA
# wrapper is constructed.  ``enable_source_semantic_router`` is forced off so
# a stale profile cannot re-enable the retired semantic-routing branch.
DUSAFE_VARIANT_RUNTIME_HPARAMS = {
    "full": {
        "dusafe_variant": "spline_residual",
        "enable_source_semantic_router": False,
    },
    "no_ssaw": {
        "dusafe_variant": "confidence_raw",
        "enable_ssaw": False,
        "enable_source_semantic_router": False,
    },
    "no_confidence_gate": {
        "dusafe_variant": "spline_residual",
        "enable_source_semantic_router": False,
        "enable_confidence_gate": False,
    },
}
DEFAULT_CORRUPTIONS = [
    "signal_freeze",
    "blackout",
    "attenuation",
    "amplitude_drift",
    "packet_loss",
    "saturation",
]

REQUIRED_SAFETY_METRICS = (
    "coverage",
    "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
)

NATIVE_RISK_CURVE_COLUMNS = (
    "dataset",
    "scenario",
    "method",
    "variant",
    "corruption",
    "severity",
    "corruption_seed",
    "coverage",
    "selective_risk",
    "count",
    "source_seed",
    "stream_seed",
    "protocol_signature",
    "correctness_column",
)

EATA_FISHER_PROTOCOL = f"official_eata_batch_gradient_diagonal_v{FISHER_CACHE_VERSION}"
SAFETY_PROTOCOL_VERSION = "controlled_safety_known_mask_v5_canonical_source_hash"
PROBABILITY_RECORD_SCHEMA = "full_multiclass_logits_probabilities_v1"


def effective_method_registry(args, method):
    """Resolve the registry for one method in a mixed safety panel.

    The benchmark registry owns the cited baseline adapters (including the
    source-only Fisher injection for EATA), while its historical ``DuSafe``
    alias points at the pre-spline base class.  A safety panel that requests
    both therefore has to route DuSafe through the production registry and
    keep every other method in the benchmark registry.  The choice is
    recorded in the signed protocol payload and per-job metadata.
    """

    requested = str(getattr(args, "registry", "production")).strip().lower()
    if requested not in {"production", "benchmark"}:
        raise ValueError(f"unknown algorithm registry: {requested!r}")
    if requested == "benchmark" and str(method) == "DuSafe":
        return "production"
    return requested


def parse_list(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def load_flowwise_source_profiles(path, datasets):
    """Load source-training configs without importing target-time settings."""

    if path is None or not str(path).strip():
        return {}
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected = set(str(dataset) for dataset in datasets)
    result = {}
    for key, record in payload.items():
        dataset, separator, scenario = str(key).partition(":")
        if not separator or dataset not in selected:
            continue
        config = record.get("source_config") if isinstance(record, dict) else None
        if not isinstance(config, dict):
            raise ValueError(f"flowwise source profile lacks source_config: {key}")
        result[(dataset, scenario)] = dict(config)
    return result


def load_source_checkpoint_references(path):
    if path is None or not str(path).strip():
        return {}
    source = Path(path).expanduser().resolve()
    frame = pd.read_csv(source)
    required = {"dataset", "scenario", "source_seed", "source_model_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source reference table lacks columns: {missing}")
    references = {}
    for key, group in frame.groupby(["dataset", "scenario", "source_seed"]):
        hashes = tuple(group["source_model_sha256"].dropna().astype(str).unique())
        if len(hashes) != 1:
            raise ValueError(f"ambiguous source checkpoint reference for {key}")
        references[(str(key[0]), str(key[1]), int(key[2]))] = hashes[0]
    return references


def read_csv_records(path):
    """Read a resumable CSV, treating a zero-byte artifact as no rows."""

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def safety_job_key(row):
    return (
        str(row["dataset"]),
        str(row["scenario"]),
        str(row["method"]),
        str(row.get("variant", "full")),
        str(row["corruption"]),
        str(row["severity"]),
        int(row["source_seed"]),
        int(row["stream_seed"]),
        int(row.get("corruption_seed", row["source_seed"])),
    )


def try_safety_job_key(row):
    """Return a resumable key, or None for a malformed legacy row."""

    try:
        return safety_job_key(row)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def safety_record_name(key):
    (
        dataset,
        scenario,
        method,
        variant,
        corruption,
        severity,
        source_seed,
        stream_seed,
        corruption_seed,
    ) = key
    scenario_token = "".join(
        character if character.isalnum() else "_" for character in str(scenario)
    ).strip("_")
    return (
        f"{dataset}_{scenario_token}_{method}_{variant}_{corruption}_{severity}_"
        f"source{source_seed}_stream{stream_seed}_"
        f"corruption{corruption_seed}.csv"
    )


def safety_protocol_signature(args, dataset, method, variant):
    """Hash every protocol/config input that controls one resumable job."""

    method_registry = effective_method_registry(args, method)
    hparams_class = (
        get_benchmark_hparams_class(dataset)
        if method_registry == "benchmark"
        else get_hparams_class(dataset)
    )
    hparams = hparams_class()
    base_algorithm = dict(hparams.alg_hparams[method])
    train_params = dict(hparams.train_params)
    scenario = "->".join(getattr(args, "scenario_map", {}).get(dataset, ()))
    flow_profile = dict(
        getattr(args, "flow_profile_overrides", {}).get((dataset, scenario), {})
    )
    source_profile = dict(
        getattr(args, "flowwise_source_profiles", {}).get((dataset, scenario), {})
    )
    expected_source_hashes = {
        str(seed): str(value)
        for (reference_dataset, reference_scenario, seed), value in getattr(
            args, "source_checkpoint_references", {}
        ).items()
        if str(reference_dataset) == str(dataset)
        and str(reference_scenario) == str(scenario)
    }
    payload = {
        "protocol_version": SAFETY_PROTOCOL_VERSION,
        "production_code_sha256": production_code_sha256(),
        "registry": str(args.registry),
        "effective_method_registry": method_registry,
        "dataset": str(dataset),
        "scenario": list(getattr(args, "scenario_map", {}).get(dataset, ())),
        "method": str(method),
        "variant": str(variant),
        "variant_overrides": dict(CONTROLLED_VARIANTS[variant]),
        "variant_runtime_hparams": dict(
            DUSAFE_VARIANT_RUNTIME_HPARAMS.get(variant, {})
        ),
        "backbone": str(args.backbone),
        "data_path": str(Path(args.data_path).resolve()),
        "pretrain_cache_dir": str(Path(args.pretrain_cache_dir).resolve()),
        "base_algorithm_hparams": base_algorithm,
        "train_params": train_params,
        "runtime_overrides": dict(args.overrides),
        "paper_flow_profile_json": str(
            getattr(args, "flow_profile_json", DEFAULT_PAPER_FLOW_PROFILE_JSON)
        ),
        "paper_flow_profile_overrides": flow_profile,
        "flowwise_source_profile": source_profile,
        "expected_source_model_sha256_by_seed": expected_source_hashes,
        # The reviewed per-flow profile owns DuSafe's deployment batch size;
        # baselines retain the checked-in dataset value.  Include the resolved
        # value in the signature so a profile change invalidates resumable
        # safety rows even when the base dataset hparams are unchanged.
        "effective_common_batch_size": int(
            flow_profile.get("batch_size", train_params["batch_size"])
        ),
        "corruption_fraction": float(args.corruption_fraction),
        "physical_protocol": bool(getattr(args, "physical_protocol", False)),
        "probability_record_schema": PROBABILITY_RECORD_SCHEMA,
        "calibration_bins": int(getattr(args, "calibration_bins", 15)),
        "eata_fisher_protocol": EATA_FISHER_PROTOCOL,
        "eata_fisher_samples": int(args.eata_fisher_samples),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"{SAFETY_PROTOCOL_VERSION}:{hashlib.sha256(encoded).hexdigest()}"


def sample_record_matches(path, key, protocol_signature):
    """Validate the first row without loading a potentially large record."""

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        first = pd.read_csv(path, nrows=1)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    if first.empty or "protocol_signature" not in first.columns:
        return False
    row = first.iloc[0].to_dict()
    return (
        try_safety_job_key(row) == key
        and str(row.get("protocol_signature", ""))
        == str(protocol_signature)
    )


def parse_overrides(entries):
    """Parse reproducible dataset-level hparam overrides for calibration runs."""
    overrides = {}
    for entry in entries or []:
        text = str(entry)
        if "=" not in text:
            raise ValueError(
                f"Invalid --override value {text!r}; expected key=value"
            )
        key, raw_value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --override value {text!r}; empty key")
        try:
            value = ast.literal_eval(raw_value.strip())
        except (ValueError, SyntaxError):
            value = raw_value.strip()
        overrides[key] = value
    return overrides


def parse_scenarios(text, datasets):
    scenarios = dict(DEFAULT_SCENARIOS)
    for item in parse_list(text):
        try:
            dataset, flow = item.split(":", 1)
            source, target = flow.split("->", 1)
        except ValueError as exc:
            raise ValueError(
                "Scenario entries must use DATASET:source->target."
            ) from exc
        scenarios[dataset.strip()] = (source.strip(), target.strip())
    missing = sorted(set(datasets) - set(scenarios))
    if missing:
        raise ValueError(f"No controlled scenario configured for: {missing}")
    return {dataset: scenarios[dataset] for dataset in datasets}


def _tensor_state_sha256(model):
    """Match the shared Fisher/overhead source-checkpoint hash convention."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def deterministic_mask_fn(fraction, seed):
    threshold = int(round(float(fraction) * 10000))

    def make_mask(data, labels, indices, step, total_steps):
        indices = torch.as_tensor(indices, dtype=torch.int64)
        hashed = (indices * 1103515245 + int(seed) * 12345 + 1013904223) % 10000
        return hashed < threshold

    return make_mask


def risk_coverage(records, job_meta):
    if job_meta.get("risk_coverage_status") != "available":
        return []
    score_column = (
        "admission_risk_score"
        if "admission_risk_score" in records
        else "raw_top1_nll"
    )
    correctness_column = (
        "pre_final_update_correct"
        if "pre_final_update_correct" in records
        else "correct"
    )
    required = {score_column, correctness_column}
    if records.empty or not required.issubset(records.columns):
        return []
    scores = pd.to_numeric(records[score_column], errors="coerce")
    correctness = _coerce_bool_series(records[correctness_column])
    valid = (
        scores.notna()
        & np.isfinite(scores.to_numpy(dtype=float))
        & correctness.notna()
    )
    if not valid.any():
        return []
    ordered = pd.DataFrame(
        {
            "score": scores[valid].to_numpy(dtype=float),
            "correct": correctness[valid].to_numpy(dtype=bool),
        }
    ).sort_values("score", ascending=True, kind="stable").reset_index(drop=True)
    rows = []
    counts = np.unique(
        np.maximum(
            1,
            np.rint(
                len(ordered) * np.arange(0.1, 1.01, 0.1)
            ).astype(int),
        )
    )
    for count in counts:
        subset = ordered.iloc[:count]
        rows.append({
            **job_meta,
            "coverage": float(count / len(ordered)),
            "selective_risk": 1.0
            - float(subset["correct"].mean()),
            "count": count,
            "correctness_column": correctness_column,
        })
    return rows


COMMON_RISK_META_COLUMNS = (
    "dataset",
    "scenario",
    "method",
    "variant",
    "corruption",
    "severity",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    "protocol_signature",
)


def _coerce_bool_series(series):
    """Parse CSV/in-memory booleans without treating the string False as true."""

    series = pd.Series(series)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    numeric_valid = numeric.notna()
    result.loc[numeric_valid] = numeric.loc[numeric_valid].ne(0).to_numpy()
    remaining = ~numeric_valid & series.notna()
    if remaining.any():
        normalized = series.loc[remaining].astype(str).str.strip().str.lower()
        mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "t": True,
            "f": False,
        }
        mapped = normalized.map(mapping)
        mapped_valid = mapped.notna()
        result.loc[mapped.index[mapped_valid]] = mapped[mapped_valid].to_numpy()
    return result


def active_batch_conditional_metrics(
    records: pd.DataFrame,
    *,
    eligible_coverage_threshold: float = 0.25,
) -> dict[str, float]:
    """Summarize label-free active batches without making a clean-F1 ladder.

    Batch selection uses only ``admitted``/``selected`` masks.  True labels are
    read after selection for descriptive conditional endpoints.  The overall
    eligible coverage is always emitted beside the conditional values; when no
    batch crosses the threshold the conditional fields are NaN.
    """

    names = {
        "eligible_coverage_overall": float("nan"),
        "eligible_active_batch_fraction": float("nan"),
        "eligible_active_batch_count": 0.0,
        "active_batch_clean_f1": float("nan"),
        "active_batch_corrupted_f1": float("nan"),
        "active_batch_admitted_accuracy": float("nan"),
        "active_batch_incorrect_admission_rate": float("nan"),
        "active_batch_unsafe_admission_rate": float("nan"),
    }
    required = {
        "batch_index",
        "admitted",
        "selected",
        "corrupted",
        "label",
        "prediction",
        "admission_pseudo_label_correct",
    }
    if records.empty or not required.issubset(records.columns):
        return names
    admitted = _coerce_bool_series(records["admitted"])
    selected = _coerce_bool_series(records["selected"])
    corrupted = _coerce_bool_series(records["corrupted"])
    admission_correct = _coerce_bool_series(
        records["admission_pseudo_label_correct"]
    )
    valid = admitted.notna() & selected.notna() & corrupted.notna() & admission_correct.notna()
    if not valid.any():
        return names
    values = records.loc[valid].copy()
    values["_admitted"] = admitted.loc[valid].astype(bool).to_numpy()
    values["_selected"] = selected.loc[valid].astype(bool).to_numpy()
    values["_corrupted"] = corrupted.loc[valid].astype(bool).to_numpy()
    values["_admission_correct"] = admission_correct.loc[valid].astype(bool).to_numpy()
    batch_rows = []
    for batch_index, group in values.groupby("batch_index", sort=True):
        admitted_count = int(group["_admitted"].sum())
        selected_count = int(group["_selected"].sum())
        batch_rows.append(
            {
                "batch_index": batch_index,
                "eligible_coverage": (
                    selected_count / admitted_count if admitted_count else 0.0
                ),
            }
        )
    batch_frame = pd.DataFrame(batch_rows)
    if batch_frame.empty:
        return names
    overall_admitted = int(values["_admitted"].sum())
    overall_selected = int(values["_selected"].sum())
    names["eligible_coverage_overall"] = (
        overall_selected / overall_admitted if overall_admitted else 0.0
    )
    active_batches = batch_frame[
        batch_frame["eligible_coverage"] >= float(eligible_coverage_threshold)
    ]
    names["eligible_active_batch_count"] = float(len(active_batches))
    names["eligible_active_batch_fraction"] = float(
        len(active_batches) / len(batch_frame)
    )
    if active_batches.empty:
        return names
    active_values = values[values["batch_index"].isin(active_batches["batch_index"])]
    labels = pd.to_numeric(active_values["label"], errors="raise").to_numpy(dtype=int)
    predictions = pd.to_numeric(active_values["prediction"], errors="raise").to_numpy(dtype=int)
    max_label = int(labels.max()) if labels.size else 0
    max_prediction = int(predictions.max()) if predictions.size else 0
    class_count = int(max(max_label, max_prediction) + 1)
    for name, mask in (
        ("active_batch_clean_f1", ~active_values["_corrupted"].to_numpy(dtype=bool)),
        ("active_batch_corrupted_f1", active_values["_corrupted"].to_numpy(dtype=bool)),
    ):
        if not mask.any():
            continue
        names[name] = float(
            f1_score(
                labels[mask],
                predictions[mask],
                labels=np.arange(class_count),
                average="macro",
                zero_division=0,
            )
        )
    admitted_active = active_values[active_values["_admitted"]]
    if not admitted_active.empty:
        names["active_batch_admitted_accuracy"] = float(
            admitted_active["_admission_correct"].mean()
        )
        incorrect = ~admitted_active["_admission_correct"]
        unsafe = incorrect | admitted_active["_corrupted"]
        names["active_batch_incorrect_admission_rate"] = float(incorrect.mean())
        names["active_batch_unsafe_admission_rate"] = float(unsafe.mean())
    return names


def common_predictive_risk_coverage(records, job_meta, *, stage="pre_update"):
    """Return a method-agnostic predictive risk--coverage curve.

    This curve is deliberately separate from ``risk_coverage``.  It ranks
    every method by the same confidence definition and evaluates the matching
    prediction.  ``pre_update`` is an admission-time diagnostic; ``post_update``
    is aligned with the final predictions used by the reported F1.  Neither is
    the method's native online admission policy.
    """

    stage_columns = {
        "pre_update": (
            "pre_final_update_confidence",
            "pre_final_update_correct",
            "common_pre_update_top1_nll",
        ),
        "post_update": (
            "confidence",
            "correct",
            "common_post_update_top1_nll",
        ),
    }
    if stage not in stage_columns:
        raise ValueError(f"Unknown predictive risk stage: {stage}")
    confidence_column, correctness_column, policy = stage_columns[stage]
    required = {confidence_column, correctness_column}
    if records.empty or not required.issubset(records.columns):
        return []
    confidence = pd.to_numeric(
        records[confidence_column], errors="coerce"
    )
    correct = _coerce_bool_series(records[correctness_column])
    finite = (
        confidence.notna()
        & np.isfinite(confidence.to_numpy(dtype=float))
        & correct.notna()
    )
    if not finite.any():
        return []
    comparison = pd.DataFrame(
        {
            "score": -np.log(
                confidence[finite]
                .clip(lower=np.finfo(float).tiny, upper=1.0)
                .to_numpy(dtype=float)
            ),
            "correct": correct[finite].to_numpy(dtype=bool),
        }
    ).sort_values("score", ascending=True, kind="stable")
    sample_count = len(comparison)
    counts = np.unique(
        np.ceil(
            np.linspace(1, sample_count, num=min(100, sample_count))
        ).astype(int)
    )
    rows = []
    for count in counts:
        subset = comparison.iloc[: int(count)]
        rows.append(
            {
                **job_meta,
                "risk_policy": policy,
                "score_column": f"-log({confidence_column})",
                "correctness_column": correctness_column,
                "coverage": float(count / sample_count),
                "selective_risk": 1.0
                - float(subset["correct"].mean()),
                "count": int(count),
            }
        )
    return rows


def _aurc_from_curves(curve_df, *, group_columns):
    rows = []
    if curve_df.empty:
        return pd.DataFrame(columns=[*group_columns, "aurc"])
    for keys, frame in curve_df.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ordered = frame.sort_values("coverage", kind="stable")
        coverage = np.concatenate(
            [[0.0], ordered["coverage"].to_numpy(dtype=float)]
        )
        risk = np.concatenate(
            [
                [float(ordered["selective_risk"].iloc[0])],
                ordered["selective_risk"].to_numpy(dtype=float),
            ]
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "aurc": float(np.trapezoid(risk, coverage)),
            }
        )
    return pd.DataFrame(rows, columns=[*group_columns, "aurc"])


def _record_matches_expected_signatures(records, expected_signatures):
    if expected_signatures is None:
        return True
    if records.empty:
        return False
    row = records.iloc[0].to_dict()
    key = try_safety_job_key(row)
    if key is None or key not in expected_signatures:
        return False
    return str(row.get("protocol_signature", "")) == str(
        expected_signatures[key]
    )


def write_native_risk_artifacts(
    records_dir, output_dir, *, expected_signatures=None
):
    """Rebuild method-native admission curves from partitioned sample records."""

    records_dir = Path(records_dir)
    output_dir = Path(output_dir)
    curve_rows = []
    for record_path in sorted(records_dir.glob("*.csv")):
        record_rows = read_csv_records(record_path)
        if not record_rows:
            continue
        records = pd.DataFrame(record_rows)
        if not _record_matches_expected_signatures(
            records, expected_signatures
        ):
            continue
        required_meta = {
            *COMMON_RISK_META_COLUMNS,
            "risk_coverage_status",
        }
        if not required_meta.issubset(records.columns):
            continue
        first = records.iloc[0]
        job_meta = {
            column: first[column] for column in COMMON_RISK_META_COLUMNS
        }
        job_meta.update(
            {
                "risk_coverage_status": first["risk_coverage_status"],
                "risk_score_policy": first.get("risk_score_policy", ""),
                "risk_score_components": first.get(
                    "risk_score_components", ""
                ),
            }
        )
        curve_rows.extend(risk_coverage(records, job_meta))

    curve_df = pd.DataFrame(
        curve_rows, columns=NATIVE_RISK_CURVE_COLUMNS
    )
    atomic_write_csv(
        curve_df, output_dir / "risk_coverage_raw.csv", index=False
    )
    aggregate_keys = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "corruption_seed",
        "protocol_signature",
        "coverage",
    ]
    if curve_df.empty:
        curve_aggregate = pd.DataFrame(
            columns=[
                *aggregate_keys,
                "selective_risk_mean",
                "selective_risk_std",
            ]
        )
    else:
        curve_aggregate = (
            curve_df.groupby(aggregate_keys, as_index=False)
            .agg(
                selective_risk_mean=("selective_risk", "mean"),
                selective_risk_std=("selective_risk", "std"),
            )
        )
    atomic_write_csv(
        curve_aggregate,
        output_dir / "risk_coverage_aggregate.csv",
        index=False,
    )

    aurc_keys = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
        "protocol_signature",
    ]
    aurc = _aurc_from_curves(curve_df, group_columns=aurc_keys)
    atomic_write_csv(
        aurc, output_dir / "aurc_per_source_seed.csv", index=False
    )
    aggregate_aurc_keys = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "corruption_seed",
        "protocol_signature",
    ]
    if aurc.empty:
        aurc_aggregate = pd.DataFrame(
            columns=[*aggregate_aurc_keys, "aurc_mean", "aurc_std"]
        )
    else:
        aurc_aggregate = (
            aurc.groupby(aggregate_aurc_keys, as_index=False)
            .agg(aurc_mean=("aurc", "mean"), aurc_std=("aurc", "std"))
        )
    atomic_write_csv(
        aurc_aggregate, output_dir / "aurc_aggregate.csv", index=False
    )
    return curve_df, aurc


def write_common_predictive_risk_artifacts(
    records_dir, output_dir, *, expected_signatures=None
):
    """Rebuild comparable risk--coverage/AURC artifacts from saved jobs."""

    records_dir = Path(records_dir)
    output_dir = Path(output_dir)
    curve_rows = []
    for record_path in sorted(records_dir.glob("*.csv")):
        record_rows = read_csv_records(record_path)
        if not record_rows:
            continue
        records = pd.DataFrame(record_rows)
        if records.empty:
            continue
        if not _record_matches_expected_signatures(
            records, expected_signatures
        ):
            continue
        missing_meta = [
            column
            for column in COMMON_RISK_META_COLUMNS
            if column not in records.columns
        ]
        if missing_meta:
            continue
        first = records.iloc[0]
        job_meta = {
            column: first[column] for column in COMMON_RISK_META_COLUMNS
        }
        for stage in ("pre_update", "post_update"):
            curve_rows.extend(
                common_predictive_risk_coverage(
                    records, job_meta, stage=stage
                )
            )

    curve_columns = [
        *COMMON_RISK_META_COLUMNS,
        "risk_policy",
        "score_column",
        "correctness_column",
        "coverage",
        "selective_risk",
        "count",
    ]
    curve_df = pd.DataFrame(curve_rows, columns=curve_columns)
    atomic_write_csv(
        curve_df,
        output_dir / "predictive_risk_coverage_raw.csv",
        index=False,
    )
    aggregate_keys = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "corruption_seed",
        "protocol_signature",
        "risk_policy",
        "coverage",
    ]
    if curve_df.empty:
        curve_aggregate = pd.DataFrame(
            columns=[
                *aggregate_keys,
                "selective_risk_mean",
                "selective_risk_std",
            ]
        )
    else:
        curve_aggregate = (
            curve_df.groupby(aggregate_keys, as_index=False)
            .agg(
                selective_risk_mean=("selective_risk", "mean"),
                selective_risk_std=("selective_risk", "std"),
            )
        )
    atomic_write_csv(
        curve_aggregate,
        output_dir / "predictive_risk_coverage_aggregate.csv",
        index=False,
    )

    aurc_keys = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
        "protocol_signature",
        "risk_policy",
    ]
    aurc = _aurc_from_curves(curve_df, group_columns=aurc_keys)
    atomic_write_csv(
        aurc,
        output_dir / "predictive_aurc_per_source_seed.csv",
        index=False,
    )
    aggregate_aurc_keys = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "corruption_seed",
        "protocol_signature",
        "risk_policy",
    ]
    if aurc.empty:
        aurc_aggregate = pd.DataFrame(
            columns=[*aggregate_aurc_keys, "aurc_mean", "aurc_std"]
        )
    else:
        aurc_aggregate = (
            aurc.groupby(aggregate_aurc_keys, as_index=False)
            .agg(aurc_mean=("aurc", "mean"), aurc_std=("aurc", "std"))
        )
    atomic_write_csv(
        aurc_aggregate,
        output_dir / "predictive_aurc_aggregate.csv",
        index=False,
    )
    return curve_df, aurc


def admission_risk_score(
    records,
    confidence_enabled,
    confidence_threshold,
    semantic_enabled=False,
):
    """Reconstruct admission with source-normalized enabled-gate scores."""
    risk_parts = []
    risk_components = []
    if confidence_enabled:
        risk_parts.append(
            records["raw_top1_nll"].to_numpy(dtype=float)
            / max(confidence_threshold, np.finfo(float).eps)
        )
        risk_components.append("source_normalized_top1_nll")
    if semantic_enabled:
        semantic_prediction = records[
            "source_semantic_prediction"
        ].to_numpy(dtype=float)
        # DuSafe computes the fixed-source semantic gate from the raw logits
        # before the online update.  ``prediction`` is the post-update output
        # recorded by the trainer and can legitimately differ after an
        # admitted update; using it here creates a false admission mismatch.
        raw_prediction_column = (
            "pre_final_update_prediction"
            if "pre_final_update_prediction" in records
            else "prediction"
        )
        raw_prediction = records[raw_prediction_column].to_numpy(dtype=float)
        semantic_disagreement = semantic_prediction != raw_prediction
        risk_parts.append(
            np.where(semantic_disagreement, 2.0, 0.0)
        )
        risk_components.append("fixed_source_semantic_disagreement")
    if risk_parts:
        return np.maximum.reduce(risk_parts), risk_components

    # A binary selection mask is not a continuous risk score. Returning zeros
    # here would fabricate a risk ranking and make risk--coverage/AURC look
    # measured when the adapter exposes no such policy.
    return np.full(len(records), np.nan, dtype=float), [
        "no_continuous_admission_score"
    ]


def attach_probability_records(
    records,
    post_update_logits,
    pre_final_update_logits,
    labels,
    *,
    calibration_bins=15,
):
    """Attach complete multiclass outputs and return standard metric summaries."""

    records = records.copy()
    post_update_logits = torch.as_tensor(post_update_logits).detach().cpu().float()
    pre_final_update_logits = (
        torch.as_tensor(pre_final_update_logits).detach().cpu().float()
    )
    labels = torch.as_tensor(labels).detach().cpu().long().view(-1)
    expected_shape = (len(records),)
    if post_update_logits.ndim != 2 or pre_final_update_logits.ndim != 2:
        raise ValueError("Complete probability logging requires [N, K] logits")
    if (
        post_update_logits.shape != pre_final_update_logits.shape
        or post_update_logits.shape[0] != expected_shape[0]
        or labels.shape != expected_shape
    ):
        raise ValueError("Probability tensors do not align with sample records")
    if post_update_logits.size(1) < 2:
        raise ValueError("Probability logging requires at least two classes")
    record_labels = pd.to_numeric(records["label"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(record_labels, labels.numpy()):
        raise ValueError("Logged labels do not align with the evaluation tensors")
    post_probabilities = post_update_logits.softmax(dim=1).numpy()
    pre_probabilities = pre_final_update_logits.softmax(dim=1).numpy()
    post_logits = post_update_logits.numpy()
    pre_logits = pre_final_update_logits.numpy()
    for class_index in range(post_update_logits.size(1)):
        records[f"post_update_logit_{class_index}"] = post_logits[:, class_index]
        records[f"post_update_probability_{class_index}"] = post_probabilities[
            :, class_index
        ]
        records[f"pre_final_update_logit_{class_index}"] = pre_logits[:, class_index]
        records[f"pre_final_update_probability_{class_index}"] = pre_probabilities[
            :, class_index
        ]
    records["probability_record_schema"] = PROBABILITY_RECORD_SCHEMA
    post_metrics = summarize_probability_metrics(
        labels.numpy(), post_probabilities, calibration_bins=int(calibration_bins)
    )
    pre_metrics = summarize_probability_metrics(
        labels.numpy(), pre_probabilities, calibration_bins=int(calibration_bins)
    )
    summary = {
        **{f"post_update_{key}": value for key, value in post_metrics.items()},
        **{f"pre_final_update_{key}": value for key, value in pre_metrics.items()},
        "probability_class_count": int(post_update_logits.size(1)),
        "probability_record_schema": PROBABILITY_RECORD_SCHEMA,
        "calibration_bins": int(calibration_bins),
    }
    return records, summary


def corruption_conditioned_probability_metrics(records, *, calibration_bins=15):
    """Compute post-update metrics separately on corrupted and clean samples."""

    records = records.copy()
    required = {"label", "prediction", "corrupted"}
    if not required.issubset(records.columns):
        raise ValueError("Corruption-conditioned metrics require labels and mask")
    probability_columns = sorted(
        (
            column
            for column in records.columns
            if column.startswith("post_update_probability_")
        ),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if len(probability_columns) < 2:
        raise ValueError("Complete post-update probabilities are unavailable")
    probabilities = records[probability_columns].to_numpy(dtype=float)
    labels = pd.to_numeric(records["label"], errors="raise").to_numpy(
        dtype=np.int64
    )
    predictions = pd.to_numeric(
        records["prediction"], errors="raise"
    ).to_numpy(dtype=np.int64)
    corrupted = _coerce_bool_series(records["corrupted"])
    if corrupted.isna().any():
        raise ValueError("Corruption mask contains an invalid boolean")
    output = {}
    for name, mask in (
        ("corrupted", corrupted.to_numpy(dtype=bool)),
        ("clean", ~corrupted.to_numpy(dtype=bool)),
    ):
        count = int(mask.sum())
        output[f"{name}_sample_count"] = count
        if count == 0:
            for metric in (
                "macro_f1",
                "nll",
                "brier",
                "ece",
                "classwise_ece",
                "aurc",
                "oracle_aurc",
                "eaurc",
            ):
                output[f"{name}_post_update_{metric}"] = float("nan")
            continue
        output[f"{name}_post_update_macro_f1"] = float(
            f1_score(
                labels[mask],
                predictions[mask],
                labels=np.arange(probabilities.shape[1]),
                average="macro",
                zero_division=0,
            )
        )
        metrics = summarize_probability_metrics(
            labels[mask],
            probabilities[mask],
            calibration_bins=int(calibration_bins),
        )
        output.update(
            {f"{name}_post_update_{key}": value for key, value in metrics.items()}
        )
    return output


def run_job(
    args, dataset, method, variant, corruption, severity, source_seed,
    stream_seed,
):
    src_id, trg_id = args.scenario_map[dataset]
    method_registry = effective_method_registry(args, method)
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method=method,
        backbone=args.backbone,
        exp_name=f"safety_{method}_{source_seed}",
        seed=stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        algorithm_registry=method_registry,
    )
    if args.overrides:
        trainer.set_runtime_hparams(args.overrides)
    scenario_label = f"{src_id}->{trg_id}"
    source_profile = dict(
        getattr(args, "flowwise_source_profiles", {}).get(
            (dataset, scenario_label), {}
        )
    )
    if source_profile:
        trainer.source_hparams.update(source_profile)
    flow_profile = dict(
        getattr(args, "flow_profile_overrides", {}).get((dataset, scenario_label), {})
    )
    if method == "DuSafe" and flow_profile:
        # The paper JSON is the reviewed per-flow TTA profile.  It is applied
        # after generic CLI overrides so the signed profile cannot be
        # accidentally replaced by a stale dataset-level value.
        trainer.set_runtime_hparams(flow_profile)
    if method == "DuSafe":
        # Select the production implementation before ``create_tta_model``
        # resolves the registry class.  The old post-construction variant
        # loop below remains responsible only for simple boolean toggles.
        trainer.set_runtime_hparams(DUSAFE_VARIANT_RUNTIME_HPARAMS[variant])
        trainer.set_runtime_hparams(
            {
                "dusafe_logging_mode": "evidence",
                "record_per_sample_evidence": True,
            }
        )
    common_batch_size = enforce_common_batch_size(
        trainer,
        src_id,
        trg_id,
        batch_size=(
            flow_profile.get("batch_size")
            if method == "DuSafe" and "batch_size" in flow_profile
            else None
        ),
    )
    tta_model = pre_trained_model = None
    canonical_source_model_sha256 = ""
    fisher_metadata = {
        "fisher_enabled": False,
    }
    try:
        def pre_tta_hook(hook_trainer, hook_model):
            nonlocal canonical_source_model_sha256
            canonical_source_model_sha256 = _tensor_state_sha256(hook_model)
            if method != "EATA":
                return
            fisher_metadata.update(
                ensure_source_fisher(
                    model=hook_model,
                    source_loader=hook_trainer.src_train_dl,
                    cache_dir=args.fisher_cache_dir,
                    dataset=dataset,
                    source_seed=source_seed,
                    source_checkpoint_sha256=canonical_source_model_sha256,
                    samples=int(
                        hook_trainer.hparams.get(
                            "fisher_samples", args.eata_fisher_samples
                        )
                    ),
                    adapt_keywords=hook_trainer.hparams.get(
                        "adapt_keywords", ("classifier", "adapter")
                    ),
                )
            )
            if not fisher_metadata.get("fisher_enabled"):
                raise RuntimeError(
                    "EATA Fisher calibration did not return fisher_enabled=True"
                )
            hook_trainer.hparams["fisher_enabled"] = True
            hook_trainer.hparams["fisher_path"] = fisher_metadata[
                "fisher_cache_path"
            ]

        tta_model, pre_trained_model = create_tta_model(
            trainer,
            src_id,
            trg_id,
            run_seed=stream_seed,
            pre_tta_hook=pre_tta_hook,
        )
        if method == "EATA" and not bool(
            getattr(tta_model, "fisher_enabled", False)
        ):
            raise RuntimeError(
                "EATA safety job constructed without a validated source Fisher"
            )
        if not canonical_source_model_sha256:
            raise RuntimeError(
                "pre-adapter source checkpoint identity was not captured"
            )
        expected_source_hash = str(
            getattr(args, "source_checkpoint_references", {}).get(
                (dataset, scenario_label, int(source_seed)), ""
            )
        )
        if expected_source_hash and canonical_source_model_sha256 != expected_source_hash:
            raise RuntimeError(
                "source checkpoint mismatch: "
                f"{canonical_source_model_sha256} != {expected_source_hash}"
            )
        variant_overrides = CONTROLLED_VARIANTS[variant]
        if method != "DuSafe" and variant_overrides:
            raise ValueError(
                "Controlled DuSafe variants can only be used with DuSafe"
            )
        for attribute_path, value in variant_overrides.items():
            owner = tta_model
            parts = attribute_path.split(".")
            for part in parts[:-1]:
                if not hasattr(owner, part):
                    raise AttributeError(
                        f"{method} does not expose controlled variant setting "
                        f"{attribute_path!r}."
                    )
                owner = getattr(owner, part)
            attribute = parts[-1]
            if not hasattr(owner, attribute):
                raise AttributeError(
                    f"{method} does not expose controlled variant setting "
                    f"{attribute_path!r}."
                )
            setattr(owner, attribute, value)
        corruption_seed = (
            int(source_seed)
            if args.corruption_seed is None
            else int(args.corruption_seed)
        )
        use_physical_protocol = bool(getattr(args, "physical_protocol", False))
        transform_registry = (
            PHYSICAL_CORRUPTION_REGISTRY
            if use_physical_protocol
            else CORRUPTION_REGISTRY
        )
        transform_meta = (
            physical_corruption_metadata(corruption, severity)
            if use_physical_protocol
            else {
                "corruption": corruption,
                "severity_name": str(severity),
                "normalized_severity": float("nan"),
                "physical_parameters": {},
            }
        )
        trainer.trg_whole_dl = BatchTransformLoader(
            trainer.trg_whole_dl,
            transform_registry[corruption],
            severity,
            sample_mask_fn=deterministic_mask_fn(
                args.corruption_fraction, corruption_seed
            ),
            meta={
                "corruption_type": corruption,
                "severity": severity,
                **transform_meta,
            },
            transform_seed=corruption_seed + 20_000,
        )
        metrics = trainer.calculate_metrics(tta_model)
        records = trainer.last_safety_records.copy()
        records, probability_summary = attach_probability_records(
            records,
            trainer.full_preds,
            trainer.full_pre_final_update_preds,
            trainer.full_labels,
            calibration_bins=getattr(args, "calibration_bins", 15),
        )
        probability_summary.update(
            corruption_conditioned_probability_metrics(
                records,
                calibration_bins=getattr(args, "calibration_bins", 15),
            )
        )
        confidence_enabled = bool(
            getattr(tta_model, "enable_confidence_gate", False)
        )
        semantic_enabled = bool(
            getattr(tta_model, "enable_source_semantic_gate", False)
        )
        confidence_threshold = float(
            getattr(
                tta_model,
                "confidence_nll_threshold",
                torch.tensor(float("nan")),
            ).detach().item()
        )
        risk_score, risk_components = admission_risk_score(
            records,
            confidence_enabled,
            confidence_threshold,
            semantic_enabled,
        )
        risk_score_available = (
            "no_continuous_admission_score" not in risk_components
        )
        records["admission_risk_score"] = risk_score
        if risk_score_available:
            records["admission_score_pass"] = records[
                "admission_risk_score"
            ].le(1.0)
        else:
            # Benchmark adapters expose their actual binary selection mask but
            # no continuous risk score. Keep that decision without inventing a
            # ranking from a constant or raw-NLL fallback.
            records["admission_score_pass"] = records["admitted"].astype(bool)
        if risk_score_available and method == "DuSafe" and not records[
            "admission_score_pass"
        ].astype(bool).equals(records["admitted"].astype(bool)):
            raise RuntimeError(
                "Source-normalized risk score does not reproduce the online "
                "DuSafe admission decision"
            )
        role_conditional_metrics = active_batch_conditional_metrics(records)
        job_meta = {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "method": method,
            "variant": variant,
            "corruption": corruption,
            "severity": severity,
            "severity_name": str(transform_meta["severity_name"]),
            "normalized_severity": float(transform_meta["normalized_severity"]),
            "physical_parameters": json.dumps(
                transform_meta["physical_parameters"], sort_keys=True
            ),
            "source_seed": int(source_seed),
            "stream_seed": int(stream_seed),
            "corruption_seed": corruption_seed,
            "protocol_signature": safety_protocol_signature(
                args, dataset, method, variant
            ),
            "production_code_sha256": production_code_sha256(),
            # Canonical fixed-source identity is captured in the hook before
            # an adapter changes BN modes/buffers.  Keep the configured state
            # separate: different TTA algorithms legitimately configure the
            # same checkpoint differently and must not look like distinct
            # source checkpoints in paired analysis.
            "source_model_sha256": canonical_source_model_sha256,
            "source_checkpoint_sha256": canonical_source_model_sha256,
            "adapter_configured_model_sha256": _tensor_state_sha256(
                pre_trained_model
            ),
            "common_batch_size": int(common_batch_size),
            "confidence_nll_threshold": confidence_threshold,
            "risk_coverage_status": (
                "available"
                if risk_score_available
                else "not_available_no_continuous_score"
            ),
            "risk_score_policy": (
                "max(" + ",".join(risk_components) + ")"
                if risk_score_available
                else "not_available; use_adapter_admission_mask_only"
            ),
            "risk_score_components": ",".join(risk_components),
            "eata_fisher_status": (
                "validated_source_fisher"
                if method == "EATA"
                else "not_applicable"
            ),
            "eata_fisher_cache_path": fisher_metadata.get(
                "fisher_cache_path", ""
            ),
            "eata_fisher_cache_hash": fisher_metadata.get(
                "fisher_cache_hash", ""
            ),
            "eata_fisher_source_checkpoint_sha256": fisher_metadata.get(
                "fisher_source_checkpoint_sha256", ""
            ),
            "eata_fisher_cache_hit": fisher_metadata.get(
                "fisher_cache_hit", False
            ),
            "eata_fisher_sample_count": fisher_metadata.get(
                "fisher_samples", 0
            ),
            "eata_fisher_batch_count": fisher_metadata.get(
                "fisher_batches", 0
            ),
            "eata_fisher_compute_seconds": fisher_metadata.get(
                "fisher_compute_seconds", 0.0
            ),
            "runtime_hparams": json.dumps(
                trainer.hparams, sort_keys=True, default=str
            ),
        }
        for key, value in job_meta.items():
            records[key] = value
        summary = {
            **job_meta,
            "acc": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            **probability_summary,
            **trainer.last_safety_summary,
            **getattr(trainer, "last_gate_contribution_summary", {}),
            **{
                f"diag_{key}": value
                for key, value in trainer.last_batch_log_summary.items()
            },
        }
        # Explicit representative-panel aliases.  Keep the canonical source
        # columns above unchanged, but expose the requested quantities without
        # forcing downstream consumers to infer them from the probability or
        # safety namespace.  IAR is wrong pseudo-labels conditional on
        # admission; the broader protocol unsafe-admission quantity (which
        # also counts annotated corruption) remains separately named.
        admitted_accuracy = summary.get(
            "admitted_pseudo_label_accuracy", float("nan")
        )
        incorrect_admission_rate = (
            1.0 - float(admitted_accuracy)
            if pd.notna(admitted_accuracy)
            else float("nan")
        )
        summary.update(
            {
                "clean_f1": summary.get(
                    "clean_post_update_macro_f1", float("nan")
                ),
                "corrupted_f1": summary.get(
                    "corrupted_post_update_macro_f1", float("nan")
                ),
                "admitted_accuracy": admitted_accuracy,
                "incorrect_admission_rate": incorrect_admission_rate,
                "incorrect_admission_label_rate": (
                    incorrect_admission_rate
                ),
                "unsafe_admission_rate": summary.get(
                    "admission_unsafe_rate", float("nan")
                ),
                **role_conditional_metrics,
            }
        )
        return summary, records, risk_coverage(records, job_meta)
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--registry",
        choices=("production", "benchmark"),
        default="production",
        help="Explicit algorithm registry; benchmark is isolated from production DuSafe.",
    )
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument(
        "--methods", default="NoAdap,DuSafe"
    )
    parser.add_argument(
        "--variants",
        default="full",
        help="Comma-separated DuSafe variants declared in CONTROLLED_VARIANTS.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help=(
            "Dataset-level hparam override for a declared calibration run, "
            "e.g. --override confidence_keep_fraction=1.0."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help=(
            "Optional comma-separated overrides such as "
            "EEG:0->11,HAR:9->18,FD:0->1."
        ),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(DEFAULT_PAPER_FLOW_PROFILE_JSON),
        help=(
            "Per-flow TTA override JSON. Source-training settings remain in "
            "the source checkpoint; defaults to configs/paper_flow_profiles_v1.json."
        ),
    )
    parser.add_argument(
        "--flowwise-source-profile-json",
        default="",
        help=(
            "Optional selected_profiles.json supplying source_config per flow. "
            "TTA settings continue to come from --flow-profile-json."
        ),
    )
    parser.add_argument(
        "--source-reference-csv",
        default="",
        help="Optional paired table used to enforce source checkpoint identity.",
    )
    parser.add_argument("--corruptions", default=",".join(DEFAULT_CORRUPTIONS))
    parser.add_argument("--severities", default="moderate,severe")
    parser.add_argument(
        "--physical_protocol",
        action="store_true",
        help=(
            "Use the pre-registered continuous s0...s6 physical panel instead "
            "of the legacy categorical corruption registry."
        ),
    )
    parser.add_argument(
        "--calibration_bins",
        type=int,
        default=15,
        help="Equal-width bins for ECE; NLL/Brier/AURC do not use bins.",
    )
    parser.add_argument("--source_seeds", default="1,2,3")
    parser.add_argument(
        "--stream_seeds",
        default="42",
        help=(
            "Paired target-time RNG control. Fixed target loaders are not "
            "reshuffled, so stream seeds are not independent repetitions."
        ),
    )
    parser.add_argument("--corruption_fraction", type=float, default=0.5)
    parser.add_argument(
        "--corruption_seed",
        type=int,
        default=None,
        help=(
            "Fixed seed for the corruption mask and transform. When omitted, "
            "legacy behavior derives it from source_seed. Set it explicitly "
            "to evaluate every source checkpoint on the same corruptions."
        ),
    )
    parser.add_argument(
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument(
        "--fisher_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "benchmark_fisher"),
        help="Cache for source-checkpoint-tied EATA diagonal Fisher states.",
    )
    parser.add_argument(
        "--eata_fisher_samples",
        type=int,
        default=2000,
        help="Maximum source-training samples used for the EATA empirical Fisher.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "reviewer_rerun" / "controlled_safety"),
    )
    parser.add_argument(
        "--finalize_only",
        action="store_true",
        help=(
            "Do not execute missing GPU jobs. Verify that every requested key "
            "has a signed matching sample record, then rebuild aggregates."
        ),
    )
    parser.add_argument(
        "--defer_artifacts",
        action="store_true",
        help=(
            "Write resumable per-job records and raw summaries but defer global "
            "risk/aggregate reconstruction to a later --finalize_only call."
        ),
    )
    args = parser.parse_args()
    args.overrides = parse_overrides(args.override)
    datasets = parse_list(args.datasets)
    methods = parse_list(args.methods)
    if "EATA" in methods and args.registry != "benchmark":
        parser.error(
            "EATA safety requires --registry benchmark so a validated source "
            "Fisher diagonal is injected; fisher-disabled EATA is forbidden."
        )
    if args.eata_fisher_samples < 1:
        parser.error("--eata_fisher_samples must be positive")
    if args.calibration_bins < 2:
        parser.error("--calibration_bins must be at least two")
    variants = parse_list(args.variants)
    unknown_variants = sorted(set(variants) - set(CONTROLLED_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown controlled variants: {unknown_variants}")
    args.scenario_map = parse_scenarios(args.scenarios, datasets)
    args.flowwise_source_profiles = load_flowwise_source_profiles(
        args.flowwise_source_profile_json, datasets
    )
    args.source_checkpoint_references = load_source_checkpoint_references(
        args.source_reference_csv
    )
    try:
        args.flow_profile_overrides = load_paper_flow_profiles(
            args.flow_profile_json, datasets
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    requested_corruptions = parse_list(args.corruptions)
    requested_severities = parse_list(args.severities)
    selected_registry = (
        PHYSICAL_CORRUPTION_REGISTRY
        if args.physical_protocol
        else CORRUPTION_REGISTRY
    )
    unknown_corruptions = sorted(set(requested_corruptions) - set(selected_registry))
    if unknown_corruptions:
        parser.error(f"Unknown corruptions: {unknown_corruptions}")
    if args.physical_protocol:
        for corruption in requested_corruptions:
            for severity in requested_severities:
                try:
                    resolve_severity(corruption, severity)
                except (KeyError, ValueError) as exc:
                    parser.error(str(exc))
    output_dir = ensure_dir(args.output_dir)
    summary_path = output_dir / "summary_raw.csv"
    records_dir = ensure_dir(output_dir / "sample_records")
    failure_path = output_dir / "failures.csv"
    stale_summary_path = output_dir / "stale_summary_rows.csv"
    loaded_summary_rows = read_csv_records(summary_path)
    failure_rows = read_csv_records(failure_path)
    stale_summary_rows = read_csv_records(stale_summary_path)
    malformed_summary_rows = []
    summary_by_key = {}
    for row in loaded_summary_rows:
        key = try_safety_job_key(row)
        if key is None:
            malformed_summary_rows.append(row)
            continue
        summary_by_key[key] = row
    for row in malformed_summary_rows:
        archived = dict(row)
        archived["stale_reason"] = "malformed_summary_key"
        archived["archived_at_unix"] = time.time()
        stale_summary_rows.append(archived)
    summary_rows = list(summary_by_key.values())
    summary_keys = set(summary_by_key)
    failure_rows = [
        row
        for row in failure_rows
        if try_safety_job_key(row) not in summary_keys
    ]
    completed = set()
    requested_keys = set()

    for dataset in datasets:
        for method in methods:
            method_variants = variants if method == "DuSafe" else ["full"]
            for variant in method_variants:
                for corruption in requested_corruptions:
                    if corruption not in selected_registry:
                        raise ValueError(f"Unknown corruption: {corruption}")
                    for severity in requested_severities:
                        for source_seed in parse_list(args.source_seeds, int):
                            for stream_seed in parse_list(
                                args.stream_seeds, int
                            ):
                                scenario = "->".join(args.scenario_map[dataset])
                                key = (
                                    dataset, scenario, method, variant,
                                    corruption, severity, source_seed,
                                    stream_seed,
                                    int(
                                        source_seed
                                        if args.corruption_seed is None
                                        else args.corruption_seed
                                    ),
                                )
                                requested_keys.add(key)
                                expected_signature = safety_protocol_signature(
                                    args, dataset, method, variant
                                )
                                record_path = records_dir / safety_record_name(key)
                                old = summary_by_key.get(key)
                                observed_signature = (
                                    str(old.get("protocol_signature", ""))
                                    if old is not None
                                    else ""
                                )
                                if args.finalize_only:
                                    is_complete = bool(observed_signature) and (
                                        sample_record_matches(
                                            record_path,
                                            key,
                                            observed_signature,
                                        )
                                    )
                                else:
                                    is_complete = (
                                        observed_signature == expected_signature
                                        and sample_record_matches(
                                            record_path,
                                            key,
                                            expected_signature,
                                        )
                                    )
                                if is_complete:
                                    completed.add(key)
                                    continue
                                if args.finalize_only:
                                    continue
                                if old is not None:
                                    archived = dict(old)
                                    archived["stale_reason"] = (
                                        "protocol_signature_or_sample_record_mismatch"
                                    )
                                    archived["expected_protocol_signature"] = (
                                        expected_signature
                                    )
                                    archived["archived_at_unix"] = time.time()
                                    stale_summary_rows.append(archived)
                                    summary_by_key.pop(key, None)
                                    summary_rows = list(summary_by_key.values())
                                    atomic_write_csv(
                                        pd.DataFrame(summary_rows),
                                        summary_path,
                                        index=False,
                                    )
                                    atomic_write_csv(
                                        pd.DataFrame(stale_summary_rows),
                                        stale_summary_path,
                                        index=False,
                                    )
                                print(f"[Safety] {key}", flush=True)
                                try:
                                    summary, records, _curves = run_job(
                                        args, dataset, method, variant, corruption,
                                        severity, source_seed, stream_seed,
                                    )
                                except Exception as exc:
                                    error_text = str(exc)
                                    lowered = error_text.lower()
                                    is_oom = (
                                        isinstance(exc, torch.cuda.OutOfMemoryError)
                                        or "out of memory" in lowered
                                        or "cuda error" in lowered
                                        and "memory" in lowered
                                    )
                                    failure_rows = [
                                        row
                                        for row in failure_rows
                                        if try_safety_job_key(row) != key
                                    ]
                                    failure_rows.append(
                                        {
                                            "dataset": dataset,
                                            "scenario": scenario,
                                            "method": method,
                                            "variant": variant,
                                            "corruption": corruption,
                                            "severity": severity,
                                            "source_seed": int(source_seed),
                                            "stream_seed": int(stream_seed),
                                            "corruption_seed": int(
                                                source_seed
                                                if args.corruption_seed is None
                                                else args.corruption_seed
                                            ),
                                            "protocol_signature": expected_signature,
                                            "status": "oom" if is_oom else "error",
                                            "error_type": type(exc).__name__,
                                            "error": error_text,
                                            "traceback": traceback.format_exc(),
                                            "recorded_at_unix": time.time(),
                                        }
                                    )
                                    atomic_write_csv(
                                        pd.DataFrame(failure_rows),
                                        failure_path,
                                        index=False,
                                    )
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    print(
                                        f"[Safety failure] {key}: "
                                        f"{failure_rows[-1]['status']} {error_text}",
                                        flush=True,
                                    )
                                    continue
                                summary_by_key[key] = summary
                                summary_rows = list(summary_by_key.values())
                                completed.add(key)
                                failure_rows = [
                                    row
                                    for row in failure_rows
                                    if try_safety_job_key(row) != key
                                ]
                                # Sample-level evidence is partitioned per job.
                                atomic_write_csv(
                                    records,
                                    record_path,
                                    index=False,
                                )
                                atomic_write_csv(
                                    pd.DataFrame(summary_rows),
                                    summary_path,
                                    index=False,
                                )

    summary_df = pd.DataFrame(summary_rows)
    for role_metric in (
        "eligible_coverage_overall",
        "eligible_active_batch_fraction",
        "eligible_active_batch_count",
        "active_batch_clean_f1",
        "active_batch_corrupted_f1",
        "active_batch_admitted_accuracy",
        "active_batch_incorrect_admission_rate",
        "active_batch_unsafe_admission_rate",
    ):
        if role_metric not in summary_df.columns:
            summary_df[role_metric] = float("nan")

    # ``--finalize_only`` can consume a summary written by an older runner
    # that predates the representative-panel aliases.  Reconstruct the
    # aliases before building the aggregate specification so finalization is
    # a read-only operation rather than a schema migration failure.  The
    # legacy fallbacks are intentionally conservative: ``f1`` is only used
    # when the old row has no clean/corrupted split, and the broad unsafe
    # admission rate remains unavailable unless its canonical source exists.
    if "clean_f1" not in summary_df.columns:
        summary_df["clean_f1"] = summary_df.get(
            "clean_post_update_macro_f1", summary_df.get("f1", float("nan"))
        )
    if "corrupted_f1" not in summary_df.columns:
        summary_df["corrupted_f1"] = summary_df.get(
            "corrupted_post_update_macro_f1", summary_df.get("f1", float("nan"))
        )
    if "admitted_accuracy" not in summary_df.columns:
        summary_df["admitted_accuracy"] = summary_df.get(
            "admitted_pseudo_label_accuracy",
            summary_df.get("accepted_pseudo_label_accuracy", float("nan")),
        )
    if "incorrect_admission_rate" not in summary_df.columns:
        summary_df["incorrect_admission_rate"] = (
            1.0 - pd.to_numeric(summary_df["admitted_accuracy"], errors="coerce")
        )
    if "incorrect_admission_label_rate" not in summary_df.columns:
        summary_df["incorrect_admission_label_rate"] = summary_df[
            "incorrect_admission_rate"
        ]
    if "unsafe_admission_rate" not in summary_df.columns:
        summary_df["unsafe_admission_rate"] = summary_df.get(
            "admission_unsafe_rate", float("nan")
        )
    aggregate_columns = [
        "dataset", "scenario", "method", "variant", "corruption",
        "severity", "severity_name", "normalized_severity", "corruption_seed",
        "f1_mean", "f1_std",
        "clean_f1_mean", "corrupted_f1_mean",
        "admitted_accuracy_mean", "incorrect_admission_rate_mean",
        "incorrect_admission_label_rate_mean",
        "eligible_coverage_overall_mean",
        "eligible_active_batch_fraction_mean",
        "eligible_active_batch_count_mean",
        "active_batch_clean_f1_mean",
        "active_batch_corrupted_f1_mean",
        "active_batch_admitted_accuracy_mean",
        "active_batch_incorrect_admission_rate_mean",
        "active_batch_unsafe_admission_rate_mean",
        "coverage_mean", "accepted_accuracy_mean",
        "corruption_recall_mean", "clean_correct_false_rejection_mean",
        "admission_corruption_recall_mean",
        "admission_clean_correct_false_rejection_mean",
        "admitted_corruption_rate_mean", "unsafe_update_rate_mean",
    ]
    if summary_df.empty:
        aggregate = pd.DataFrame(columns=aggregate_columns)
    else:
        aggregate_group_columns = [
            "dataset", "scenario", "method", "variant", "corruption",
            "severity", "corruption_seed",
        ]
        for optional_group in ("severity_name", "normalized_severity"):
            if optional_group in summary_df.columns:
                aggregate_group_columns.append(optional_group)
        aggregate_spec = {
            "f1_mean": ("f1", "mean"),
            "f1_std": ("f1", "std"),
            "clean_f1_mean": ("clean_f1", "mean"),
            "corrupted_f1_mean": ("corrupted_f1", "mean"),
            "admitted_accuracy_mean": ("admitted_accuracy", "mean"),
            "incorrect_admission_rate_mean": (
                "incorrect_admission_rate", "mean"
            ),
            "incorrect_admission_label_rate_mean": (
                "incorrect_admission_label_rate", "mean"
            ),
            "eligible_coverage_overall_mean": (
                "eligible_coverage_overall", "mean"
            ),
            "eligible_active_batch_fraction_mean": (
                "eligible_active_batch_fraction", "mean"
            ),
            "eligible_active_batch_count_mean": (
                "eligible_active_batch_count", "mean"
            ),
            "active_batch_clean_f1_mean": (
                "active_batch_clean_f1", "mean"
            ),
            "active_batch_corrupted_f1_mean": (
                "active_batch_corrupted_f1", "mean"
            ),
            "active_batch_admitted_accuracy_mean": (
                "active_batch_admitted_accuracy", "mean"
            ),
            "active_batch_incorrect_admission_rate_mean": (
                "active_batch_incorrect_admission_rate", "mean"
            ),
            "active_batch_unsafe_admission_rate_mean": (
                "active_batch_unsafe_admission_rate", "mean"
            ),
            "coverage_mean": ("coverage", "mean"),
            "accepted_accuracy_mean": (
                "accepted_pseudo_label_accuracy", "mean"
            ),
            "corruption_recall_mean": (
                "corruption_rejection_recall", "mean"
            ),
            "clean_correct_false_rejection_mean": (
                "clean_correct_false_rejection_rate", "mean"
            ),
            "admission_corruption_recall_mean": (
                "admission_corruption_rejection_recall", "mean"
            ),
            "admission_clean_correct_false_rejection_mean": (
                "admission_clean_correct_false_rejection_rate", "mean"
            ),
            "admitted_corruption_rate_mean": (
                "admitted_corruption_rate", "mean"
            ),
            "unsafe_update_rate_mean": ("unsafe_update_rate", "mean"),
        }
        for probability_metric in (
            "post_update_nll",
            "post_update_brier",
            "post_update_ece",
            "post_update_classwise_ece",
            "post_update_aurc",
            "post_update_eaurc",
            "pre_final_update_nll",
            "pre_final_update_brier",
            "pre_final_update_ece",
            "pre_final_update_classwise_ece",
            "pre_final_update_aurc",
            "pre_final_update_eaurc",
            "corrupted_post_update_macro_f1",
            "corrupted_post_update_nll",
            "corrupted_post_update_brier",
            "corrupted_post_update_ece",
            "corrupted_post_update_classwise_ece",
            "corrupted_post_update_aurc",
            "corrupted_post_update_eaurc",
            "clean_post_update_macro_f1",
            "clean_post_update_nll",
            "clean_post_update_brier",
            "clean_post_update_ece",
            "clean_post_update_classwise_ece",
            "clean_post_update_aurc",
            "clean_post_update_eaurc",
        ):
            if probability_metric in summary_df.columns:
                aggregate_spec[f"{probability_metric}_mean"] = (
                    probability_metric,
                    "mean",
                )
        aggregate = (
            summary_df.groupby(aggregate_group_columns)
            .agg(**aggregate_spec)
            .reset_index()
        )
    if not args.defer_artifacts:
        atomic_write_csv(
            aggregate, output_dir / "summary_aggregate.csv", index=False
        )
    active_protocol_signatures = {
        key: str(row.get("protocol_signature", ""))
        for key, row in summary_by_key.items()
        if str(row.get("protocol_signature", ""))
    }
    if args.defer_artifacts:
        native_curve_df = pd.DataFrame()
        native_aurc = pd.DataFrame()
        predictive_curve_df = pd.DataFrame()
        predictive_aurc = pd.DataFrame()
    else:
        native_curve_df, native_aurc = write_native_risk_artifacts(
            records_dir,
            output_dir,
            expected_signatures=active_protocol_signatures,
        )
        predictive_curve_df, predictive_aurc = (
            write_common_predictive_risk_artifacts(
                records_dir,
                output_dir,
                expected_signatures=active_protocol_signatures,
            )
        )
    failure_columns = [
        "dataset", "scenario", "method", "variant", "corruption",
        "severity", "source_seed", "stream_seed", "corruption_seed",
        "protocol_signature",
        "status", "error_type", "error", "traceback",
        "recorded_at_unix",
    ]
    # Keep an explicit empty failure artifact when every job succeeds.  This
    # makes the absence of failures auditable instead of conflating a missing
    # file with an unrecorded or interrupted run.
    atomic_write_csv(
        pd.DataFrame(failure_rows, columns=failure_columns),
        failure_path,
        index=False,
    )
    atomic_write_csv(
        pd.DataFrame(summary_rows), summary_path, index=False
    )
    atomic_write_csv(
        pd.DataFrame(stale_summary_rows), stale_summary_path, index=False
    )
    missing_requested_keys = sorted(requested_keys - completed)
    manifest = {
        "production_code_sha256": production_code_sha256(),
        "annotation": "sample-level corruption mask generated independently from model outputs",
        "online_target_labels_used_by_dusafe": False,
        "offline_hyperparameter_provenance": (
            "not enforced by code; declare the dataset-level selection split "
            "when reporting these results"
        ),
        "result_scope": (
            "controlled safety diagnosis using the dataset-level DuSafe configuration"
        ),
        "corruption_fraction": args.corruption_fraction,
        "corruption_seed": args.corruption_seed,
        "physical_protocol": bool(args.physical_protocol),
        "physical_severity_policy": (
            "pre_registered_s0_to_s6_normalized_with_physical_parameters"
            if args.physical_protocol
            else "legacy_categorical"
        ),
        "probability_record_schema": PROBABILITY_RECORD_SCHEMA,
        "probability_recording": (
            "complete pre-final and post-update logits/probabilities for every class"
        ),
        "calibration_bins": int(args.calibration_bins),
        "standard_probability_metrics": [
            "true_label_nll",
            "summed_multiclass_brier",
            "top_label_ece",
            "macro_classwise_ece",
            "samplewise_error_aurc",
            "excess_aurc",
        ],
        "scenarios": args.scenario_map,
        "paper_flow_profile_json": str(args.flow_profile_json),
        "flowwise_source_profile_json": str(args.flowwise_source_profile_json),
        "source_reference_csv": str(args.source_reference_csv),
        "flowwise_source_profile_applied": bool(args.flowwise_source_profiles),
        "source_checkpoint_identity_enforced": bool(
            args.source_checkpoint_references
        ),
        "paper_flow_profile_overrides": {
            f"{dataset}:{scenario}": dict(values)
            for (dataset, scenario), values in getattr(
                args, "flow_profile_overrides", {}
            ).items()
        },
        "variants": variants,
        "methods_requested": methods,
        "methods_completed": sorted(
            {str(row.get("method")) for row in summary_rows}
        ),
        "source_seeds": parse_list(args.source_seeds, int),
        "stream_seeds": parse_list(args.stream_seeds, int),
        "required_safety_metrics": list(REQUIRED_SAFETY_METRICS),
        "metric_grain": (
            "coverage, accepted pseudo-label accuracy, rejection and unsafe-update "
            "rates are computed at the common inner-step-by-sample grain"
        ),
        "representative_panel_metrics": {
            "clean_f1": "clean_post_update_macro_f1",
            "corrupted_f1": "corrupted_post_update_macro_f1",
            "admitted_accuracy": "admitted_pseudo_label_accuracy",
            "incorrect_admission_rate": (
                "1 - admitted_accuracy = admitted wrong pseudo-label only, "
                "conditional on admission"
            ),
            "unsafe_admission_rate": (
                "admission_unsafe_rate = admitted wrong pseudo-label OR "
                "annotated corrupted sample, conditional on admission"
            ),
        },
        "evidence_roles": {
            "A_confidence_accept_all_vs_admitted": {
                "source": "controlled safety rows",
                "reported": [
                    "clean_f1",
                    "corrupted_f1",
                    "coverage",
                    "admitted_accuracy",
                    "incorrect_admission_rate",
                    ],
                    "future_horizon": 5,
                    "future_horizon_source": "run_representative_causal_ablation",
                    "state_isolation_audit_source": "run_full_no_ssaw_horizon_queue",
                },
            "C_active_batch_conditional": {
                "eligible_coverage_threshold": 0.25,
                "selection_must_be_label_free": True,
                "overall_coverage_must_be_reported_alongside": True,
                "conditional_columns": [
                    "active_batch_clean_f1",
                    "active_batch_corrupted_f1",
                    "active_batch_admitted_accuracy",
                    "active_batch_incorrect_admission_rate",
                    "active_batch_unsafe_admission_rate",
                ],
                "not_a_clean_f1_primary_ladder": True,
            },
            "two_by_two_grid": "audit_only",
        },
        "sample_records": "one CSV per job under sample_records/",
        "failure_records": "failures.csv; failed/OOM jobs are not assigned metric values",
        "method_policy": (
            "DuSafe uses the declared per-flow TTA profile and, when supplied, "
            "the declared per-flow source-training profile; paired variants "
            "share an enforced source checkpoint identity"
        ),
        "per_job_runtime_hparams": (
            "summary_raw.csv and each sample record contain the actual "
            "runtime_hparams used by that independently resumable job"
        ),
        "no_adap_policy": (
            "NoAdap is source-only and all-rejects update admission by design; "
            "it is a reference bound, not a fair adaptive competitor and must "
            "not be ranked against updating methods by rejection recall alone"
        ),
        "adaptive_comparison_policy": (
            "Compare updating methods jointly on every required safety metric; "
            "do not select a winner from corruption rejection recall alone"
        ),
        "algorithm_registry": args.registry,
        "effective_method_registry": {
            method: effective_method_registry(args, method)
            for method in methods
        },
        "method_provenance": (
            __import__("configs.benchmark_baselines", fromlist=["PROVENANCE"]).PROVENANCE
            if args.registry == "benchmark"
            else {}
        ),
        "risk_coverage_score": (
            "maximum of each enabled DuSafe gate score divided by its fixed-source "
            "threshold; score <= 1 exactly reproduces online admission; benchmark "
            "adapters expose binary masks only, so their risk coverage/AURC is "
            "not available rather than fabricated; artifacts are rebuilt from "
            "partitioned sample records on every successful invocation"
        ),
        "native_risk_curve_rows": int(len(native_curve_df)),
        "native_aurc_rows": int(len(native_aurc)),
        "cross_method_predictive_risk_coverage": {
            "status": "available_for_methods_with_complete_sample_records",
            "policies": {
                "common_pre_update_top1_nll": {
                    "score": "-log(pre_final_update_confidence)",
                    "outcome": "pre_final_update_correct",
                    "semantics": (
                        "model state after prior online batches/inner updates "
                        "and before the current batch's final update"
                    ),
                },
                "common_post_update_top1_nll": {
                    "score": "-log(confidence)",
                    "outcome": "correct",
                    "semantics": "final predictions aligned with reported F1",
                },
            },
            "cross_method_primary_policy": "common_post_update_top1_nll",
            "posthoc_only": True,
            "online_admission_policy": False,
            "aurc_origin_convention": (
                "coverage zero inherits the first measured selective risk; "
                "compare only artifacts generated by this protocol"
            ),
            "curve_rows": int(len(predictive_curve_df)),
            "aurc_rows": int(len(predictive_aurc)),
            "outputs": [
                "predictive_risk_coverage_raw.csv",
                "predictive_risk_coverage_aggregate.csv",
                "predictive_aurc_per_source_seed.csv",
                "predictive_aurc_aggregate.csv",
            ],
        },
        "eata_fisher": {
            "required": "EATA jobs are forbidden without a validated source Fisher",
            "protocol": EATA_FISHER_PROTOCOL,
            "source_training_data_only": True,
            "max_samples": int(args.eata_fisher_samples),
            "cache_dir": str(Path(args.fisher_cache_dir)),
        },
        "failure_count": len(failure_rows),
        "failure_statuses": sorted(
            {str(row.get("status")) for row in failure_rows}
        ),
        "requested_job_count": int(len(requested_keys)),
        "requested_completed_job_count": int(
            len(requested_keys.intersection(completed))
        ),
        "requested_missing_job_count": int(len(missing_requested_keys)),
        "requested_missing_job_keys": [list(key) for key in missing_requested_keys],
        "protocol_signature_version": SAFETY_PROTOCOL_VERSION,
        "signed_sample_record_required_for_completion": True,
        "finalize_only": bool(args.finalize_only),
        "global_artifacts_deferred": bool(args.defer_artifacts),
        "malformed_summary_rows_archived": int(len(malformed_summary_rows)),
        "stale_summary_archive": str(stale_summary_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {output_dir}")
    return 1 if missing_requested_keys else 0


if __name__ == "__main__":
    raise SystemExit(main())
