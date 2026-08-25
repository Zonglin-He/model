"""Trace where the SSAW certificate loses useful signal.

This is an offline diagnostic runner.  Target labels are read only after each
admission decision to measure discrimination; they never enter DuSafe's
forward or update path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.ssaw_components import (  # noqa: E402
    SSAWEveryStepCertificateAdmissionRunner,
)
from scripts.run_optuna_stepwise import (  # noqa: E402
    parse_csv,
    scenario_label,
    scenario_pairs,
)
from scripts.run_ssaw_internal_ablation import load_json  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    move_data_to_device,
)


FORMAL_MANIFESTS = {
    "EEG": ROOT
    / "results"
    / "ablation"
    / "stepmetrics_v2_post_eeg_full"
    / "manifest.json",
    "HAR": ROOT
    / "results"
    / "ablation"
    / "stepmetrics_v2_post_har_full"
    / "manifest.json",
    "FD": ROOT
    / "results"
    / "ablation"
    / "stepmetrics_v2_post_fd_full"
    / "manifest.json",
}


def _primary(data):
    return data[0] if isinstance(data, (tuple, list)) else data


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _binary_auc(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


class PipelineAccumulator:
    """Accumulate exact step-by-sample counts without retaining model tensors."""

    MASK_NAMES = (
        "base_admission",
        "near_boundary",
        "adapting_failure",
        "source_failure",
        "physical_failure",
        "veto_base_near",
        "veto_pre_kl",
        "veto",
        "exactly_one_raw_signal",
        "rescue_stability",
        "rescue_pre_nll",
        "rescue_raw_nll_pass",
        "rescue",
        "final_admission",
    )

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.decisions = 0
        self.correct = 0
        self.counts = defaultdict(int)
        self.correct_counts = defaultdict(int)
        self.wrong_counts = defaultdict(int)
        self.sums = defaultdict(float)
        self.value_counts = defaultdict(int)
        self.risk_labels: list[int] = []
        self.kl_scores: list[float] = []
        self.flip_scores: list[float] = []
        self.source_failure_scores: list[float] = []
        self.class_counts = {
            name: np.zeros(self.num_classes, dtype=np.int64)
            for name in ("all", "base_admission", "veto", "rescue", "final_admission")
        }

    def add_values(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().float()
        finite = torch.isfinite(values)
        if finite.any():
            self.sums[name] += float(values[finite].sum().item())
            self.value_counts[name] += int(finite.sum().item())

    def add_mask(
        self,
        name: str,
        mask: torch.Tensor,
        correct: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> None:
        mask = mask.detach().bool()
        count = int(mask.sum().item())
        self.counts[name] += count
        self.correct_counts[name] += int((mask & correct).sum().item())
        self.wrong_counts[name] += int((mask & (~correct)).sum().item())
        if name in self.class_counts and count:
            bincount = torch.bincount(
                pseudo_labels[mask].detach().cpu(), minlength=self.num_classes
            ).numpy()
            self.class_counts[name] += bincount

    def add_step(self, model_inputs, labels: torch.Tensor, adapter) -> None:
        state = adapter._ssaw_admission_state
        if state is None:
            raise RuntimeError("SSAW admission state was not populated")

        raw_inputs = _primary(model_inputs["data"])
        raw_logits = adapter.ssaw.last_reference_logits
        candidate_logits = adapter.ssaw.last_candidate_logits
        candidate_inputs = adapter.ssaw.last_candidate_inputs
        if raw_logits is None or candidate_logits is None or candidate_inputs is None:
            raise RuntimeError("SSAW candidate tensors were not retained")

        pseudo_labels = raw_logits.argmax(dim=1)
        correct = pseudo_labels.eq(labels)
        raw_log_probabilities = raw_logits.log_softmax(dim=1)
        raw_probabilities = raw_log_probabilities.exp()
        raw_nll = -raw_log_probabilities.gather(
            1, pseudo_labels[:, None]
        ).squeeze(1)
        candidate_log_probabilities = candidate_logits.log_softmax(dim=2)
        label_index = pseudo_labels[:, None, None].expand(
            -1, candidate_logits.size(1), 1
        )
        candidate_nll = -candidate_log_probabilities.gather(
            2, label_index
        ).squeeze(2)
        stress_nll = candidate_nll.mean(dim=1)
        candidate_labels = candidate_logits.argmax(dim=2)
        candidate_agreement = candidate_labels.eq(pseudo_labels[:, None])
        agreement_fraction = candidate_agreement.float().mean(dim=1)
        prediction_kl = (
            raw_probabilities[:, None]
            * (
                raw_log_probabilities[:, None]
                - candidate_log_probabilities
            )
        ).sum(dim=2).clamp_min(0.0).mean(dim=1)

        confidence_mask = state["confidence_mask"].bool()
        semantic_mask = state["semantic_mask"].bool()
        base_admission = confidence_mask & semantic_mask
        label_agreement = agreement_fraction.ge(
            adapter.ssaw_admission_min_agreement
        )
        source_agreement_fraction = state["source_agreement_fraction"]
        source_label_agreement = source_agreement_fraction.ge(
            adapter.ssaw_admission_min_agreement
        )
        near_boundary = raw_nll.ge(
            adapter.confidence_nll_threshold * adapter.ssaw_veto_nll_ratio
        )
        adapting_failure = ~label_agreement
        source_failure = ~source_label_agreement
        physical_failure = (
            adapting_failure & source_failure
            if adapter.require_joint_veto_failure
            else adapting_failure | source_failure
        )
        veto_base_near = base_admission & near_boundary
        veto_pre_kl = veto_base_near & physical_failure
        veto = veto_pre_kl & prediction_kl.gt(
            adapter.ssaw_veto_kl_threshold
        )

        exactly_one = confidence_mask ^ semantic_mask
        rescue_stability = label_agreement & source_label_agreement & prediction_kl.le(
            adapter.ssaw_rescue_kl_threshold
        )
        rescue_limit = (
            adapter.confidence_nll_threshold
            * adapter.ssaw_rescue_nll_multiplier
        )
        rescue_pre_nll = exactly_one & rescue_stability
        rescue_raw_nll_pass = rescue_pre_nll & raw_nll.le(rescue_limit)
        rescue = (
            rescue_raw_nll_pass
            & stress_nll.le(rescue_limit)
        )
        final_admission = (base_admission & (~veto)) | rescue

        recorded_veto = state["veto_mask"].bool()
        recorded_rescue = state["rescue_mask"].bool()
        recorded_admission = state["admission_mask"].bool()
        if not (
            torch.equal(veto, recorded_veto)
            and torch.equal(rescue, recorded_rescue)
            and torch.equal(final_admission, recorded_admission)
        ):
            raise RuntimeError("Probe reconstruction differs from runner decisions")

        batch_size = int(labels.numel())
        self.decisions += batch_size
        self.correct += int(correct.sum().item())
        self.add_mask("all", torch.ones_like(correct), correct, pseudo_labels)
        masks = {
            "base_admission": base_admission,
            "near_boundary": near_boundary,
            "adapting_failure": adapting_failure,
            "source_failure": source_failure,
            "physical_failure": physical_failure,
            "veto_base_near": veto_base_near,
            "veto_pre_kl": veto_pre_kl,
            "veto": veto,
            "exactly_one_raw_signal": exactly_one,
            "rescue_stability": rescue_stability,
            "rescue_pre_nll": rescue_pre_nll,
            "rescue_raw_nll_pass": rescue_raw_nll_pass,
            "rescue": rescue,
            "final_admission": final_admission,
        }
        for name, mask in masks.items():
            self.add_mask(name, mask, correct, pseudo_labels)

        input_delta = candidate_inputs - raw_inputs[:, None]
        input_relative_rms = input_delta.flatten(2).pow(2).mean(dim=2).sqrt() / (
            raw_inputs[:, None].flatten(2).pow(2).mean(dim=2).sqrt().clamp_min(1e-8)
        )
        flip_fraction = (~candidate_agreement).float().mean(dim=1)
        confidence_gradient_proxy = 1.0 - raw_probabilities.gather(
            1, pseudo_labels[:, None]
        ).squeeze(1)

        self.add_values("input_relative_rms", input_relative_rms)
        self.add_values("prediction_kl", prediction_kl)
        self.add_values("candidate_flip_fraction", flip_fraction)
        self.add_values("source_agreement_fraction", source_agreement_fraction)
        self.add_values("prediction_kl_correct", prediction_kl[correct])
        self.add_values("prediction_kl_wrong", prediction_kl[~correct])
        self.add_values("candidate_flip_fraction_correct", flip_fraction[correct])
        self.add_values("candidate_flip_fraction_wrong", flip_fraction[~correct])
        self.add_values("raw_nll_all", raw_nll)
        self.add_values("raw_gradient_proxy_all", confidence_gradient_proxy)
        for name, mask in (
            ("base", base_admission),
            ("veto", veto),
            ("rescue", rescue),
            ("final", final_admission),
        ):
            self.add_values(f"raw_nll_{name}", raw_nll[mask])
            self.add_values(
                f"raw_gradient_proxy_{name}", confidence_gradient_proxy[mask]
            )

        self.risk_labels.extend((~correct).detach().cpu().int().tolist())
        self.kl_scores.extend(prediction_kl.detach().cpu().tolist())
        self.flip_scores.extend(flip_fraction.detach().cpu().tolist())
        self.source_failure_scores.extend(
            (1.0 - source_agreement_fraction).detach().cpu().tolist()
        )

    def summary(self) -> dict[str, float]:
        decisions = self.decisions
        wrong = decisions - self.correct
        result = {
            "step_decisions": decisions,
            "raw_pseudo_label_accuracy": _safe_div(self.correct, decisions),
            "raw_wrong_count": wrong,
            "kl_auc_for_wrong_pseudo_label": _binary_auc(
                self.risk_labels, self.kl_scores
            ),
            "flip_auc_for_wrong_pseudo_label": _binary_auc(
                self.risk_labels, self.flip_scores
            ),
            "source_failure_auc_for_wrong_pseudo_label": _binary_auc(
                self.risk_labels, self.source_failure_scores
            ),
        }
        for name in self.MASK_NAMES:
            count = self.counts[name]
            result[f"{name}_rate"] = _safe_div(count, decisions)
            result[f"{name}_correct_fraction"] = _safe_div(
                self.correct_counts[name], count
            )
            result[f"{name}_wrong_fraction"] = _safe_div(
                self.wrong_counts[name], count
            )
        for name, total in self.sums.items():
            result[f"{name}_mean"] = _safe_div(total, self.value_counts[name])

        helpful = self.correct_counts["rescue"] + self.wrong_counts["veto"]
        harmful = self.wrong_counts["rescue"] + self.correct_counts["veto"]
        interventions = self.counts["rescue"] + self.counts["veto"]
        result.update(
            {
                "helpful_decision_changes": helpful,
                "harmful_decision_changes": harmful,
                "net_helpful_decision_changes": helpful - harmful,
                "intervention_precision": _safe_div(helpful, interventions),
                "wrong_veto_recall": _safe_div(
                    self.wrong_counts["veto"], wrong
                ),
                "correct_rescue_recall": _safe_div(
                    self.correct_counts["rescue"], self.correct
                ),
            }
        )
        return result

    def class_rows(self, metadata: dict) -> list[dict]:
        rows = []
        for mask_name, counts in self.class_counts.items():
            total = int(counts.sum())
            for class_index, count in enumerate(counts.tolist()):
                rows.append(
                    {
                        **metadata,
                        "mask": mask_name,
                        "pseudo_class": class_index,
                        "count": int(count),
                        "fraction_within_mask": _safe_div(count, total),
                    }
                )
        return rows


def _load_formal_config(dataset: str, manifest_path: Path | None) -> dict:
    path = FORMAL_MANIFESTS[dataset] if manifest_path is None else manifest_path
    manifest = load_json(path)
    return dict(manifest["effective_tta_configs"][dataset])


def run_job(args, dataset: str, scenario: tuple[str, str], test_time_seed: int):
    state = load_json(args.tuning_dir / dataset / "state.json")
    source_seed = int(state.get("signature", {}).get("source_seed", 1))
    tta_config = _load_formal_config(dataset, args.manifest)
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="ssaw_pipeline_diagnostic",
        seed=test_time_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    adapter = source_model = None
    try:
        trainer.get_tta_model_class = lambda: SSAWEveryStepCertificateAdmissionRunner
        trainer.source_hparams.update(dict(state["source_config"]))
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model = create_tta_model(
            trainer,
            scenario[0],
            scenario[1],
            run_seed=test_time_seed,
        )
        accumulator = PipelineAccumulator(trainer.dataset_configs.num_classes)
        for data, labels, target_indices in trainer.trg_whole_dl:
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            model_inputs = {
                "data": data,
                "labels": labels,
                "meta": {"trg_idx": torch.as_tensor(target_indices).view(-1).tolist()},
            }
            for _ in range(adapter.steps):
                adapter.forward_and_adapt(
                    model_inputs,
                    adapter.model,
                    adapter.optimizer,
                    target_indices,
                )
                accumulator.add_step(model_inputs, labels, adapter)
        metadata = {
            "dataset": dataset,
            "scenario": scenario_label(scenario),
            "source_seed": source_seed,
            "test_time_seed": int(test_time_seed),
        }
        return {**metadata, **accumulator.summary()}, accumulator.class_rows(metadata)
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional single source->target pair; valid only for one dataset.",
    )
    parser.add_argument("--test-time-seeds", default="1")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4",
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest override; valid only for one dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "ssaw_pipeline_v1",
    )
    args = parser.parse_args(argv)
    datasets = [name.upper() for name in parse_csv(args.datasets)]
    seeds = parse_csv(args.test_time_seeds, int)
    if (args.scenario or args.manifest) and len(datasets) != 1:
        parser.error("--scenario/--manifest requires exactly one dataset")

    output_dir = ensure_dir(args.output_dir)
    summaries = []
    class_rows = []
    for dataset in datasets:
        scenarios = scenario_pairs(dataset)
        if args.scenario:
            pieces = args.scenario.replace("->", ",").split(",")
            if len(pieces) != 2:
                parser.error("--scenario must look like 0->11")
            requested = (pieces[0].strip(), pieces[1].strip())
            if requested not in scenarios:
                parser.error(f"Unknown {dataset} scenario: {args.scenario}")
            scenarios = [requested]
        for scenario in scenarios:
            for seed in seeds:
                print(
                    f"[SSAW pipeline] {dataset} {scenario_label(scenario)} seed={seed}",
                    flush=True,
                )
                summary, per_class = run_job(args, dataset, scenario, seed)
                summaries.append(summary)
                class_rows.extend(per_class)
                atomic_write_csv(
                    pd.DataFrame(summaries), output_dir / "summary.csv", index=False
                )
                atomic_write_csv(
                    pd.DataFrame(class_rows), output_dir / "class_counts.csv", index=False
                )

    manifest = {
        "diagnostic_only": True,
        "target_labels_used_for_updates": False,
        "unit": "inner-step by sample decision",
        "runner": SSAWEveryStepCertificateAdmissionRunner.__name__,
        "datasets": datasets,
        "test_time_seeds": seeds,
        "interpretation": {
            "input_relative_rms": "strength of the physical input perturbation",
            "kl_auc_for_wrong_pseudo_label": "whether SSAW response ranks wrong raw pseudo-labels above correct ones",
            "veto_pre_kl_rate": "samples reaching the final veto KL threshold",
            "rescue_pre_nll_rate": "samples reaching the rescue NLL checks",
            "raw_gradient_proxy": "1 - pseudo-label probability; descriptive proxy, not an exact gradient norm",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
