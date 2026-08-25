import hashlib
import json
import os
import sys
import ast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader
import collections
import argparse
import warnings
import sklearn.exceptions
from datetime import datetime
import numpy as np

from utils.utils import fix_randomness, starting_logs, AverageMeter
from trainers.tta_abstract_trainer import TTAAbstractTrainer
from optim.optimizer import build_optimizer
from pre_train_model.build import state_dict_to_cpu
from configs.data_model_configs import validate_scenario

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


def _parse_override_value(raw_value):
    text = str(raw_value).strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_cli_overrides(entries):
    overrides = {}
    for entry in entries or []:
        if "=" not in str(entry):
            raise ValueError(
                f"Invalid --override value '{entry}'. Expected key=value."
            )
        key, raw_value = str(entry).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --override value '{entry}'. Empty key.")
        overrides[key] = _parse_override_value(raw_value)
    return overrides


class TTATrainer(TTAAbstractTrainer):
    """Main training loop for our test-time adaptation methods."""

    def __init__(self, args):
        super(TTATrainer, self).__init__(args)
        self.seed = getattr(args, "seed", 42)
        if getattr(args, 'seeds', None):
            self._seeds_list = [int(s.strip()) for s in str(args.seeds).split(',') if s.strip()]
        else:
            self._seeds_list = None
        if self._seeds_list:
            self.num_runs = max(int(self.num_runs), len(self._seeds_list))
            self.seed = int(self._seeds_list[0])
        self._current_run_seed = self.seed
        self.source_seed = int(getattr(args, "source_seed", 1))
        self._current_source_seed = self.source_seed
        self.source_confidence_metadata = None
        self.source_semantic_metadata = None
        fix_randomness(self.seed)
        self.pretrain_cache_dir = None if getattr(args, "disable_pretrain_cache", False) else getattr(args, "pretrain_cache_dir", None)
        if self.pretrain_cache_dir:
            self.pretrain_cache_dir = os.path.abspath(self.pretrain_cache_dir)
            os.makedirs(self.pretrain_cache_dir, exist_ok=True)
        self._current_scenario = None
        self.exp_log_dir = os.path.join(
            self.home_path,
            self.save_dir,
            self.experiment_description,
            f"{self.run_description}",
        )
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.summary_f1_scores = open(
            os.path.join(self.exp_log_dir, 'summary_f1_scores.txt'), 'w'
        )

    def test_time_adaptation(self):
        """Entry point for running test-time adaptation."""
        results_columns = ["scenario", "seed", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)
        risks_columns = ["scenario", "seed", "run", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Reset caches so repeated calls do not leak state
        self.scenario_metrics = {}
        self.last_table_results = None
        self.last_table_risks = None

        for src_id, trg_id in self.dataset_configs.scenarios:
            self.set_scenario_hparams(src_id, trg_id)
            self._current_scenario = (str(src_id), str(trg_id))
            if hasattr(self.dataset_configs, "_active_scenario"):
                self.dataset_configs._active_scenario = self._current_scenario
            else:
                setattr(self.dataset_configs, "_active_scenario", self._current_scenario)
            scenario = f"{src_id}_to_{trg_id}"
            cur_scenario_f1_ret = []
            cur_scenario_metrics = []
            cur_scenario_gate_logs = []

            for run_id in range(self.num_runs):
                self.run_id = run_id
                if self._seeds_list and len(self._seeds_list) >= self.num_runs:
                    current_seed = self._seeds_list[run_id]
                else:
                    current_seed = self.seed + run_id
                fix_randomness(current_seed)
                self.set_test_time_seed(current_seed)
                self.set_scenario_hparams(src_id, trg_id)
                self._current_source_seed = self.source_seed
                print(f"[Seed] run_id={run_id}, seed={current_seed}")
                print(run_id)
                self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id
                )
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                self.load_data_demo(
                    src_id,
                    trg_id,
                    current_seed,
                    source_seed=self._current_source_seed,
                )

                print('Total test datasize:', len(self.trg_whole_dl.dataset))
                if bool(
                    self.hparams.get("record_target_label_histogram", False)
                ):
                    all_labels = torch.zeros(self.dataset_configs.num_classes)
                    for _, (_, target, _) in enumerate(self.trg_whole_dl):
                        all_labels += torch.bincount(
                            target.to(dtype=torch.long).flatten(),
                            minlength=self.dataset_configs.num_classes,
                        ).to(dtype=all_labels.dtype)
                    print('trg whole labels:', all_labels)
                else:
                    print('trg whole labels: disabled (not used by TTA)')

                fix_randomness(self._current_source_seed)
                non_adapted_model_state, pre_trained_model = self.pre_train()
                # Source training and online adaptation have independent RNG
                # streams. Reset here so cache hits and cache misses generate
                # exactly the same target-time randomness.
                fix_randomness(current_seed)
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state)

                optimizer = build_optimizer(self.hparams)
                if self.da_method == 'NoAdap':
                    tta_model = pre_trained_model
                    tta_model.eval()
                else:
                    tta_model_class = self.get_tta_model_class()
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)

                    if hasattr(
                        tta_model, "load_source_normalization_reference"
                    ):
                        normalization_stats = getattr(
                            self.src_train_dl.dataset,
                            "normalization_stats",
                            None,
                        )
                        if normalization_stats is None:
                            raise RuntimeError(
                                "Physical SSAW requires fixed source "
                                "normalization stats"
                            )
                        tta_model.load_source_normalization_reference(
                            *normalization_stats
                        )
                    if getattr(tta_model, "enable_confidence_gate", False):
                        if self.source_confidence_metadata is None:
                            raise RuntimeError(
                                "DuSafe requires label-free source confidence "
                                "metadata from the source checkpoint stage"
                            )
                        tta_model.load_source_confidence_reference(
                            self.source_confidence_metadata
                        )
                    if getattr(
                        tta_model, "enable_source_semantic_gate", False
                    ):
                        if self.source_semantic_metadata is None:
                            raise RuntimeError(
                                "DuSafe requires labelled source semantic "
                                "metadata from the source checkpoint stage"
                            )
                        tta_model.load_source_semantic_reference(
                            self.source_semantic_metadata
                        )
                tta_model.to(self.device)
                metrics = self.calculate_metrics(tta_model)
                cur_scenario_metrics.append(metrics)
                cur_scenario_f1_ret.append(metrics[1])
                batch_log_summary = getattr(self, "last_batch_log_summary", None)
                if isinstance(batch_log_summary, dict) and batch_log_summary:
                    cur_scenario_gate_logs.append(dict(batch_log_summary))
                table_results = self.append_results_to_tables(
                    table_results, scenario, run_id, metrics[:3], seed=current_seed
                )
                table_risks = self.append_results_to_tables(
                    table_risks, scenario, run_id, metrics[-1], seed=current_seed
                )

            if cur_scenario_metrics:
                metrics_array = np.array(cur_scenario_metrics)
                avg_metrics = metrics_array.mean(axis=0)
                std_metrics = metrics_array.std(axis=0)
                cur_avg_f1_raw = float(avg_metrics[1])
                cur_std_f1_raw = float(std_metrics[1])
                cur_avg_f1_scores = 100.0 * cur_avg_f1_raw
                cur_std_f1_scores = 100.0 * cur_std_f1_raw
            else:
                avg_metrics = np.full(4, np.nan)
                std_metrics = np.full(4, np.nan)
                cur_avg_f1_raw = float('nan')
                cur_std_f1_raw = float('nan')
                cur_avg_f1_scores = float('nan')
                cur_std_f1_scores = float('nan')

            print('Average current f1_scores::', cur_avg_f1_scores, 'Std:', cur_std_f1_scores)
            print(
                scenario,
                ' : ',
                np.around(cur_avg_f1_scores, 2),
                '/',
                np.around(cur_std_f1_scores, 2),
                sep='',
                file=self.summary_f1_scores,
            )

            scenario_key = (str(src_id), str(trg_id))
            scenario_payload = {
                "acc_mean": float(avg_metrics[0]),
                "f1_mean": cur_avg_f1_raw,
                "auroc_mean": float(avg_metrics[2]),
                "trg_risk_mean": float(avg_metrics[3]),
                "acc_std": float(std_metrics[0]),
                "f1_std": cur_std_f1_raw,
                "auroc_std": float(std_metrics[2]),
                "trg_risk_std": float(std_metrics[3]),
            }
            if cur_scenario_gate_logs:
                gate_df = pd.DataFrame(cur_scenario_gate_logs)
                scenario_payload["gate_means"] = {
                    col: float(gate_df[col].mean())
                    for col in gate_df.columns
                }
            self.scenario_metrics[scenario_key] = scenario_payload

        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
        self.last_table_results = table_results
        self.last_table_risks = table_risks
        self.save_tables_to_file(table_results, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + '_results')
        self.save_tables_to_file(table_risks, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + '_risks')

        self.summary_f1_scores.close()

    def _pretrain_cache_path(self):
        if not self.pretrain_cache_dir or not self._current_scenario:
            return None
        signature = {
            "pretrain_protocol_version": 2,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "src": self._current_scenario[0],
            "source_seed": int(self._current_source_seed),
        }
        pretrain_keys = [
            "pre_learning_rate",
            "num_epochs",
            "batch_size",
            "weight_decay",
            "steps",
            "momentum",
            "optim_method",
        ]
        signature.update({key: self.source_hparams.get(key) for key in pretrain_keys if key in self.source_hparams})
        backbone_overrides = {
            attr: getattr(self.dataset_configs, attr)
            for attr in getattr(self, "_backbone_attr_names", [])
            if hasattr(self.dataset_configs, attr)
        }
        signature["backbone_overrides"] = backbone_overrides
        digest = hashlib.md5(json.dumps(signature, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        filename = f"{self.dataset}_{self.backbone}_src{self._current_scenario[0]}_{digest}.pt"
        return os.path.join(self.pretrain_cache_dir, filename)

    def _source_confidence_metadata_config(self):
        dusafe_hparams = self.hparams_class.alg_hparams.get("DuSafe", {})
        return {
            "reference_samples": int(
                dusafe_hparams.get("confidence_reference_samples", 4096)
            ),
            "bn_statistics": str(
                dusafe_hparams.get("bn_statistics", "batch")
            ).strip().lower(),
            "disable_dropout": True,
            # TTBN confidence depends on the batch population.  Calibrate in
            # the same batch context used by the deployment stream without
            # coupling this value to source-model training.
            "source_batch_size": int(
                self.hparams.get(
                    "batch_size",
                    getattr(self.src_test_dl, "batch_size", 1),
                )
            ),
        }

    def _source_confidence_context_key(self):
        config = self._source_confidence_metadata_config()
        return json.dumps(config, sort_keys=True, default=str)

    def _source_confidence_metadata_matches(self, metadata):
        if not isinstance(metadata, dict):
            return False
        from algorithms.dusafe import SOURCE_CONFIDENCE_METADATA_VERSION

        config = self._source_confidence_metadata_config()
        return (
            int(metadata.get("version", -1))
            == SOURCE_CONFIDENCE_METADATA_VERSION
            and int(metadata.get("reference_samples", -1))
            == config["reference_samples"]
            and str(metadata.get("bn_statistics", "")).strip().lower()
            == config["bn_statistics"]
            and bool(metadata.get("disable_dropout"))
            == config["disable_dropout"]
            and int(metadata.get("source_batch_size", -1))
            == config["source_batch_size"]
            and torch.as_tensor(metadata.get("top1_nll", [])).numel() > 0
        )

    def _collect_source_confidence_metadata(self, model):
        from algorithms.dusafe import collect_source_confidence_metadata

        config = self._source_confidence_metadata_config()
        source_batch_size = int(config.pop("source_batch_size"))
        calibration_loader = DataLoader(
            self.src_test_dl.dataset,
            batch_size=source_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return collect_source_confidence_metadata(
            calibration_loader,
            model,
            **config,
        )

    def _source_semantic_metadata_config(self):
        dusafe_hparams = self.hparams_class.alg_hparams.get("DuSafe", {})
        return {
            "reference_samples": int(
                dusafe_hparams.get("source_semantic_reference_samples", 4096)
            ),
            "bn_statistics": str(
                dusafe_hparams.get(
                    "source_semantic_bn_statistics", "frozen"
                )
            ).strip().lower(),
            "disable_dropout": True,
            "num_classes": int(self.dataset_configs.num_classes),
            "source_batch_size": int(
                self.hparams.get(
                    "batch_size",
                    getattr(self.src_test_dl, "batch_size", 1),
                )
            ),
        }

    def _source_semantic_context_key(self):
        return json.dumps(
            self._source_semantic_metadata_config(),
            sort_keys=True,
            default=str,
        )

    def _source_semantic_metadata_matches(self, metadata):
        if not isinstance(metadata, dict):
            return False
        from algorithms.dusafe import SOURCE_SEMANTIC_METADATA_VERSION

        config = self._source_semantic_metadata_config()
        prototypes = torch.as_tensor(metadata.get("prototypes", []))
        class_counts = torch.as_tensor(metadata.get("class_counts", []))
        bn_state = metadata.get("feature_extractor_bn_state")
        return (
            int(metadata.get("version", -1))
            == SOURCE_SEMANTIC_METADATA_VERSION
            and int(metadata.get("reference_samples", -1))
            == config["reference_samples"]
            and str(metadata.get("bn_statistics", "")).strip().lower()
            == config["bn_statistics"]
            and bool(metadata.get("disable_dropout"))
            == config["disable_dropout"]
            and int(metadata.get("source_batch_size", -1))
            == config["source_batch_size"]
            and int(metadata.get("num_classes", -1))
            == config["num_classes"]
            and prototypes.dim() == 2
            and prototypes.size(0) == config["num_classes"]
            and torch.isfinite(prototypes).all().item()
            and class_counts.numel() == config["num_classes"]
            and class_counts.gt(0).all().item()
            and isinstance(bn_state, dict)
            and int(metadata.get("bn_calibration_samples", 0)) > 0
        )

    def _collect_source_semantic_metadata(self, model):
        from algorithms.dusafe import collect_source_semantic_metadata

        config = self._source_semantic_metadata_config()
        source_batch_size = int(config.pop("source_batch_size"))
        calibration_loader = DataLoader(
            self.src_test_dl.dataset,
            batch_size=source_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return collect_source_semantic_metadata(
            calibration_loader,
            model,
            **config,
        )

    def _requires_source_semantic_metadata(self):
        """Return whether the selected method explicitly uses old semantics."""
        if str(getattr(self, "da_method", "")) != "DuSafe":
            return False
        hparams = getattr(self, "hparams", {}) or {}
        return bool(
            hparams.get("enable_source_semantic_gate", False)
            or hparams.get("enable_source_semantic_router", False)
        )

    def pre_train(self):
        requires_semantic = self._requires_source_semantic_metadata()
        cache_path = self._pretrain_cache_path()
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached pre-training weights from {cache_path}")
            try:
                # Keep all serialized copies on CPU. Loading both checkpoint
                # states directly onto CUDA temporarily tripled model memory
                # before the adapted model was even constructed.
                payload = torch.load(cache_path, map_location="cpu")
                if int(payload.get("pretrain_protocol_version", -1)) != 2:
                    raise ValueError("stale pre-training cache protocol")
                if int(payload.get("source_seed", -1)) != int(self._current_source_seed):
                    raise ValueError("pre-training cache source seed mismatch")
                cached_model = self.initialize_pretrained_model()
                cached_model.load_state_dict(payload["model_state"])
                cached_model = cached_model.to(self.device)
                confidence_context = self._source_confidence_context_key()
                confidence_by_context = payload.get(
                    "source_confidence_metadata_by_context", {}
                )
                if not isinstance(confidence_by_context, dict):
                    confidence_by_context = {}
                confidence_metadata = confidence_by_context.get(
                    confidence_context
                )
                legacy_confidence = payload.get(
                    "source_confidence_metadata"
                )
                if (
                    not self._source_confidence_metadata_matches(
                        confidence_metadata
                    )
                    and self._source_confidence_metadata_matches(
                        legacy_confidence
                    )
                ):
                    confidence_metadata = legacy_confidence
                if not self._source_confidence_metadata_matches(
                    confidence_metadata
                ):
                    print(
                        "Adding deployment-batch-matched source confidence "
                        "metadata to the pre-training cache."
                    )
                    confidence_metadata = (
                        self._collect_source_confidence_metadata(cached_model)
                    )
                confidence_by_context[confidence_context] = (
                    confidence_metadata
                )
                payload["source_confidence_metadata_by_context"] = (
                    confidence_by_context
                )
                # Keep the active context at the legacy key for explicit
                # checkpoint consumers while the map prevents cache thrash.
                payload["source_confidence_metadata"] = confidence_metadata
                self.source_confidence_metadata = confidence_metadata

                if requires_semantic:
                    semantic_context = self._source_semantic_context_key()
                    semantic_by_context = payload.get(
                        "source_semantic_metadata_by_context", {}
                    )
                    if not isinstance(semantic_by_context, dict):
                        semantic_by_context = {}
                    semantic_metadata = semantic_by_context.get(semantic_context)
                    legacy_semantic = payload.get("source_semantic_metadata")
                    if (
                        not self._source_semantic_metadata_matches(
                            semantic_metadata
                        )
                        and self._source_semantic_metadata_matches(legacy_semantic)
                    ):
                        semantic_metadata = legacy_semantic
                    if not self._source_semantic_metadata_matches(
                        semantic_metadata
                    ):
                        print(
                            "Adding source-BN-frozen semantic "
                            "metadata to the pre-training cache."
                        )
                        semantic_metadata = self._collect_source_semantic_metadata(
                            cached_model
                        )
                    semantic_by_context[semantic_context] = semantic_metadata
                    payload["source_semantic_metadata_by_context"] = (
                        semantic_by_context
                    )
                    payload["source_semantic_metadata"] = semantic_metadata
                    self.source_semantic_metadata = semantic_metadata
                else:
                    self.source_semantic_metadata = None
                torch.save(payload, cache_path)
                return payload["non_adapted"], cached_model
            except Exception as exc:
                print(f"Failed to load cache ({exc}); re-training from scratch and refreshing cache.")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        non_adapted_model_state, pre_trained_model = super(TTATrainer, self).pre_train()
        self.source_confidence_metadata = (
            self._collect_source_confidence_metadata(pre_trained_model)
        )
        self.source_semantic_metadata = (
            self._collect_source_semantic_metadata(pre_trained_model)
            if requires_semantic
            else None
        )
        if cache_path:
            payload = {
                "pretrain_protocol_version": 2,
                "source_seed": int(self._current_source_seed),
                "source_hparams": dict(self.source_hparams),
                "non_adapted": non_adapted_model_state,
                "model_state": state_dict_to_cpu(pre_trained_model),
                "source_confidence_metadata": self.source_confidence_metadata,
                "source_confidence_metadata_by_context": {
                    self._source_confidence_context_key(): (
                        self.source_confidence_metadata
                    )
                },
            }
            if requires_semantic:
                payload.update(
                    {
                        "source_semantic_metadata": (
                            self.source_semantic_metadata
                        ),
                        "source_semantic_metadata_by_context": {
                            self._source_semantic_context_key(): (
                                self.source_semantic_metadata
                            )
                        },
                    }
                )
            torch.save(payload, cache_path)
            print(f"Cached pre-training weights at {cache_path}")

        return non_adapted_model_state, pre_trained_model


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='All_Trg', type=str, help='experiment name')
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='DuSafe', choices=('DuSafe', 'NoAdap'))
    # ========= Select the DATASET ==============
    parser.add_argument('--data-path', default='data/Dataset', type=str)
    parser.add_argument('--dataset', default='EEG', choices=('EEG', 'HAR', 'FD', 'HHAR'))
    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', choices=('CNN', 'TimesNet'))
    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--seed', default=42, type=int, help='Random seed applied to every run in this invocation')
    parser.add_argument(
        '--source_seed',
        default=1,
        type=int,
        help='Independent source-training seed; shared across methods for paired comparisons.',
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help=(
            "Comma-separated target-time control seeds (e.g., '42'). "
            "Independent source checkpoints use --source_seed; fixed target "
            "loaders are not reshuffled by repeating this option."
        ),
    )
    parser.add_argument(
        '--pretrain_cache_dir',
        type=str,
        default=None,
        help="Optional directory to cache/reuse pre-training weights across runs.",
    )
    parser.add_argument(
        '--disable_pretrain_cache',
        action='store_true',
        help="Force pre-training from scratch even if a cache directory is provided.",
    )
    parser.add_argument(
        '--override',
        action='append',
        default=None,
        help=(
            "Optional hyperparameter override in key=value form. "
            "Repeat to override multiple values, e.g. "
            "--override batch_size=97 --override learning_rate=3e-5."
        ),
    )
    parser.add_argument(
        '--scenario',
        action='append',
        default=None,
        help=(
            "Optional src->trg scenario filter. "
            "Example: --scenario 7->18 --scenario 16->1. "
            "If omitted, all dataset-defined scenarios will be evaluated."
        ),
    )

    args = parser.parse_args()
    override_values = _parse_cli_overrides(args.override)
    if args.disable_pretrain_cache:
        args.pretrain_cache_dir = None

    def _run_single(seed_args):
        trainer = TTATrainer(seed_args)
        if seed_args.scenario:
            selected_pairs = []
            for entry in seed_args.scenario:
                if '->' in entry:
                    src, trg = entry.split('->', 1)
                elif ',' in entry:
                    src, trg = entry.split(',', 1)
                else:
                    raise ValueError(f"Invalid scenario format '{entry}'. Expected 'src->trg'.")
                selected_pairs.append(validate_scenario(seed_args.dataset, src, trg))
            trainer.dataset_configs.scenarios = selected_pairs
        if override_values:
            trainer.set_runtime_hparams(override_values)
        trainer.test_time_adaptation()

    if args.seeds:
        try:
            seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except Exception as exc:
            raise ValueError(f"Unable to parse --seeds='{args.seeds}'") from exc
    else:
        seed_list = [getattr(args, 'seed', 42)]

    if seed_list:
        args.seed = seed_list[0]
        args.seeds = ",".join(str(seed) for seed in seed_list)
        args.num_runs = max(int(args.num_runs), len(seed_list))
    _run_single(args)
