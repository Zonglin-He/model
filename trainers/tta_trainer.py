import hashlib
import json
import os
import sys
import ast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ADATIME_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'ADATIME'))
if ADATIME_PATH not in sys.path:
    sys.path.append(ADATIME_PATH)

import pandas as pd
import torch
import collections
import argparse
import warnings
import sklearn.exceptions
from datetime import datetime
import numpy as np

from utils.utils import fix_randomness, starting_logs, AverageMeter
from trainers.tta_abstract_trainer import TTAAbstractTrainer
from optim.optimizer import build_optimizer
from utils.utils import EATAMemory, select_eata_indices

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
        fix_randomness(self.seed)
        self.pretrain_cache_dir = getattr(args, "pretrain_cache_dir", None)
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
        self.load_pretrained_checkpoint = os.path.join(
            self.home_path,
            self.save_dir,
            self.experiment_description,
            "NoAdap_All_Trg",
        )
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.summary_f1_scores = open(
            os.path.join(self.exp_log_dir, 'summary_f1_scores.txt'), 'w'
        )

        # Initialize EATA memory (length is configurable via hparams)
        mem_len = int(self.hparams.get('memory_size', 4096)) if hasattr(self, 'hparams') else 4096
        self.eata_memory = EATAMemory(maxlen=mem_len, device=self.device)

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
                self._current_run_seed = current_seed
                print(f"[Seed] run_id={run_id}, seed={current_seed}")
                print(run_id)
                self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id
                )
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Refresh memory buffer for each run to avoid cross-scenario leakage
                mem_len = int(self.hparams.get('memory_size', 4096)) if hasattr(self, 'hparams') else 4096
                self.eata_memory = EATAMemory(maxlen=mem_len, device=self.device)

                if self.da_method == "NoAdap":
                    self.load_data(src_id, trg_id)
                else:
                    self.load_data_demo(src_id, trg_id, current_seed)

                print('Total test datasize:', len(self.trg_whole_dl.dataset))
                all_labels = torch.zeros(self.dataset_configs.num_classes)
                for _, (_, target, _) in enumerate(self.trg_whole_dl):
                    for idx in range(target.shape[0]):
                        all_labels[target[idx]] += 1
                print('trg whole labels:', all_labels)

                non_adapted_model_state, pre_trained_model = self.pre_train()
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state)

                optimizer = build_optimizer(self.hparams)
                if self.da_method == 'NoAdap':
                    tta_model = pre_trained_model
                    tta_model.eval()
                else:
                    tta_model_class = self.get_tta_model_class()
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)

                    if hasattr(tta_model, "set_eata_memory"):
                        tta_model.set_eata_memory(self.eata_memory)
                    else:
                        tta_model.eata_memory = self.eata_memory

                    tta_model.select_eata_indices = select_eata_indices
                    # 记录目标集总样本，用于成本统计
                    try:
                        tta_model._total_samples = len(self.trg_whole_dl.dataset)
                        tta_model._selected_counter = 0
                    except Exception:
                        pass

                tta_model.to(self.device)
                pre_trained_model.eval()

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

                # 输出/保存选样统计（若算法有记录）
                sel_cnt = getattr(tta_model, "_selected_counter", None)
                total_cnt = getattr(tta_model, "_total_samples", None)
                if sel_cnt is not None and total_cnt is not None:
                    stat_line = (
                        f"[SelStats] scenario={scenario} seed={current_seed} "
                        f"selected={sel_cnt}/{total_cnt} ({100.0*sel_cnt/total_cnt:.2f}%)"
                    )
                    print(stat_line)
                    try:
                        with open(os.path.join(self.scenario_log_dir, "selected_stats.txt"), "a") as f:
                            f.write(stat_line + "\n")
                    except Exception:
                        pass

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
            "dataset": self.dataset,
            "backbone": self.backbone,
            "src": self._current_scenario[0],
        }
        pretrain_keys = [
            "pre_learning_rate",
            "num_epochs",
            "batch_size",
            "weight_decay",
            "step_size",
            "lr_decay",
            "steps",
            "momentum",
            "optim_method",
        ]
        signature.update({key: self.hparams.get(key) for key in pretrain_keys if key in self.hparams})
        backbone_overrides = {
            attr: getattr(self.dataset_configs, attr)
            for attr in getattr(self, "_backbone_attr_names", [])
            if hasattr(self.dataset_configs, attr)
        }
        signature["backbone_overrides"] = backbone_overrides
        digest = hashlib.md5(json.dumps(signature, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        filename = f"{self.dataset}_{self.backbone}_src{self._current_scenario[0]}_{digest}.pt"
        return os.path.join(self.pretrain_cache_dir, filename)

    def pre_train(self):
        cache_path = self._pretrain_cache_path()
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached pre-training weights from {cache_path}")
            try:
                payload = torch.load(cache_path, map_location=self.device)
                cached_model = self.initialize_pretrained_model()
                cached_model.load_state_dict(payload["model_state"])
                cached_model = cached_model.to(self.device)
                return payload["non_adapted"], cached_model
            except Exception as exc:
                print(f"Failed to load cache ({exc}); re-training from scratch and refreshing cache.")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        non_adapted_model_state, pre_trained_model = super(TTATrainer, self).pre_train()

        if cache_path:
            torch.save(
                {
                    "non_adapted": non_adapted_model_state,
                    "model_state": pre_trained_model.state_dict(),
                },
                cache_path,
            )
            print(f"Cached pre-training weights at {cache_path}")

        return non_adapted_model_state, pre_trained_model


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='All_Trg', type=str, help='experiment name')
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='ACCUP', type=str, help='ACCUP, NoAdap')
    # ========= Select the DATASET ==============
    parser.add_argument('--data-path', default=r'D:\PyCharm Project\ACCUP + EATA\data\Dataset', type=str)
    parser.add_argument('--dataset', default='EEG', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')
    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--seed', default=42, type=int, help='Random seed applied to every run in this invocation')
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help="Comma-separated seeds to run sequentially (e.g., '41,42,43'). Overrides --seed when provided.",
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
                selected_pairs.append((str(src), str(trg)))
            trainer.dataset_configs.scenarios = selected_pairs
        target_pairs = [(str(src), str(trg)) for src, trg in trainer.dataset_configs.scenarios]
        if override_values:
            trainer._train_params.update(override_values)
            trainer.hparams.update(override_values)
            for src_id, trg_id in target_pairs:
                existing_override = trainer.get_scenario_override(src_id, trg_id)
                merged_override = dict(existing_override)
                merged_override.update(override_values)
                trainer.store_scenario_override(src_id, trg_id, merged_override)
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
