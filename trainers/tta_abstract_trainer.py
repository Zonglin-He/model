import sys
from copy import deepcopy

sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, whole_targe_data_generator
from dataloader.demo_dataloader import data_generator_demo, whole_targe_data_generator_demo
from configs.data_model_configs import get_dataset_class
from configs.tta_hparams_new import get_hparams_class
from algorithms.get_tta_class import get_algorithm_class

from utils.utils import safe_torch_load

from models.da_models import get_backbone_class
from pre_train_model.pre_train_model import PreTrainModel
from pre_train_model.build import pre_train_model
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class TTAAbstractTrainer(object):
    """
    主要功能：实现不同的训练算法
    1. 初始化模型和数据
    2. 预训练模型
    3. 评估模型
    4. 计算指标和风险
    5. 保存和加载检查点
    6. 日志记录和结果保存
    7. 辅助功能：创建保存目录，获取配置等
    8. 处理不同的数据加载器
    """
    def __init__(self, args): #初始化
        self.da_method = args.da_method 
        self.dataset = args.dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)

        self.run_description = f"{args.da_method}_{args.exp_name}"
        self.experiment_description = args.dataset

        self.home_path = os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.num_runs = args.num_runs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        alg_hparams_all = dict(self.hparams_class.alg_hparams[self.da_method])
        self._scenario_hparam_overrides = alg_hparams_all.pop("scenario_overrides", {})
        self._base_alg_hparams = alg_hparams_all
        self._train_params = dict(self.hparams_class.train_params)
        self.hparams = {**self._base_alg_hparams, **self._train_params}

        self._backbone_attr_names = (
            "times_hidden_channels",
            "times_num_layers",
            "times_patch_lens",
            "times_dropout",
            "times_ffn_expansion",
        )
        self._backbone_key = str(self.backbone)
        self._dataset_backbone_defaults = {
            attr: deepcopy(getattr(self.dataset_configs, attr))
            for attr in self._backbone_attr_names
            if hasattr(self.dataset_configs, attr)
        }

        self.num_classes = self.dataset_configs.num_classes
        # 准备评估指标：Accuracy（多分类），宏F1，AUROC（多分类）
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)

        # Cache latest experiment metrics so external tools (e.g., Optuna) can read them
        self.scenario_metrics = {}
        self.last_table_results = None
        self.last_table_risks = None

    def sweep(self):
        pass

    def initialize_pretrained_model(self): #初始化预训练模型
        backbone_fe = get_backbone_class(self.backbone) #获取backbone
        pretrained_model = PreTrainModel(backbone_fe, self.dataset_configs, self.hparams) #预训练模型
        pretrained_model = pretrained_model.to(self.device)

        return pretrained_model

    def pre_train(self): #预训练
        backbone_fe = get_backbone_class(self.backbone)
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        non_adapted_model_state, pre_trained_model = pre_train_model(backbone_fe, self.dataset_configs, self.hparams, self.src_train_dl, self.pre_loss_avg_meters, self.logger, self.device)

        return non_adapted_model_state, pre_trained_model

    def evaluate(self, test_loader, tta_model):
        """Run evaluation and keep cached tensors on CPU to avoid GPU bloat."""
        total_loss, preds_list, labels_list = [], [], []
        gate_log_rows = []

        for data, labels, trg_idx in test_loader:
            if isinstance(data, list):
                data = [tensor.float().to(self.device) for tensor in data]
            else:
                data = data.float().to(self.device)
            labels = labels.view(-1).long().to(self.device)

            predictions = tta_model(data)
            loss = F.cross_entropy(predictions, labels)
            total_loss.append(loss.item())
            preds_list.append(predictions.detach().cpu())
            labels_list.append(labels.cpu())
            batch_gate_log = getattr(tta_model, "_last_batch_log", None)
            if isinstance(batch_gate_log, dict) and batch_gate_log:
                numeric_row = {}
                for key, value in batch_gate_log.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        numeric_row[key] = float(value)
                if numeric_row:
                    gate_log_rows.append(numeric_row)

        self.loss = torch.tensor(total_loss, dtype=torch.float32).mean()
        self.full_preds = torch.cat(preds_list)
        self.full_labels = torch.cat(labels_list)
        if gate_log_rows:
            gate_df = pd.DataFrame(gate_log_rows)
            self.last_batch_log_summary = {
                col: float(gate_df[col].mean())
                for col in gate_df.columns
            }
        else:
            self.last_batch_log_summary = {}

    def get_configs(self): #获取当前数据集对应的配置类实例和超参数类实例
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def _scenario_override_candidates(self, src_id, trg_id):
        """Yield candidate keys for scenario overrides ordered by backbone specificity."""
        src = str(src_id)
        trg = str(trg_id)
        backbone = self._backbone_key
        backbone_variants = {backbone, backbone.lower(), backbone.upper()}

        for name in backbone_variants:
            yield (name, src, trg)
        for name in backbone_variants:
            yield f"{name}:{src}->{trg}"
        yield (src, trg)
        yield f"{src}->{trg}"
        yield f"{src}_to_{trg}"

    def _scenario_override_storage_key(self, src_id, trg_id):
        """Return the canonical storage key for overrides tied to the active backbone."""
        return (self._backbone_key, str(src_id), str(trg_id))

    def _resolve_scenario_override(self, src_id, trg_id):
        """Lookup overrides for the given scenario honoring backbone-specific entries."""
        for candidate in self._scenario_override_candidates(src_id, trg_id):
            if candidate in self._scenario_hparam_overrides:
                payload = self._scenario_hparam_overrides.get(candidate, {})
                return dict(payload) if isinstance(payload, dict) else {}, candidate
        return {}, None

    def get_scenario_override(self, src_id, trg_id):
        """Expose read-only view of the merged override for the given scenario."""
        overrides, _ = self._resolve_scenario_override(src_id, trg_id)
        return overrides

    def store_scenario_override(self, src_id, trg_id, overrides):
        """Persist overrides under the backbone-aware key without mutating base configs."""
        key = self._scenario_override_storage_key(src_id, trg_id)
        existing = dict(self._scenario_hparam_overrides.get(key, {}))
        existing.update(overrides or {})
        self._scenario_hparam_overrides[key] = existing
        return key

    def set_scenario_hparams(self, src_id, trg_id):
        """
        Refresh active hyperparameters for a specific source→target scenario.
        Allows per-scenario tuning by merging overrides on top of base + train params.
        """
        combined = {**self._base_alg_hparams, **self._train_params}
        overrides, _ = self._resolve_scenario_override(src_id, trg_id)
        if overrides:
            combined.update(overrides)
        self._apply_backbone_overrides(combined)
        self.hparams = combined
        return self.hparams

    def _apply_backbone_overrides(self, hparams):
        """Reset dataset backbone params to defaults, then apply overrides if provided."""
        for attr, value in self._dataset_backbone_defaults.items():
            setattr(self.dataset_configs, attr, deepcopy(value))
        for attr in self._backbone_attr_names:
            if attr in hparams:
                setattr(self.dataset_configs, attr, deepcopy(hparams[attr]))

    def get_tta_model_class(self): #获取指定的 TTA 模型类
        tta_model_class = get_algorithm_class(self.da_method)

        return tta_model_class

    def load_data(self, src_id, trg_id): # 加载数据集（不带增强版本）
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

        self.trg_whole_dl = whole_targe_data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams)

    def load_data_demo(self, src_id, trg_id, run_id = 0): #加载数据集（带增强版本）
        self.src_train_dl = data_generator_demo(
            self.data_path, src_id, self.dataset_configs, self.hparams, "train", seed_id=run_id
        )
        self.src_test_dl = data_generator_demo(
            self.data_path, src_id, self.dataset_configs, self.hparams, "test", seed_id=run_id
        )
        self.trg_whole_dl = whole_targe_data_generator_demo(self.data_path, trg_id, self.dataset_configs, self.hparams, seed_id = run_id)

    def create_save_dir(self, save_dir): #创建保存目录
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self): # 计算源测试集和目标测试集上的风险（分类误差）
        self.evaluate(self.src_test_dl) # 调用 evaluate 分别在源测试集、目标测试集上运行当前模型
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        preds_cpu = self.full_preds if self.full_preds.device.type == 'cpu' else self.full_preds.cpu()
        labels_cpu = self.full_labels if self.full_labels.device.type == 'cpu' else self.full_labels.cpu()
        pred_labels = preds_cpu.argmax(dim=1)

        acc = self.ACC(pred_labels, labels_cpu).item()
        f1 = self.F1(pred_labels, labels_cpu).item()
        auroc = self.AUROC(preds_cpu, labels_cpu).item()
        self.ACC.reset()
        self.F1.reset()
        self.AUROC.reset()

        risks = src_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self, table_results, name):
        # 保存结果表格到 CSV 文件
        table_results.to_csv(os.path.join(self.exp_log_dir, f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, non_adapted):
        save_dict = {
            "non_adapted": non_adapted
        }
        # 保存模型checkpoint
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def load_checkpoint(self, model_dir): # 从指定目录加载 checkpoint.pt，提取non_adapted模型参数并返回
        checkpoint = safe_torch_load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        pretrained_model = checkpoint['non_adapted']

        return pretrained_model

    def calculate_avg_std_wandb_table(self, results): #计算平均值和标准差，并将其添加到结果表中
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks): # 计算表格（wandb.Table）的各指标列的平均值和标准差，

        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}

        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)
        # 将mean和std作为新行添加到表末尾，并返回更新后的表和summary_metrics字典

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # 使用Weights & Biases记录结果表、风险表、超参数表和汇总指标
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self, tta_model):
        # ������Ӧ��ģ��������Ŀ���������ϵ�ָ��
        self.evaluate(self.trg_whole_dl, tta_model)
        preds_cpu = self.full_preds if self.full_preds.device.type == 'cpu' else self.full_preds.cpu()
        labels_cpu = self.full_labels if self.full_labels.device.type == 'cpu' else self.full_labels.cpu()
        pred_labels = preds_cpu.argmax(dim=1)

        acc = self.ACC(pred_labels, labels_cpu).item()
        f1 = self.F1(pred_labels, labels_cpu).item()
        auroc = self.AUROC(preds_cpu, labels_cpu).item()
        self.ACC.reset()
        self.F1.reset()
        self.AUROC.reset()
        trg_risk = self.loss.item()

        return acc, f1, auroc, trg_risk

    def calculate_risks(self): #计算源测试集和目标测试集上的风险（分类误差）
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics, seed=None):
        # 将新的结果添加到表中
        row = [scenario]
        if "seed" in table.columns:
            row.append(seed if seed is not None else getattr(self, "seed", None))
        row.append(run_id)

        if isinstance(metrics, float):
            row.append(metrics)
        elif isinstance(metrics, tuple):
            row.extend(metrics)

        # Create new dataframes for each row
        results_df = pd.DataFrame([row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table

    def add_mean_std_table(self, table, columns): #计算表格的各指标列的平均值和标准差，并将其添加到结果表中
        # Calculate average and standard deviation for metrics
        metric_start_idx = 3 if "seed" in columns else 2
        metric_cols = columns[metric_start_idx:]
        avg_metrics = [table[metric].mean() for metric in metric_cols]
        std_metrics = [table[metric].std() for metric in metric_cols]

        # Create dataframes for mean and std values
        prefix = ['mean', '-']
        prefix_std = ['std', '-']
        if "seed" in columns:
            prefix.insert(1, '-')
            prefix_std.insert(1, '-')
        mean_metrics_df = pd.DataFrame([prefix + avg_metrics], columns=columns)
        std_metrics_df = pd.DataFrame([prefix_std + std_metrics], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        if hasattr(table, "map"):
            table = table.map(format_func)
        else:
            table = table.applymap(format_func)

        return table
