import argparse
import collections
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optim.optimizer import build_optimizer
from trainers.tta_trainer import TTATrainer
from utils.utils import AverageMeter, EATAMemory, fix_randomness, select_eata_indices, starting_logs


RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"
PRETRAIN_CACHE_ROOT = ROOT / "results" / "pretrain_cache"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def scenario_name(src_id, trg_id):
    return f"{src_id}to{trg_id}"


def make_base_args(
    data_path,
    device,
    dataset,
    da_method="ACCUP",
    backbone="CNN",
    exp_name="supplementary",
    save_dir="results/tta_experiments_logs",
    num_runs=1,
    seed=42,
    seeds=None,
    pretrain_cache_dir=None,
    disable_pretrain_cache=False,
    scenario=None,
):
    return argparse.Namespace(
        save_dir=save_dir,
        exp_name=exp_name,
        da_method=da_method,
        data_path=str(data_path),
        dataset=dataset,
        backbone=backbone,
        num_runs=int(num_runs),
        device=device,
        seed=int(seed),
        seeds=seeds,
        pretrain_cache_dir=pretrain_cache_dir,
        disable_pretrain_cache=disable_pretrain_cache,
        scenario=scenario,
    )


def extract_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("model_state", "non_adapted", "model_dict", "state_dict", "network", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
        if raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError("Unrecognized checkpoint format.")


def load_state_dict_flex(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except Exception:
        pass
    if hasattr(model, "network"):
        try:
            model.network.load_state_dict(state_dict, strict=True)
            return
        except Exception:
            pass
    model.load_state_dict(state_dict, strict=False)


class SupplementaryTTATrainer(TTATrainer):
    def __init__(self, args, tta_model_class=None, pretrained_checkpoint=None):
        self._supplementary_tta_model_class = tta_model_class
        self._pretrained_checkpoint = pretrained_checkpoint
        super().__init__(args)

    def get_tta_model_class(self):
        if self._supplementary_tta_model_class is not None:
            return self._supplementary_tta_model_class
        return super().get_tta_model_class()

    def pre_train(self):
        if not self._pretrained_checkpoint:
            return super().pre_train()
        raw = torch.load(self._pretrained_checkpoint, map_location=self.device)
        state_dict = extract_state_dict(raw)
        pretrained_model = self.initialize_pretrained_model()
        load_state_dict_flex(pretrained_model, state_dict)
        pretrained_model = pretrained_model.to(self.device)
        non_adapted = raw.get("non_adapted", state_dict) if isinstance(raw, dict) else state_dict
        return non_adapted, pretrained_model


def build_trainer(
    data_path,
    device,
    dataset,
    da_method="ACCUP",
    backbone="CNN",
    exp_name="supplementary",
    seed=42,
    num_runs=1,
    seeds=None,
    tta_model_class=None,
    pretrain_cache_dir=None,
    pretrained_checkpoint=None,
):
    if pretrain_cache_dir is None:
        pretrain_cache_dir = str(ensure_dir(PRETRAIN_CACHE_ROOT))
    args = make_base_args(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method=da_method,
        backbone=backbone,
        exp_name=exp_name,
        seed=seed,
        num_runs=num_runs,
        seeds=seeds,
        pretrain_cache_dir=pretrain_cache_dir,
    )
    return SupplementaryTTATrainer(
        args,
        tta_model_class=tta_model_class,
        pretrained_checkpoint=pretrained_checkpoint,
    )


def prepare_scenario(trainer, src_id, trg_id, run_seed=42, run_id=0):
    fix_randomness(int(run_seed))
    trainer._current_run_seed = int(run_seed)
    trainer.run_id = run_id
    trainer.set_scenario_hparams(src_id, trg_id)
    trainer._current_scenario = (str(src_id), str(trg_id))
    if hasattr(trainer.dataset_configs, "_active_scenario"):
        trainer.dataset_configs._active_scenario = trainer._current_scenario
    else:
        setattr(trainer.dataset_configs, "_active_scenario", trainer._current_scenario)
    if trainer.da_method == "NoAdap":
        trainer.load_data(src_id, trg_id)
    else:
        trainer.load_data_demo(src_id, trg_id, int(run_seed))
    trainer.logger, trainer.scenario_log_dir = starting_logs(
        trainer.dataset,
        trainer.da_method,
        trainer.exp_log_dir,
        str(src_id),
        str(trg_id),
        run_id,
    )
    trainer.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
    trainer.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
    mem_len = int(trainer.hparams.get("memory_size", 4096))
    trainer.eata_memory = EATAMemory(maxlen=mem_len, device=trainer.device)


def create_tta_model(trainer, src_id, trg_id, run_seed=42, run_id=0, save_checkpoint=False):
    prepare_scenario(trainer, src_id, trg_id, run_seed=run_seed, run_id=run_id)
    non_adapted_model_state, pre_trained_model = trainer.pre_train()
    if save_checkpoint:
        trainer.save_checkpoint(trainer.home_path, trainer.scenario_log_dir, non_adapted_model_state)

    if trainer.da_method == "NoAdap":
        tta_model = pre_trained_model
        tta_model.eval()
    else:
        optimizer = build_optimizer(trainer.hparams)
        tta_model_class = trainer.get_tta_model_class()
        tta_model = tta_model_class(trainer.dataset_configs, trainer.hparams, pre_trained_model, optimizer)
        if hasattr(tta_model, "set_eata_memory"):
            tta_model.set_eata_memory(trainer.eata_memory)
        else:
            tta_model.eata_memory = trainer.eata_memory
        tta_model.select_eata_indices = select_eata_indices
        try:
            tta_model._total_samples = len(trainer.trg_whole_dl.dataset)
            tta_model._selected_counter = 0
        except Exception:
            pass
    tta_model = tta_model.to(trainer.device)
    pre_trained_model.eval()
    return tta_model, pre_trained_model


def cleanup_trainer(trainer, *models, close_summary=True):
    for model in models:
        if model is not None:
            del model

    summary_handle = getattr(trainer, "summary_f1_scores", None)
    if close_summary and summary_handle is not None and not summary_handle.closed:
        summary_handle.close()

    logger = getattr(trainer, "logger", None)
    if logger is not None:
        for handler in list(logger.handlers):
            try:
                handler.close()
            finally:
                logger.removeHandler(handler)

    for attr in (
        "src_train_dl",
        "src_val_dl",
        "src_test_dl",
        "trg_train_dl",
        "trg_val_dl",
        "trg_test_dl",
        "trg_whole_dl",
        "eata_memory",
        "pre_loss_avg_meters",
        "loss_avg_meters",
    ):
        if hasattr(trainer, attr):
            setattr(trainer, attr, None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_primary_tensor(data):
    return data[0] if isinstance(data, (list, tuple)) else data


def replace_primary_tensor(data, new_primary):
    if isinstance(data, tuple):
        items = list(data)
        items[0] = new_primary
        return tuple(items)
    if isinstance(data, list):
        items = list(data)
        items[0] = new_primary
        return items
    return new_primary


def move_data_to_device(data, device):
    if isinstance(data, tuple):
        return tuple(move_data_to_device(item, device) for item in data)
    if isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    if torch.is_tensor(data):
        return data.float().to(device)
    return data


def apply_corruption_to_data(data, transform_fn, severity, sample_mask=None):
    primary = extract_primary_tensor(data)
    primary_corrupted = primary.clone()
    if sample_mask is None:
        primary_corrupted = transform_fn(primary_corrupted, severity)
    else:
        sample_mask = sample_mask.to(primary.device)
        if sample_mask.any():
            primary_corrupted[sample_mask] = transform_fn(primary_corrupted[sample_mask], severity)
    return replace_primary_tensor(data, primary_corrupted)


class BatchTransformLoader:
    def __init__(self, base_loader, transform_fn, severity, sample_mask_fn=None):
        self.base_loader = base_loader
        self.transform_fn = transform_fn
        self.severity = severity
        self.sample_mask_fn = sample_mask_fn
        self.dataset = base_loader.dataset
        self.batch_size = getattr(base_loader, "batch_size", None)

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        total_steps = len(self)
        for step, batch in enumerate(self.base_loader):
            data, labels, indices = batch
            mask = None
            if self.sample_mask_fn is not None:
                mask = self.sample_mask_fn(data, labels, indices, step, total_steps)
            yield apply_corruption_to_data(data, self.transform_fn, self.severity, sample_mask=mask), labels, indices


def macro_f1_from_logits(logits, labels):
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    refs = labels.detach().cpu().numpy()
    return float(f1_score(refs, preds, average="macro"))


def summarize_metric(df, group_cols, value_col):
    grouped = df.groupby(group_cols)[value_col]
    return grouped.agg(["mean", "std"]).reset_index()


def rolling_macro_f1(y_true, y_pred, window):
    scores = []
    for idx in range(len(y_true)):
        start = max(0, idx - window + 1)
        scores.append(f1_score(y_true[start : idx + 1], y_pred[start : idx + 1], average="macro"))
    return scores


def dataset_scenarios(trainer):
    return [(str(src), str(trg)) for src, trg in trainer.dataset_configs.scenarios]
