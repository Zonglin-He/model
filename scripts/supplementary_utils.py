import argparse
import collections
import gc
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optim.optimizer import build_optimizer
from trainers.tta_trainer import TTATrainer
from utils.utils import AverageMeter, fix_randomness, starting_logs


RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"
PRETRAIN_CACHE_ROOT = ROOT / "results" / "pretrain_cache"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_csv(frame, path, **to_csv_kwargs):
    """Write a complete CSV and atomically publish it at ``path``."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    to_csv_kwargs.setdefault("index", False)
    try:
        frame.to_csv(temporary, **to_csv_kwargs)
        # Antivirus and spreadsheet readers can briefly lock a CSV on
        # Windows. Keep the complete temporary file and retry publication.
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def scenario_name(src_id, trg_id):
    return f"{src_id}to{trg_id}"


def enforce_common_batch_size(trainer, src_id, trg_id, batch_size=None):
    """Pin a dataset-level stream batch size across methods in paired audits."""
    del src_id, trg_id
    if batch_size is None:
        batch_size = trainer._train_params["batch_size"]
    trainer.set_runtime_hparams({"batch_size": int(batch_size)})
    return int(batch_size)


def make_base_args(
    data_path,
    device,
    dataset,
    da_method="DuSafe",
    backbone="CNN",
    exp_name="supplementary",
    save_dir="results/tta_experiments_logs",
    num_runs=1,
    seed=42,
    source_seed=1,
    seeds=None,
    pretrain_cache_dir=None,
    disable_pretrain_cache=False,
    scenario=None,
    ablation_mode=None,
    algorithm_registry="production",
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
        source_seed=int(source_seed),
        seeds=seeds,
        pretrain_cache_dir=pretrain_cache_dir,
        disable_pretrain_cache=disable_pretrain_cache,
        scenario=scenario,
        ablation_mode=ablation_mode,
        algorithm_registry=str(algorithm_registry),
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
    def __init__(self, args, pretrained_checkpoint=None):
        self._pretrained_checkpoint = pretrained_checkpoint
        super().__init__(args)

    def pre_train(self):
        if not self._pretrained_checkpoint:
            return super().pre_train()
        raw = torch.load(
            self._pretrained_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = extract_state_dict(raw)
        pretrained_model = self.initialize_pretrained_model()
        load_state_dict_flex(pretrained_model, state_dict)
        pretrained_model = pretrained_model.to(self.device)
        non_adapted = raw.get("non_adapted", state_dict) if isinstance(raw, dict) else state_dict
        confidence_metadata = (
            raw.get("source_confidence_metadata")
            if isinstance(raw, dict)
            else None
        )
        if not self._source_confidence_metadata_matches(confidence_metadata):
            raise ValueError(
                "The explicit source checkpoint has no compatible "
                "source_confidence_metadata. Regenerate it with the current "
                "source-preparation code before source-free TTA."
            )
        self.source_confidence_metadata = confidence_metadata
        if self._requires_source_semantic_metadata():
            semantic_metadata = (
                raw.get("source_semantic_metadata")
                if isinstance(raw, dict)
                else None
            )
            if not self._source_semantic_metadata_matches(semantic_metadata):
                raise ValueError(
                    "The explicit source checkpoint has no compatible "
                    "source_semantic_metadata. Regenerate it with the current "
                    "source-preparation code before source-free TTA."
                )
            self.source_semantic_metadata = semantic_metadata
        else:
            self.source_semantic_metadata = None
        return non_adapted, pretrained_model


def build_trainer(
    data_path,
    device,
    dataset,
    da_method="DuSafe",
    backbone="CNN",
    exp_name="supplementary",
    seed=42,
    source_seed=1,
    num_runs=1,
    seeds=None,
    pretrain_cache_dir=None,
    pretrained_checkpoint=None,
    ablation_mode=None,
    algorithm_registry="production",
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
        source_seed=source_seed,
        num_runs=num_runs,
        seeds=seeds,
        pretrain_cache_dir=pretrain_cache_dir,
        ablation_mode=ablation_mode,
        algorithm_registry=algorithm_registry,
    )
    return SupplementaryTTATrainer(
        args,
        pretrained_checkpoint=pretrained_checkpoint,
    )


def prepare_scenario(trainer, src_id, trg_id, run_seed=42, run_id=0):
    fix_randomness(int(run_seed))
    trainer.set_test_time_seed(int(run_seed))
    trainer._current_source_seed = int(getattr(trainer, "source_seed", 1))
    trainer.run_id = run_id
    trainer.set_scenario_hparams(src_id, trg_id)
    trainer._current_scenario = (str(src_id), str(trg_id))
    if hasattr(trainer.dataset_configs, "_active_scenario"):
        trainer.dataset_configs._active_scenario = trainer._current_scenario
    else:
        setattr(trainer.dataset_configs, "_active_scenario", trainer._current_scenario)
    trainer.load_data_demo(
        src_id,
        trg_id,
        int(run_seed),
        source_seed=trainer._current_source_seed,
    )
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


def create_tta_model(
    trainer,
    src_id,
    trg_id,
    run_seed=42,
    run_id=0,
    save_checkpoint=False,
    pre_tta_hook=None,
):
    prepare_scenario(trainer, src_id, trg_id, run_seed=run_seed, run_id=run_id)
    fix_randomness(trainer._current_source_seed)
    non_adapted_model_state, pre_trained_model = trainer.pre_train()
    if pre_tta_hook is not None:
        pre_tta_hook(trainer, pre_trained_model)
    fix_randomness(int(run_seed))
    if save_checkpoint:
        trainer.save_checkpoint(trainer.home_path, trainer.scenario_log_dir, non_adapted_model_state)

    if trainer.da_method == "NoAdap":
        tta_model = pre_trained_model
        tta_model.eval()
    else:
        optimizer = build_optimizer(trainer.hparams)
        tta_model_class = trainer.get_tta_model_class()
        tta_model = tta_model_class(trainer.dataset_configs, trainer.hparams, pre_trained_model, optimizer)
        if hasattr(tta_model, "load_source_normalization_reference"):
            normalization_stats = getattr(
                trainer.src_train_dl.dataset, "normalization_stats", None
            )
            if normalization_stats is None:
                raise RuntimeError(
                    "Physical SSAW requires fixed source normalization stats"
                )
            tta_model.load_source_normalization_reference(*normalization_stats)
        if getattr(tta_model, "enable_confidence_gate", False):
            if trainer.source_confidence_metadata is None:
                raise RuntimeError(
                    "DuSafe requires label-free source confidence metadata "
                    "from the source checkpoint stage"
                )
            tta_model.load_source_confidence_reference(
                trainer.source_confidence_metadata
            )
        if getattr(tta_model, "enable_source_semantic_gate", False):
            if trainer.source_semantic_metadata is None:
                raise RuntimeError(
                    "DuSafe requires labelled source semantic metadata from "
                    "the source checkpoint stage"
                )
            tta_model.load_source_semantic_reference(
                trainer.source_semantic_metadata
            )
    tta_model = tta_model.to(trainer.device)
    return tta_model, pre_trained_model


def cleanup_trainer(trainer, *models, close_summary=True):
    for model in models:
        if model is not None:
            # Callers still own their local references until the surrounding
            # finally block returns.  Merely deleting this loop variable leaves
            # the CUDA tensors alive while empty_cache() runs, which fragments
            # long reviewer sweeps.  These objects are at end-of-life here, so
            # release optimizer state and move every registered module to CPU
            # before clearing the allocator.
            optimizer = getattr(model, "optimizer", None)
            if optimizer is not None:
                optimizer.state.clear()
            # These diagnostics are ordinary attributes rather than buffers,
            # so Module.to("cpu") would otherwise leave their CUDA tensors
            # alive until the caller's finally block exits.
            for attr, empty_value in (
                ("_last_gate_log", {}),
                ("_last_batch_log", {}),
            ):
                if hasattr(model, attr):
                    setattr(model, attr, empty_value)
            try:
                model.to("cpu")
            except (AttributeError, RuntimeError):
                pass

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
        "pre_loss_avg_meters",
        "loss_avg_meters",
        "source_confidence_metadata",
        "source_semantic_metadata",
        "full_preds",
        "full_pre_final_update_preds",
        "full_labels",
        "last_safety_records",
        "last_safety_summary",
        "last_batch_log_summary",
    ):
        if hasattr(trainer, attr):
            setattr(trainer, attr, None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
    def __init__(
        self,
        base_loader,
        transform_fn,
        severity,
        sample_mask_fn=None,
        meta=None,
        transform_seed=None,
    ):
        self.base_loader = base_loader
        self.transform_fn = transform_fn
        self.severity = severity
        self.sample_mask_fn = sample_mask_fn
        self.transform_seed = None if transform_seed is None else int(transform_seed)
        self.dataset = base_loader.dataset
        self.batch_size = getattr(base_loader, "batch_size", None)
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        total_steps = len(self)
        for step, batch in enumerate(self.base_loader):
            data, labels, indices = batch
            mask = None
            if self.sample_mask_fn is not None:
                mask = self.sample_mask_fn(data, labels, indices, step, total_steps)
            if mask is None:
                primary = extract_primary_tensor(data)
                mask = torch.ones(primary.size(0), dtype=torch.bool)
            self.meta["corruption_mask"] = torch.as_tensor(mask, dtype=torch.bool).view(-1).tolist()
            self.meta["corruption_severity"] = str(self.severity)
            if self.transform_seed is None:
                transformed = apply_corruption_to_data(
                    data, self.transform_fn, self.severity, sample_mask=mask
                )
            else:
                # Isolate corruption randomness from the TTA method. Otherwise
                # an algorithm that draws more augmentations in batch t changes
                # the corruption realization seen in batch t+1.
                batch_seed = (self.transform_seed + step * 1_000_003) % (2**63 - 1)
                primary = extract_primary_tensor(data)
                cuda_devices = []
                if torch.is_tensor(primary) and primary.is_cuda:
                    cuda_devices = [primary.device.index or 0]
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(batch_seed)
                    if cuda_devices:
                        torch.cuda.manual_seed_all(batch_seed)
                    transformed = apply_corruption_to_data(
                        data, self.transform_fn, self.severity, sample_mask=mask
                    )
                self.meta["corruption_transform_seed"] = int(batch_seed)
            yield transformed, labels, indices


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
