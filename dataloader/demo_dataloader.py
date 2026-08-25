"""Raw time-series loaders for the fixed-source DuSafe protocol."""

import os

import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils import safe_torch_load


class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs, normalization_stats=None):
        self.num_channels = int(dataset_configs.input_channels)
        samples = torch.as_tensor(dataset["samples"])
        labels = dataset.get("labels")
        labels = None if labels is None else torch.as_tensor(labels)

        if samples.ndim == 2:
            samples = samples.unsqueeze(1)
        elif samples.ndim == 3 and samples.shape[1] != self.num_channels:
            samples = samples.transpose(1, 2)
        self.x_data = samples.float()
        self.y_data = None if labels is None else labels.long()

        self.normalization_stats = None
        if dataset_configs.normalize:
            if normalization_stats is None:
                mean = self.x_data.mean(dim=(0, 2))
                std = self.x_data.std(dim=(0, 2))
            else:
                mean, std = normalization_stats
                mean = torch.as_tensor(mean, dtype=self.x_data.dtype)
                std = torch.as_tensor(std, dtype=self.x_data.dtype)
            std = std.clamp_min(1e-6)
            self.normalization_stats = (
                mean.detach().clone(),
                std.detach().clone(),
            )
            self.x_data = (
                self.x_data - mean[None, :, None]
            ) / std[None, :, None]

    def __getitem__(self, index):
        label = None if self.y_data is None else self.y_data[index]
        return self.x_data[index], label, index

    def __len__(self):
        return self.x_data.shape[0]


def data_generator_demo(
    data_path,
    domain_id,
    dataset_configs,
    hparams,
    dtype,
    seed_id=1,
    normalization_stats=None,
):
    del seed_id
    dataset_file = safe_torch_load(
        os.path.join(data_path, f"{dtype}_{domain_id}.pt")
    )
    dataset = Load_Dataset(
        dataset_file,
        dataset_configs,
        normalization_stats=normalization_stats,
    )
    is_test = dtype == "test"
    return DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=False if is_test else dataset_configs.shuffle,
        drop_last=(
            hparams.get("drop_last_test", False)
            if is_test
            else dataset_configs.drop_last
        ),
        num_workers=0,
    )


def whole_targe_data_generator_demo(
    data_path,
    domain_id,
    dataset_configs,
    hparams,
    seed_id=1,
    normalization_stats=None,
):
    del seed_id
    dataset_file = safe_torch_load(
        os.path.join(data_path, f"test_{domain_id}.pt")
    )
    dataset = Load_Dataset(
        dataset_file,
        dataset_configs,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        drop_last=hparams.get(
            "drop_last_eval", hparams.get("drop_last_test", False)
        ),
        num_workers=0,
    )


__all__ = [
    "Load_Dataset",
    "data_generator_demo",
    "whole_targe_data_generator_demo",
]
