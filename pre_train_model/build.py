"""Fixed-source supervised training used before DuSafe deployment."""

import torch
from sklearn.metrics import accuracy_score, f1_score

from models.loss import CrossEntropyLabelSmooth
from pre_train_model.pre_train_model import PreTrainModel


def state_dict_to_cpu(module):
    """Clone a module state on CPU so checkpoints do not retain CUDA memory."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def pre_train_model(
    backbone, configs, hparams, src_dataloader, avg_meter, logger, device
):
    if len(src_dataloader) == 0:
        raise ValueError(
            "Source training loader has zero batches. Reduce source batch_size "
            "or disable drop_last for this training configuration."
        )
    model = PreTrainModel(backbone, configs, hparams).to(device)
    optimizer = torch.optim.Adam(
        model.network.parameters(),
        lr=hparams["pre_learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    criterion = CrossEntropyLabelSmooth(
        configs.num_classes, device, epsilon=0.1
    )

    for epoch in range(1, hparams["num_epochs"] + 1):
        predictions, labels = [], []
        for source_x, source_y, _ in src_dataloader:
            source_x = source_x.float().to(device)
            source_y = source_y.long().to(device)
            logits = model(source_x)
            loss = criterion(logits, source_y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    "Source training produced non-finite logits or loss"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter["source_classification_loss"].update(
                loss.item(), source_x.size(0)
            )
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels.extend(source_y.detach().cpu().tolist())

        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro")
        logger.debug(
            f"[Epoch {epoch}/{hparams['num_epochs']}] "
            f"source_acc={accuracy:.4f} source_f1={macro_f1:.4f} "
            f"loss={avg_meter['source_classification_loss'].avg:.4f}"
        )

    return state_dict_to_cpu(model.network), model


__all__ = ["pre_train_model", "state_dict_to_cpu"]
