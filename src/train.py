from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn, optim

from utils import count_correct_predictions


def train_model(
    model: nn.Module,
    train_loader,
    train_eval_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    early_stopping: bool,
    patience: int,
    weight_decay: float = 0.0,
    pos_weight: float | None = None,
    use_amp: bool = False,
    use_channels_last: bool = False,
):
    criterion_kwargs = {}
    if pos_weight is not None:
        criterion_kwargs["pos_weight"] = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(**criterion_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale_epochs = 0

    for _epoch in range(epochs):
        train_loss, _train_aug_accuracy = _run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            training=True,
            scaler=scaler,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )
        train_eval_loss, train_accuracy = _run_epoch(
            model=model,
            data_loader=train_eval_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            training=False,
            scaler=None,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            training=False,
            scaler=None,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )

        history["train_loss"].append(train_loss)
        history.setdefault("train_eval_loss", []).append(train_eval_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
            torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
            if early_stopping and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def _run_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer,
    training: bool,
    scaler,
    use_amp: bool,
    use_channels_last: bool,
) -> tuple[float, float]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if use_channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        batch_size = labels.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * batch_size
        total_correct += count_correct_predictions(logits.detach(), labels.detach())
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No batches were produced. Check the dataset and dataloader settings.")

    return total_loss / total_samples, total_correct / total_samples
