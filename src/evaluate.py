from __future__ import annotations

import torch

from utils import count_correct_predictions


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device: torch.device,
    use_amp: bool = False,
    use_channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if use_channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
            dtype=torch.float16,
        ):
            logits = model(images)
        total_correct += count_correct_predictions(logits, labels)
        total_samples += labels.size(0)

    if total_samples == 0:
        raise RuntimeError("No batches were produced during evaluation.")

    return {"accuracy": total_correct / total_samples}
