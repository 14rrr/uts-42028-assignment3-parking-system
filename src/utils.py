from __future__ import annotations

import csv
import json
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(slots=True)
class RuntimeConfig:
    device: torch.device
    cuda_usable: bool
    device_name: str
    use_amp: bool
    use_channels_last: bool
    use_pin_memory: bool
    non_blocking: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_runtime_config(use_amp: bool = True, use_channels_last: bool = True) -> RuntimeConfig:
    if torch.cuda.is_available():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                probe = torch.randn((8, 3, 128, 128), device="cuda")
                layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to("cuda")
                _ = layer(probe)
                torch.cuda.synchronize()

                for warning in caught:
                    if "not compatible with the current PyTorch installation" in str(
                        warning.message
                    ):
                        raise RuntimeError(str(warning.message))
                torch.backends.cudnn.benchmark = True
                return RuntimeConfig(
                    device=torch.device("cuda"),
                    cuda_usable=True,
                    device_name=torch.cuda.get_device_name(0),
                    use_amp=use_amp,
                    use_channels_last=use_channels_last,
                    use_pin_memory=True,
                    non_blocking=True,
                )
            except Exception:
                pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return RuntimeConfig(
            device=torch.device("mps"),
            cuda_usable=False,
            device_name="Apple MPS",
            use_amp=False,
            use_channels_last=False,
            use_pin_memory=False,
            non_blocking=False,
        )
    return RuntimeConfig(
        device=torch.device("cpu"),
        cuda_usable=False,
        device_name="CPU",
        use_amp=False,
        use_channels_last=False,
        use_pin_memory=False,
        non_blocking=False,
    )


def count_correct_predictions(logits: torch.Tensor, labels: torch.Tensor) -> int:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return int((preds == labels).sum().item())


def benchmark_batch_sizes(
    model_builder,
    image_size: int,
    device: torch.device,
    candidates: list[int],
    use_amp: bool,
    use_channels_last: bool,
) -> tuple[int | None, list[dict[str, float | int | str]]]:
    if device.type != "cuda":
        return None, []

    results: list[dict[str, float | int | str]] = []
    best_batch_size: int | None = None

    for batch_size in sorted(dict.fromkeys(candidates)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = model_builder().to(device)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        try:
            for _ in range(2):
                images = torch.randn(batch_size, 3, image_size, image_size, device=device)
                labels = torch.rand(batch_size, device=device)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                    dtype=torch.float16,
                ):
                    logits = model(images)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            start = time.perf_counter()
            steps = 5
            for _ in range(steps):
                images = torch.randn(batch_size, 3, image_size, image_size, device=device)
                labels = torch.rand(batch_size, device=device)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                    dtype=torch.float16,
                ):
                    logits = model(images)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "images_per_second": (batch_size * steps) / max(elapsed, 1e-6),
                    "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                }
            )
            best_batch_size = batch_size
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": batch_size, "status": "oom"})
            break
        finally:
            del model
            del optimizer
            torch.cuda.empty_cache()

    return best_batch_size, results


def save_history_plot(history: dict[str, list[float]], output_path: Path, model_name: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_results_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fieldnames = ["model", "train_accuracy", "validation_accuracy", "test_accuracy"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    rows: list[dict[str, float | str]],
    output_path: Path,
    title: str,
    dataset_name: str,
    image_size: int,
    batch_size: int,
    epochs: int,
    dataset_variant: str,
    total_crops: int,
    empty_crops: int,
    occupied_crops: int,
    class_weighting: str,
    model_group_description: str,
    results_heading: str,
) -> None:
    settings = (
        f"The experiments were conducted on the {dataset_name} dataset for binary image "
        f"classification of parking-space occupancy (occupied vs empty). Images were resized "
        f"to {image_size}x{image_size} and normalized before training. The data split used "
        f"70% training, 15% validation, and 15% testing with a fixed random seed for "
        f"reproducibility. {model_group_description} Models used PyTorch with the Adam "
        f"optimizer, binary cross-entropy via BCEWithLogitsLoss, "
        f"a batch size of {batch_size}, and up to {epochs} epochs. Dropout was applied in "
        f"models where configured, data augmentation was used where configured, and early "
        f"stopping was enabled according to the experiment configuration. Dataset usage: "
        f"{dataset_variant}. Total processed crops: "
        f"{total_crops} ({empty_crops} empty, {occupied_crops} occupied). Class weighting: "
        f"{class_weighting}."
    )

    lines = [
        f"# {title}",
        "",
        "## 1. Task Type",
        "",
        "Image classification",
        "",
        "## 2. Experimental Settings",
        "",
        settings,
        "",
        f"## 3. {results_heading}",
        "",
        "| Model | Train Accuracy | Validation Accuracy | Test Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} | {row['train_accuracy']:.4f} | "
            f"{row['validation_accuracy']:.4f} | {row['test_accuracy']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
