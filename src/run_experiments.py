from __future__ import annotations

import argparse
from pathlib import Path
import re
import torch

from config import ExperimentConfig, MODEL_SETTINGS
from dataset import (
    DatasetPreparationError,
    build_dataloaders,
    ensure_dataset_available,
    has_processed_dataset,
    prepare_dataset_from_local_imagefolders,
)
from evaluate import evaluate_model
from models import build_model
from train import train_model
from utils import (
    benchmark_batch_sizes,
    ensure_directories,
    get_runtime_config,
    save_history_plot,
    save_json,
    seed_everything,
    write_markdown_summary,
    write_results_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Parkivo Part C CNN experiments.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--hf-max-samples", type=int, default=None)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument(
        "--batch-size-candidates",
        type=int,
        nargs="+",
        default=[512, 1024, 1536, 2048, 2304, 2560],
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return slug


def print_device_sanity(runtime) -> None:
    print(f"Detected device: {runtime.device}")
    if runtime.device.type == "cuda":
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_gb = props.total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(device_index) / (1024**3)
        print(
            "CUDA sanity:"
            f" name={props.name}, total_vram_gb={total_gb:.2f},"
            f" allocated_gb={allocated_gb:.2f}, reserved_gb={reserved_gb:.2f}"
        )
        print(
            "CUDA runtime:"
            f" amp={runtime.use_amp}, channels_last={runtime.use_channels_last},"
            f" cudnn_benchmark={torch.backends.cudnn.benchmark}"
        )
    else:
        if torch.cuda.is_available():
            print(
                "Resource sanity: CUDA is visible but failed the real CUDA kernel probe, "
                "so the pipeline is falling back to a non-CUDA device."
            )
        else:
            print("Resource sanity: running without CUDA acceleration.")


def get_processed_class_counts(processed_dir: Path) -> dict[str, int]:
    counts = {}
    for class_name in ("empty", "occupied"):
        class_dir = processed_dir / class_name
        counts[class_name] = sum(1 for path in class_dir.rglob("*") if path.is_file())
    return counts


def compute_pos_weight(train_dataset) -> float | None:
    positives = sum(sample.label for sample in train_dataset.samples)
    negatives = len(train_dataset.samples) - positives
    if positives == 0 or negatives == 0:
        return None
    return negatives / positives


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        max_samples_per_class=args.max_samples_per_class,
        hf_max_samples=args.hf_max_samples,
    )

    ensure_directories(
        [
            config.raw_data_dir,
            config.processed_data_dir,
            config.checkpoints_dir,
            config.plots_dir,
            config.reports_dir,
        ]
    )
    seed_everything(config.seed)

    if not has_processed_dataset(config.processed_data_dir):
        mirrored = prepare_dataset_from_local_imagefolders(
            source_dir=config.raw_data_dir,
            processed_dir=config.processed_data_dir,
        )
        if not mirrored:
            ensure_dataset_available(config)

    if args.prepare_only:
        print(f"Processed dataset is ready at: {config.processed_data_dir}")
        return

    runtime = get_runtime_config(use_amp=config.use_amp, use_channels_last=config.use_channels_last)
    config.use_pin_memory = runtime.use_pin_memory
    print_device_sanity(runtime)
    processed_class_counts = get_processed_class_counts(config.processed_data_dir)
    total_processed_crops = sum(processed_class_counts.values())
    dataset_variant = "full PKLot" if config.max_samples_per_class is None else "PKLot subset"

    if args.auto_batch_size and runtime.device.type == "cuda":
        recommended_batch_size, batch_benchmark = benchmark_batch_sizes(
            model_builder=lambda: build_model("resnet18", config.image_size),
            image_size=config.image_size,
            device=runtime.device,
            candidates=args.batch_size_candidates,
            use_amp=runtime.use_amp,
            use_channels_last=runtime.use_channels_last,
        )
        if recommended_batch_size is not None:
            config.batch_size = recommended_batch_size
            print(f"Selected batch size from benchmark: {config.batch_size}")
        for row in batch_benchmark:
            if row["status"] == "ok":
                print(
                    "Batch benchmark:"
                    f" batch_size={row['batch_size']},"
                    f" images_per_second={row['images_per_second']:.2f},"
                    f" peak_memory_gb={row['peak_memory_gb']:.2f}"
                )
            else:
                print(f"Batch benchmark: batch_size={row['batch_size']}, status=oom")

    model_settings = MODEL_SETTINGS
    csv_path = config.reports_dir / "lecture_aligned_initial_results.csv"
    md_path = config.reports_dir / "lecture_aligned_part_c_ready.md"
    summary_title = "Part C Ready Summary"
    model_group_description = (
        "Three lecture-aligned CNN architectures were trained from scratch: "
        "LeNet-5 CNN, AlexNet CNN, and ResNet-18 CNN. Legacy custom CNN results are "
        "preserved separately under legacy_custom_cnn_* report files."
    )

    results = []

    for model_name, settings in model_settings.items():
        train_loader, train_eval_loader, val_loader, test_loader, split_counts = build_dataloaders(
            config=config,
            augmentation=settings["augmentation"],
        )
        pos_weight = compute_pos_weight(train_loader.dataset)
        class_weighting = (
            f"pos_weight={pos_weight:.4f} from train split"
            if pos_weight is not None
            else "not used"
        )
        model = build_model(settings["builder"], config.image_size).to(runtime.device)
        if runtime.use_channels_last and runtime.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        model_slug = slugify_model_name(model_name)
        checkpoint_path = config.checkpoints_dir / f"{model_slug}.pt"
        history_path = config.reports_dir / f"{model_slug}_history.json"
        plot_path = config.plots_dir / f"{model_slug}.png"

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            device=runtime.device,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            checkpoint_path=checkpoint_path,
            early_stopping=settings["early_stopping"],
            patience=config.patience,
            weight_decay=config.weight_decay,
            pos_weight=pos_weight,
            use_amp=runtime.use_amp,
            use_channels_last=runtime.use_channels_last,
        )

        train_metrics = evaluate_model(
            model,
            train_eval_loader,
            runtime.device,
            use_amp=runtime.use_amp,
            use_channels_last=runtime.use_channels_last,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            runtime.device,
            use_amp=runtime.use_amp,
            use_channels_last=runtime.use_channels_last,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            runtime.device,
            use_amp=runtime.use_amp,
            use_channels_last=runtime.use_channels_last,
        )
        save_history_plot(history, plot_path, model_name)
        save_json(
            {
                "model": model_name,
                "device": str(runtime.device),
                "device_name": runtime.device_name,
                "dataset_variant": dataset_variant,
                "processed_class_counts": processed_class_counts,
                "total_processed_crops": total_processed_crops,
                "class_weighting": class_weighting,
                "split_counts": split_counts,
                "history": history,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            history_path,
        )

        results.append(
            {
                "model": model_name,
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    write_results_csv(results, csv_path)
    write_markdown_summary(
        rows=results,
        output_path=md_path,
        title=summary_title,
        dataset_name="PKLot",
        image_size=config.image_size,
        batch_size=config.batch_size,
        epochs=config.epochs,
        dataset_variant=dataset_variant,
        total_crops=total_processed_crops,
        empty_crops=processed_class_counts["empty"],
        occupied_crops=processed_class_counts["occupied"],
        class_weighting=class_weighting,
        model_group_description=model_group_description,
        results_heading="Lecture-Aligned Initial Results",
    )

    print("Experiments complete.")
    print(f"Results CSV: {csv_path}")
    print(f"Markdown summary: {md_path}")


if __name__ == "__main__":
    try:
        main()
    except DatasetPreparationError as error:
        raise SystemExit(str(error)) from error
