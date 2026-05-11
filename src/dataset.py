from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import ExperimentConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES = {
    "empty": "empty",
    "vacant": "empty",
    "free": "empty",
    "not occupied": "empty",
    "occupied": "occupied",
    "full": "occupied",
}


class DatasetPreparationError(RuntimeError):
    """Raised when PKLot data is not ready for training."""


@dataclass(slots=True)
class Sample:
    path: Path
    label: int


class ParkingSpotDataset(Dataset):
    def __init__(self, samples: list[Sample], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return image, label


def build_transforms(image_size: int, augmentation: str):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]

    if augmentation == "basic":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if augmentation == "strong":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(base)


def build_dataloaders(
    config: ExperimentConfig,
    augmentation: str,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict[str, int]]:
    samples = load_processed_samples(config.processed_data_dir, config.max_samples_per_class, config.seed)
    split_samples = stratified_split(
        samples=samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_dataset = ParkingSpotDataset(
        split_samples["train"],
        transform=build_transforms(config.image_size, augmentation=augmentation),
    )
    eval_transform = build_transforms(config.image_size, augmentation="none")
    train_eval_dataset = ParkingSpotDataset(split_samples["train"], transform=eval_transform)
    val_dataset = ParkingSpotDataset(split_samples["val"], transform=eval_transform)
    test_dataset = ParkingSpotDataset(split_samples["test"], transform=eval_transform)

    num_workers = config.resolved_num_workers()
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": config.use_pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs = dict(loader_kwargs)
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = config.persistent_workers
        eval_loader_kwargs["persistent_workers"] = False

    train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kwargs)
    train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **eval_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs)

    counts = {name: len(items) for name, items in split_samples.items()}
    return train_loader, train_eval_loader, val_loader, test_loader, counts


def ensure_dataset_available(config: ExperimentConfig) -> None:
    if has_processed_dataset(config.processed_data_dir):
        return
    try:
        prepare_dataset_from_huggingface(config)
    except Exception as error:
        raise DatasetPreparationError(
            "PKLot dataset is not ready. First try `python3 src/run_experiments.py --prepare-only` "
            "after installing dependencies. If the Hugging Face/FiftyOne path is unavailable, "
            "manually place real parking-space crops under "
            "`data/processed/pklot_binary/{empty,occupied}`."
        ) from error


def has_processed_dataset(processed_dir: Path) -> bool:
    empty_dir = processed_dir / "empty"
    occupied_dir = processed_dir / "occupied"
    return _count_images(empty_dir) > 0 and _count_images(occupied_dir) > 0


def load_processed_samples(
    processed_dir: Path,
    max_samples_per_class: int | None,
    seed: int,
) -> list[Sample]:
    if not has_processed_dataset(processed_dir):
        raise DatasetPreparationError(
            f"Processed dataset not found at {processed_dir}. "
            "Prepare PKLot first or populate empty/occupied class folders."
        )

    rng = random.Random(seed)
    samples: list[Sample] = []
    class_to_label = {"empty": 0, "occupied": 1}

    for class_name, label in class_to_label.items():
        class_dir = processed_dir / class_name
        class_samples = [
            Sample(path=image_path, label=label)
            for image_path in sorted(class_dir.rglob("*"))
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_samples_per_class is not None and len(class_samples) > max_samples_per_class:
            class_samples = rng.sample(class_samples, max_samples_per_class)
        samples.extend(class_samples)

    if not samples:
        raise DatasetPreparationError(f"No images were found in {processed_dir}.")
    return samples


def stratified_split(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for label_samples in grouped.values():
        rng.shuffle(label_samples)
        total = len(label_samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if train_end == 0 or val_end == train_end or val_end >= total:
            raise DatasetPreparationError(
                "Dataset split failed because one class does not have enough examples."
            )

        splits["train"].extend(label_samples[:train_end])
        splits["val"].extend(label_samples[train_end:val_end])
        splits["test"].extend(label_samples[val_end:])

    rng.shuffle(splits["train"])
    rng.shuffle(splits["val"])
    rng.shuffle(splits["test"])
    return splits


def prepare_dataset_from_huggingface(config: ExperimentConfig) -> None:
    try:
        import fiftyone as fo
        import fiftyone.core.labels as fol
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError as error:
        raise DatasetPreparationError(
            "Optional dependency `fiftyone` is required for Hugging Face PKLot preparation."
        ) from error

    processed_dir = config.processed_data_dir
    if has_processed_dataset(processed_dir):
        return

    processed_dir.mkdir(parents=True, exist_ok=True)
    for class_name in ("empty", "occupied"):
        (processed_dir / class_name).mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {}
    if config.hf_max_samples is not None:
        dataset_kwargs["max_samples"] = config.hf_max_samples

    dataset = load_from_hub(config.hf_dataset_name, **dataset_kwargs)
    class_counts = {"empty": 0, "occupied": 0}
    max_per_class = config.max_samples_per_class

    for sample in dataset.iter_samples(progress=True):
        image_path = Path(sample.filepath)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        for field_name in sample.field_names:
            field_value = sample[field_name]
            if not isinstance(field_value, fol.Polylines):
                continue

            for index, polyline in enumerate(field_value.polylines):
                class_name = _resolve_class_name(polyline)
                if class_name is None:
                    continue
                if max_per_class is not None and class_counts[class_name] >= max_per_class:
                    continue

                crop = _crop_from_polyline(image, polyline.points, width, height)
                if crop is None:
                    continue

                stem = f"{image_path.stem}_{field_name}_{index:04d}.png"
                output_path = processed_dir / class_name / stem
                crop.save(output_path)
                class_counts[class_name] += 1

        if max_per_class is not None and all(
            class_counts[name] >= max_per_class for name in class_counts
        ):
            break

    if not has_processed_dataset(processed_dir):
        raise DatasetPreparationError(
            "Hugging Face PKLot download completed, but no binary parking-space crops were exported. "
            "Inspect the dataset fields or use manual setup."
        )

    if isinstance(dataset, fo.Dataset):
        dataset.delete()


def prepare_dataset_from_local_imagefolders(source_dir: Path, processed_dir: Path) -> bool:
    if not source_dir.exists():
        return False

    processed_dir.mkdir(parents=True, exist_ok=True)
    found = False
    for folder in source_dir.iterdir():
        if not folder.is_dir():
            continue
        class_name = CLASS_ALIASES.get(folder.name.lower())
        if class_name is None:
            continue
        target_dir = processed_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in folder.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            shutil.copy2(image_path, target_dir / image_path.name)
            found = True
    return found and has_processed_dataset(processed_dir)


def _resolve_class_name(polyline) -> str | None:
    candidates = [getattr(polyline, "label", None)]
    candidates.extend(getattr(polyline, "tags", []) or [])
    candidates.append(getattr(polyline, "occupancy_status", None))
    attributes = getattr(polyline, "attributes", {}) or {}
    candidates.extend(attributes.values())

    for value in candidates:
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in CLASS_ALIASES:
            return CLASS_ALIASES[normalized]
    return None


def _crop_from_polyline(image: Image.Image, points, width: int, height: int) -> Image.Image | None:
    flat_points = []
    for poly in points:
        for x_ratio, y_ratio in poly:
            flat_points.append((x_ratio * width, y_ratio * height))

    if not flat_points:
        return None

    xs = [point[0] for point in flat_points]
    ys = [point[1] for point in flat_points]
    left = max(int(min(xs)), 0)
    top = max(int(min(ys)), 0)
    right = min(int(max(xs)), width)
    bottom = min(int(max(ys)), height)

    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
