from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class ExperimentConfig:
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "pklot_binary"
    )
    checkpoints_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "checkpoints"
    )
    plots_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "plots")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "reports")
    image_size: int = 128
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    num_workers: int = -1
    patience: int = 4
    max_samples_per_class: int | None = None
    use_pin_memory: bool = True
    use_amp: bool = True
    use_channels_last: bool = True
    use_cudnn_benchmark: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    hf_dataset_name: str = "Voxel51/PKLot"
    hf_max_samples: int | None = None

    def resolved_num_workers(self) -> int:
        if self.num_workers >= 0:
            return self.num_workers
        cpu_count = os.cpu_count() or 4
        return max(2, min(8, cpu_count // 2))


MODEL_SETTINGS = {
    "LeNet-5 CNN": {
        "builder": "lenet5",
        "early_stopping": True,
        "augmentation": "none",
    },
    "AlexNet CNN": {
        "builder": "alexnet",
        "early_stopping": True,
        "augmentation": "basic",
    },
    "ResNet-18 CNN": {
        "builder": "resnet18",
        "early_stopping": True,
        "augmentation": "strong",
    },
}
