from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


class LeNet5CNN(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        flattened_dim = self._infer_flattened_dim(image_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 1),
        )

    def _infer_flattened_dim(self, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            return self.features(dummy).numel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.classifier(x)
        return x.squeeze(1)


class AlexNetCNN(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.model = models.alexnet(weights=None, num_classes=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).squeeze(1)


class ResNet18CNN(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.model = models.resnet18(weights=None, num_classes=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).squeeze(1)


MODEL_REGISTRY = {
    "lenet5": LeNet5CNN,
    "alexnet": AlexNetCNN,
    "resnet18": ResNet18CNN,
}


TRAINED_CHECKPOINTS = {
    "lenet5": Path("outputs/checkpoints/lenet_5_cnn.pt"),
    "alexnet": Path("outputs/checkpoints/alexnet_cnn.pt"),
    "resnet18": Path("outputs/checkpoints/resnet_18_cnn.pt"),
}


def build_model(model_key: str, image_size: int) -> nn.Module:
    try:
        return MODEL_REGISTRY[model_key](image_size=image_size)
    except KeyError as error:
        raise ValueError(f"Unknown model key: {model_key}") from error


def load_trained_model(
    model_key: str,
    image_size: int,
    project_root: Path | None = None,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    model = build_model(model_key, image_size=image_size)
    checkpoint_path = TRAINED_CHECKPOINTS[model_key]
    if project_root is not None:
        checkpoint_path = project_root / checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    return model
