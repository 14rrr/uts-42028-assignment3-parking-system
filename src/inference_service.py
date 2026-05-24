"""Load CNN checkpoints once and run batched ROI classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Adjust to match your repo (e.g. src.models, src.model_factory).
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"

MODEL_CHOICES = {
    "LeNet-5 CNN": "lenet",
    "AlexNet CNN": "alexnet",
    "ResNet-18 CNN": "resnet18",
}

IMAGENET_NORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass(frozen=True)
class SlotPrediction:
    slot_id: str
    label: str
    confidence: float
    occupied: bool


@lru_cache(maxsize=3)
def _load_model(model_key: str) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINT_DIR / f"{model_key}_best.pt"
    if not ckpt_path.exists():
        alt = CHECKPOINT_DIR / f"{model_key}.pth"
        ckpt_path = alt if alt.exists() else ckpt_path

    model = build_model(model_key, num_classes=2)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, device


def _preprocess(crop: Image.Image) -> torch.Tensor:
    return IMAGENET_NORM(crop.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def predict_slots(
    crops: Iterable[tuple[str, Image.Image]],
    model_display_name: str,
    occupied_index: int = 1,
    confidence_threshold: float = 0.55,
) -> list[SlotPrediction]:
    model_key = MODEL_CHOICES[model_display_name]
    model, device = _load_model(model_key)

    items = list(crops)
    if not items:
        return []

    batch = torch.cat([_preprocess(img) for _, img in items], dim=0).to(device)
    probs = torch.softmax(model(batch), dim=1).cpu().numpy()

    class_names = ["available", "occupied"]
    results: list[SlotPrediction] = []
    for (slot_id, _), row in zip(items, probs):
        pred_idx = int(row.argmax())
        confidence = float(row[pred_idx])
        if confidence < confidence_threshold:
            label = "uncertain"
        else:
            label = class_names[pred_idx]
        occupied = pred_idx == occupied_index and label != "uncertain"
        results.append(
            SlotPrediction(slot_id=slot_id, label=label, confidence=confidence, occupied=occupied)
        )
    return results
