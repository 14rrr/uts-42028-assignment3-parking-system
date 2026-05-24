"""ROI map I/O, cropping, and overlay drawing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLOT_MAP = ROOT / "assets" / "custom_slot_map.json"


def load_slot_map(path: Path | None = None) -> dict[str, Any]:
    map_path = path or DEFAULT_SLOT_MAP
    with map_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "slots" not in data and isinstance(data, list):
        return {"slots": data}
    return data


def crop_slot_rois(frame_bgr: np.ndarray, slot_map: dict[str, Any]) -> list[tuple[str, Image.Image]]:
    slots = slot_map.get("slots", slot_map)
    h, w = frame_bgr.shape[:2]
    crops: list[tuple[str, Image.Image]] = []

    for slot in slots:
        slot_id = str(slot.get("id", slot.get("slot_id", len(crops))))
        points = np.array(slot["polygon"], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        x, y, bw, bh = cv2.boundingRect(points)
        roi = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue
        crops.append((slot_id, Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))))

    return crops


def draw_slot_overlay(
    frame_bgr: np.ndarray,
    slot_map: dict[str, Any],
    predictions: dict[str, tuple[str, float]],
    highlight_ids: set[str] | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    slots = slot_map.get("slots", slot_map)
    highlight_ids = highlight_ids or set()
    colors = {
        "available": (46, 204, 113),
        "occupied": (231, 76, 60),
        "uncertain": (241, 196, 15),
    }

    for slot in slots:
        slot_id = str(slot.get("id", slot.get("slot_id", "")))
        label, _ = predictions.get(slot_id, ("uncertain", 0.0))
        color = (255, 255, 0) if slot_id in highlight_ids else colors.get(label, (149, 165, 166))
        pts = np.array(slot["polygon"], dtype=np.int32)
        cv2.polylines(out, [pts], True, color, 2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.putText(out, slot_id, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out
