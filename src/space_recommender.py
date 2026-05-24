"""Pick the best available parking spaces for the driver display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference_service import SlotPrediction


@dataclass(frozen=True)
class RecommendedSpace:
    slot_id: str
    confidence: float
    score: float
    reason: str


def _centroid(polygon: list[list[int]]) -> tuple[float, float]:
    arr = np.array(polygon, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def recommend_spaces(
    predictions: list[SlotPrediction],
    slot_map: dict,
    entry_point: tuple[float, float] | None = None,
    top_k: int = 3,
) -> list[RecommendedSpace]:
    slots = {str(s.get("id", s.get("slot_id"))): s for s in slot_map.get("slots", slot_map)}
    if entry_point is None:
        pts = [p for s in slots.values() for p in s["polygon"]]
        entry_point = (float(np.mean([p[0] for p in pts])), float(max(p[1] for p in pts)))

    ranked: list[RecommendedSpace] = []
    for pred in predictions:
        if pred.occupied or pred.label != "available":
            continue
        slot = slots.get(pred.slot_id)
        if not slot:
            continue
        cx, cy = _centroid(slot["polygon"])
        dist = float(np.hypot(cx - entry_point[0], cy - entry_point[1]))
        score = pred.confidence * 100.0 - dist * 0.05
        ranked.append(
            RecommendedSpace(
                slot_id=pred.slot_id,
                confidence=pred.confidence,
                score=score,
                reason=f"High confidence ({pred.confidence:.0%}) and close to entry",
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]
