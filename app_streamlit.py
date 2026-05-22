from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
CUSTOM_ROI_CONFIG_PATH = ASSETS_DIR / "custom_slot_map.json"
DEMO_FRAME_CLEAN_PATH = ASSETS_DIR / "demo_frame_clean.jpg"
LEGACY_FRAME_PATH = ASSETS_DIR / "image.jpg"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ExperimentConfig, MODEL_SETTINGS  # noqa: E402
from models import load_trained_model  # noqa: E402


MODEL_OPTIONS = {
    "LeNet-5 CNN": "lenet5",
    "AlexNet CNN": "alexnet",
    "ResNet-18 CNN": "resnet18",
}

LABEL_MAPPING = {
    0: "Empty",
    1: "Occupied",
}

SIMULATED_OCCUPIED_SLOTS = {
    "A01",
    "A02",
    "A05",
    "A09",
    "B03",
    "B04",
    "B08",
    "B10",
    "C01",
    "C02",
    "C06",
    "C09",
    "D04",
    "D07",
    "D08",
}


def load_roi_config(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ROI configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    slots = data.get("slots", data)
    if not isinstance(slots, list):
        raise ValueError("ROI configuration must contain a list named 'slots'.")
    return normalize_roi_slots(slots)


def load_active_roi_config() -> tuple[list[dict[str, Any]], str, Path | None]:
    if CUSTOM_ROI_CONFIG_PATH.exists():
        return load_roi_config(CUSTOM_ROI_CONFIG_PATH), "Custom ROI map", CUSTOM_ROI_CONFIG_PATH
    raise FileNotFoundError(f"Official ROI map is required: {CUSTOM_ROI_CONFIG_PATH}")


def derive_roi_coverage_boundary(roi_slots: list[dict[str, Any]], padding: int = 12) -> list[list[int]]:
    points: list[tuple[float, float]] = []
    for slot in roi_slots:
        if slot.get("polygon"):
            points.extend((float(x), float(y)) for x, y in slot["polygon"])
        else:
            x, y, width, height = slot_bounds(slot)
            points.extend(
                [
                    (float(x), float(y)),
                    (float(x + width), float(y)),
                    (float(x + width), float(y + height)),
                    (float(x), float(y + height)),
                ]
            )
    unique_points = sorted(set(points))
    if len(unique_points) < 3:
        return []

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    centroid_x = sum(x for x, _ in hull) / len(hull)
    centroid_y = sum(y for _, y in hull) / len(hull)
    expanded: list[list[int]] = []
    for x, y in hull:
        dx = x - centroid_x
        dy = y - centroid_y
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        expanded.append([int(round(x + padding * dx / length)), int(round(y + padding * dy / length))])
    return expanded


def default_frame_path() -> Path | None:
    if DEMO_FRAME_CLEAN_PATH.exists():
        return DEMO_FRAME_CLEAN_PATH
    if LEGACY_FRAME_PATH.exists():
        return LEGACY_FRAME_PATH
    return None


def get_default_demo_frame() -> tuple[Image.Image | None, Path | None]:
    frame_path = default_frame_path()
    if frame_path is None:
        return None, None
    return Image.open(frame_path).convert("RGB"), frame_path


def normalize_roi_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, slot in enumerate(slots, start=1):
        slot_id = str(slot.get("slot_id") or slot.get("Slot ID") or f"A{index:02d}").upper()
        row = str(slot.get("row") or slot.get("Row") or slot_id[0]).upper()
        spot = int(slot.get("spot") or slot.get("Spot") or slot_id[1:] or index)
        polygon = slot.get("polygon")
        if polygon:
            normalized_polygon = [[int(round(float(x))), int(round(float(y)))] for x, y in polygon]
            normalized_slot = {
                "slot_id": slot_id,
                "row": row,
                "spot": spot,
                "polygon": normalized_polygon,
            }
            for key in ("original_id", "ground_truth_status"):
                if key in slot:
                    normalized_slot[key] = slot[key]
            normalized.append(normalized_slot)
            continue

        roi = slot.get("roi")
        if roi is None:
            roi = [slot.get("x", 0), slot.get("y", 0), slot.get("width", 64), slot.get("height", 82)]
        x, y, width, height = [int(float(value)) for value in roi]
        normalized.append(
            {
                "slot_id": slot_id,
                "row": row,
                "spot": spot,
                "roi": [x, y, max(1, width), max(1, height)],
            }
        )
    return normalized


def preprocess_image(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_parking_slot(model: torch.nn.Module, image: Image.Image, image_size: int = 128) -> dict[str, float | str]:
    start_time = time.perf_counter()
    tensor = preprocess_image(image, image_size=image_size)

    with torch.no_grad():
        logit = model(tensor).reshape(-1)[0]
        probability = torch.sigmoid(logit).item()

    inference_ms = (time.perf_counter() - start_time) * 1000
    label_id = 1 if probability >= 0.5 else 0
    confidence = probability if label_id == 1 else 1.0 - probability

    return {
        "label": LABEL_MAPPING[label_id],
        "occupied_probability": probability,
        "confidence": confidence,
        "logit": logit.item(),
        "inference_ms": inference_ms,
    }


def roi_as_text(roi: list[int | float]) -> str:
    x, y, width, height = roi
    return f"x={int(x)}, y={int(y)}, w={int(width)}, h={int(height)}"


def slot_bounds(slot: dict[str, Any]) -> list[int]:
    if slot.get("polygon"):
        xs = [int(point[0]) for point in slot["polygon"]]
        ys = [int(point[1]) for point in slot["polygon"]]
        left = min(xs)
        top = min(ys)
        return [left, top, max(1, max(xs) - left), max(1, max(ys) - top)]
    x, y, width, height = [int(value) for value in slot["roi"]]
    return [x, y, width, height]


def slot_coordinates_text(slot: dict[str, Any]) -> str:
    if slot.get("polygon"):
        return f"polygon ({len(slot['polygon'])} points)"
    return roi_as_text(slot["roi"])


def generate_ground_truth_slots(roi_slots: list[dict[str, Any]]) -> list[dict[str, str | float]]:
    slots: list[dict[str, str | float]] = []
    for roi_slot in roi_slots:
        status_text = str(roi_slot.get("ground_truth_status", "")).strip().lower()
        status = "occupied" if status_text == "occupied" else "available"
        slots.append(
            {
                "Slot ID": str(roi_slot["slot_id"]),
                "Row": str(roi_slot["row"]),
                "Spot": f"{int(roi_slot['spot']):02d}",
                "ROI coordinates": slot_coordinates_text(roi_slot),
                "Status": status,
                "Confidence": "PKLot annotation",
                "Occupied probability": "",
                "Crop bounds": roi_as_text(slot_bounds(roi_slot)),
                "Last updated": datetime.now().strftime("%H:%M:%S"),
            }
        )

    return apply_recommendations(slots)


def row_count_summary(roi_slots: list[dict[str, Any]]) -> str:
    rows: dict[str, int] = {}
    for slot in roi_slots:
        row = str(slot["row"])
        rows[row] = rows.get(row, 0) + 1
    return ", ".join(f"{row}: {count}" for row, count in sorted(rows.items()))


def base_slot_status(slot: dict[str, str | float]) -> str:
    status = str(slot.get("Status", "")).strip().lower()
    if status in {"recommended", "recommended_secondary"}:
        return "available"
    return status


def rank_recommended_slots(slots: list[dict[str, str | float]], max_recommendations: int = 3) -> list[str]:
    slot_by_position: dict[tuple[str, int], dict[str, str | float]] = {}
    available_slots: list[dict[str, str | float]] = []
    for slot in slots:
        row = str(slot["Row"])
        spot = int(str(slot["Spot"]))
        slot_by_position[(row, spot)] = slot
        if base_slot_status(slot) == "available":
            available_slots.append(slot)

    scored: list[tuple[int, str, int, str]] = []
    for slot in available_slots:
        row = str(slot["Row"])
        spot = int(str(slot["Spot"]))
        left = slot_by_position.get((row, spot - 1))
        right = slot_by_position.get((row, spot + 1))
        left_available = left is not None and base_slot_status(left) == "available"
        right_available = right is not None and base_slot_status(right) == "available"
        has_left = left is not None
        has_right = right is not None

        score = 0
        if left_available:
            score += 2
        if right_available:
            score += 2
        if has_left and has_right:
            score += 1

        scored.append((score, row, spot, str(slot["Slot ID"])))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [slot_id for _, _, _, slot_id in scored[:max_recommendations]]


def apply_recommendations(slots: list[dict[str, str | float]], max_recommendations: int = 3) -> list[dict[str, str | float]]:
    updated: list[dict[str, str | float]] = []
    for slot in slots:
        slot_copy = dict(slot)
        if base_slot_status(slot_copy) != "occupied":
            slot_copy["Status"] = "available"
        else:
            slot_copy["Status"] = "occupied"
        updated.append(slot_copy)

    recommended_ids = rank_recommended_slots(updated, max_recommendations=max_recommendations)
    for index, slot_id in enumerate(recommended_ids):
        for slot in updated:
            if str(slot["Slot ID"]) == slot_id and slot["Status"] != "occupied":
                slot["Status"] = "recommended" if index == 0 else "recommended_secondary"
                break
    return updated


def recommendation_reason(slot_id: str, slots: list[dict[str, str | float]]) -> str:
    slot = next((item for item in slots if str(item["Slot ID"]) == slot_id), None)
    if slot is None:
        return "Available space"
    row = str(slot["Row"])
    spot = int(str(slot["Spot"]))
    by_position = {(str(item["Row"]), int(str(item["Spot"]))): item for item in slots}
    left = by_position.get((row, spot - 1))
    right = by_position.get((row, spot + 1))
    available_neighbours = sum(
        neighbour is not None and base_slot_status(neighbour) == "available"
        for neighbour in (left, right)
    )
    if available_neighbours == 2:
        return "Open space with both neighbouring bays available"
    if available_neighbours == 1:
        return "Open space with one neighbouring bay available"
    return "Available space"


def recommended_slot_ids(slots: list[dict[str, str | float]]) -> list[str]:
    primary = [str(slot["Slot ID"]) for slot in slots if slot["Status"] == "recommended"]
    secondary = [str(slot["Slot ID"]) for slot in slots if slot["Status"] == "recommended_secondary"]
    return primary + secondary


def slot_counts(slots: list[dict[str, str | float]]) -> tuple[int, int, list[str]]:
    available = sum(1 for slot in slots if base_slot_status(slot) == "available")
    occupied = sum(1 for slot in slots if slot["Status"] == "occupied")
    return available, occupied, recommended_slot_ids(slots)


def choose_recommended_slot(slots: list[dict[str, str | float]]) -> str:
    recommended = rank_recommended_slots(slots, max_recommendations=1)
    return recommended[0] if recommended else ""


def crop_slot(frame: Image.Image, slot: dict[str, Any]) -> Image.Image:
    x, y, width, height = slot_bounds(slot)
    frame_width, frame_height = frame.size
    left = max(0, min(x, frame_width))
    top = max(0, min(y, frame_height))
    right = max(left + 1, min(x + width, frame_width))
    bottom = max(top + 1, min(y + height, frame_height))
    return frame.crop((left, top, right, bottom))


def classify_rois(
    model: torch.nn.Module,
    frame: Image.Image,
    roi_slots: list[dict[str, Any]],
    image_size: int,
) -> list[dict[str, str | float]]:
    results: list[dict[str, str | float]] = []
    for roi_slot in roi_slots:
        crop = crop_slot(frame, roi_slot)
        prediction = predict_parking_slot(model, crop, image_size=image_size)
        status = "occupied" if prediction["label"] == "Occupied" else "available"
        x, y, width, height = slot_bounds(roi_slot)
        results.append(
            {
                "Slot ID": str(roi_slot["slot_id"]),
                "Row": str(roi_slot["row"]),
                "Spot": f"{int(roi_slot['spot']):02d}",
                "ROI coordinates": slot_coordinates_text(roi_slot),
                "Crop bounds": roi_as_text([x, y, width, height]),
                "Status": status,
                "Confidence": f"{float(prediction['confidence']):.2f}",
                "Occupied probability": f"{float(prediction['occupied_probability']):.4f}",
                "Inference ms": f"{float(prediction['inference_ms']):.2f}",
                "Last updated": datetime.now().strftime("%H:%M:%S"),
            }
        )

    return apply_recommendations(results)


def create_simulated_camera_frame(roi_slots: list[dict[str, Any]]) -> Image.Image:
    max_right = max((slot_bounds(slot)[0] + slot_bounds(slot)[2] for slot in roi_slots), default=930)
    max_bottom = max((slot_bounds(slot)[1] + slot_bounds(slot)[3] for slot in roi_slots), default=535)
    width = max(1000, min(1280, max_right + 80))
    height = max(560, min(720, max_bottom + 80))
    image = Image.new("RGB", (width, height), "#6f7778")
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 30, width - 70, height - 25), outline="#e9ecef", width=3)
    draw.rectangle((width - 60, height // 2 - 75, width - 15, height // 2 + 75), fill="#d8dee2", outline="#23313a", width=2)
    draw.text((width - 56, height // 2 - 60), "ENTRY", fill="#23313a")

    for roi_slot in roi_slots:
        slot_id = str(roi_slot["slot_id"])
        x, y, width, height = slot_bounds(roi_slot)
        is_occupied = slot_id in SIMULATED_OCCUPIED_SLOTS
        fill = "#bfc7cc" if not is_occupied else "#b44d4d"
        if roi_slot.get("polygon"):
            points = [(int(px), int(py)) for px, py in roi_slot["polygon"]]
            draw.polygon(points, fill=fill, outline="#ffffff")
        else:
            draw.rectangle((x, y, x + width, y + height), fill=fill, outline="#ffffff", width=2)
        if is_occupied and width > 16 and height > 16:
            inset_x = min(8, max(2, width // 5))
            inset_y = min(18, max(2, height // 5))
            draw.rounded_rectangle((x + inset_x, y + inset_y, x + width - inset_x, y + height - inset_y), radius=7, fill="#842929")
        draw.text((x + 9, y + height - 20), slot_id, fill="#101820")
    return image


def draw_roi_overlay(
    frame: Image.Image,
    slots: list[dict[str, str | float]],
    roi_slots: list[dict[str, Any]],
    boundary: list[list[int]] | None = None,
    show_boundary: bool = False,
) -> Image.Image:
    overlay = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    status_by_id = {str(slot["Slot ID"]): str(slot["Status"]) for slot in slots}
    color_by_status = {
        "available": "#f3c623",
        "occupied": "#d82d2d",
        "recommended": "#168a35",
        "recommended_secondary": "#8bdc8f",
    }
    if show_boundary and boundary:
        boundary_points = [(int(x), int(y)) for x, y in boundary]
        draw.line(boundary_points + [boundary_points[0]], fill="#1d6bff", width=4)
    for roi_slot in roi_slots:
        slot_id = str(roi_slot["slot_id"])
        status = status_by_id.get(slot_id, "available")
        color = color_by_status.get(status, "#f3c623")
        line_width = 6 if status == "recommended" else 4
        if roi_slot.get("polygon"):
            points = [(int(x), int(y)) for x, y in roi_slot["polygon"]]
            draw.line(points + [points[0]], fill=color, width=line_width)
            label_x, label_y = points[0]
        else:
            x, y, width, height = slot_bounds(roi_slot)
            draw.rectangle((x, y, x + width, y + height), outline=color, width=line_width)
            label_x, label_y = x, y
        label_fill = "#101820" if status in {"available", "recommended_secondary"} else "#ffffff"
        draw.rectangle((label_x, label_y, label_x + 48, label_y + 18), fill=color)
        draw.text((label_x + 4, label_y + 2), slot_id, fill=label_fill)
    return overlay


def draw_roi_geometry_overlay(
    frame: Image.Image,
    roi_slots: list[dict[str, Any]],
    boundary: list[list[int]] | None = None,
    show_boundary: bool = False,
) -> Image.Image:
    overlay = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    if show_boundary and boundary:
        boundary_points = [(int(x), int(y)) for x, y in boundary]
        draw.line(boundary_points + [boundary_points[0]], fill="#1d6bff", width=4)

    for roi_slot in roi_slots:
        slot_id = str(roi_slot["slot_id"])
        if roi_slot.get("polygon"):
            points = [(int(x), int(y)) for x, y in roi_slot["polygon"]]
            draw.line(points + [points[0]], fill="#ffffff", width=3)
            label_x, label_y = points[0]
        else:
            x, y, width, height = slot_bounds(roi_slot)
            draw.rectangle((x, y, x + width, y + height), outline="#ffffff", width=3)
            label_x, label_y = x, y
        draw.rectangle((label_x, label_y, label_x + 48, label_y + 18), fill="#101820")
        draw.text((label_x + 4, label_y + 2), slot_id, fill="#ffffff")
    return overlay


def render_status_cards(st, total: int, available: int, occupied: int) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total spaces", total)
    col2.metric("Available spaces", available)
    col3.metric("Occupied spaces", occupied)


def render_recommendation_card(st, recommendations: list[str], slots: list[dict[str, str | float]]) -> None:
    if not recommendations:
        value = "No spaces available"
        text = "No available spaces detected"
        other_options = ""
    else:
        primary = recommendations[0]
        value = f"Row {primary[0]}, Spot {primary[1:]}"
        text = recommendation_reason(primary, slots)
        secondary = recommendations[1:3]
        if secondary:
            option_text = ", ".join(f"Row {slot_id[0]}, Spot {slot_id[1:]}" for slot_id in secondary)
            other_options = f"<div class='recommendation-options'><strong>Other good options:</strong> {option_text}</div>"
        else:
            other_options = ""

    st.markdown(
        f"""
        <div class='recommendation-card'>
            <div class='recommendation-label'>Best parking space</div>
            <div class='recommendation-value'>{value}</div>
            <div class='recommendation-text'>{text}</div>
            {other_options}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_parking_map(st, slots: list[dict[str, str | float]]) -> None:
    rows = sorted({str(slot["Row"]) for slot in slots})
    for row in rows:
        row_slots = [slot for slot in slots if slot["Row"] == row]
        st.markdown(f"**Row {row}**")
        columns = st.columns(len(row_slots))
        for column, slot in zip(columns, row_slots):
            status = str(slot["Status"])
            css_class = {
                "available": "slot-available",
                "occupied": "slot-occupied",
                "recommended": "slot-recommended",
                "recommended_secondary": "slot-recommended-secondary",
            }[status]
            label = f"{slot['Slot ID']}"
            if status == "recommended":
                label = f"{slot['Slot ID']}<br><span>BEST</span>"
            elif status == "recommended_secondary":
                label = f"{slot['Slot ID']}<br><span>ALT</span>"
            column.markdown(
                f"<div class='parking-slot {css_class}'>{label}</div>",
                unsafe_allow_html=True,
            )


def render_parking_legend(st) -> None:
    st.markdown(
        """
        <div class='map-legend'>
            <span><i class='legend-dot legend-occupied'></i>Red: Occupied</span>
            <span><i class='legend-dot legend-available'></i>Yellow: Available</span>
            <span><i class='legend-dot legend-recommended'></i>Green: Best recommendation</span>
            <span><i class='legend-dot legend-recommended-secondary'></i>Light green: Alternative recommendation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_app() -> None:
    try:
        import streamlit as st
    except ImportError as error:
        raise SystemExit(
            "Streamlit is not installed. Install it with: "
            r"C:\Users\pc\miniconda3\envs\ai\python.exe -m pip install streamlit"
        ) from error

    config = ExperimentConfig()
    roi_slots, roi_map_label, roi_map_path = load_active_roi_config()
    roi_boundary = derive_roi_coverage_boundary(roi_slots)
    default_demo_frame, current_default_frame_path = get_default_demo_frame()

    @st.cache_resource(show_spinner="Loading trained CNN checkpoint...")
    def cached_model(model_key: str) -> torch.nn.Module:
        return load_trained_model(
            model_key=model_key,
            image_size=config.image_size,
            project_root=PROJECT_ROOT,
            map_location="cpu",
        )

    st.set_page_config(page_title="Parkivo Parking Availability System", page_icon="P", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.85rem; padding-bottom: 1.25rem; }
        footer { visibility: hidden; }
        .recommendation-card {
            border-radius: 8px;
            border: 2px solid #168a35;
            background: #fff9c7;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
        }
        .recommendation-label { font-size: 1rem; font-weight: 700; color: #32404a; }
        .recommendation-value { font-size: 2.7rem; font-weight: 800; color: #1f2a30; line-height: 1.1; }
        .recommendation-text { font-size: 1.1rem; color: #32404a; margin-top: 0.3rem; }
        .recommendation-options { font-size: 0.95rem; color: #32404a; margin-top: 0.8rem; }
        .parking-slot {
            width: 100%;
            box-sizing: border-box;
            min-height: 50px;
            padding: 0.35rem 0.2rem;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-weight: 800;
            border: 1px solid rgba(49, 51, 63, 0.18);
            margin-bottom: 0.35rem;
            line-height: 1.05;
        }
        .parking-slot span { font-size: 0.72rem; }
        .slot-available { background: #fff2a8; color: #594a00; }
        .slot-occupied {
            background: #d82d2d;
            color: #ffffff;
            border: 3px solid #991b1b;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
        }
        .slot-recommended { background: #168a35; color: #ffffff; border: 4px solid #0d5f23; box-shadow: inset 0 0 0 2px rgba(255,255,255,0.55); }
        .slot-recommended-secondary { background: #d8f5df; color: #145a27; border: 3px solid #168a35; }
        .map-legend {
            display: flex;
            gap: 0.9rem;
            flex-wrap: wrap;
            color: #4c5963;
            font-size: 0.9rem;
            margin: 0.35rem 0 0.7rem 0;
        }
        .legend-dot {
            display: inline-block;
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 50%;
            margin-right: 0.3rem;
            border: 1px solid rgba(49, 51, 63, 0.18);
            vertical-align: -0.05rem;
        }
        .legend-occupied { background: #ffd9d9; }
        .legend-available { background: #fff2a8; }
        .legend-recommended { background: #168a35; }
        .legend-recommended-secondary { background: #d8f5df; }
        .status-line {
            color: #53616b;
            font-size: 0.92rem;
            margin: 0.2rem 0 0.8rem 0;
        }
        .note {
            border-left: 4px solid #2477bf;
            padding: 0.7rem 1rem;
            background: #f5f9fc;
            margin: 0.75rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "slot_results" not in st.session_state or st.session_state.get("source_mode") == "Simulation":
        st.session_state.slot_results = generate_ground_truth_slots(roi_slots)
        st.session_state.source_mode = "PKLot annotation status"
        st.session_state.status_note = "Initial status is loaded from official PKLot annotations. Run CNN Detection to update from the current frame."
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_detection_summary = None

    slots = st.session_state.slot_results
    st.session_state.slot_results = slots
    available, occupied, recommended = slot_counts(slots)
    total_spaces = len(slots)

    detection_tab, driver_tab, methodology_tab = st.tabs(["Parking Detection", "Driver Display", "ROI & Methodology"])

    with detection_tab:
        st.title("Parkivo Parking Availability System")
        st.caption("Fixed-camera parking-zone availability using CNN classification on cropped ROI images.")

        top_left, top_right = st.columns([1.35, 1])

        with top_right:
            selected_camera_model = st.selectbox("CNN model", list(MODEL_OPTIONS.keys()), index=2)
            uploaded_frame = st.file_uploader(
                "Upload full-frame image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="full_frame_upload",
            )

        if uploaded_frame is not None:
            frame = Image.open(uploaded_frame).convert("RGB")
            source_label = "uploaded frame"
            can_classify_frame = True
        elif default_demo_frame is not None:
            frame = default_demo_frame.copy()
            source_label = "default clean PKLot frame"
            can_classify_frame = True
        else:
            frame = create_simulated_camera_frame(roi_slots)
            source_label = "simulated fallback frame"
            can_classify_frame = False

        with top_left:
            st.image(
                draw_roi_overlay(frame, slots, roi_slots),
                caption="Current parking-zone frame with ROI status overlay",
                use_container_width=True,
            )

        with top_right:
            if st.button("Run CNN Detection", type="primary"):
                if can_classify_frame:
                    model_key = MODEL_OPTIONS[selected_camera_model]
                    model = cached_model(model_key)
                    detected_slots = classify_rois(model, frame, roi_slots, image_size=config.image_size)
                    detected_available, detected_occupied, detected_recommended = slot_counts(detected_slots)
                    st.session_state.slot_results = detected_slots
                    st.session_state.source_mode = f"CNN detection from {source_label} ({selected_camera_model})"
                    st.session_state.status_note = "Status is based on CNN predictions from the current frame."
                    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.last_detection_summary = {
                        "Model used": selected_camera_model,
                        "Total processed slots": len(detected_slots),
                        "Available count": detected_available,
                        "Occupied count": detected_occupied,
                        "Best recommended slot": detected_recommended[0] if detected_recommended else "None",
                        "Alternative recommended slots": ", ".join(detected_recommended[1:]) if len(detected_recommended) > 1 else "None",
                    }
                    st.rerun()
                else:
                    st.warning("No full-frame image is available for CNN detection.")

            render_recommendation_card(st, recommended, slots)
            render_status_cards(st, total_spaces, available, occupied)
            st.caption("Detection runs on the current frame shown on the left. Each official ROI is cropped and classified by the selected CNN.")

        render_parking_legend(st)
        st.subheader("Parking Zone Status")
        render_parking_map(st, slots)

        if st.session_state.get("last_detection_summary"):
            st.subheader("Detection Summary")
            st.table([st.session_state.last_detection_summary])
        else:
            st.info("Initial status is loaded from PKLot annotations. Run CNN Detection to update from the current frame.")

        with st.expander("Advanced / debug details", expanded=False):
            show_boundary = st.checkbox("Show monitored-zone boundary", value=False, key="detection_show_boundary")
            if show_boundary:
                st.image(
                    draw_roi_overlay(frame, slots, roi_slots, boundary=roi_boundary, show_boundary=True),
                    caption="Boundary debug overlay",
                    use_container_width=True,
                )
            active_path_text = str(roi_map_path.relative_to(PROJECT_ROOT)) if roi_map_path else "built-in defaults"
            st.write(f"Active ROI map: `{active_path_text}`")
            st.write(f"Source mode: `{st.session_state.source_mode}`")
            st.write(f"Last updated: `{st.session_state.last_updated}`")
            st.write("Boundary is derived from active official ROI polygons.")
            st.dataframe(slots, use_container_width=True, hide_index=True)

    with driver_tab:
        st.title("Recommended Parking")
        render_recommendation_card(st, recommended, slots)
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Available spaces", available)
        metric_right.metric("Occupied spaces", occupied)
        render_parking_legend(st)
        st.subheader("Parking Zone Status")
        render_parking_map(st, slots)
        st.caption(f"Updated: {st.session_state.last_updated}")
        if st.session_state.source_mode == "PKLot annotation status":
            st.info("Initial parking status shown. Run CNN Detection from the Parking Detection tab to update from the current frame.")

    with methodology_tab:
        st.header("ROI & Methodology")

        summary_left, summary_right = st.columns([1, 1.25])
        with summary_left:
            st.subheader("Official ROI Map")
            summary_rows = [
                {"Item": "ROI source", "Value": "Official PKLot/Voxel51 annotations"},
                {"Item": "Slot count", "Value": len(roi_slots)},
                {"Item": "Rows", "Value": row_count_summary(roi_slots)},
                {"Item": "Default frame", "Value": "assets/demo_frame_clean.jpg" if DEMO_FRAME_CLEAN_PATH.exists() else "assets/image.jpg"},
            ]
            st.table(summary_rows)
            st.write("The prototype monitors a selected fixed-camera parking zone rather than every bay in the wider facility.")
            st.write("Each official polygon ROI is cropped from the full frame and passed to a CNN classifier.")
            st.write("The task is cropped ROI image classification; the CNN does not localize cars in the full frame.")
            st.write("A full deployment could use multiple cameras or zones to cover a larger parking facility.")
            st.write("CNN models: LeNet-5 CNN, AlexNet CNN, ResNet-18 CNN.")

        with summary_right:
            calibration_frame = default_demo_frame.copy() if default_demo_frame is not None else create_simulated_camera_frame(roi_slots)
            st.image(
                draw_roi_geometry_overlay(calibration_frame, roi_slots),
                caption="Read-only official PKLot ROI map for the monitored zone",
                use_container_width=True,
            )

        preview_path = PROJECT_ROOT / "outputs" / "gui_roi_debug" / "official_roi_preview_clean.jpg"
        with st.expander("Advanced / developer ROI tools", expanded=False):
            st.warning("Do not overwrite the official ROI map unless recalibrating the camera view.")
            show_methodology_boundary = st.checkbox("Show monitored-zone boundary", value=False, key="methodology_show_boundary")
            if show_methodology_boundary:
                debug_frame = default_demo_frame.copy() if default_demo_frame is not None else create_simulated_camera_frame(roi_slots)
                st.image(
                    draw_roi_geometry_overlay(debug_frame, roi_slots, boundary=roi_boundary, show_boundary=True),
                    caption="Boundary debug view",
                    use_container_width=True,
                )
            active_path_text = str(roi_map_path.relative_to(PROJECT_ROOT)) if roi_map_path else "built-in defaults"
            st.write(f"Active ROI map: `{active_path_text}`")
            st.write(f"Display boundary points: `{len(roi_boundary)}`")
            st.write("Boundary is derived from active official ROI polygons.")
            st.write("Rebuild script: `tools/rebuild_official_pklot_roi_map.py`")
            if preview_path.exists():
                st.image(Image.open(preview_path).convert("RGB"), caption="Saved debug/reference preview", use_container_width=True)
            st.dataframe(
                [
                    {
                        "Slot ID": slot["slot_id"],
                        "Row": slot["row"],
                        "Spot": slot["spot"],
                        "Original PKLot ID": slot.get("original_id", ""),
                        "Ground truth": slot.get("ground_truth_status", ""),
                        "ROI": slot_coordinates_text(slot),
                    }
                    for slot in roi_slots
                ],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    run_app()
