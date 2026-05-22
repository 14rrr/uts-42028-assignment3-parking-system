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
SAMPLE_ROI_CONFIG_PATH = ASSETS_DIR / "sample_slot_map.json"
CUSTOM_ROI_CONFIG_PATH = ASSETS_DIR / "custom_slot_map.json"
MANUAL_STATUS_PATH = ASSETS_DIR / "manual_slot_status.json"
PKLOT_DEMO_FEED_PATH = ASSETS_DIR / "pklot_demo_feed.mp4"
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

MODEL_RESULTS = [
    {"Model": "LeNet-5 CNN", "Train Accuracy": "0.998105", "Validation Accuracy": "0.998144", "Test Accuracy": "0.997629"},
    {"Model": "AlexNet CNN", "Train Accuracy": "0.998467", "Validation Accuracy": "0.998387", "Test Accuracy": "0.998309"},
    {"Model": "ResNet-18 CNN", "Train Accuracy": "0.999300", "Validation Accuracy": "0.999291", "Test Accuracy": "0.999164"},
]

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
DEFAULT_RECOMMENDED_SLOT = "C05"


def default_roi_slots() -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    row_y = {"A": 70, "B": 180, "C": 310, "D": 420}
    for row in ("A", "B", "C", "D"):
        for number in range(1, 11):
            x = 70 + (number - 1) * 88
            y = row_y[row]
            slots.append(
                {
                    "slot_id": f"{row}{number:02d}",
                    "row": row,
                    "spot": number,
                    "roi": [x, y, 64, 82],
                }
            )
    return slots


def load_roi_config(path: Path = SAMPLE_ROI_CONFIG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return default_roi_slots()

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    slots = data.get("slots", data)
    if not isinstance(slots, list):
        raise ValueError("ROI configuration must contain a list named 'slots'.")
    return normalize_roi_slots(slots)


def load_active_roi_config() -> tuple[list[dict[str, Any]], str, Path | None]:
    if CUSTOM_ROI_CONFIG_PATH.exists():
        return load_roi_config(CUSTOM_ROI_CONFIG_PATH), "Custom ROI map", CUSTOM_ROI_CONFIG_PATH
    if SAMPLE_ROI_CONFIG_PATH.exists():
        return load_roi_config(SAMPLE_ROI_CONFIG_PATH), "Sample ROI map", SAMPLE_ROI_CONFIG_PATH
    return default_roi_slots(), "Built-in default map", None


def load_roi_boundary(path: Path | None) -> list[list[int]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    boundary = data.get("parking_zone_boundary", [])
    if not isinstance(boundary, list):
        return []
    return [[int(round(float(x))), int(round(float(y)))] for x, y in boundary]


def default_frame_path() -> Path | None:
    if DEMO_FRAME_CLEAN_PATH.exists():
        return DEMO_FRAME_CLEAN_PATH
    if LEGACY_FRAME_PATH.exists():
        return LEGACY_FRAME_PATH
    return None


def load_manual_status_override(path: Path = MANUAL_STATUS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"occupied": [], "available": [], "recommended": None}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {
        "occupied": [str(slot_id).upper() for slot_id in data.get("occupied", [])],
        "available": [str(slot_id).upper() for slot_id in data.get("available", [])],
        "recommended": str(data["recommended"]).upper() if data.get("recommended") else None,
    }


def apply_manual_status_override(slots: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    override = load_manual_status_override()
    occupied_ids = set(override["occupied"])
    available_ids = set(override["available"])
    recommended_id = override["recommended"]

    updated: list[dict[str, str | float]] = []
    for slot in slots:
        slot_copy = dict(slot)
        slot_id = str(slot_copy["Slot ID"]).upper()
        if slot_id in occupied_ids:
            slot_copy["Status"] = "occupied"
        elif slot_id in available_ids:
            slot_copy["Status"] = "available"
        updated.append(slot_copy)

    if recommended_id:
        for slot in updated:
            if slot["Status"] == "recommended":
                slot["Status"] = "available"
        for slot in updated:
            if str(slot["Slot ID"]).upper() == recommended_id and slot["Status"] != "occupied":
                slot["Status"] = "recommended"
                break
    return updated


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


def save_roi_config(slots: list[dict[str, Any]], path: Path = CUSTOM_ROI_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera": {
            "name": "Custom fixed camera calibration",
            "calibration_note": "Custom ROI coordinates for the current fixed camera view.",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "slots": normalize_roi_slots(slots),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def roi_slots_to_editor_rows(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in normalize_roi_slots(slots):
        x, y, width, height = slot_bounds(slot)
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "row": slot["row"],
                "spot": slot["spot"],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
    return rows


def editor_rows_to_roi_slots(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")

    slots: list[dict[str, Any]] = []
    for row in rows:
        slot_id = str(row.get("slot_id") or row.get("Slot ID") or "").strip().upper()
        if not slot_id:
            continue
        slots.append(
            {
                "slot_id": slot_id,
                "row": str(row.get("row") or row.get("Row") or slot_id[0]).strip().upper() or slot_id[0],
                "spot": int(row.get("spot") or row.get("Spot") or slot_id[1:] or 1),
                "roi": [
                    int(row.get("x", 0)),
                    int(row.get("y", 0)),
                    max(1, int(row.get("width", 64))),
                    max(1, int(row.get("height", 82))),
                ],
            }
        )
    return normalize_roi_slots(slots)


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


def generate_parking_slots(roi_slots: list[dict[str, Any]] | None = None) -> list[dict[str, str | float]]:
    slots: list[dict[str, str | float]] = []
    roi_slots = roi_slots or default_roi_slots()

    for roi_slot in roi_slots:
        slot_id = str(roi_slot["slot_id"])
        row = str(roi_slot["row"])
        status = "occupied" if slot_id in SIMULATED_OCCUPIED_SLOTS else "available"
        confidence_seed = (ord(row) + int(roi_slot["spot"]) * 7) % 9
        confidence = 0.91 + confidence_seed / 100
        if slot_id == DEFAULT_RECOMMENDED_SLOT:
            status = "recommended"
            confidence = 0.98
        slots.append(
            {
                "Slot ID": slot_id,
                "Row": row,
                "Spot": f"{int(roi_slot['spot']):02d}",
                "ROI coordinates": slot_coordinates_text(roi_slot),
                "Status": status,
                "Confidence": f"{confidence:.2f}",
                "Last updated": datetime.now().strftime("%H:%M:%S"),
            }
        )
    return slots


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

    recommended = choose_recommended_slot(slots)
    for slot in slots:
        if slot["Slot ID"] == recommended and slot["Status"] == "available":
            slot["Status"] = "recommended"
    return slots


def row_count_summary(roi_slots: list[dict[str, Any]]) -> str:
    rows: dict[str, int] = {}
    for slot in roi_slots:
        row = str(slot["row"])
        rows[row] = rows.get(row, 0) + 1
    return ", ".join(f"{row}: {count}" for row, count in sorted(rows.items()))


def slot_counts(slots: list[dict[str, str | float]]) -> tuple[int, int, str]:
    available = sum(1 for slot in slots if slot["Status"] in {"available", "recommended"})
    occupied = sum(1 for slot in slots if slot["Status"] == "occupied")
    recommended = next((str(slot["Slot ID"]) for slot in slots if slot["Status"] == "recommended"), "None")
    return available, occupied, recommended


def choose_recommended_slot(slots: list[dict[str, str | float]]) -> str:
    for slot in slots:
        if slot["Status"] == "available":
            return str(slot["Slot ID"])
    return ""


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

    recommended = choose_recommended_slot(results)
    for result in results:
        if result["Slot ID"] == recommended and result["Status"] == "available":
            result["Status"] = "recommended"
    return results


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
    }
    if show_boundary and boundary:
        boundary_points = [(int(x), int(y)) for x, y in boundary]
        draw.line(boundary_points + [boundary_points[0]], fill="#1d6bff", width=4)
    for roi_slot in roi_slots:
        slot_id = str(roi_slot["slot_id"])
        status = status_by_id.get(slot_id, "available")
        color = color_by_status.get(status, "#13a538")
        if roi_slot.get("polygon"):
            points = [(int(x), int(y)) for x, y in roi_slot["polygon"]]
            draw.line(points + [points[0]], fill=color, width=4)
            label_x, label_y = points[0]
        else:
            x, y, width, height = slot_bounds(roi_slot)
            draw.rectangle((x, y, x + width, y + height), outline=color, width=4)
            label_x, label_y = x, y
        label_fill = "#101820" if status == "available" else "#ffffff"
        draw.rectangle((label_x, label_y, label_x + 48, label_y + 18), fill=color)
        draw.text((label_x + 4, label_y + 2), slot_id, fill=label_fill)
    return overlay


def render_status_cards(st, total: int, available: int, occupied: int, recommended: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total spaces", total)
    col2.metric("Available spaces", available)
    col3.metric("Occupied spaces", occupied)
    col4.metric("Recommended space", recommended)


def render_recommendation_card(st, recommended: str) -> None:
    if recommended == "None":
        value = "No spaces available"
        text = "Please wait for the next available bay"
    else:
        value = f"Row {recommended[0]}, Spot {recommended[1:]}"
        text = "Please proceed to the highlighted bay"

    st.markdown(
        f"""
        <div class='recommendation-card'>
            <div class='recommendation-label'>Recommended parking space</div>
            <div class='recommendation-value'>{value}</div>
            <div class='recommendation-text'>{text}</div>
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
            }[status]
            label = f"{slot['Slot ID']}"
            if status == "recommended":
                label = f"{slot['Slot ID']}<br><span>REC</span>"
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
            <span><i class='legend-dot legend-recommended'></i>Green: Recommended</span>
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
    roi_boundary = load_roi_boundary(roi_map_path)
    current_default_frame_path = default_frame_path()

    @st.cache_resource(show_spinner="Loading trained CNN checkpoint...")
    def cached_model(model_key: str) -> torch.nn.Module:
        return load_trained_model(
            model_key=model_key,
            image_size=config.image_size,
            project_root=PROJECT_ROOT,
            map_location="cpu",
        )

    st.set_page_config(page_title="Parkivo Parking Availability", page_icon="P", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        .recommendation-card {
            border-radius: 8px;
            border: 2px solid #1c70c8;
            background: #fff9c7;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
        }
        .recommendation-label { font-size: 1rem; font-weight: 700; color: #32404a; }
        .recommendation-value { font-size: 2.7rem; font-weight: 800; color: #1f2a30; line-height: 1.1; }
        .recommendation-text { font-size: 1.1rem; color: #32404a; margin-top: 0.3rem; }
        .parking-slot {
            min-height: 46px;
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
        .slot-occupied { background: #ffd9d9; color: #8a1f1f; }
        .slot-recommended { background: #d8f5df; color: #145a27; border: 3px solid #168a35; }
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
        .legend-recommended { background: #d8f5df; }
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
        st.session_state.slot_results = apply_manual_status_override(generate_ground_truth_slots(roi_slots))
        st.session_state.source_mode = "PKLot annotation status"
        st.session_state.status_note = "Initial status is loaded from official PKLot annotations. Run CNN Detection to update from the current frame."
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_detection_summary = None

    slots = apply_manual_status_override(st.session_state.slot_results)
    st.session_state.slot_results = slots
    available, occupied, recommended = slot_counts(slots)
    total_spaces = len(slots)

    dashboard_tab, detection_tab, calibration_tab, about_tab = st.tabs(
        ["Dashboard", "Detection Demo", "ROI Calibration", "About / Methodology"]
    )

    with dashboard_tab:
        st.title("Parkivo Parking Availability System")
        st.caption("Fixed-camera parking-zone availability using CNN classification on cropped ROI images.")

        left, right = st.columns([1.3, 1])
        with left:
            show_dashboard_boundary = False
            with st.expander("Advanced display options", expanded=False):
                active_path_text = str(roi_map_path.relative_to(PROJECT_ROOT)) if roi_map_path else "built-in defaults"
                st.write(f"Active ROI map: `{active_path_text}`")
                st.write(f"Source mode: `{st.session_state.source_mode}`")
                st.write(f"Last updated: `{st.session_state.last_updated}`")
                show_dashboard_boundary = st.checkbox("Show monitored-zone boundary", value=False, key="dashboard_show_boundary")

            if current_default_frame_path is not None:
                dashboard_frame = Image.open(current_default_frame_path).convert("RGB")
            else:
                dashboard_frame = create_simulated_camera_frame(roi_slots)
            st.image(
                draw_roi_overlay(dashboard_frame, slots, roi_slots, boundary=roi_boundary, show_boundary=show_dashboard_boundary),
                caption="Monitored parking zone with official ROI overlay",
                use_container_width=True,
            )

        with right:
            render_recommendation_card(st, recommended)
            render_status_cards(st, total_spaces, available, occupied, recommended)
            render_parking_legend(st)
            st.info(str(st.session_state.get("status_note", "")))

        st.subheader("Parking Zone Status")
        render_parking_map(st, slots)

    with detection_tab:
        st.header("Detection Demo")
        st.write("Run CNN classification on each official ROI crop from the current full-frame parking-zone image.")
        selected_camera_model = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=2)

        with st.expander("Upload a full-frame parking-zone image", expanded=False):
            uploaded_frame = st.file_uploader(
                "Full-frame image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="full_frame_upload",
            )

        if uploaded_frame is not None:
            frame = Image.open(uploaded_frame).convert("RGB")
            source_label = "uploaded frame"
            can_classify_frame = True
        elif current_default_frame_path is not None:
            frame = Image.open(current_default_frame_path).convert("RGB")
            source_label = "default clean PKLot frame"
            can_classify_frame = True
        else:
            frame = create_simulated_camera_frame(roi_slots)
            source_label = "simulated fallback frame"
            can_classify_frame = False

        if st.button("Run CNN Detection", type="primary"):
            if can_classify_frame:
                model_key = MODEL_OPTIONS[selected_camera_model]
                model = cached_model(model_key)
                detected_slots = classify_rois(model, frame, roi_slots, image_size=config.image_size)
                detected_slots = apply_manual_status_override(detected_slots)
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
                    "Recommended slot": detected_recommended,
                }
                st.rerun()
            else:
                st.warning("No full-frame image is available for CNN detection.")

        st.image(
            draw_roi_overlay(frame, slots, roi_slots),
            caption=f"Current frame: {source_label}",
            use_container_width=True,
        )

        if st.session_state.get("last_detection_summary"):
            st.subheader("Detection Summary")
            st.table([st.session_state.last_detection_summary])
        else:
            st.info("Initial status is loaded from PKLot annotations. Run CNN Detection to update from the current frame.")

        with st.expander("Advanced prediction details", expanded=False):
            st.dataframe(slots, use_container_width=True, hide_index=True)
            st.subheader("Single-slot crop test")
            uploaded_crop = st.file_uploader(
                "Upload one parking-space crop",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="single_slot_upload",
            )
            if uploaded_crop is not None:
                crop_image = Image.open(uploaded_crop).convert("RGB")
                test_left, test_right = st.columns([1, 1])
                test_left.image(crop_image, caption="Uploaded crop", use_container_width=True)
                if st.button("Run single-slot CNN prediction", type="secondary"):
                    model_key = MODEL_OPTIONS[selected_camera_model]
                    model = cached_model(model_key)
                    result = predict_parking_slot(model, crop_image, image_size=config.image_size)
                    test_right.metric("Prediction", str(result["label"]))
                    test_right.metric("Confidence", f"{float(result['confidence']):.3f}")
                    test_right.write(f"Occupied probability: `{float(result['occupied_probability']):.6f}`")
                    test_right.write(f"Inference time: `{float(result['inference_ms']):.2f} ms`")

    with calibration_tab:
        st.header("ROI Calibration")
        st.write("The ROI map was extracted from official PKLot polygon annotations for the selected monitored parking zone.")
        summary_rows = [
            {"Item": "ROI source", "Value": "Official PKLot/Voxel51 annotations"},
            {"Item": "Slot count", "Value": len(roi_slots)},
            {"Item": "Rows", "Value": row_count_summary(roi_slots)},
            {"Item": "Default frame", "Value": "assets/demo_frame_clean.jpg" if DEMO_FRAME_CLEAN_PATH.exists() else "assets/image.jpg"},
        ]
        st.table(summary_rows)

        preview_path = PROJECT_ROOT / "outputs" / "gui_roi_debug" / "official_roi_preview_clean.jpg"
        if preview_path.exists():
            st.image(Image.open(preview_path).convert("RGB"), caption="Clean official ROI preview", use_container_width=True)
        else:
            st.image(
                draw_roi_overlay(Image.open(current_default_frame_path).convert("RGB"), slots, roi_slots) if current_default_frame_path else create_simulated_camera_frame(roi_slots),
                caption="Generated ROI preview",
                use_container_width=True,
            )

        with st.expander("Advanced / developer ROI tools", expanded=False):
            st.warning("Changing this may overwrite the official ROI map. The submitted demo should keep the official PKLot polygon ROIs.")
            active_path_text = str(roi_map_path.relative_to(PROJECT_ROOT)) if roi_map_path else "built-in defaults"
            st.write(f"Active ROI map: `{active_path_text}`")
            st.write(f"Boundary points: `{len(roi_boundary)}`")
            st.write(f"Status override file: `{MANUAL_STATUS_PATH.relative_to(PROJECT_ROOT)}`")
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

    with about_tab:
        st.header("About / Methodology")
        st.write(
            "This prototype monitors a selected fixed-camera parking zone. A clean PKLot full-frame image is paired "
            "with official PKLot polygon ROIs that define each monitored parking space."
        )
        st.write(
            "Each ROI is cropped from the full frame and passed to a CNN that classifies the parking-space crop as "
            "occupied or empty. The task is image classification on cropped parking-space images, not object detection."
        )
        st.write(
            "A full deployment could use multiple cameras or monitored zones to cover a larger parking facility."
        )
        st.write("Available CNN models: LeNet-5 CNN, AlexNet CNN, and ResNet-18 CNN.")


if __name__ == "__main__":
    run_app()
