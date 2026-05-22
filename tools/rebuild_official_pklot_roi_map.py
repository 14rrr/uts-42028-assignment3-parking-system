from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gui_roi_debug"
DATASET_NAME = "Voxel51/PKLot"
SOURCE_IMAGE = ASSETS_DIR / "image.jpg"
DEMO_FRAME_CLEAN_PATH = ASSETS_DIR / "demo_frame_clean.jpg"
CUSTOM_SLOT_MAP = ASSETS_DIR / "custom_slot_map.json"
RAW_ROIS_PATH = OUTPUT_DIR / "pklot_official_rois_raw.json"
PREVIEW_PATH = OUTPUT_DIR / "official_roi_preview.jpg"
PREVIEW_CLEAN_PATH = OUTPUT_DIR / "official_roi_preview_clean.jpg"
PREVIEW_WITH_BOUNDARY_PATH = OUTPUT_DIR / "official_roi_preview_with_boundary.jpg"

DEFAULT_BOUNDARY = [
    [512, 55],
    [584, 51],
    [842, 42],
    [1022, 42],
    [1040, 73],
    [1075, 160],
    [1123, 286],
    [1169, 443],
    [1181, 498],
    [866, 502],
    [472, 497],
    [348, 491],
    [389, 365],
]


def hf_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=DATASET_NAME, repo_type="dataset", filename=filename))


def load_samples() -> list[dict[str, Any]]:
    samples_path = hf_download("samples.json")
    with samples_path.open("r", encoding="utf-8") as file:
        return json.load(file)["samples"]


def print_annotation_structure(samples: list[dict[str, Any]], count: int = 3) -> None:
    print("TASK 1 - PKLot/Voxel51 annotation structure")
    for index, sample in enumerate(samples[:count], start=1):
        metadata = sample.get("metadata", {})
        parking_spaces = sample.get("parking_spaces", {})
        polylines = parking_spaces.get("polylines", [])
        first_space = polylines[0] if polylines else {}
        print(f"\nSample {index}")
        print("sample keys:", list(sample.keys()))
        print("source:", sample.get("source"))
        print("weather:", sample.get("weather", {}).get("label"))
        print("date:", sample.get("date", {}).get("$date"))
        print("parking_timestamp:", sample.get("parking_timestamp", {}).get("$date"))
        print("image size:", metadata.get("width"), "x", metadata.get("height"))
        print("parking_spaces structure:", {"_cls": parking_spaces.get("_cls"), "count": len(polylines)})
        print("polygon points storage:", "parking_spaces.polylines[*].points[0] as normalized [x,y] pairs")
        print("first polygon points:", first_space.get("points"))
        print("occupancy status storage:", "parking_spaces.polylines[*].occupancy_status")
        print("first occupancy status:", first_space.get("occupancy_status"))


def point_in_polygon(x: float, y: float, polygon: list[list[int]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            intersect_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < intersect_x:
                inside = not inside
        j = i
    return inside


def polygon_centroid(points: list[list[int]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def official_polygons(sample: dict[str, Any]) -> list[dict[str, Any]]:
    width = int(sample.get("metadata", {}).get("width", 1280))
    height = int(sample.get("metadata", {}).get("height", 720))
    rois: list[dict[str, Any]] = []
    for polyline in sample.get("parking_spaces", {}).get("polylines", []):
        normalized_points = polyline.get("points", [[]])[0]
        polygon = [
            [int(round(float(x) * width)), int(round(float(y) * height))]
            for x, y in normalized_points
        ]
        cx, cy = polygon_centroid(polygon)
        rois.append(
            {
                "original_id": str(polyline.get("space_id") or polyline.get("index")),
                "polygon": polygon,
                "ground_truth_status": normalize_status(polyline.get("occupancy_status")),
                "centroid": [cx, cy],
            }
        )
    return rois


def normalize_status(status: Any) -> str:
    text = str(status or "unknown").strip().lower()
    if text in {"occupied", "1", "true"}:
        return "occupied"
    if text in {"not occupied", "empty", "available", "0", "false"}:
        return "empty"
    return text or "unknown"


def load_boundary() -> list[list[int]]:
    config_path = ASSETS_DIR / "roi_row_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        boundary = data.get("parking_zone_boundary")
        if boundary:
            return [[int(round(float(x))), int(round(float(y)))] for x, y in boundary]
    return DEFAULT_BOUNDARY


def filtered_occupancy_count(sample: dict[str, Any], boundary: list[list[int]]) -> int:
    count = 0
    for roi in official_polygons(sample):
        cx, cy = roi["centroid"]
        if point_in_polygon(cx, cy, boundary) and roi["ground_truth_status"] == "occupied":
            count += 1
    return count


def mean_abs_diff(left: Image.Image, right: Image.Image) -> float:
    diff = ImageChops.difference(left, right)
    histogram = diff.histogram()
    pixels = left.size[0] * left.size[1] * 3
    return sum(
        value * freq
        for channel in range(3)
        for value, freq in enumerate(histogram[channel * 256 : (channel + 1) * 256])
    ) / pixels


def image_similarity_score(target: Image.Image, candidate: Image.Image) -> float:
    boxes = [
        (0, 0, 1280, 720),
        (330, 40, 1190, 520),
        (450, 40, 1185, 505),
    ]
    weights = [0.35, 0.30, 0.35]
    score = 0.0
    for box, weight in zip(boxes, weights):
        target_small = target.crop(box).resize((160, 90), Image.Resampling.BILINEAR)
        candidate_small = candidate.crop(box).resize((160, 90), Image.Resampling.BILINEAR)
        score += weight * mean_abs_diff(target_small, candidate_small)
    return score


def find_matching_sample(
    samples: list[dict[str, Any]],
    boundary: list[list[int]],
    target_image: Path,
    candidate_limit: int,
    match_threshold: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    target = Image.open(target_image).convert("RGB")
    target_size = target.size
    candidates = []
    for sample in samples:
        metadata = sample.get("metadata", {})
        if (metadata.get("width"), metadata.get("height")) != target_size:
            continue
        # The selected frame is a sparse sunny PUCPR frame. This keeps the search
        # bounded while still using official metadata, not manual ROI geometry.
        if sample.get("source") != "pucpr":
            continue
        if sample.get("weather", {}).get("label") != "sunny":
            continue
        occupied = filtered_occupancy_count(sample, boundary)
        if occupied <= 10:
            candidates.append((sample, occupied))

    best: tuple[float, dict[str, Any], int] | None = None
    inspected = 0
    for sample, occupied in candidates[:candidate_limit]:
        inspected += 1
        image_path = hf_download(sample["filepath"])
        candidate = Image.open(image_path).convert("RGB")
        score = image_similarity_score(target, candidate)
        if best is None or score < best[0]:
            best = (score, sample, occupied)
        if score <= match_threshold:
            break

    summary = {
        "target_size": list(target_size),
        "candidate_count": len(candidates),
        "candidates_inspected": inspected,
        "best_score": best[0] if best else None,
        "best_filtered_occupied": best[2] if best else None,
    }
    if best and best[0] <= match_threshold:
        return best[1], summary
    return None, summary


def group_slots(filtered: list[dict[str, Any]], row_gap: float = 45.0) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[list[dict[str, Any]]] = []
    for roi in sorted(filtered, key=lambda item: (item["centroid"][1], item["centroid"][0])):
        if not rows:
            rows.append([roi])
            continue
        row_mean_y = sum(item["centroid"][1] for item in rows[-1]) / len(rows[-1])
        if roi["centroid"][1] - row_mean_y > row_gap:
            rows.append([roi])
        else:
            rows[-1].append(roi)

    slots: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row_index, row_items in enumerate(rows):
        row_name = string.ascii_uppercase[row_index]
        sorted_row = sorted(row_items, key=lambda item: item["centroid"][0])
        counts[row_name] = len(sorted_row)
        for spot, item in enumerate(sorted_row, start=1):
            slots.append(
                {
                    "slot_id": f"{row_name}{spot:02d}",
                    "row": row_name,
                    "spot": spot,
                    "polygon": item["polygon"],
                    "original_id": item["original_id"],
                    "ground_truth_status": item["ground_truth_status"],
                }
            )
    return slots, counts


def draw_preview(
    source_image: Path,
    boundary: list[list[int]],
    slots: list[dict[str, Any]],
    output_path: Path,
    show_boundary: bool = False,
) -> None:
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    if show_boundary:
        boundary_points = [tuple(point) for point in boundary]
        draw.line(boundary_points + [boundary_points[0]], fill="#1d6bff", width=4)

    for slot in slots:
        points = [tuple(point) for point in slot["polygon"]]
        status = slot.get("ground_truth_status")
        color = "#d64040" if status == "occupied" else "#fff4a3"
        draw.line(points + [points[0]], fill=color, width=3)
        cx, cy = polygon_centroid(slot["polygon"])
        label = slot["slot_id"]
        label_box = draw.textbbox((0, 0), label)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        left = cx - label_width / 2 - 2
        top = cy - label_height / 2 - 2
        draw.rectangle((left, top, left + label_width + 4, top + label_height + 4), fill=(0, 0, 0))
        draw.text((left + 2, top + 2), label, fill="#ffffff")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def write_outputs(
    sample: dict[str, Any],
    boundary: list[list[int]],
    all_rois: list[dict[str, Any]],
    slots: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "dataset": DATASET_NAME,
        "matched_sample": sample_metadata(sample),
        "parking_spaces": all_rois,
    }
    with RAW_ROIS_PATH.open("w", encoding="utf-8") as file:
        json.dump(raw_payload, file, indent=2)

    final_payload = {
        "camera": {
            "name": "PKLot monitored parking zone",
            "source_image": "assets/image.jpg",
            "roi_source": "Official PKLot parking-space annotations",
            "note": "ROIs were extracted from PKLot annotations and filtered to the monitored camera zone.",
        },
        "parking_zone_boundary": boundary,
        "slots": slots,
    }
    with CUSTOM_SLOT_MAP.open("w", encoding="utf-8") as file:
        json.dump(final_payload, file, indent=2)


def sample_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "filepath": sample.get("filepath"),
        "source": sample.get("source"),
        "weather": sample.get("weather", {}).get("label"),
        "date": sample.get("date", {}).get("$date"),
        "parking_timestamp": sample.get("parking_timestamp", {}).get("$date"),
        "image_width": sample.get("metadata", {}).get("width"),
        "image_height": sample.get("metadata", {}).get("height"),
        "parking_spaces": len(sample.get("parking_spaces", {}).get("polylines", [])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild GUI ROI map from official PKLot/Voxel51 annotations.")
    parser.add_argument("--candidate-limit", type=int, default=180)
    parser.add_argument("--match-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples()
    print_annotation_structure(samples)

    boundary = load_boundary()
    matched, match_summary = find_matching_sample(
        samples=samples,
        boundary=boundary,
        target_image=SOURCE_IMAGE,
        candidate_limit=args.candidate_limit,
        match_threshold=args.match_threshold,
    )
    print("\nTASK 2 - Match assets/image.jpg")
    print("match summary:", match_summary)
    if matched is None:
        print("Exact/near-exact match not found within search limit.")
        raise SystemExit(1)

    print("matched sample metadata:", sample_metadata(matched))
    all_rois = official_polygons(matched)
    filtered = [
        roi
        for roi in all_rois
        if point_in_polygon(roi["centroid"][0], roi["centroid"][1], boundary)
    ]
    slots, counts = group_slots(filtered)
    write_outputs(matched, boundary, all_rois, slots)
    preview_source = DEMO_FRAME_CLEAN_PATH if DEMO_FRAME_CLEAN_PATH.exists() else SOURCE_IMAGE
    draw_preview(preview_source, boundary, slots, PREVIEW_CLEAN_PATH, show_boundary=False)
    draw_preview(preview_source, boundary, slots, PREVIEW_WITH_BOUNDARY_PATH, show_boundary=True)
    draw_preview(preview_source, boundary, slots, PREVIEW_PATH, show_boundary=True)

    print("\nTASKS 3-7 - Outputs")
    print("official polygons extracted:", len(all_rois))
    print("polygons after monitored-zone filtering:", len(filtered))
    print("row counts:", dict(counts))
    print("raw debug file:", RAW_ROIS_PATH.relative_to(PROJECT_ROOT))
    print("final ROI map:", CUSTOM_SLOT_MAP.relative_to(PROJECT_ROOT))
    print("clean preview image:", PREVIEW_CLEAN_PATH.relative_to(PROJECT_ROOT))
    print("boundary debug preview image:", PREVIEW_WITH_BOUNDARY_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
