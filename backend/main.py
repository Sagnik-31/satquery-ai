
from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from satellite_service import (
    RetrievedPair,
    SceneRetrievalError,
    get_cached_pair,
    pair_cache_key,
    retrieve_temporal_pair,
    store_cached_pair,
    EARTH_SEARCH_URL,
    STAC_COLLECTION,
    RASTERIO_AVAILABLE,
)
try:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:
    AutoImageProcessor = None
    AutoModelForSemanticSegmentation = None
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


APP_NAME = "SatQuery AI Change Intelligence API"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

_processor = None
_segmenter = None


CATEGORY_KEYWORDS = {
    "building": ["building", "house", "skyscraper", "structure"],
    "vegetation": ["tree", "grass", "plant", "flower", "field", "palm"],
    "terrain": ["earth", "mountain", "hill", "sand", "dirt"],
    "water": ["water", "sea", "river", "lake"],
    "road": ["road", "path"],
}

# ADE20k classes that dominate when SegFormer is applied to aerial/satellite
# imagery it was not trained for (indoor/outdoor photo scene labels).
INDOOR_MISFIT_LABELS = {
    "wall", "ceiling", "floor", "bed", "sofa", "table", "chair", "window",
    "door", "mirror", "painting", "sconce", "lamp", "cabinet", "shelf",
    "curtain", "pillow", "blanket", "rug", "desk", "wardrobe", "toilet",
}

# False-positive ADE20k label fragments for keyword routing.
EXCLUDED_CLASS_MATCHES = {
    "water": {"seat", "pool table", "swimming pool", "waterfall"},
    "vegetation": {"streetlight"},
    "road": {"streetlight"},
    "terrain": set(),
    "building": set(),
}


def load_segmenter():
    global _processor, _segmenter

    if not TRANSFORMERS_AVAILABLE:
        return None, None

    if _segmenter is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _segmenter = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
        _segmenter.to(DEVICE)
        _segmenter.eval()

    return _processor, _segmenter


def read_image(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image


def encode_png(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode result image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def resize_to_match(before: np.ndarray, after: np.ndarray):
    h, w = after.shape[:2]
    return cv2.resize(before, (w, h), interpolation=cv2.INTER_AREA), after


def align_images(before: np.ndarray, after: np.ndarray):
    """
    Robust registration:
    - ORB feature matching
    - Lowe ratio filtering
    - RANSAC homography
    - conservative rejection of weak transforms
    """

    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(
        nfeatures=6000,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=8,
    )

    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return before.copy(), False, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.73 * n.distance:
            good.append(m)

    if len(good) < 15:
        return before.copy(), False, len(good), 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        4.0,
    )

    if H is None or inlier_mask is None:
        return before.copy(), False, len(good), 0.0

    inliers = int(inlier_mask.sum())
    ratio = inliers / max(len(good), 1)

    # Conservative acceptance criteria.
    if inliers < 12 or ratio < 0.22:
        return before.copy(), False, inliers, ratio

    h, w = after.shape[:2]

    aligned = cv2.warpPerspective(
        before,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return aligned, True, inliers, ratio


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize luminance while retaining chromatic information.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    l = clahe.apply(l)

    result = cv2.merge((l, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def calculate_difference(before: np.ndarray, after: np.ndarray):
    """
    Combine LAB color difference with edge difference.
    This is more stable than raw RGB subtraction.
    """

    b_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)

    color_diff = np.linalg.norm(a_lab - b_lab, axis=2)

    b_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    b_edges = cv2.Canny(b_gray, 60, 140).astype(np.float32)
    a_edges = cv2.Canny(a_gray, 60, 140).astype(np.float32)

    edge_diff = cv2.GaussianBlur(
        np.abs(a_edges - b_edges),
        (5, 5),
        0,
    )

    color_norm = cv2.normalize(
        color_diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    edge_norm = cv2.normalize(
        edge_diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    combined = (0.82 * color_norm) + (0.18 * edge_norm)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    return np.clip(combined, 0, 255).astype(np.uint8)


def adaptive_min_area(contours: list, min_area: int, total_pixels: int) -> int:
    """
    If too many tiny regions are detected, raise the minimum area
    automatically so the main visualization stays readable.
    """
    if len(contours) <= 40:
        return min_area

    target = min_area
    for _ in range(4):
        target = min(int(target * 1.6), max(500, int(total_pixels * 0.002)))
        filtered = [c for c in contours if cv2.contourArea(c) >= target]
        if len(filtered) <= 40:
            return target
        contours = filtered

    return target


def semantic_for_response(semantic: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip non-JSON-serializable arrays before returning to the client."""
    if not semantic:
        return None

    safe: dict[str, Any] = {}
    for key, value in semantic.items():
        if key == "mask":
            continue
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (np.floating, np.float32, np.float64)):
            safe[key] = float(value)
        elif isinstance(value, (np.integer, np.int32, np.int64)):
            safe[key] = int(value)
        else:
            safe[key] = value
    return safe


def create_change_mask(difference: np.ndarray, min_area: int):
    """
    Otsu is combined with a high-percentile floor to avoid
    extremely permissive thresholds on low-contrast scenes.
    """

    otsu_threshold, _ = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    percentile_threshold = float(np.percentile(difference, 82))
    threshold = max(float(otsu_threshold), percentile_threshold)

    _, mask = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    # Gentle morphology to remove speckle and bridge small gaps.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned, float(threshold)


def get_contours(mask: np.ndarray, min_area: int):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]

    contours.sort(
        key=cv2.contourArea,
        reverse=True,
    )

    return contours


def make_heatmap(difference: np.ndarray, mask: np.ndarray):
    heatmap = cv2.applyColorMap(
        difference,
        cv2.COLORMAP_TURBO,
    )

    heatmap[mask == 0] = (28, 32, 38)
    return heatmap


def make_overlay(after: np.ndarray, contours):
    overlay = after.copy()
    filled = after.copy()

    cv2.drawContours(
        filled,
        contours,
        -1,
        (0, 0, 255),
        thickness=cv2.FILLED,
    )

    overlay = cv2.addWeighted(
        after,
        0.72,
        filled,
        0.28,
        0,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 0, 255),
        2,
    )

    for i, contour in enumerate(contours[:15], start=1):
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2

        cv2.circle(
            overlay,
            (cx, cy),
            13,
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            overlay,
            str(i),
            (cx - 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 25, 20),
            1,
            cv2.LINE_AA,
        )

    return overlay


def classify_query(query: str):
    q = query.lower()

    if any(x in q for x in [
        "building", "buildings", "house", "houses",
        "construction", "structure", "urban",
    ]):
        return "building"

    if any(x in q for x in [
        "vegetation", "tree", "trees", "forest",
        "grass", "plant", "plants", "crop", "crops",
        "greenery", "green cover",
    ]):
        return "vegetation"

    if any(x in q for x in [
        "terrain", "land", "soil", "earth",
        "mountain", "mountains", "hill", "hills",
        "sand", "ground", "surface",
    ]):
        return "terrain"

    if any(x in q for x in [
        "water", "river", "lake", "pool",
        "flood", "flooding",
    ]):
        return "water"

    if any(x in q for x in [
        "road", "roads", "street", "path",
    ]):
        return "road"

    return "general"


def find_category_ids(
    category: str,
    class_names: dict[int, str],
):
    keywords = CATEGORY_KEYWORDS.get(category, [])
    excluded = EXCLUDED_CLASS_MATCHES.get(category, set())

    ids = []

    for class_id, name in class_names.items():
        lowered = name.lower()

        if any(ex in lowered for ex in excluded):
            continue

        if any(k in lowered for k in keywords):
            ids.append(class_id)

    return ids


def segformer_scene_mismatch(
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    class_names: dict[int, str],
) -> bool:
    """
    Detect when ADE20k SegFormer predictions are dominated by indoor-scene
    labels, which indicates the model is a poor fit for aerial/satellite data.
    """
    combined = np.concatenate([before_labels.ravel(), after_labels.ravel()])
    unique, counts = np.unique(combined, return_counts=True)
    total = max(int(combined.size), 1)

    misfit_pixels = 0
    for class_id, count in zip(unique, counts):
        label = class_names.get(int(class_id), "").lower()
        if label in INDOOR_MISFIT_LABELS or any(
            token in label for token in ("wall", "ceiling", "mirror", "bed", "sofa")
        ):
            misfit_pixels += int(count)

    return (misfit_pixels / total) >= 0.30


def build_heuristic_semantic(
    aligned: np.ndarray,
    after: np.ndarray,
    general_mask: np.ndarray,
    category: str,
) -> dict[str, Any]:
    """Compare heuristic category masks inside the general temporal-change mask."""
    before_mask, before_conf = heuristic_category_mask(aligned, category)
    after_mask, after_conf = heuristic_category_mask(after, category)

    before_category = before_mask > 0
    after_category = after_mask > 0
    changed = general_mask > 0

    new_regions = after_category & (~before_category) & changed
    removed_regions = before_category & (~after_category) & changed

    total_pixels = general_mask.size
    new_pixels = int(np.count_nonzero(new_regions))
    removed_pixels = int(np.count_nonzero(removed_regions))

    category_present = before_category | after_category
    if np.any(category_present):
        confidence_value = float(
            (
                before_conf[category_present].mean()
                + after_conf[category_present].mean()
            ) / 2
        )
    else:
        confidence_value = 0.0

    if new_pixels == 0 and removed_pixels == 0:
        direction = "none"
    elif new_pixels > removed_pixels * 1.25:
        direction = "increase"
    elif removed_pixels > new_pixels * 1.25:
        direction = "decrease"
    else:
        direction = "mixed"

    semantic_mask = (new_regions | removed_regions).astype(np.uint8) * 255
    semantic_contours = get_contours(
        semantic_mask,
        max(80, int(total_pixels * 0.00005)),
    )

    return {
        "available": True,
        "category": category,
        "new_pixels": new_pixels,
        "removed_pixels": removed_pixels,
        "new_percentage": new_pixels / total_pixels * 100 if total_pixels else 0.0,
        "removed_percentage": removed_pixels / total_pixels * 100 if total_pixels else 0.0,
        "direction": direction,
        "confidence": confidence_value,
        "mask": semantic_mask,
        "region_count": len(semantic_contours),
        "backend": "deterministic_visual_fallback",
    }


def _resize_band_dict(bands: dict[str, np.ndarray], width: int, height: int) -> dict[str, np.ndarray]:
    resized = {}
    for key, arr in bands.items():
        if arr.shape[:2] == (height, width):
            resized[key] = arr
        else:
            resized[key] = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir.astype(np.float32) - red.astype(np.float32)) / (
        nir.astype(np.float32) + red.astype(np.float32) + 1e-6
    )


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green.astype(np.float32) - nir.astype(np.float32)) / (
        green.astype(np.float32) + nir.astype(np.float32) + 1e-6
    )


def build_spectral_semantic(
    before_bands: dict[str, np.ndarray],
    after_bands: dict[str, np.ndarray],
    general_mask: np.ndarray,
    category: str,
) -> dict[str, Any] | None:
    """NDVI/NDWI-based semantic analysis when multispectral bands are available."""
    height, width = general_mask.shape[:2]
    before_bands = _resize_band_dict(before_bands, width, height)
    after_bands = _resize_band_dict(after_bands, width, height)
    changed = general_mask > 0
    total_pixels = general_mask.size

    if category == "vegetation" and {"B04", "B08"}.issubset(before_bands) and {"B04", "B08"}.issubset(after_bands):
        ndvi_t1 = compute_ndvi(before_bands["B04"], before_bands["B08"])
        ndvi_t2 = compute_ndvi(after_bands["B04"], after_bands["B08"])
        gain = changed & (ndvi_t2 - ndvi_t1 > 0.08) & (ndvi_t2 > 0.20)
        loss = changed & (ndvi_t1 - ndvi_t2 > 0.08) & (ndvi_t1 > 0.20)
        if np.any(changed):
            confidence = float(min(0.95, 0.65 + 0.30 * np.nanmean(np.abs(ndvi_t2 - ndvi_t1)[changed])))
        else:
            confidence = 0.0
        backend = "sentinel2_ndvi"
    elif category == "water" and {"B03", "B08"}.issubset(before_bands) and {"B03", "B08"}.issubset(after_bands):
        ndwi_t1 = compute_ndwi(before_bands["B03"], before_bands["B08"])
        ndwi_t2 = compute_ndwi(after_bands["B03"], after_bands["B08"])
        gain = changed & (ndwi_t2 - ndwi_t1 > 0.08) & (ndwi_t2 > 0.0)
        loss = changed & (ndwi_t1 - ndwi_t2 > 0.08) & (ndwi_t1 > 0.0)
        if np.any(changed):
            confidence = float(min(0.92, 0.60 + 0.32 * np.nanmean(np.abs(ndwi_t2 - ndwi_t1)[changed])))
        else:
            confidence = 0.0
        backend = "sentinel2_ndwi"
    else:
        return None

    new_pixels = int(np.count_nonzero(gain))
    removed_pixels = int(np.count_nonzero(loss))

    if new_pixels == 0 and removed_pixels == 0:
        direction = "none"
    elif new_pixels > removed_pixels * 1.25:
        direction = "increase"
    elif removed_pixels > new_pixels * 1.25:
        direction = "decrease"
    else:
        direction = "mixed"

    semantic_mask = (gain | loss).astype(np.uint8) * 255
    semantic_contours = get_contours(semantic_mask, max(80, int(total_pixels * 0.00005)))

    return {
        "available": True,
        "category": category,
        "new_pixels": new_pixels,
        "removed_pixels": removed_pixels,
        "new_percentage": new_pixels / total_pixels * 100 if total_pixels else 0.0,
        "removed_percentage": removed_pixels / total_pixels * 100 if total_pixels else 0.0,
        "direction": direction,
        "confidence": confidence,
        "mask": semantic_mask,
        "region_count": len(semantic_contours),
        "backend": backend,
    }


def heuristic_category_mask(image: np.ndarray, category: str):
    """
    Fast, deterministic semantic proxy used when a general semantic
    segmentation model cannot be imported/downloaded. It is deliberately
    conservative and returns a soft confidence-like map based on colour
    and texture cues. This keeps the hackathon demo functional while the
    stronger model remains optional.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(image)
    h, s, v = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    if category == "vegetation":
        # Green hue + healthy saturation/value.
        mask = (((h >= 30) & (h <= 95)) & (s >= 45) & (v >= 35)).astype(np.uint8) * 255
    elif category == "water":
        # Blue/cyan, generally darker low-texture surfaces.
        blue = (b.astype(np.int16) - r.astype(np.int16))
        mask = ((((h >= 85) & (h <= 135) & (s >= 40) & (v >= 25)) |
                 ((blue > 25) & (s >= 35) & (v <= 190)))).astype(np.uint8) * 255
    elif category == "terrain":
        # Earth/sand/dirt-like surfaces: warm hue or low-saturation midtones.
        warm = ((h >= 5) & (h <= 35) & (s >= 35) & (v >= 45) & (v <= 220))
        neutral_ground = ((s < 70) & (v >= 55) & (v <= 190) & (A > 120) & (B > 120))
        mask = (warm | neutral_ground).astype(np.uint8) * 255
    elif category == "building":
        # Built surfaces tend to be neutral/bright and edged.
        neutral = (s < 70) & (v > 85)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        mask = ((neutral & (dilated_edges > 0)) |
                (neutral & (v > 135))).astype(np.uint8) * 255
    elif category == "road":
        # Neutral midtone linear surfaces.
        mask = ((s < 65) & (v >= 55) & (v <= 200)).astype(np.uint8) * 255
    else:
        mask = np.zeros_like(h, dtype=np.uint8)

    # Conservative cleanup and region-size filtering.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )

    # Soft confidence proxy = local agreement / compactness.
    binary = (mask > 0).astype(np.uint8)
    blur = cv2.GaussianBlur(binary.astype(np.float32), (15, 15), 0)
    confidence = np.clip(0.55 + 0.45 * blur, 0, 1).astype(np.float32)
    return mask, confidence


def segment_image(image: np.ndarray, category: str | None = None):
    """
    Prefer SegFormer when available. Otherwise use the deterministic
    category-specific visual fallback.
    """
    processor, model = load_segmenter()

    if processor is None or model is None:
        if category is None:
            return None, None, None, "heuristic"
        mask, confidence = heuristic_category_mask(image, category)
        return mask, confidence, None, "heuristic"

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    h, w = image.shape[:2]
    logits = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    probabilities = torch.softmax(logits, dim=1)
    confidence, labels = torch.max(probabilities, dim=1)
    labels_np = labels[0].cpu().numpy()
    confidence_np = confidence[0].cpu().numpy()
    class_names = {int(k): str(v) for k, v in model.config.id2label.items()}
    return labels_np, confidence_np, class_names, "segformer"

def analyze_semantic_category(
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    class_names: dict[int, str],
    category: str,
    general_mask: np.ndarray,
    before_conf: np.ndarray,
    after_conf: np.ndarray,
):
    ids = find_category_ids(
        category,
        class_names,
    )

    if not ids:
        return {
            "available": False,
            "category": category,
        }

    before_category = np.isin(
        before_labels,
        ids,
    )

    after_category = np.isin(
        after_labels,
        ids,
    )

    changed = general_mask > 0

    new_regions = (
        after_category
        & (~before_category)
        & changed
    )

    removed_regions = (
        before_category
        & (~after_category)
        & changed
    )

    total_pixels = general_mask.size

    new_pixels = int(np.count_nonzero(new_regions))
    removed_pixels = int(np.count_nonzero(removed_regions))

    new_pct = new_pixels / total_pixels * 100
    removed_pct = removed_pixels / total_pixels * 100

    category_present = before_category | after_category

    if np.any(category_present):
        semantic_confidence = float(
            (
                before_conf[category_present].mean()
                + after_conf[category_present].mean()
            ) / 2
        )
    else:
        semantic_confidence = 0.0

    if new_pixels == 0 and removed_pixels == 0:
        direction = "none"
    elif new_pixels > removed_pixels * 1.25:
        direction = "increase"
    elif removed_pixels > new_pixels * 1.25:
        direction = "decrease"
    else:
        direction = "mixed"

    semantic_mask = (
        new_regions | removed_regions
    ).astype(np.uint8) * 255

    contours = get_contours(
        semantic_mask,
        max(80, int(total_pixels * 0.00005)),
    )

    return {
        "available": True,
        "category": category,
        "new_pixels": new_pixels,
        "removed_pixels": removed_pixels,
        "new_percentage": float(new_pct),
        "removed_percentage": float(removed_pct),
        "direction": direction,
        "confidence": semantic_confidence,
        "mask": semantic_mask,
        "region_count": len(contours),
    }


def make_semantic_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    direction: str,
):
    if direction == "increase":
        color = (0, 210, 255)  # orange/yellow
    elif direction == "decrease":
        color = (0, 70, 255)   # red
    else:
        color = (255, 170, 0)  # blue/orange

    overlay = image.copy()
    color_layer = image.copy()

    color_layer[mask > 0] = color

    overlay = cv2.addWeighted(
        image,
        0.66,
        color_layer,
        0.34,
        0,
    )

    contours = get_contours(
        mask,
        80,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        color,
        2,
    )

    return overlay


def answer_for_category(
    category: str,
    semantic: dict[str, Any],
):
    if not semantic.get("available"):
        return (
            f"The semantic segmentation model could not map "
            f"the requested category '{category}' reliably."
        )

    new_pct = semantic["new_percentage"]
    removed_pct = semantic["removed_percentage"]
    confidence = semantic["confidence"] * 100
    direction = semantic["direction"]
    regions = semantic["region_count"]

    if category == "terrain":
        caveat = (
            "Here, terrain means visible surface/ground-cover "
            "classes; it is not a measurement of elevation change."
        )
    else:
        caveat = (
            "The result is a semantic visual estimate, not a "
            "ground-truth accuracy score."
        )

    if direction == "increase":
        summary = (
            f"Likely increase in {category} coverage. "
            f"New {category} regions account for about "
            f"{new_pct:.2f}% of the image, with about "
            f"{removed_pct:.2f}% showing possible loss."
        )
    elif direction == "decrease":
        summary = (
            f"Likely decrease in {category} coverage. "
            f"About {removed_pct:.2f}% of the image shows "
            f"{category} present earlier but not later, while "
            f"{new_pct:.2f}% shows possible new {category}."
        )
    elif direction == "mixed":
        summary = (
            f"Mixed {category} changes were detected. "
            f"Approximately {new_pct:.2f}% appears to be newly "
            f"detected {category}, while {removed_pct:.2f}% "
            f"appears to be lost."
        )
    else:
        summary = (
            f"No significant {category} change was detected "
            f"inside the general change regions."
        )

    return (
        f"{summary} "
        f"{regions} semantic region(s) were identified. "
        f"Semantic signal confidence: {confidence:.1f}%. "
        f"{caveat}"
    )


def bbox_features(
    contours,
    width: int,
    height: int,
    category: str | None = None,
):
    """
    Pixel-coordinate features for frontend display/export.
    Geographic conversion can be added later when image CRS
    and bounds are available.
    """

    features = []

    for i, contour in enumerate(contours[:50], start=1):
        x, y, w, h = cv2.boundingRect(contour)

        features.append({
            "type": "Feature",
            "properties": {
                "name": (
                    f"{category.title()} change {i}"
                    if category
                    else f"Change region {i}"
                ),
                "category": category or "general",
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "normalized": [
                    round(x / width, 6),
                    round(y / height, 6),
                    round((x + w) / width, 6),
                    round((y + h) / height, 6),
                ],
            },
            "geometry": None,
        })

    return features



def run_change_analysis(
    before: np.ndarray,
    after: np.ndarray,
    query: str,
    min_area: int = 3000,
    spectral: dict[str, Any] | None = None,
    geospatial_aligned: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    before, after = resize_to_match(before, after)
    if spectral and spectral.get("before_bands") and spectral.get("after_bands"):
        h, w = after.shape[:2]
        spectral = {
            **spectral,
            "before_bands": _resize_band_dict(spectral["before_bands"], w, h),
            "after_bands": _resize_band_dict(spectral["after_bands"], w, h),
        }

    general = {
        "before": before.copy(),
        "after": after.copy(),
    }

    if geospatial_aligned:
        aligned, alignment_success, matches, inlier_ratio = before.copy(), True, 0, 1.0
    else:
        aligned, alignment_success, matches, inlier_ratio = align_images(
            before,
            after,
        )

    general["aligned"] = aligned
    general["alignment_success"] = alignment_success
    general["feature_matches"] = matches
    general["inlier_ratio"] = inlier_ratio

    normalized_before = normalize_image(aligned)
    normalized_after = normalize_image(after)

    difference = calculate_difference(
        normalized_before,
        normalized_after,
    )

    effective_min_area = max(200, min(int(min_area), 10000))
    # Catalog AOIs are often a few hundred pixels on a side; 3000px
    # would erase every region. Scale with image area.
    image_pixels = after.shape[0] * after.shape[1]
    area_scale = image_pixels / 1_000_000.0
    scaled = int(effective_min_area * max(area_scale, 0.04))
    effective_min_area = max(60, min(scaled, max(120, int(image_pixels * 0.015))))

    mask, threshold = create_change_mask(
        difference,
        effective_min_area,
    )

    contours = get_contours(
        mask,
        effective_min_area,
    )

    if len(contours) > 40:
        effective_min_area = adaptive_min_area(
            contours,
            effective_min_area,
            mask.size,
        )
        mask, threshold = create_change_mask(
            difference,
            effective_min_area,
        )
        contours = get_contours(
            mask,
            effective_min_area,
        )
        warnings_pre = [
            f"Detected {len(contours)} regions after adaptive filtering "
            f"(minimum area raised to {effective_min_area}px)."
        ]
    else:
        warnings_pre = []

    total_pixels = mask.size
    changed_pixels = int(np.count_nonzero(mask))
    change_pct = changed_pixels / total_pixels * 100 if total_pixels else 0.0

    largest_region_pct = 0.0

    if contours:
        largest_region_pct = (
            cv2.contourArea(contours[0])
            / total_pixels
            * 100
        )

    heatmap = make_heatmap(
        difference,
        mask,
    )

    overlay = make_overlay(
        after,
        contours,
    )

    category = classify_query(query)
    semantic = None
    semantic_overlay = None

    warnings = list(warnings_pre)

    if not alignment_success:
        warnings.append(
            "Registration confidence low; fallback alignment used."
        )

    if category != "general":
        spectral_sem = None
        if (
            spectral
            and spectral.get("has_nir")
            and spectral.get("before_bands")
            and spectral.get("after_bands")
            and category in ("vegetation", "water")
        ):
            spectral_sem = build_spectral_semantic(
                spectral["before_bands"],
                spectral["after_bands"],
                mask,
                category,
            )

        if spectral_sem is not None:
            semantic = spectral_sem
            semantic_overlay = make_semantic_overlay(
                after,
                semantic["mask"],
                semantic["direction"],
            )
            warnings.append(
                f"{category.title()} analysis used Sentinel-2 spectral indices "
                f"({'NDVI' if category == 'vegetation' else 'NDWI'}) on the same AOI."
            )
        else:
            try:
                before_labels, before_conf, class_names, semantic_backend = segment_image(
                    aligned, category
                )
                after_labels, after_conf, _, _ = segment_image(
                    after, category
                )

                use_heuristic = semantic_backend == "heuristic"

                if semantic_backend == "segformer" and class_names is not None:
                    if segformer_scene_mismatch(before_labels, after_labels, class_names):
                        use_heuristic = True
                        warnings.append(
                            "SegFormer ADE20k is not suited for this aerial/satellite scene; "
                            "deterministic visual classifier fallback active."
                        )
                    else:
                        semantic = analyze_semantic_category(
                            before_labels,
                            after_labels,
                            class_names,
                            category,
                            mask,
                            before_conf,
                            after_conf,
                        )
                        semantic["backend"] = "segformer"

                        if (
                            semantic.get("available")
                            and semantic.get("new_pixels", 0) == 0
                            and semantic.get("removed_pixels", 0) == 0
                            and changed_pixels > 0
                        ):
                            use_heuristic = True
                            warnings.append(
                                "SegFormer did not map this category inside detected "
                                "change regions; deterministic visual fallback used."
                            )

                if use_heuristic:
                    semantic = build_heuristic_semantic(
                        aligned,
                        after,
                        mask,
                        category,
                    )

                if semantic and semantic.get("available"):
                    semantic_overlay = make_semantic_overlay(
                        after,
                        semantic["mask"],
                        semantic["direction"],
                    )

                if semantic and semantic.get("backend") == "deterministic_visual_fallback":
                    warnings.append(
                        "Semantic category analysis is using a deterministic visual classifier fallback; "
                        "results are intended for hackathon demonstration, not geospatial ground-truth accuracy."
                    )

                if category == "terrain":
                    warnings.append(
                        "Terrain is interpreted as visible surface/ground-cover classes, not elevation or digital surface-model change."
                    )

            except Exception as exc:
                warnings.append(
                    f"Semantic enrichment fallback activated: {type(exc).__name__}."
                )
                semantic = build_heuristic_semantic(
                    aligned,
                    after,
                    mask,
                    category,
                )
                semantic_overlay = make_semantic_overlay(
                    after,
                    semantic["mask"],
                    semantic["direction"],
                )

    if category == "general":
        answer = (
            f"Detected {len(contours)} significant visual-change "
            f"region(s), covering approximately {change_pct:.2f}% "
            f"of the compared image."
        )
    elif semantic:
        answer = answer_for_category(
            category,
            semantic,
        )
    else:
        answer = (
            f"General change detection completed, but the semantic "
            f"{category} analysis was unavailable."
        )

    regions = bbox_features(
        contours,
        after.shape[1],
        after.shape[0],
        category=None,
    )

    semantic_regions = []

    if semantic and semantic.get("available"):
        semantic_contours = get_contours(
            semantic["mask"],
            80,
        )

        semantic_regions = bbox_features(
            semantic_contours,
            after.shape[1],
            after.shape[0],
            category=category,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "ok": True,
        "query": query,
        "category": category,
        "answer": answer,
        "method": [
            "ORB + RANSAC registration",
            "LAB illumination normalization",
            "multi-channel temporal difference",
            "Otsu + percentile adaptive threshold",
            "morphological denoising",
            "connected-component filtering",
            "semantic enrichment for category questions",
        ],
        "alignment": {
            "success": alignment_success,
            "matches_or_inliers": matches,
            "inlier_ratio": round(inlier_ratio, 4),
        },
        "metrics": [
            {
                "label": "Change regions",
                "value": str(len(contours)),
            },
            {
                "label": "Visual difference",
                "value": f"{change_pct:.2f}%",
            },
            {
                "label": "Largest region",
                "value": f"{largest_region_pct:.2f}%",
            },
            {
                "label": "Processing",
                "value": f"{elapsed_ms / 1000:.2f}s",
            },
        ],
        "technical": {
            "threshold": round(threshold, 2),
            "min_area_px": effective_min_area,
            "alignment_success": alignment_success,
            "feature_matches_or_inliers": matches,
            "inlier_ratio": round(inlier_ratio, 4),
            "processing_ms": round(elapsed_ms, 1),
            "semantic_backend": (
                semantic.get("backend")
                if semantic
                else ("none" if category == "general" else "unavailable")
            ),
        },
        "semantic": semantic_for_response(semantic),
        "regions": regions,
        "semantic_regions": semantic_regions,
        "warnings": warnings,
        "visuals": {
            "overlay_png": encode_png(overlay),
            "heatmap_png": encode_png(heatmap),
            "mask_png": encode_png(mask),
            "semantic_overlay_png": (
                encode_png(semantic_overlay)
                if semantic_overlay is not None
                else None
            ),
        },
        "timing_ms": round(elapsed_ms, 1),
        "disclaimer": (
            "Visual difference percentage is not model accuracy. "
            "Quantitative accuracy requires ground-truth change masks "
            "and should be reported with IoU, precision, recall, and F1."
        ),
    }


@app.get("/health")
def health():
    semantic_loaded = _segmenter is not None
    return {
        "status": "ok",
        "service": APP_NAME,
        "device": str(DEVICE),
        "semantic_model": MODEL_NAME if TRANSFORMERS_AVAILABLE else "deterministic_visual_fallback",
        "semantic_transformers_available": TRANSFORMERS_AVAILABLE,
        "semantic_model_loaded": semantic_loaded,
        "transformers_import_error": TRANSFORMERS_IMPORT_ERROR,
        "satellite_catalog": EARTH_SEARCH_URL,
        "satellite_collection": STAC_COLLECTION,
        "rasterio_available": RASTERIO_AVAILABLE,
    }


@app.post("/analyze")
async def analyze(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    query: str = Form("What changed between these images?"),
    min_area: int = Form(3000),
):
    started = time.perf_counter()

    if not before_image.content_type or not before_image.content_type.startswith("image/"):
        raise HTTPException(400, "The earlier file must be an image.")

    if not after_image.content_type or not after_image.content_type.startswith("image/"):
        raise HTTPException(400, "The later file must be an image.")

    try:
        before_raw = await before_image.read()
        after_raw = await after_image.read()

        if len(before_raw) > 35 * 1024 * 1024:
            raise HTTPException(413, "Earlier image is larger than 35 MB.")

        if len(after_raw) > 35 * 1024 * 1024:
            raise HTTPException(413, "Later image is larger than 35 MB.")

        before = read_image(before_raw)
        after = read_image(after_raw)

        if before.shape[0] * before.shape[1] > 100_000_000:
            raise HTTPException(413, "Earlier image is too large.")

        if after.shape[0] * after.shape[1] > 100_000_000:
            raise HTTPException(413, "Later image is too large.")

        before, after = resize_to_match(before, after)
        return run_change_analysis(before, after, query, min_area)
    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}",
        ) from exc

class TemporalAnalysisRequest(BaseModel):
    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    before_date: str
    cloud_threshold: float = Field(default=20.0, ge=5.0, le=80.0)
    query: str = "What changed between these dates?"
    min_area: int = Field(default=3000, ge=200, le=10000)
    aoi_radius_km: float = Field(default=2.5, ge=0.5, le=15.0)
    bbox: list[float] | None = None
    pair_id: str | None = None


def _analysis_from_pair(pair: RetrievedPair, query: str, min_area: int, pair_id: str) -> dict[str, Any]:
    spectral = None
    if pair.has_nir and pair.before_bands and pair.after_bands:
        spectral = {
            "has_nir": True,
            "before_bands": pair.before_bands,
            "after_bands": pair.after_bands,
        }

    result = run_change_analysis(
        pair.before_bgr,
        pair.after_bgr,
        query,
        min_area,
        spectral=spectral,
        geospatial_aligned=True,
    )

    result["pair_id"] = pair_id
    result["analysis_source"] = "satellite_catalog"
    result["location"] = {
        **pair.location,
        "bbox_wgs84": pair.aoi_bbox,
    }
    result["before"] = {
        "date": pair.before_meta.date,
        "datetime_utc": pair.before_meta.datetime_utc,
        "sensor": pair.before_meta.sensor,
        "cloud_cover": pair.before_meta.cloud_cover,
        "source": pair.before_meta.source,
        "stac_id": pair.before_meta.stac_id,
        "mgrs_tile": pair.before_meta.mgrs_tile,
    }
    result["after"] = {
        "date": pair.after_meta.date,
        "datetime_utc": pair.after_meta.datetime_utc,
        "sensor": pair.after_meta.sensor,
        "cloud_cover": pair.after_meta.cloud_cover,
        "source": pair.after_meta.source,
        "stac_id": pair.after_meta.stac_id,
        "mgrs_tile": pair.after_meta.mgrs_tile,
        "label": "Latest available acquisition",
    }
    result["provenance"] = pair.provenance()
    result["general_change"] = {
        "changed_area_percent": next(
            (m["value"] for m in result.get("metrics", []) if "Visual" in m.get("label", "")),
            None,
        ),
        "regions": next(
            (m["value"] for m in result.get("metrics", []) if "Change regions" in m.get("label", "")),
            None,
        ),
    }
    result["warnings"] = pair.warnings + result.get("warnings", [])
    result["visuals"] = result.get("visuals", {})
    result["visuals"]["before_png"] = encode_png(pair.before_bgr)
    result["visuals"]["after_png"] = encode_png(pair.after_bgr)
    if not pair.has_nir and result.get("category") in ("vegetation", "water"):
        result["warnings"].append(
            "NIR band unavailable for this scene; RGB/heuristic semantic analysis used."
        )
    return result


@app.post("/api/temporal-analysis")
async def temporal_analysis(request: TemporalAnalysisRequest):
    cache_id = pair_cache_key(
        location=request.location,
        latitude=request.latitude,
        longitude=request.longitude,
        before_date=request.before_date,
        cloud_threshold=request.cloud_threshold,
        bbox=request.bbox,
        aoi_radius_km=request.aoi_radius_km,
    )
    pair = get_cached_pair(request.pair_id) or get_cached_pair(cache_id)

    if pair is None:
        try:
            pair = await asyncio.to_thread(
                retrieve_temporal_pair,
                location=request.location,
                latitude=request.latitude,
                longitude=request.longitude,
                before_date=request.before_date,
                cloud_threshold=request.cloud_threshold,
                bbox=request.bbox,
                aoi_radius_km=request.aoi_radius_km,
            )
        except SceneRetrievalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Satellite retrieval failed: {type(exc).__name__}: {exc}",
            ) from exc
        store_cached_pair(cache_id, pair)

    try:
        return _analysis_from_pair(pair, request.query, request.min_area, cache_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}",
        ) from exc

