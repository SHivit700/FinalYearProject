"""Color Harmony Score (CHS) — measures whether a diagram's combined palette
(background fills + text foreground colors) is restrained and harmonious.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_K_MAX_DEFAULT = 8
_K_TEXT_DEFAULT = 4
_MIN_WEIGHT_DEFAULT = 0.02
_LAB_MERGE_THRESHOLD_DEFAULT = 10.0
_PALETTE_THRESHOLD_DEFAULT = 5
_PALETTE_WEIGHT_DEFAULT = 8.0

_NEUTRAL_CHROMA_THRESHOLD = 12.0
_MONOCHROME_SCORE = 85

_SMALL_IMAGE_K_MAX = 4
_SMALL_IMAGE_PIXELS = 100 * 100

# OpenCV uint8 Lab: L_cv > 230 ≈ L* > 90 — treated as near-white background bleed
_NEAR_WHITE_L_CV = 230
# Minimum dark pixels in a text box region to bother sampling
_MIN_TEXT_PIXELS = 5
# Minimum bounding-box area (px²) to attempt text color sampling
_MIN_TEXT_BOX_AREA = 20
# Lab distance threshold below which text and background are flagged as a contrast violation
_CONTRAST_VIOLATION_THRESHOLD = 20.0


# ---------------------------------------------------------------------------
# Colour-space helpers
# ---------------------------------------------------------------------------

def _bgr_to_lab(bgr_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab).astype(np.float32)


def _lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Convert Lab (…,3) array to LCH.

    OpenCV uint8 Lab stores a and b with a +128 offset (neutral = 128, not 0).
    Subtract 128 so white correctly gets C≈0 and hues are meaningful.
    """
    L = lab[..., 0]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=-1)


def _lab_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Approximate CIE76 ΔE between two colors in OpenCV uint8 Lab encoding."""
    dL = (float(c1[0]) - float(c2[0])) * 100.0 / 255.0
    da = float(c1[1]) - float(c2[1])
    db = float(c1[2]) - float(c2[2])
    return math.sqrt(dL * dL + da * da + db * db)


def _labs_to_hex(lab_centers: np.ndarray) -> list[str]:
    """Convert OpenCV uint8 Lab centers to hex strings via uint8 round-trip."""
    hex_colors = []
    for lab in lab_centers:
        lab_uint8 = np.array([[lab]], dtype=np.uint8)
        bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_Lab2BGR)
        b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
        hex_colors.append(f"#{r:02X}{g:02X}{b:02X}")
    return hex_colors


# ---------------------------------------------------------------------------
# Harmony classification
# ---------------------------------------------------------------------------

def _classify_pair(h1: float, h2: float, c1: float, c2: float) -> bool | None:
    """Return True/False for a chromatic pair, None to skip neutral pairs.

    Achromatic colors (white, grey, black) are excluded from the ratio entirely —
    they should neither reward nor penalise chromatic harmony.
    """
    if c1 < _NEUTRAL_CHROMA_THRESHOLD or c2 < _NEUTRAL_CHROMA_THRESHOLD:
        return None

    delta = abs(h1 - h2)
    if delta > 180.0:
        delta = 360.0 - delta

    if delta <= 30.0:
        return True   # Analogous
    if 165.0 <= delta <= 195.0:
        return True   # Complementary
    if 115.0 <= delta <= 125.0:
        return True   # Triadic (~120°; 240° folds to the same range)
    if 150.0 <= delta <= 165.0 or 195.0 <= delta <= 210.0:
        return True   # Split-complementary
    return False


# ---------------------------------------------------------------------------
# K-means helper
# ---------------------------------------------------------------------------

def _kmeans_active(pixels: np.ndarray, k: int, min_weight: float) -> np.ndarray:
    """Run k-means and return centers whose pixel-weight >= min_weight."""
    if pixels.shape[0] == 0 or k < 1:
        return np.empty((0, 3), dtype=np.float32)
    k = min(k, pixels.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    try:
        _, labels_km, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error as exc:
        logger.warning("color_harmony: k-means failed: %s", exc)
        return np.empty((0, 3), dtype=np.float32)
    counts = np.bincount(labels_km.flatten(), minlength=k)
    weights = counts / pixels.shape[0]
    return centers[weights >= min_weight]


# ---------------------------------------------------------------------------
# Text color extraction
# ---------------------------------------------------------------------------

def _extract_text_colors(
    lab_image: np.ndarray,
    labels: list[dict],
    k_text: int,
    min_weight: float,
) -> tuple[np.ndarray, int]:
    """
    Sample foreground (text) colors from OCR label bounding boxes.

    For each box: crop the region, discard near-white pixels (background bleed,
    L_cv > 230 ≈ L* > 90), collect the remaining dark pixels as text candidates.
    All candidates are pooled and clustered with k-means.

    Returns (text_centers, skipped_count).
    """
    h_img, w_img = lab_image.shape[:2]
    text_pixels_list: list[np.ndarray] = []
    skipped = 0

    for lbl in labels:
        bbox = lbl.get("bbox", [])
        if not bbox:
            skipped += 1
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1 = max(0, int(min(xs)))
        x2 = min(w_img, int(max(xs)))
        y1 = max(0, int(min(ys)))
        y2 = min(h_img, int(max(ys)))

        if (x2 - x1) * (y2 - y1) < _MIN_TEXT_BOX_AREA:
            skipped += 1
            continue

        region = lab_image[y1:y2, x1:x2].reshape(-1, 3)
        # Keep only non-white pixels (potential ink/text pixels)
        dark_mask = region[:, 0] <= _NEAR_WHITE_L_CV
        dark_pixels = region[dark_mask]

        if dark_pixels.shape[0] < _MIN_TEXT_PIXELS:
            skipped += 1
            continue

        text_pixels_list.append(dark_pixels)

    if not text_pixels_list:
        return np.empty((0, 3), dtype=np.float32), skipped

    all_text = np.vstack(text_pixels_list).astype(np.float32)
    centers = _kmeans_active(all_text, k_text, min_weight)
    return centers, skipped


# ---------------------------------------------------------------------------
# Palette deduplication
# ---------------------------------------------------------------------------

def _merge_colors(
    bg_centers: np.ndarray,
    text_centers: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Merge near-identical colors from background and text palettes using
    union-find on pairs with CIE76 ΔE < threshold.

    Returns the deduplicated unified palette as an array of Lab centers.
    """
    all_centers: list[list[float]] = []
    if bg_centers.shape[0] > 0:
        all_centers.extend(bg_centers.tolist())
    if text_centers.shape[0] > 0:
        all_centers.extend(text_centers.tolist())

    n = len(all_centers)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            ci = np.array(all_centers[i])
            cj = np.array(all_centers[j])
            if _lab_distance(ci, cj) < threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    seen: set[int] = set()
    unified: list[list[float]] = []
    for i in range(n):
        root = find(i)
        if root not in seen:
            seen.add(root)
            unified.append(all_centers[i])

    return np.array(unified, dtype=np.float32)


# ---------------------------------------------------------------------------
# Contrast violation diagnostic
# ---------------------------------------------------------------------------

def _detect_contrast_violation(
    bg_centers: np.ndarray,
    text_centers: np.ndarray,
) -> bool:
    """Return True if any text color is perceptually too close to a background color."""
    for tc in text_centers:
        for bc in bg_centers:
            if _lab_distance(tc, bc) < _CONTRAST_VIOLATION_THRESHOLD:
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_color_harmony_score(
    image: np.ndarray,
    labels: list[dict],
    k_max: int = _K_MAX_DEFAULT,
    k_text: int = _K_TEXT_DEFAULT,
    min_weight: float = _MIN_WEIGHT_DEFAULT,
    lab_merge_threshold: float = _LAB_MERGE_THRESHOLD_DEFAULT,
    palette_threshold: int = _PALETTE_THRESHOLD_DEFAULT,
    palette_weight: float = _PALETTE_WEIGHT_DEFAULT,
) -> dict[str, Any]:
    """
    Compute Color Harmony Score (CHS) from a diagram image and its OCR labels.

    Extracts background colors from the full image via k-means and text colors
    from label bounding boxes by sampling dark (non-white) pixels. The two sets
    are merged into a unified palette (near-duplicates collapsed by Lab distance),
    then scored on palette restraint and hue-angle harmony among chromatic colors.

    Args:
        image:   BGR image as numpy array.
        labels:  OCR label detections from run_label_detection (list of dicts
                 with "bbox" polygon). Used to isolate text foreground colors.

    Returns dict with:
      color_harmony_score, palette_size, harmony_ratio,
      background_palette_size, text_palette_size, palette_penalty,
      background_colors_lab, text_colors_lab, dominant_colors_hex,
      monochrome, low_confidence, text_colors_unavailable,
      skipped_labels, contrast_violation.
    """
    _null: dict[str, Any] = {
        "color_harmony_score": None,
        "palette_size": None,
        "harmony_ratio": None,
        "background_palette_size": None,
        "text_palette_size": None,
        "palette_penalty": None,
        "background_colors_lab": None,
        "text_colors_lab": None,
        "dominant_colors_hex": None,
        "monochrome": False,
        "low_confidence": False,
        "text_colors_unavailable": False,
        "skipped_labels": 0,
        "contrast_violation": False,
    }

    if image is None or image.size == 0:
        logger.warning("color_harmony: received empty or None image.")
        return _null

    if image.ndim != 3 or image.shape[2] != 3:
        logger.warning(
            "color_harmony: expected 3-channel BGR image, got shape %s.", image.shape
        )
        return _null

    h_img, w_img = image.shape[:2]
    low_confidence = (h_img * w_img) < _SMALL_IMAGE_PIXELS
    effective_k = _SMALL_IMAGE_K_MAX if low_confidence else k_max

    lab_image = _bgr_to_lab(image)

    # --- Step 1a: background colors via k-means on full image ---
    all_pixels = lab_image.reshape(-1, 3).astype(np.float32)
    bg_centers = _kmeans_active(all_pixels, effective_k, min_weight)

    if bg_centers.shape[0] == 0:
        logger.warning("color_harmony: no background clusters above min_weight.")
        return _null

    # --- Step 1b: text colors from label bounding boxes ---
    text_colors_unavailable = False
    skipped_labels = 0
    if not labels:
        text_colors_unavailable = True
        text_centers: np.ndarray = np.empty((0, 3), dtype=np.float32)
    else:
        text_centers, skipped_labels = _extract_text_colors(
            lab_image, labels, k_text, min_weight
        )
        if text_centers.shape[0] == 0:
            text_colors_unavailable = True

    # --- Step 1c: merge into unified palette ---
    unified = _merge_colors(bg_centers, text_centers, lab_merge_threshold)

    if unified.shape[0] == 0:
        logger.warning("color_harmony: unified palette is empty after merge.")
        return _null

    # --- Step 2: palette size and penalty ---
    bg_palette_size = int(bg_centers.shape[0])
    text_palette_size = int(text_centers.shape[0])
    palette_size = int(unified.shape[0])

    palette_penalty = (
        0.0 if palette_size <= palette_threshold
        else (palette_size - palette_threshold) * palette_weight
    )

    lch_centers = _lab_to_lch(unified)

    # --- Monochrome check ---
    if np.all(lch_centers[:, 1] < _NEUTRAL_CHROMA_THRESHOLD):
        hex_colors = _labs_to_hex(unified)
        contrast_violation_mono = (
            _detect_contrast_violation(bg_centers, text_centers)
            if text_centers.shape[0] > 0
            else False
        )
        return {
            "color_harmony_score": _MONOCHROME_SCORE,
            "palette_size": palette_size,
            "harmony_ratio": 1.0,
            "background_palette_size": bg_palette_size,
            "text_palette_size": text_palette_size,
            "palette_penalty": 0.0,
            "background_colors_lab": bg_centers.tolist(),
            "text_colors_lab": text_centers.tolist() if text_centers.shape[0] > 0 else [],
            "dominant_colors_hex": hex_colors,
            "monochrome": True,
            "low_confidence": low_confidence,
            "text_colors_unavailable": text_colors_unavailable,
            "skipped_labels": skipped_labels,
            "contrast_violation": contrast_violation_mono,
        }

    # --- Single unified color ---
    if palette_size == 1:
        hex_colors = _labs_to_hex(unified)
        return {
            "color_harmony_score": 100.0,
            "palette_size": 1,
            "harmony_ratio": 1.0,
            "background_palette_size": bg_palette_size,
            "text_palette_size": text_palette_size,
            "palette_penalty": 0.0,
            "background_colors_lab": bg_centers.tolist(),
            "text_colors_lab": text_centers.tolist() if text_centers.shape[0] > 0 else [],
            "dominant_colors_hex": hex_colors,
            "monochrome": False,
            "low_confidence": True,
            "text_colors_unavailable": text_colors_unavailable,
            "skipped_labels": skipped_labels,
            "contrast_violation": False,
        }

    # --- Step 3: harmony reward (chromatic pairs only) ---
    n = palette_size
    harmonic_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            result = _classify_pair(
                lch_centers[i, 2], lch_centers[j, 2],
                lch_centers[i, 1], lch_centers[j, 1],
            )
            if result is None:
                continue  # achromatic — excluded from ratio
            total_pairs += 1
            if result:
                harmonic_pairs += 1

    # No chromatic pairs → palette is effectively achromatic (borderline chroma)
    harmony_ratio = harmonic_pairs / total_pairs if total_pairs > 0 else 1.0

    # --- Step 4: composite score ---
    raw_score = (harmony_ratio * 100.0) - palette_penalty
    color_harmony_score = round(max(0.0, min(100.0, raw_score)), 2)

    contrast_violation = (
        _detect_contrast_violation(bg_centers, text_centers)
        if text_centers.shape[0] > 0
        else False
    )
    hex_colors = _labs_to_hex(unified)

    return {
        "color_harmony_score": color_harmony_score,
        "palette_size": palette_size,
        "harmony_ratio": round(harmony_ratio, 4),
        "background_palette_size": bg_palette_size,
        "text_palette_size": text_palette_size,
        "palette_penalty": round(palette_penalty, 2),
        "background_colors_lab": bg_centers.tolist(),
        "text_colors_lab": text_centers.tolist() if text_centers.shape[0] > 0 else [],
        "dominant_colors_hex": hex_colors,
        "monochrome": False,
        "low_confidence": low_confidence,
        "text_colors_unavailable": text_colors_unavailable,
        "skipped_labels": skipped_labels,
        "contrast_violation": contrast_violation,
    }
