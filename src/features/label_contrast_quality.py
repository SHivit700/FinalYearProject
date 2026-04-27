"""Label Contrast Quality (LCQ) — measures luminance difference (ΔL in CIE Lab L*
space) between text pixels and the background immediately behind each label.

Scores highest in the moderate contrast zone (ΔL 40–80), penalises low contrast
(muddy/unreadable text) and extreme contrast combined with thick outlines (visually
harsh, overdesigned styling).
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BOX_AREA = 100   # px² — boxes smaller than this skip outline detection


# ---------------------------------------------------------------------------
# Per-label helpers
# ---------------------------------------------------------------------------

def _aabb_from_bbox(bbox: list, img_h: int, img_w: int) -> tuple[int, int, int, int]:
    """Convert polygon bbox to clamped (x1, y1, x2, y2)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1 = max(0, int(min(xs)))
    x2 = min(img_w, int(max(xs)))
    y1 = max(0, int(min(ys)))
    y2 = min(img_h, int(max(ys)))
    return x1, y1, x2, y2


def _sample_background(
    L_true: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    border: int,
) -> float | None:
    """
    Collect pixels from a ring of width `border` just outside the label box and
    return their median L* value. Returns None if no border pixels are available
    (label at image edge with all sides clipped).

    Falls back to the 90th-percentile of the interior (brightest region, likely
    the shape fill) when the border ring is fully clipped.
    """
    img_h, img_w = L_true.shape[:2]
    strips = [
        L_true[max(0, y1 - border):y1, x1:x2],     # top
        L_true[y2:min(img_h, y2 + border), x1:x2],    # bottom
        L_true[y1:y2,  max(0, x1 - border):x1],              # left
        L_true[y1:y2,  x2:min(img_w, x2 + border)],          # right
    ]
    border_pixels = np.concatenate(
        [s.ravel() for s in strips if s.size > 0]
    )
    if border_pixels.size == 0:
        return None
    return float(np.median(border_pixels))


def _sample_text_luminance(
    label_region: np.ndarray,
    L_background: float,
) -> float:
    """
    Return the L* value that best represents text pixels — whichever interior
    extreme (p10 or p90) is furthest from the known background luminance.

    This handles both dark-on-light (p10 = text) and light-on-dark (p90 = text)
    label styles without assuming a fixed polarity.
    """
    p10 = float(np.percentile(label_region, 10))
    p90 = float(np.percentile(label_region, 90))
    if abs(p10 - L_background) >= abs(p90 - L_background):
        return p10
    return p90


def _outline_pixel_ratio(label_bgr: np.ndarray) -> float:
    """
    Estimate the fraction of pixels that belong to text outlines / thick strokes
    by converting to greyscale, thresholding with Otsu, then eroding twice and
    measuring how much of the foreground mass was removed.

    A high ratio means thick outlines are present (aesthetically harsh styling).
    Returns 0.0 if the region is uniform or Otsu fails.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV + OTSU: foreground = dark ink (text + outlines)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ret == 0:
        return 0.0

    total_fg = int(np.sum(binary > 0))
    if total_fg == 0:
        return 0.0

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=2)
    eroded_fg = int(np.sum(eroded > 0))

    removed = total_fg - eroded_fg
    return removed / max(binary.size, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_label_contrast_quality(
    image: np.ndarray,
    labels: list[dict],
    border_width: int = 6,
    optimal_low: float = 40.0,
    optimal_high: float = 80.0,
    harsh_threshold: float = 90.0,
    min_acceptable: float = 25.0,
    low_penalty_weight: float = 2.0,
    high_penalty_weight: float = 0.5,
    min_penalty_weight: float = 1.5,
    outline_threshold: float = 0.15,
    harsh_penalty: float = 15.0,
    low_ratio_floor: float = 0.20,
    low_ratio_penalty_weight: float = 1.0,
) -> dict[str, Any]:
    """
    Compute Label Contrast Quality (LCQ) for a diagram.

    Measures luminance difference (ΔL in CIE Lab L* space) between text pixels
    and their local background for each label region. Scores highest in the
    moderate contrast zone (ΔL 40–80), penalises low contrast (muddy text) and
    extreme contrast with thick outlines (harsh, overdesigned styling).

    Args:
        image:        BGR image as numpy array.
        labels:       OCR label detections from run_label_detection (list of dicts
                      with "bbox" polygon). Same format used by all other metrics.
        border_width: Pixel width of background-sampling ring outside each label box.

    Returns dict with:
        label_contrast_score    — primary 0–100 score (None if no labels)
        mean_delta_L            — mean ΔL across all labels
        min_delta_L             — worst-case (lowest) ΔL across all labels
        std_delta_L             — ΔL consistency across labels
        mean_outline_ratio      — mean outline pixel ratio across eligible labels
        low_contrast_label_ratio — fraction of labels below optimal_low
        per_label_delta_L       — diagnostic list of per-label ΔL values
        harsh_contrast_flag     — True when extreme ΔL + thick outlines detected
        skipped_labels          — count of labels skipped due to missing/tiny bbox
        skipped_outline_count   — count of labels that skipped outline detection
    """
    _null = {
        "label_contrast_score": None,
        "mean_delta_L": None,
        "min_delta_L": None,
        "std_delta_L": None,
        "mean_outline_ratio": None,
        "low_contrast_label_ratio": None,
        "per_label_delta_L": [],
        "harsh_contrast_flag": False,
        "skipped_labels": 0,
        "skipped_outline_count": 0,
    }

    if not labels:
        return _null

    if image is None or image.size == 0:
        logger.warning("LCQ: received empty image — returning null result")
        return _null

    img_h, img_w = image.shape[:2]

    # Convert to CIE Lab and extract L* channel on the true 0–100 scale.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_true = lab[:, :, 0].astype(np.float64) * (100.0 / 255.0)

    delta_L_values: list[float] = []
    outline_ratios: list[float] = []
    skipped = 0
    skipped_outline = 0

    for lbl in labels:
        bbox = lbl.get("bbox", [])
        if not bbox:
            skipped += 1
            continue

        x1, y1, x2, y2 = _aabb_from_bbox(bbox, img_h, img_w)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            skipped += 1
            logger.warning(
                "LCQ: label bbox collapsed to zero area after clamping — skipped"
            )
            continue

        # --- Background luminance ---
        L_background = _sample_background(L_true, x1, y1, x2, y2, border_width)
        if L_background is None:
            # Edge label with no usable border: fall back to the 90th-percentile
            # of the interior (likely the shape fill, not the text strokes).
            L_background = float(np.percentile(L_true[y1:y2, x1:x2], 90))

        # --- Text luminance ---
        label_region = L_true[y1:y2, x1:x2]
        L_text = _sample_text_luminance(label_region, L_background)

        delta_L = abs(L_text - L_background)
        delta_L_values.append(delta_L)

        # --- Outline detection (skip tiny boxes) ---
        box_area = (x2 - x1) * (y2 - y1)
        if box_area >= _MIN_BOX_AREA:
            label_bgr = image[y1:y2, x1:x2]
            outline_ratios.append(_outline_pixel_ratio(label_bgr))
        else:
            skipped_outline += 1

    if not delta_L_values:
        result = dict(_null)
        result["skipped_labels"] = skipped
        return result

    # --- Aggregate ---
    mean_dL = float(np.mean(delta_L_values))
    min_dL = float(np.min(delta_L_values))
    std_dL = float(np.std(delta_L_values))
    mean_outline = float(np.mean(outline_ratios)) if outline_ratios else 0.0

    low_contrast_count = sum(1 for d in delta_L_values if d < optimal_low)
    low_contrast_ratio = low_contrast_count / len(delta_L_values)

    # --- Three-zone scoring ---
    if optimal_low <= mean_dL <= optimal_high:
        base_score = 100.0
    elif mean_dL < optimal_low:
        deficit = optimal_low - mean_dL
        base_score = max(0.0, 100.0 - deficit * low_penalty_weight)
    else:
        excess = mean_dL - optimal_high
        base_score = max(0.0, 100.0 - excess * high_penalty_weight)

    # Minimum ΔL penalty — catches local failure even when mean looks acceptable.
    if min_dL < min_acceptable:
        min_penalty = (min_acceptable - min_dL) * min_penalty_weight
        base_score = max(0.0, base_score - min_penalty)

    # Harshness penalty — extreme ΔL combined with thick outlines.
    harsh_flag = mean_dL > harsh_threshold and mean_outline > outline_threshold
    if harsh_flag:
        base_score = max(0.0, base_score - harsh_penalty)

    # Low-contrast label spread penalty — fires when a large fraction of individual
    # labels fall below optimal_low even if the mean looks acceptable. Prevents a
    # majority of high-ΔL labels from masking a large block of low-contrast text.
    if low_contrast_ratio > low_ratio_floor:
        excess_ratio = low_contrast_ratio - low_ratio_floor
        ratio_penalty = excess_ratio * 100.0 * low_ratio_penalty_weight
        base_score = max(0.0, base_score - ratio_penalty)

    return {
        "label_contrast_score": round(base_score, 4),
        "mean_delta_L": round(mean_dL, 4),
        "min_delta_L": round(min_dL, 4),
        "std_delta_L": round(std_dL, 4),
        "mean_outline_ratio": round(mean_outline, 4),
        "low_contrast_label_ratio": round(low_contrast_ratio, 4),
        "per_label_delta_L": [round(d, 4) for d in delta_L_values],
        "harsh_contrast_flag": harsh_flag,
        "skipped_labels": skipped,
        "skipped_outline_count": skipped_outline,
    }
