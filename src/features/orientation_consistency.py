"""Orientation Consistency (OC) — measures how uniformly labels are aligned
to a dominant text orientation within the diagram.

Computes a dominant orientation (mode of per-label angles) and scores the
fraction of labels whose angle falls within ±15° of that dominant orientation.
High consistency → orderly; many labels at odd angles → visually noisy.

Angle extraction handles two distinct cases that arise from EasyOCR output:
  • Non-axis-aligned bboxes (dy ≠ 0): the bottom-edge vector gives the true
    rotation angle directly (used for diagonal / tilted text).
  • Axis-aligned bboxes (dy == 0): the bottom-edge carries no rotation signal,
    so orientation is inferred from the AABB aspect ratio — tall-and-narrow
    boxes indicate vertical (90°-rotated) text; wide-and-short indicate
    horizontal (0°) text; square-ish bboxes are skipped as ambiguous.

Scoring amplifies inconsistency: each percentage point of non-dominant labels
reduces the score by _INCONSISTENCY_AMPLIFICATION points, so even a small
fraction of rotated labels produces a clearly low score.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CONSISTENCY_WINDOW_DEG = 15.0
_MIN_LABELS = 2
_BIN_WIDTH_DEG = 5.0
# AABB aspect-ratio thresholds for the axis-aligned fallback path.
_SQUARE_RATIO_LO = 0.6   # h/w below this → horizontal  (w ≥ 1.67 × h)
_SQUARE_RATIO_HI = 1.6   # h/w above this → vertical    (h ≥ 1.6  × w)
# How much each unit of inconsistent fraction penalises the score.
# With 4.0: having 15 % inconsistent labels → score ≈ 40.
_INCONSISTENCY_AMPLIFICATION = 4.0


def _label_angle_deg(bbox: list) -> float | None:
    """Return an orientation angle in [-90, 90) degrees for a label bbox.

    Two-path logic:
    1. If the bottom edge has a non-zero dy (non-axis-aligned bbox), compute the
       angle from that vector directly — this handles diagonally-rotated text
       whose rotation EasyOCR preserves in the quad corners.
    2. If dy == 0 (axis-aligned bbox), use the AABB aspect ratio:
         h/w < _SQUARE_RATIO_LO  →  0°   (horizontal)
         h/w > _SQUARE_RATIO_HI  →  -90° (vertical)
         between the two          →  None (ambiguous single-char glyph, skip)

    Returns None for degenerate or ambiguous bboxes.
    """
    if not bbox or len(bbox) < 2:
        return None

    try:
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
    except (IndexError, TypeError, ValueError):
        return None

    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    if w == 0.0 and h == 0.0:
        return None

    # --- Path 1: non-axis-aligned bbox → use bottom-edge vector ---
    if len(bbox) >= 4:
        try:
            bl = bbox[3]
            br = bbox[2]
            dx = float(br[0]) - float(bl[0])
            dy = float(br[1]) - float(bl[1])
        except (IndexError, TypeError, ValueError):
            dx = dy = 0.0

        if dy != 0.0:
            angle = math.degrees(math.atan2(dy, dx))
            if angle >= 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
            return angle

    # --- Path 2: axis-aligned bbox → infer from aspect ratio ---
    if w == 0.0:
        return -90.0
    if h == 0.0:
        return 0.0

    ratio = h / w
    if ratio < _SQUARE_RATIO_LO:
        return 0.0
    if ratio > _SQUARE_RATIO_HI:
        return -90.0

    return None   # square-ish: orientation-ambiguous, skip


def _dominant_orientation(angles: list[float], bin_width: float = _BIN_WIDTH_DEG) -> float:
    """Find the dominant orientation by histogram bin voting + circular refinement."""
    arr = np.array(angles, dtype=np.float64)

    bins = np.arange(-90.0, 90.0 + bin_width, bin_width)
    counts, edges = np.histogram(arr, bins=bins)

    best_idx = int(np.argmax(counts))
    dominant = float(edges[best_idx] + bin_width / 2.0)

    # Refine with circular mean of angles near the winning bin.
    candidate_mask = np.abs(arr - dominant) <= bin_width
    for shift in (-180.0, 180.0):
        candidate_mask |= np.abs(arr - (dominant + shift)) <= bin_width
    candidates = arr[candidate_mask]

    if len(candidates) == 0:
        return dominant

    doubled = np.deg2rad(candidates * 2.0)
    sin_mean = np.mean(np.sin(doubled))
    cos_mean = np.mean(np.cos(doubled))
    return float(math.degrees(math.atan2(sin_mean, cos_mean)) / 2.0)


def _angular_distance(a: float, b: float) -> float:
    """Smallest angular distance between two angles in [-90, 90) degrees."""
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def compute_orientation_consistency(
    labels: list[dict],
    consistency_window: float = _CONSISTENCY_WINDOW_DEG,
    inconsistency_amplification: float = _INCONSISTENCY_AMPLIFICATION,
) -> dict[str, Any]:
    """Compute Orientation Consistency (OC) for a diagram.

    Args:
        labels:                    OCR label detections with "bbox" polygons.
        consistency_window:        Half-width (°) around dominant; labels within
                                   this are counted consistent (default 15°).
        inconsistency_amplification: Multiplier on the inconsistent fraction
                                   before subtracting from 100 (default 4.0).

    Returns dict with:
        orientation_consistency_score — 0–100 (None if <2 scorable labels)
        dominant_orientation_deg      — modal orientation in degrees
        consistent_label_fraction     — raw fraction within the window
        total_labels                  — total OCR label count
        scored_labels                 — labels with a resolved angle
        skipped_labels                — labels dropped (ambiguous/degenerate bbox)
        low_confidence                — True when scored_labels < _MIN_LABELS
    """
    _null: dict[str, Any] = {
        "orientation_consistency_score": None,
        "dominant_orientation_deg": None,
        "consistent_label_fraction": None,
        "total_labels": len(labels),
        "scored_labels": 0,
        "skipped_labels": 0,
        "low_confidence": True,
    }

    if not labels:
        return _null

    angles: list[float] = []
    skipped = 0
    label_angle_pairs: list[tuple[dict, float]] = []

    for lbl in labels:
        angle = _label_angle_deg(lbl.get("bbox", []))
        if angle is None:
            skipped += 1
        else:
            angles.append(angle)
            label_angle_pairs.append((lbl, angle))

    _null["skipped_labels"] = skipped
    _null["scored_labels"] = len(angles)

    if len(angles) < _MIN_LABELS:
        logger.debug(
            "OC: only %d valid angle(s) — returning null (need ≥%d)",
            len(angles),
            _MIN_LABELS,
        )
        return _null

    dominant = _dominant_orientation(angles)

    per_label_info: list[dict] = []
    consistent_count = 0
    for lbl, angle in label_angle_pairs:
        is_consistent = _angular_distance(angle, dominant) <= consistency_window
        if is_consistent:
            consistent_count += 1
        bbox = lbl.get("bbox", [])
        xs = [float(p[0]) for p in bbox] if bbox else []
        ys = [float(p[1]) for p in bbox] if bbox else []
        per_label_info.append({
            "bbox": bbox,
            "text": lbl.get("text", ""),
            "angle_deg": round(angle, 4),
            "consistent": is_consistent,
            "x1": int(min(xs)) if xs else 0,
            "y1": int(min(ys)) if ys else 0,
            "x2": int(max(xs)) if xs else 0,
            "y2": int(max(ys)) if ys else 0,
        })

    consistent_fraction = consistent_count / len(angles)
    inconsistent_fraction = 1.0 - consistent_fraction

    raw_score = max(0.0, 100.0 - inconsistent_fraction * 100.0 * inconsistency_amplification)

    return {
        "orientation_consistency_score": round(raw_score, 4),
        "dominant_orientation_deg": round(dominant, 4),
        "consistent_label_fraction": round(consistent_fraction, 4),
        "total_labels": len(labels),
        "scored_labels": len(angles),
        "skipped_labels": skipped,
        "low_confidence": len(angles) < _MIN_LABELS,
        "per_label_info": per_label_info,
    }
