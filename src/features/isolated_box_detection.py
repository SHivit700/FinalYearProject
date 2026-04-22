#!/usr/bin/env python3
"""
Isolated box detection: find diagram nodes (boxes/classes) with no connector lines touching them — "islands" that are visually present but structurally disconnected.

Pipeline:
  1. Filter shapes → candidate boxes (size + rectangularity, no touches_border exclusion)
  2. Remove shapes contained within other candidate shapes (inner sections of UML boxes)
  3. Mask box interiors out of the Canny edge image so dividers/borders are excluded
  4. HoughLinesP on masked edges → connector line segments
  5. Discard segments whose both endpoints lie near the SAME box (residual border noise)
  6. Classify each box: connected if ≥1 remaining segment endpoint is within margin px
  7. Build annotated overlay + return metrics dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
MIN_BOX_WIDTH = 80
MIN_BOX_HEIGHT = 40
MIN_RECTANGULARITY = 0.50   # filters diagonal-line contours (low coverage of bbox)
MAX_AREA_FRACTION = 0.50    # anything larger than half the image is the outer frame
CONTAINMENT_THRESHOLD = 0.85  # fraction of a box's bbox that must overlap a larger box
                               # to be considered "contained" (inner section of a class)

HOUGH_THRESHOLD = 30       # lower = more sensitive; needed for thin UML arrows
MIN_LINE_LENGTH = 30        # px; short arrows still get detected
MAX_LINE_GAP = 15           # px; bridges small breaks in dashed connectors

DEFAULT_MARGIN = 20         # px tolerance when checking if a line endpoint touches a box
BOX_INTERIOR_INSET = 0      # 0 = mask the full box rect including borders.
CANNY_LOW = 50
CANNY_HIGH = 150


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _intersection_area(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _remove_contained(boxes: list[dict]) -> list[dict]:
    """
    Remove boxes that are mostly (>= CONTAINMENT_THRESHOLD) inside a larger box.
    This eliminates the inner section contours (title/attributes/methods rows) that
    detect_shapes picks up for each UML class box, keeping only the outermost shape.
    """
    result = []
    for i, b1 in enumerate(boxes):
        area1 = float(b1["w"] * b1["h"])
        if area1 == 0:
            continue
        contained = False
        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            area2 = float(b2["w"] * b2["h"])
            if area2 <= area1:
                continue  # b2 is not larger; can't contain b1
            inter = _intersection_area(b1["x"], b1["y"], b1["w"], b1["h"],
                                        b2["x"], b2["y"], b2["w"], b2["h"])
            if inter / area1 >= CONTAINMENT_THRESHOLD:
                contained = True
                break
        if not contained:
            result.append(b1)
    return result


def _point_near_box(px: int, py: int, box: dict, margin: int) -> bool:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    return (x - margin <= px <= x + w + margin and
            y - margin <= py <= y + h + margin)


def _both_endpoints_same_box(x1, y1, x2, y2, boxes: list[dict], margin: int) -> bool:
    """True if both endpoints fall near the SAME box — residual box-border segment."""
    for box in boxes:
        if (_point_near_box(x1, y1, box, margin) and
                _point_near_box(x2, y2, box, margin)):
            return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_isolated_box_metrics(
    bgr_image: np.ndarray,
    shapes: list[dict],
    image_shape: tuple,
    margin: int = DEFAULT_MARGIN,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Detect isolated (island) boxes in a diagram.

    Parameters
    ----------
    bgr_image   : BGR image array (from run_label_detection)
    shapes      : shape dicts from run_shape_detection (x, y, w, h, area, ...)
    image_shape : (H, W, C) tuple
    margin      : px tolerance for connector-to-box proximity check
    output_path : if given, save annotated overlay PNG here

    Returns
    -------
    dict with keys:
        total_box_count, connected_count, island_count, island_fraction,
        isolated_box_score (0–100, higher = fewer islands),
        island_boxes (list of {x, y, w, h}),
        overlay_image (BGR ndarray)
    """
    h_img, w_img = image_shape[:2]
    image_area = float(h_img * w_img)

    # ------------------------------------------------------------------
    # 1. Filter candidate boxes from pre-computed shapes
    #    - sufficient size
    #    - rectangular enough (not a diagonal-line contour)
    #    - not the image-scale outer frame
    #    NOTE: touches_border is intentionally NOT used here — island boxes
    #    often sit near image edges and would be incorrectly excluded.
    # ------------------------------------------------------------------
    candidate_boxes = [
        s for s in shapes
        if s["w"] >= MIN_BOX_WIDTH
        and s["h"] >= MIN_BOX_HEIGHT
        and s.get("rectangularity", 0.0) >= MIN_RECTANGULARITY
        and (s["w"] * s["h"]) < MAX_AREA_FRACTION * image_area
    ]

    # ------------------------------------------------------------------
    # 2. Drop inner-section shapes (title/attributes/methods rows inside a
    #    UML class box) — keep only the outermost box for each class.
    # ------------------------------------------------------------------
    candidate_boxes = _remove_contained(candidate_boxes)

    if not candidate_boxes:
        overlay = bgr_image.copy()
        _save_overlay(overlay, output_path)
        return _empty_result(overlay)

    # ------------------------------------------------------------------
    # 3. Build Canny edge image, then mask out box interiors so that the
    #    internal divider lines and box-border pixels don't show up as
    #    connectors in the HoughLinesP step.
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    box_interior_mask = np.zeros(edges.shape, dtype=np.uint8)
    ins = BOX_INTERIOR_INSET
    for box in candidate_boxes:
        x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(
            box_interior_mask,
            (x + ins, y + ins),
            (x + bw - ins, y + bh - ins),
            255,
            -1,
        )

    edges_for_hough = cv2.bitwise_and(edges, cv2.bitwise_not(box_interior_mask))

    # ------------------------------------------------------------------
    # 4. Detect connector line segments via HoughLinesP
    # ------------------------------------------------------------------
    raw_lines = cv2.HoughLinesP(
        edges_for_hough,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    # ------------------------------------------------------------------
    # 5. Filter residual box-border segments (both endpoints same box)
    # ------------------------------------------------------------------
    connector_lines: list[tuple[int, int, int, int]] = []
    if raw_lines is not None:
        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            if not _both_endpoints_same_box(x1, y1, x2, y2, candidate_boxes, margin):
                connector_lines.append((x1, y1, x2, y2))

    # ------------------------------------------------------------------
    # 6. Classify each box as connected or island
    # ------------------------------------------------------------------
    connected_flags = [False] * len(candidate_boxes)
    for x1, y1, x2, y2 in connector_lines:
        for i, box in enumerate(candidate_boxes):
            if (_point_near_box(x1, y1, box, margin) or
                    _point_near_box(x2, y2, box, margin)):
                connected_flags[i] = True

    island_boxes = [
        {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
        for b, connected in zip(candidate_boxes, connected_flags)
        if not connected
    ]

    total = len(candidate_boxes)
    island_count = len(island_boxes)
    connected_count = total - island_count
    island_fraction = island_count / total if total > 0 else 0.0
    isolated_box_score = round((1.0 - island_fraction) * 100)

    # ------------------------------------------------------------------
    # 7. Build annotated overlay
    # ------------------------------------------------------------------
    overlay = bgr_image.copy()

    for x1, y1, x2, y2 in connector_lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 180, 0), 1)

    for box, connected in zip(candidate_boxes, connected_flags):
        x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
        if connected:
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        else:
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 220), 2)
            cv2.putText(
                overlay, "ISLAND",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2,
                cv2.LINE_AA,
            )

    _save_overlay(overlay, output_path)

    return {
        "total_box_count": total,
        "connected_count": connected_count,
        "island_count": island_count,
        "island_fraction": round(island_fraction, 4),
        "isolated_box_score": isolated_box_score,
        "island_boxes": island_boxes,
        "overlay_image": overlay,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_overlay(overlay: np.ndarray, output_path: str | None) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)


def _empty_result(overlay: np.ndarray) -> dict[str, Any]:
    return {
        "total_box_count": 0,
        "connected_count": 0,
        "island_count": 0,
        "island_fraction": 0.0,
        "isolated_box_score": 100,
        "island_boxes": [],
        "overlay_image": overlay,
    }
