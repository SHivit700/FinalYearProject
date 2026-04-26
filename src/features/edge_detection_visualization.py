#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 40
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 10
ENDPOINT_MARGIN = 20
TEXT_MASK_MARGIN = 4        # px padding around each label bbox before masking

# Node-box filter thresholds (mirrors isolated_box_detection.py)
MIN_BOX_WIDTH = 80
MIN_BOX_HEIGHT = 40
MIN_RECTANGULARITY = 0.50
MAX_AREA_FRACTION = 0.50
CONTAINMENT_THRESHOLD = 0.85


def _intersection_area(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _filter_node_boxes(shapes: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> list[dict[str, Any]]:
    img_area = float(image_shape[0] * image_shape[1])
    candidates = [
        s for s in shapes
        if s["w"] >= MIN_BOX_WIDTH
        and s["h"] >= MIN_BOX_HEIGHT
        and s["rectangularity"] >= MIN_RECTANGULARITY
        and (s["w"] * s["h"]) / img_area <= MAX_AREA_FRACTION
    ]
    result = []
    for i, b1 in enumerate(candidates):
        area1 = float(b1["w"] * b1["h"])
        if area1 == 0:
            continue
        contained = False
        for j, b2 in enumerate(candidates):
            if i == j:
                continue
            if float(b2["w"] * b2["h"]) <= area1:
                continue
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
    return x - margin <= px <= x + w + margin and y - margin <= py <= y + h + margin


def _both_endpoints_same_box(x1, y1, x2, y2, boxes: list[dict], margin: int) -> bool:
    for box in boxes:
        if _point_near_box(x1, y1, box, margin) and _point_near_box(x2, y2, box, margin):
            return True
    return False


def _build_text_mask(labels: list[dict[str, Any]], edge_shape: tuple[int, int], margin: int) -> np.ndarray:
    """Return a binary mask (255 = text region) covering all label bounding boxes."""
    mask = np.zeros(edge_shape, dtype=np.uint8)
    for label in labels:
        pts = np.array(label["bbox"], dtype=np.float32)
        # Expand each point outward from the centroid by margin px
        centroid = pts.mean(axis=0)
        directions = pts - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        expanded = pts + directions / norms * margin
        cv2.fillPoly(mask, [expanded.astype(np.int32)], 255)
    return mask


def compute_edge_detection_visualization(
    bgr_image: np.ndarray,
    shapes: list[dict[str, Any]],
    image_shape: tuple[int, int, int],
    labels: list[dict[str, Any]] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    node_boxes = _filter_node_boxes(shapes, image_shape)

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Erase node box regions from the edge map
    box_mask = np.zeros(edges.shape, dtype=np.uint8)
    for s in node_boxes:
        cv2.rectangle(box_mask, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), 255, -1)
    edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(box_mask))

    # Erase text regions from the edge map using OCR label bboxes
    if labels:
        text_mask = _build_text_mask(labels, edges.shape[:2], TEXT_MASK_MARGIN)
        edges_clean = cv2.bitwise_and(edges_clean, cv2.bitwise_not(text_mask))

    raw_lines = cv2.HoughLinesP(
        edges_clean,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    # Drop residual box-border segments (both endpoints on the same node box)
    arrow_lines: list[tuple[int, int, int, int]] = []
    if raw_lines is not None:
        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            if not _both_endpoints_same_box(x1, y1, x2, y2, node_boxes, ENDPOINT_MARGIN):
                arrow_lines.append((x1, y1, x2, y2))

    segment_count = len(arrow_lines)
    lengths = [float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) for x1, y1, x2, y2 in arrow_lines]

    if output_path is not None:
        canny_bgr = cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR)

        overlay = bgr_image.copy()
        for x1, y1, x2, y2 in arrow_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(overlay, (x1, y1), 4, (0, 0, 255), -1)
            cv2.circle(overlay, (x2, y2), 4, (0, 0, 255), -1)

        result = np.hstack([canny_bgr, overlay])
        cv2.putText(result, f"Arrow segments: {segment_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result)

    return {
        "segment_count": segment_count,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "min_length": float(np.min(lengths)) if lengths else 0.0,
        "max_length": float(np.max(lengths)) if lengths else 0.0,
    }
