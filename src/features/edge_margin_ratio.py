#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _label_aabb(item: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if not bbox:
        return None
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs), min(ys), max(xs), max(ys))


def _contour_aabb(contour: np.ndarray) -> Tuple[float, float, float, float]:
    x, y, w, h = cv2.boundingRect(contour)
    return (float(x), float(y), float(x + w), float(y + h))


def _min_clearance_to_edges(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
) -> float:
    """Smallest gap from the AABB to the image border (pixels)."""
    dl = x_min
    dr = width - x_max
    dt = y_min
    db = height - y_max
    return float(min(dl, dr, dt, db))


def compute_edge_margin_metrics(
    labels: List[Dict[str, Any]],
    shapes: List[Dict[str, Any]],
    image_shape: Tuple[int, ...],
    margin_fraction: float = 0.05,
) -> Dict[str, Any]:
    """
    For each label (polygon bbox → AABB) and each shape (contour → bounding rect), compute the minimum distance in pixels from that box to the nearest image edge. Compare to ``margin_fraction * min(width, height)`` and count as violating if strictly below that.
    """
    height = int(image_shape[0])
    width = int(image_shape[1])
    if width <= 0 or height <= 0:
        return {
            "feature_name": "edge_clearance",
            "margin_fraction": float(margin_fraction),
            "labels_fraction_violating": -100.0,
            "shapes_fraction_violating": -100.0,
        }

    ref = min(width, height)
    margin_px = max(0.0, float(margin_fraction)) * float(ref)

    label_violations = 0
    n_lab = 0
    per_label_violations: List[Dict[str, Any]] = []
    for item in labels:
        box = _label_aabb(item)
        if box is None:
            continue
        n_lab += 1
        x1, y1, x2, y2 = box
        if _min_clearance_to_edges(x1, y1, x2, y2, width, height) < margin_px:
            label_violations += 1
            per_label_violations.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "bbox": item.get("bbox", []),
            })

    shape_violations = 0
    n_shp = 0
    per_shape_violations: List[Dict[str, Any]] = []
    for s in shapes:
        cnt = s.get("contour")
        if cnt is None:
            continue
        n_shp += 1
        x1, y1, x2, y2 = _contour_aabb(cnt)
        if _min_clearance_to_edges(x1, y1, x2, y2, width, height) < margin_px:
            shape_violations += 1
            per_shape_violations.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            })

    return {
        "feature_name": "edge_clearance",
        "margin_fraction": float(margin_fraction),
        "labels_fraction_violating": label_violations / n_lab if n_lab else -100.0,
        "shapes_fraction_violating": shape_violations / n_shp if n_shp else -100.0,
        "per_label_violations": per_label_violations,
        "per_shape_violations": per_shape_violations,
    }
