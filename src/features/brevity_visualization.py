"""Brevity Visualization — outlines box-shaped shapes that contain more labels
than the density threshold, flagging the density penalty that drives the score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_RED_BORDER = (40, 30, 210)   # red (BGR)
_RED_THICKNESS = 3
_BADGE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BADGE_SCALE = 0.40
_BADGE_THICK = 1

_MIN_BOX_AREA = 4000  # px² — ignore tiny noise shapes


def _find_dense_boxes(shapes: list, per_label_info: list, density_threshold: float) -> list:
    """Return bounding rects of top-level box shapes with more labels than the threshold."""
    box_shapes = [
        s for s in shapes
        if s.get("parent_contour_index", -1) in (-1, 0)
        and s.get("rectangularity", 0.0) > 0.7
        and s.get("aspect_ratio", 999.0) < 8.0
    ]

    centroids = [
        ((lbl["x1"] + lbl["x2"]) // 2, (lbl["y1"] + lbl["y2"]) // 2)
        for lbl in per_label_info
    ]

    dense = []
    for shape in box_shapes:
        cnt = shape.get("contour")
        if cnt is None:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < _MIN_BOX_AREA:
            continue
        count = sum(1 for cx, cy in centroids if x <= cx <= x + w and y <= cy <= y + h)
        if count > density_threshold:
            dense.append({"x": x, "y": y, "w": w, "h": h, "label_count": count})
    return dense


def compute_brevity_visualization(
    image: np.ndarray,
    brevity_result: dict[str, Any],
    output_path: str | Path,
    shapes: list | None = None,
) -> dict[str, Any]:
    """Save an annotated copy of *image* to *output_path*.

    Draws a red outline around every top-level box shape whose label count
    exceeds the density threshold. Returns visualization_saved=False when there
    are no dense boxes to annotate.
    """
    score = brevity_result.get("brevity_quality_score")
    per_label_info = brevity_result.get("per_label_info", [])
    th = brevity_result.get("thresholds_used", {})
    density_threshold = th.get("density_threshold", 5.0)

    if score is None or not per_label_info:
        return {"visualization_saved": False, "output_path": None}

    dense_boxes = _find_dense_boxes(shapes or [], per_label_info, density_threshold)
    if not dense_boxes:
        return {"visualization_saved": False, "output_path": None}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image.copy()

    for box in dense_boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), _RED_BORDER, _RED_THICKNESS)
        badge = f"{box['label_count']} lbl"
        (tw, th_), bl = cv2.getTextSize(badge, _BADGE_FONT, _BADGE_SCALE, _BADGE_THICK)
        cv2.rectangle(vis, (x, y - th_ - bl - 2), (x + tw + 4, y), _RED_BORDER, -1)
        cv2.putText(vis, badge, (x + 2, y - bl - 1), _BADGE_FONT, _BADGE_SCALE,
                    (255, 255, 255), _BADGE_THICK, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)
    return {"visualization_saved": True, "output_path": str(output_path)}
