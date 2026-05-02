"""Label Contrast Quality Visualization — draws a red box and "Low Contrast
Detected" label around any label whose CIE Lab ΔL falls below the optimal
threshold. Only produced when the overall score is below 100.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_OPTIMAL_LOW = 40.0
_COLOR_LOW = (0, 0, 210)      # red (BGR)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def compute_label_contrast_visualization(
    image: np.ndarray,
    lcq_result: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Save an annotated copy of *image* to *output_path*.

    Returns immediately (visualization_saved=False) when the score is exactly
    100 or no per-label data is available.
    """
    score = lcq_result.get("label_contrast_score")
    per_label_info = lcq_result.get("per_label_info", [])

    if score is None or score >= 100.0 or not per_label_info:
        return {"visualization_saved": False, "output_path": None}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image.copy()

    for info in per_label_info:
        if info["delta_L"] >= _OPTIMAL_LOW:
            continue

        bbox = info.get("bbox", [])
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

        # Red polygon outline (or AABB rectangle if no polygon)
        if bbox:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=_COLOR_LOW, thickness=2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), _COLOR_LOW, 2)

        # "Low Contrast Detected" tag above the box
        tag = "Low Contrast Detected"
        (tw, th), _ = cv2.getTextSize(tag, _FONT, 0.40, 1)
        tx = x1
        ty = y1 - 5 if y1 > th + 5 else y2 + th + 5
        cv2.rectangle(vis, (tx - 1, ty - th - 2), (tx + tw + 2, ty + 2), (20, 20, 20), -1)
        cv2.putText(vis, tag, (tx, ty), _FONT, 0.40, _COLOR_LOW, 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)
    return {"visualization_saved": True, "output_path": str(output_path)}
