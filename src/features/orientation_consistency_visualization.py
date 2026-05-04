"""Orientation Consistency Visualization — highlights labels whose detected
angle deviates from the dominant orientation with a semi-transparent fill and
a compact angle badge. Only produced when the score is strictly below 100.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_COLOR_BGR = (60, 20, 220)   # red (BGR)
_COLOR_FILL = (80, 40, 240)  # slightly brighter fill for semi-transparent pass
_FILL_ALPHA = 0.28


def compute_orientation_consistency_visualization(
    image: np.ndarray,
    oc_result: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Save an annotated copy of *image* to *output_path*.

    Returns immediately (visualization_saved=False) when the score is exactly
    100, None, or no inconsistent labels are found.
    """
    score = oc_result.get("orientation_consistency_score")
    per_label_info = oc_result.get("per_label_info", [])

    if score is None or score >= 100.0 or not per_label_info:
        return {"visualization_saved": False, "output_path": None}

    inconsistent = [lbl for lbl in per_label_info if not lbl["consistent"]]
    if not inconsistent:
        return {"visualization_saved": False, "output_path": None}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image.copy()

    # Semi-transparent fill pass
    overlay = vis.copy()
    for info in inconsistent:
        bbox = info.get("bbox", [])
        if bbox:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], _COLOR_FILL)
        else:
            x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _COLOR_FILL, -1)
    cv2.addWeighted(overlay, _FILL_ALPHA, vis, 1.0 - _FILL_ALPHA, 0, vis)

    # Solid outline pass
    for info in inconsistent:
        bbox = info.get("bbox", [])
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        if bbox:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=_COLOR_BGR, thickness=2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), _COLOR_BGR, 2)

    cv2.imwrite(str(output_path), vis)
    return {"visualization_saved": True, "output_path": str(output_path)}
