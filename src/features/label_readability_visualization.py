"""Label Readability Visualization — highlights labels whose OCR confidence
falls below the readability threshold with a semi-transparent fill, polygon
outline, and a compact confidence badge. Only produced when at least one
label is below threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_COLOR_BGR = (0, 140, 255)    # orange (BGR)


def compute_label_readability_visualization(
    image: np.ndarray,
    readability_result: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Save an annotated copy of *image* to *output_path*.

    Returns immediately (visualization_saved=False) when no labels are below
    the readability threshold.
    """
    below_details = readability_result.get("labels_below_threshold_details", [])

    if not below_details:
        return {"visualization_saved": False, "output_path": None}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image.copy()

    for detail in below_details:
        bbox = detail.get("bbox") or []
        if not bbox:
            continue
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=_COLOR_BGR, thickness=2)

    cv2.imwrite(str(output_path), vis)
    return {"visualization_saved": True, "output_path": str(output_path)}
