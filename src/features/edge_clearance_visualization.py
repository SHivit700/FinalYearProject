"""Edge Clearance Violation Visualization — draws an orange box and
"Edge Clearance Violation" label around any label or shape whose bounding
box is closer than the margin threshold to the image edge. Only produced
when at least one violation exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_COLOR_VIOLATION = (0, 140, 255)  # orange (BGR)


def compute_edge_clearance_visualization(
    image: np.ndarray,
    edge_clearance_result: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Save an annotated copy of *image* to *output_path*.

    Returns immediately (visualization_saved=False) when there are no
    violations or the per-item data is unavailable.
    """
    per_label_violations = edge_clearance_result.get("per_label_violations", [])
    per_shape_violations = edge_clearance_result.get("per_shape_violations", [])

    if not per_label_violations and not per_shape_violations:
        return {"visualization_saved": False, "output_path": None}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image.copy()

    for info in per_label_violations:
        bbox = info.get("bbox", [])
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

        if bbox:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=_COLOR_VIOLATION, thickness=2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), _COLOR_VIOLATION, 2)


    for info in per_shape_violations:
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), _COLOR_VIOLATION, 2)


    cv2.imwrite(str(output_path), vis)
    return {"visualization_saved": True, "output_path": str(output_path)}
