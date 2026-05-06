"""Label Overlap Visualization — highlights labels that physically overlap (IoU > 0)
or are too close (normalised gap below threshold) with a semi-transparent fill and
outline. A legend in the top-right corner explains the colour coding.
Only produced when problematic labels exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Orange for "too close" labels (BGR)
_COLOR_CLOSE_BGR = (0, 140, 255)
_COLOR_CLOSE_FILL = (30, 160, 255)

# Red for physically overlapping labels (BGR)
_COLOR_IOU_BGR = (30, 30, 220)
_COLOR_IOU_FILL = (60, 60, 240)

_FILL_ALPHA = 0.28
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LEGEND_FONT_SCALE = 0.55
_LEGEND_FONT_THICKNESS = 1
_LEGEND_PADDING = 12
_LEGEND_ROW_GAP = 8       # extra vertical gap between rows
_LEGEND_SWATCH = 18
_LEGEND_SWATCH_BORDER = 1 # thin border around each swatch


def _draw_legend(vis: np.ndarray) -> None:
    """Draw a colour-key legend in the top-right corner of *vis* (in-place)."""
    title = "Label overlap"
    entries = [
        (_COLOR_IOU_BGR,    "Overlap (IoU > 0)"),
        (_COLOR_CLOSE_BGR,  "Too close"),
    ]

    title_scale = _LEGEND_FONT_SCALE + 0.05
    title_thickness = 2

    (ttw, tth), t_base = cv2.getTextSize(title, _FONT, title_scale, title_thickness)

    entry_sizes = [
        cv2.getTextSize(lbl, _FONT, _LEGEND_FONT_SCALE, _LEGEND_FONT_THICKNESS)
        for _, lbl in entries
    ]
    max_text_w = max(sz[0][0] for sz in entry_sizes)
    row_text_h = max(sz[0][1] + sz[1] for sz in entry_sizes)
    row_h = max(row_text_h, _LEGEND_SWATCH) + _LEGEND_ROW_GAP

    panel_w = max(ttw, _LEGEND_SWATCH + _LEGEND_PADDING + max_text_w) + _LEGEND_PADDING * 2
    title_block_h = tth + t_base + _LEGEND_PADDING
    panel_h = title_block_h + row_h * len(entries) + _LEGEND_PADDING

    img_h, img_w = vis.shape[:2]
    px = img_w - panel_w - _LEGEND_PADDING
    py = _LEGEND_PADDING

    # Opaque dark background
    overlay = vis.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)

    # Outer border
    cv2.rectangle(vis, (px, py), (px + panel_w, py + panel_h), (160, 160, 160), 1)

    # Title
    cv2.putText(
        vis, title,
        (px + _LEGEND_PADDING, py + _LEGEND_PADDING + tth),
        _FONT, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA,
    )
    # Divider under title
    div_y = py + title_block_h
    cv2.line(vis, (px + 6, div_y), (px + panel_w - 6, div_y), (100, 100, 100), 1)

    for row, ((color, label), (sz, baseline)) in enumerate(zip(entries, entry_sizes)):
        row_top = div_y + _LEGEND_PADDING // 2 + row * row_h
        sx = px + _LEGEND_PADDING

        # Swatch with a subtle border
        cv2.rectangle(vis,
                      (sx, row_top),
                      (sx + _LEGEND_SWATCH, row_top + _LEGEND_SWATCH),
                      color, -1)
        cv2.rectangle(vis,
                      (sx, row_top),
                      (sx + _LEGEND_SWATCH, row_top + _LEGEND_SWATCH),
                      (200, 200, 200), _LEGEND_SWATCH_BORDER)

        # Label text, vertically centred on the swatch
        tx = sx + _LEGEND_SWATCH + _LEGEND_PADDING // 2
        ty = row_top + _LEGEND_SWATCH // 2 + sz[1] // 2
        cv2.putText(vis, label, (tx, ty), _FONT, _LEGEND_FONT_SCALE,
                    (230, 230, 230), _LEGEND_FONT_THICKNESS, cv2.LINE_AA)


def compute_label_overlap_visualization(
    image: np.ndarray,
    overlap_result: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Save an annotated copy of *image* to *output_path*.

    Returns immediately (visualization_saved=False) when no problematic labels
    are found.
    """
    per_label_info = overlap_result.get("per_label_info", [])

    if not per_label_info:
        return {"visualization_saved": False, "output_path": None}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image.copy()

    # Semi-transparent fill pass
    overlay = vis.copy()
    for info in per_label_info:
        fill_color = _COLOR_IOU_FILL if info["has_iou_overlap"] else _COLOR_CLOSE_FILL
        bbox = info.get("bbox", [])
        if bbox:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], fill_color)
        else:
            x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
    cv2.addWeighted(overlay, _FILL_ALPHA, vis, 1.0 - _FILL_ALPHA, 0, vis)

    # Solid outline pass (no per-box text)
    for info in per_label_info:
        outline_color = _COLOR_IOU_BGR if info["has_iou_overlap"] else _COLOR_CLOSE_BGR
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        bbox = info.get("bbox", [])
        if bbox:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=outline_color, thickness=2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), outline_color, 2)

    _draw_legend(vis)

    cv2.imwrite(str(output_path), vis)
    return {"visualization_saved": True, "output_path": str(output_path)}
