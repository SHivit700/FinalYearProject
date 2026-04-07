"""Compact text lines for label/shape detections"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

ImagePathLike = Union[str, Path]


def label_preview_line(label: Dict[str, Any]) -> str:
    bbox = np.asarray(label["bbox"], dtype=float).tolist()
    return f"Label: {label['text']} position: {bbox}"


def shape_preview_line(shape: Dict[str, Any]) -> str:
    c = np.asarray(shape["contour"]).reshape(-1, 2)
    n = int(c.shape[0])
    x0, y0 = int(c[:, 0].min()), int(c[:, 1].min())
    x1, y1 = int(c[:, 0].max()), int(c[:, 1].max())
    w, h = x1 - x0, y1 - y0
    return (
        f"type={shape['type']!r} area={shape['area']:.1f} "
        f"bbox_xywh=({x0},{y0},{w},{h}) "
        f"contour_points={n} approx_vertex_count={shape['approx_vertex_count']}"
    )


def build_detection_report(
    image_path: ImagePathLike,
    image_shape: Tuple[int, ...],
    labels: List[Dict[str, Any]],
    shapes: List[Dict[str, Any]],
    *,
    max_labels: int = 200,
    max_shapes: int = 150,
) -> str:
    p = Path(image_path).as_posix()
    lines: List[str] = [
        f"image_path: {p}",
        f"image_shape: {tuple(image_shape)}",
        "",
        f"Detected {len(labels)} label(s).",
    ]
    for label in labels[:max_labels]:
        lines.append(label_preview_line(label))
    if len(labels) > max_labels:
        lines.append(f"... ({len(labels) - max_labels} more labels omitted)")
    lines.append("")
    lines.append(f"Detected {len(shapes)} contour(s).")
    for i, shape in enumerate(shapes[:max_shapes]):
        lines.append(f"Shape #{i}: {shape_preview_line(shape)}")
    if len(shapes) > max_shapes:
        lines.append(f"... ({len(shapes) - max_shapes} more shapes omitted)")
    return "\n".join(lines)
