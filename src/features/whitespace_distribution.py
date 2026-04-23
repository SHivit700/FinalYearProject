"""Whitespace Distribution Score (WDS) — measures how evenly whitespace is
spread across the diagram canvas using a grid-based pixel-density approach.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SENSITIVITY_DEFAULT = 2.0
_GRID_DEFAULT = 8

# Pixels whose grayscale value is below this are classified as "content"
# (dark borders, text). White/light fills (>220) are treated as background.
_CONTENT_THRESHOLD = 220

# Cells whose content fraction is below this are treated as empty.
# Prevents thin connector lines (which cover <2% of a cell) from artificially
# inflating density in otherwise empty inter-node whitespace regions.
_MIN_CELL_DENSITY = 0.02

# --- Node-based fallback constants (mirrors isolated_box_detection thresholds) ---
_MIN_BOX_WIDTH = 80
_MIN_BOX_HEIGHT = 40
_MAX_AREA_FRACTION = 0.40
_CONTAINMENT_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Pixel-based density grid (primary approach)
# ---------------------------------------------------------------------------

def _pixel_density_grid(
    bgr_image: np.ndarray,
    N: int,
    content_threshold: int = _CONTENT_THRESHOLD,
    min_cell_density: float = _MIN_CELL_DENSITY,
) -> np.ndarray:
    """Return an N×N density grid from the raw image.

    Each cell value is the fraction of pixels darker than content_threshold,
    zeroed out if below min_cell_density (so thin connector lines don't count).
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) if bgr_image.ndim == 3 else bgr_image
    h, w = gray.shape
    content = (gray < content_threshold).astype(np.float32)

    density_grid = np.zeros((N, N), dtype=float)
    for row in range(N):
        y1 = int(row * h / N)
        y2 = h if row == N - 1 else int((row + 1) * h / N)
        for col in range(N):
            x1 = int(col * w / N)
            x2 = w if col == N - 1 else int((col + 1) * w / N)
            raw = float(np.mean(content[y1:y2, x1:x2]))
            density_grid[row, col] = raw if raw >= min_cell_density else 0.0

    return density_grid


# ---------------------------------------------------------------------------
# Node-based density grid (fallback when no image is available)
# ---------------------------------------------------------------------------

def _intersection_area(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _structural_boxes(
    shapes: list[dict],
    canvas_width: float,
    canvas_height: float,
) -> list[tuple[float, float, float, float]]:
    canvas_area = canvas_width * canvas_height
    candidates = [
        s for s in shapes
        if s["w"] >= _MIN_BOX_WIDTH
        and s["h"] >= _MIN_BOX_HEIGHT
        and (s["w"] * s["h"]) < _MAX_AREA_FRACTION * canvas_area
    ]
    result = []
    for i, b1 in enumerate(candidates):
        area1 = float(b1["w"] * b1["h"])
        if area1 == 0:
            continue
        contained = False
        for j, b2 in enumerate(candidates):
            if i == j or float(b2["w"] * b2["h"]) <= area1:
                continue
            inter = _intersection_area(b1["x"], b1["y"], b1["w"], b1["h"],
                                        b2["x"], b2["y"], b2["w"], b2["h"])
            if inter / area1 >= _CONTAINMENT_THRESHOLD:
                contained = True
                break
        if not contained:
            result.append((float(b1["x"]), float(b1["y"]), float(b1["w"]), float(b1["h"])))
    return result


def _label_boxes(labels: list[dict]) -> list[tuple[float, float, float, float]]:
    boxes = []
    for lbl in labels:
        bbox = lbl.get("bbox", [])
        if not bbox:
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        if w > 0 and h > 0:
            boxes.append((float(x), float(y), float(w), float(h)))
    return boxes


def _node_density_grid(
    node_boxes: list[tuple[float, float, float, float]],
    canvas_width: float,
    canvas_height: float,
    N: int,
) -> np.ndarray:
    cell_w = canvas_width / N
    cell_h = canvas_height / N
    cell_area = cell_w * cell_h
    density_grid = np.zeros((N, N), dtype=float)
    for row in range(N):
        cy = row * cell_h
        for col in range(N):
            cx = col * cell_w
            total = sum(
                _intersection_area(cx, cy, cell_w, cell_h, nx, ny, nw, nh)
                for nx, ny, nw, nh in node_boxes
            )
            density_grid[row, col] = min(total, cell_area) / cell_area
    return density_grid


# ---------------------------------------------------------------------------
# Score computation (shared)
# ---------------------------------------------------------------------------

def compute_whitespace_distribution_score(
    density_grid: np.ndarray,
    sensitivity: float = _SENSITIVITY_DEFAULT,
) -> dict[str, Any]:
    """
    Compute WDS from a pre-computed N×N density grid.

    Args:
        density_grid: N×N numpy array where each value is the content density
                      of that grid cell (0 = empty, 1 = fully covered).
        sensitivity:  Controls penalty steepness. CoV ≥ sensitivity → score 0.

    Returns:
        Dict with keys: whitespace_distribution_score, density_cov, density_mean,
        density_std, empty_cell_ratio, grid_resolution, density_grid,
        low_confidence, degenerate_layout.
    """
    N = density_grid.shape[0]
    density_mean = float(np.mean(density_grid))
    density_std = float(np.std(density_grid))
    empty_cells = int(np.sum(density_grid == 0.0))
    empty_cell_ratio = empty_cells / (N * N)

    _null: dict[str, Any] = {
        "whitespace_distribution_score": None,
        "density_cov": None,
        "density_mean": round(density_mean, 4),
        "density_std": round(density_std, 4),
        "empty_cell_ratio": round(empty_cell_ratio, 4),
        "grid_resolution": N,
        "density_grid": density_grid.tolist(),
        "low_confidence": True,
        "degenerate_layout": False,
    }

    if density_mean == 0.0:
        logger.warning("whitespace_distribution: all grid cells are empty.")
        return _null

    density_cov = density_std / density_mean
    degenerate = int(np.count_nonzero(density_grid)) == 1
    score = max(0.0, 100.0 * (1.0 - density_cov / sensitivity))

    return {
        "whitespace_distribution_score": round(score, 2),
        "density_cov": round(density_cov, 4),
        "density_mean": round(density_mean, 4),
        "density_std": round(density_std, 4),
        "empty_cell_ratio": round(empty_cell_ratio, 4),
        "grid_resolution": N,
        "density_grid": density_grid.tolist(),
        "low_confidence": False,
        "degenerate_layout": degenerate,
    }


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

def compute_whitespace_distribution_from_diagram(
    labels: list[dict],
    shapes: list[dict],
    image_shape: tuple,
    bgr_image: np.ndarray | None = None,
    grid_resolution: int = _GRID_DEFAULT,
    sensitivity: float = _SENSITIVITY_DEFAULT,
) -> dict[str, Any]:
    """
    Compute WDS for a diagram.

    Primary path: pixel-based density from bgr_image (recommended — robust
    across image resolutions and diagram styles).
    Fallback path: node bounding-box coverage when no image is available.

    Args:
        labels:          OCR label detections from run_label_detection.
        shapes:          Shape detections from run_shape_detection.
        image_shape:     (height, width[, channels]) tuple from the image.
        bgr_image:       Raw BGR image array. When provided, pixel-based density
                         is used; otherwise falls back to node-based approach.
        grid_resolution: N for the N×N grid. Default 8.
        sensitivity:     Penalty steepness (CoV ≥ sensitivity → score 0). Default 1.5.
    """
    h_img, w_img = image_shape[:2]
    canvas_width = float(w_img)
    canvas_height = float(h_img)

    if bgr_image is not None:
        density_grid = _pixel_density_grid(bgr_image, grid_resolution)
    else:
        logger.warning("whitespace_distribution: no image provided — using node-based fallback.")
        node_boxes = _structural_boxes(shapes, canvas_width, canvas_height)
        if not node_boxes:
            logger.warning("whitespace_distribution: no structural boxes — falling back to labels.")
            node_boxes = _label_boxes(labels)
        if not node_boxes:
            return {
                "whitespace_distribution_score": None,
                "density_cov": None,
                "density_mean": None,
                "density_std": None,
                "empty_cell_ratio": None,
                "grid_resolution": grid_resolution,
                "density_grid": None,
                "low_confidence": True,
                "degenerate_layout": False,
            }
        density_grid = _node_density_grid(node_boxes, canvas_width, canvas_height, grid_resolution)

    return compute_whitespace_distribution_score(density_grid, sensitivity)
