"""Symmetry Score (SS) — measures how balanced a diagram's node layout is
relative to the canvas centroid across horizontal and vertical axes."""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

SENSITIVITY_DEFAULT = 2.0
HORIZONTAL_WEIGHT_DEFAULT = 0.6
VERTICAL_WEIGHT_DEFAULT = 0.4
DOMINANT_AXIS_DEADBAND = 5.0
MIN_NODES = 3

# Structural box filters: size floor, and area ceiling to exclude whole-image
# frames and giant composite blobs that the shape detector sometimes returns.
# Label clustering: labels within this fraction of the canvas diagonal are merged
# into one structural node. Chosen so that labels within a single diagram box
# (which spans at most ~8% of diagonal) collapse to one point, while labels from
# distinct boxes (typically ≥10% of diagonal apart) remain separate.
_CLUSTER_FRACTION = 0.07

# Shape-based constants kept for _structural_boxes (used by containment check).
_MIN_BOX_WIDTH = 80
_MIN_BOX_HEIGHT = 40
_MAX_AREA_FRACTION = 0.40
_CONTAINMENT_THRESHOLD = 0.85


def _intersection_area(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _structural_boxes(
    shapes: list[dict],
    canvas_width: float,
    canvas_height: float,
) -> list[dict]:
    """Return only top-level structural boxes.

    1. Size filter: w ≥ _MIN_BOX_WIDTH, h ≥ _MIN_BOX_HEIGHT.
    2. Area filter: exclude shapes that cover ≥ _MAX_AREA_FRACTION of the canvas
       (these are image frames or giant composite blobs, not individual nodes).
    3. Containment filter: drop any box that is ≥ _CONTAINMENT_THRESHOLD fraction inside a larger box (inner UML class sections).
    """
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
            if i == j:
                continue
            area2 = float(b2["w"] * b2["h"])
            if area2 <= area1:
                continue
            inter = _intersection_area(
                b1["x"], b1["y"], b1["w"], b1["h"],
                b2["x"], b2["y"], b2["w"], b2["h"],
            )
            if inter / area1 >= _CONTAINMENT_THRESHOLD:
                contained = True
                break
        if not contained:
            result.append(b1)
    return result


def _label_centres(labels: list[dict]) -> list[tuple[float, float]]:
    centres = []
    for lbl in labels:
        bbox = lbl.get("bbox", [])
        if not bbox:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        centres.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    return centres


def _cluster_positions(
    positions: list[tuple[float, float]],
    threshold: float,
) -> list[tuple[float, float]]:
    """Greedy nearest-centroid clustering.

    Each position joins the closest existing cluster whose centroid is within
    `threshold` pixels; otherwise starts a new cluster. Sorting by x then y
    before clustering improves stability for left-to-right diagram layouts.
    """
    clusters: list[list[float, float, int]] = []  # [sum_x, sum_y, count]
    for x, y in sorted(positions):
        best_idx = -1
        best_dist = float("inf")
        for i, (sx, sy, n) in enumerate(clusters):
            d = math.hypot(x - sx / n, y - sy / n)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0 and best_dist <= threshold:
            c = clusters[best_idx]
            c[0] += x
            c[1] += y
            c[2] += 1
        else:
            clusters.append([x, y, 1])
    return [(sx / n, sy / n) for sx, sy, n in clusters]


def _centre_inside_shape(cx: float, cy: float, shapes: list[dict]) -> bool:
    for s in shapes:
        if s["x"] <= cx <= s["x"] + s["w"] and s["y"] <= cy <= s["y"] + s["h"]:
            return True
    return False


def _node_positions_from_inputs(
    labels: list[dict],
    shapes: list[dict],
    canvas_width: float,
    canvas_height: float,
) -> list[tuple[float, float]]:
    """Derive structural node positions by clustering label centres.

    Every structural element in a diagram has at least one text label (class name,
    node title, etc.). Labels that belong to the same box are spatially close and
    collapse into one cluster; labels from distinct boxes are separated by more
    than the threshold and form separate clusters.

    This approach is diagram-type agnostic and does not depend on shape detection
    correctly finding complete bounding boxes — which is unreliable across the
    range of diagram styles in the dataset (row-section detection, composite-region
    detection, JPEG compression artefacts, etc.).
    """
    all_centres = _label_centres(labels)
    if not all_centres:
        return []
    diagonal = math.hypot(canvas_width, canvas_height)
    threshold = max(60.0, _CLUSTER_FRACTION * diagonal)
    return _cluster_positions(all_centres, threshold)


def _clamp_positions(
    positions: list[tuple[float, float]],
    canvas_width: float,
    canvas_height: float,
) -> list[tuple[float, float]]:
    clamped = []
    n_out = 0
    for x, y in positions:
        cx = max(0.0, min(canvas_width, x))
        cy = max(0.0, min(canvas_height, y))
        if cx != x or cy != y:
            n_out += 1
        clamped.append((cx, cy))
    if n_out:
        logger.warning(
            "symmetry: %d node(s) were outside canvas bounds and have been clamped — "
            "layout may be malformed.",
            n_out,
        )
    return clamped


def _mean_reflection_displacement(
    positions: list[tuple[float, float]],
    centroid_x: float,
    centroid_y: float,
    axis: str,
) -> float:
    total = 0.0
    for x, y in positions:
        if axis == "horizontal":
            rx, ry = 2.0 * centroid_x - x, y
        else:
            rx, ry = x, 2.0 * centroid_y - y
        min_dist = min(math.hypot(rx - nx, ry - ny) for nx, ny in positions)
        total += min_dist
    return total / len(positions)


def _axis_score(
    mean_disp: float,
    canvas_diagonal: float,
    sensitivity: float,
) -> float:
    normalised = mean_disp / canvas_diagonal
    return max(0.0, 100.0 * (1.0 - normalised * sensitivity))


def compute_symmetry_score(
    node_positions: list[tuple[float, float]],
    canvas_width: float,
    canvas_height: float,
    sensitivity: float = SENSITIVITY_DEFAULT,
    horizontal_weight: float = HORIZONTAL_WEIGHT_DEFAULT,
    vertical_weight: float = VERTICAL_WEIGHT_DEFAULT,
) -> dict[str, Any]:
    """
    Compute Symmetry Score (SS) for a diagram's node layout.

    Args:
        node_positions:    List of (x, y) centre coordinates for each node.
        canvas_width:      Total width of the diagram canvas in pixels.
        canvas_height:     Total height of the diagram canvas in pixels.
        sensitivity:       Controls penalty steepness. Higher = harsher.
        horizontal_weight: Weight for left-right symmetry in composite score.
        vertical_weight:   Weight for top-bottom symmetry in composite score.

    Returns:
        Dict with keys: symmetry_score, horizontal_symmetry, vertical_symmetry,
        dominant_axis, node_count, low_confidence.
    """
    _null = {
        "symmetry_score": None,
        "horizontal_symmetry": None,
        "vertical_symmetry": None,
        "dominant_axis": None,
        "node_count": len(node_positions),
        "low_confidence": True,
    }

    if not node_positions:
        return _null

    if len(node_positions) < MIN_NODES:
        logger.warning(
            "symmetry: only %d node(s) — need at least %d for a meaningful score.",
            len(node_positions), MIN_NODES,
        )
        return {**_null, "node_count": len(node_positions)}

    positions = _clamp_positions(node_positions, canvas_width, canvas_height)

    cx = canvas_width / 2.0
    cy = canvas_height / 2.0
    diagonal = math.hypot(canvas_width, canvas_height)

    all_at_centroid = all(math.hypot(x - cx, y - cy) < 1e-6 for x, y in positions)
    low_confidence = all_at_centroid

    h_disp = _mean_reflection_displacement(positions, cx, cy, axis="horizontal")
    v_disp = _mean_reflection_displacement(positions, cx, cy, axis="vertical")

    h_score = _axis_score(h_disp, diagonal, sensitivity)
    v_score = _axis_score(v_disp, diagonal, sensitivity)

    composite = horizontal_weight * h_score + vertical_weight * v_score

    diff = h_score - v_score
    if abs(diff) < DOMINANT_AXIS_DEADBAND:
        dominant = "balanced"
    elif diff > 0:
        dominant = "horizontal"
    else:
        dominant = "vertical"

    return {
        "symmetry_score": round(composite, 2),
        "horizontal_symmetry": round(h_score, 2),
        "vertical_symmetry": round(v_score, 2),
        "dominant_axis": dominant,
        "node_count": len(positions),
        "low_confidence": low_confidence,
    }


def compute_symmetry_score_from_diagram(
    labels: list[dict],
    shapes: list[dict],
    image_shape: tuple,
    sensitivity: float = SENSITIVITY_DEFAULT,
    horizontal_weight: float = HORIZONTAL_WEIGHT_DEFAULT,
    vertical_weight: float = VERTICAL_WEIGHT_DEFAULT,
) -> dict[str, Any]:
    """Convenience wrapper that derives node positions from labels and shapes."""
    h_img, w_img = image_shape[:2]
    positions = _node_positions_from_inputs(
        labels, shapes, float(w_img), float(h_img)
    )
    return compute_symmetry_score(
        positions,
        canvas_width=float(w_img),
        canvas_height=float(h_img),
        sensitivity=sensitivity,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
    )
