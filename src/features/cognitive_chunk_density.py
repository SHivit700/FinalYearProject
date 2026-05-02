"""Cognitive Chunk Density (CCD) — measures how many perceptual groups a
diagram's node layout creates, scored against Cowan's (2010) working memory
capacity range of 3–5 discrete chunks.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# Label clustering: labels within this fraction of canvas diagonal are merged
# into one structural node.
_CLUSTER_FRACTION = 0.07

_OPTIMAL_LOW_DEFAULT = 3
_OPTIMAL_HIGH_DEFAULT = 5
_PENALTY_PER_STEP_DEFAULT = 12.0
_EPS_FRACTION_DEFAULT = 0.15
_MIN_SAMPLES_DEFAULT = 2


# ---------------------------------------------------------------------------
# Node-position helpers
# ---------------------------------------------------------------------------

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


def _cluster_label_positions(
    positions: list[tuple[float, float]],
    threshold: float,
) -> list[tuple[float, float]]:
    """Greedy nearest-centroid clustering to merge per-box label positions."""
    clusters: list[list] = []  # [sum_x, sum_y, count]
    for x, y in sorted(positions):
        best_idx, best_dist = -1, float("inf")
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


def _node_positions_from_labels(
    labels: list[dict],
    canvas_width: float,
    canvas_height: float,
) -> list[tuple[float, float]]:
    """Derive structural node centres by clustering OCR label positions.

    Multiple labels belonging to the same diagram box are spatially close
    (within ~7% of the canvas diagonal) and collapse into one cluster.
    Labels from distinct boxes are further apart and form separate clusters.
    """
    all_centres = _label_centres(labels)
    if not all_centres:
        return []
    diagonal = math.hypot(canvas_width, canvas_height)
    threshold = max(60.0, _CLUSTER_FRACTION * diagonal)
    return _cluster_label_positions(all_centres, threshold)


# ---------------------------------------------------------------------------
# Core CCD computation
# ---------------------------------------------------------------------------

def compute_cognitive_chunk_density(
    node_boxes: list[tuple[float, float, float, float]],
    canvas_width: float,
    canvas_height: float,
    eps_fraction: float = _EPS_FRACTION_DEFAULT,
    min_samples: int = _MIN_SAMPLES_DEFAULT,
    optimal_low: int = _OPTIMAL_LOW_DEFAULT,
    optimal_high: int = _OPTIMAL_HIGH_DEFAULT,
    penalty_per_step: float = _PENALTY_PER_STEP_DEFAULT,
) -> dict[str, Any]:
    """
    Compute Cognitive Chunk Density (CCD) for a diagram's node layout.

    Clusters node centres spatially using DBSCAN to approximate the perceptual
    chunks a viewer would form when scanning the diagram. Scores highest when
    the chunk count falls within Cowan's working memory capacity range (3–5),
    and penalises both under-structured and over-complex layouts.

    Args:
        node_boxes:       List of (x, y, w, h) bounding boxes for each node.
        canvas_width:     Canvas width in pixels.
        canvas_height:    Canvas height in pixels.
        eps_fraction:     DBSCAN epsilon as a fraction of canvas diagonal.
        min_samples:      Minimum nodes to form a DBSCAN cluster.
        optimal_low:      Lower bound of optimal chunk count (inclusive).
        optimal_high:     Upper bound of optimal chunk count (inclusive).
        penalty_per_step: Score deducted per chunk outside optimal range.

    Returns:
        Dict with keys: cognitive_chunk_score, effective_chunks, cluster_count,
        singleton_count, eps_used, cluster_labels, low_confidence, degenerate_layout.
    """
    _null: dict[str, Any] = {
        "cognitive_chunk_score": None,
        "effective_chunks": None,
        "cluster_count": None,
        "singleton_count": None,
        "eps_used": None,
        "cluster_labels": None,
        "low_confidence": True,
        "degenerate_layout": False,
    }

    if len(node_boxes) < 3:
        logger.warning(
            "cognitive_chunk_density: only %d node(s) — need at least 3 for "
            "meaningful clustering.",
            len(node_boxes),
        )
        return _null

    centres = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in node_boxes]

    # Fall back to node bounding box as canvas if dimensions are unavailable.
    if canvas_width <= 0 or canvas_height <= 0:
        xs = [c[0] for c in centres]
        ys = [c[1] for c in centres]
        canvas_width = max(xs) - min(xs) or 1.0
        canvas_height = max(ys) - min(ys) or 1.0
        logger.warning(
            "cognitive_chunk_density: canvas dimensions unavailable — using "
            "node bounding box as canvas. Scores may be less reliable."
        )

    diagonal = math.hypot(canvas_width, canvas_height)
    eps = diagonal * eps_fraction

    centres_arr = np.array(centres, dtype=float)
    raw_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centres_arr)

    unique = set(raw_labels)
    cluster_count = len(unique) - (1 if -1 in unique else 0)
    singleton_count = int(np.sum(raw_labels == -1))
    effective_chunks = cluster_count + singleton_count

    if effective_chunks == 0:
        # All nodes somehow invisible to DBSCAN — shouldn't happen, but guard.
        return {**_null, "eps_used": round(eps, 2), "cluster_labels": raw_labels.tolist()}

    if optimal_low <= effective_chunks <= optimal_high:
        raw_score = 100.0
    elif effective_chunks < optimal_low:
        deficit = optimal_low - effective_chunks
        raw_score = max(0.0, 100.0 - deficit * penalty_per_step)
    else:
        excess = effective_chunks - optimal_high
        raw_score = max(0.0, 100.0 - excess * penalty_per_step)

    degenerate = singleton_count > 10

    return {
        "cognitive_chunk_score": round(raw_score, 2),
        "effective_chunks": effective_chunks,
        "cluster_count": cluster_count,
        "singleton_count": singleton_count,
        "eps_used": round(eps, 2),
        "cluster_labels": raw_labels.tolist(),
        "low_confidence": False,
        "degenerate_layout": degenerate,
    }


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

def compute_cognitive_chunk_density_from_diagram(
    labels: list[dict],
    shapes: list[dict],
    image_shape: tuple,
    eps_fraction: float = _EPS_FRACTION_DEFAULT,
    min_samples: int = _MIN_SAMPLES_DEFAULT,
    optimal_low: int = _OPTIMAL_LOW_DEFAULT,
    optimal_high: int = _OPTIMAL_HIGH_DEFAULT,
    penalty_per_step: float = _PENALTY_PER_STEP_DEFAULT,
) -> dict[str, Any]:
    """Convenience wrapper that derives node positions from OCR labels.

    Node positions are obtained by clustering label centres, which is more robust than shape detection
    across diagram styles. Each cluster centroid is treated as a zero-size
    node box (x, y, 0, 0) so that compute_cognitive_chunk_density receives
    one position per structural node.
    """
    h_img, w_img = image_shape[:2]
    canvas_width = float(w_img)
    canvas_height = float(h_img)

    node_positions = _node_positions_from_labels(labels, canvas_width, canvas_height)

    if not node_positions:
        logger.warning("cognitive_chunk_density: no node positions derived — returning null result.")
        return {
            "cognitive_chunk_score": None,
            "effective_chunks": None,
            "cluster_count": None,
            "singleton_count": None,
            "eps_used": None,
            "cluster_labels": None,
            "low_confidence": True,
            "degenerate_layout": False,
        }

    # Represent each node centre as a zero-size box so the core function
    # computes centre = (x + 0/2, y + 0/2) = (x, y) exactly.
    node_boxes = [(x, y, 0.0, 0.0) for x, y in node_positions]

    return compute_cognitive_chunk_density(
        node_boxes=node_boxes,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        eps_fraction=eps_fraction,
        min_samples=min_samples,
        optimal_low=optimal_low,
        optimal_high=optimal_high,
        penalty_per_step=penalty_per_step,
    )
