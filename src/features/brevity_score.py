"""Brevity Quality Score — label verbosity and box population density."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Per-diagram-type threshold tables ─────────────────────────────────────────
#
# Thresholds are calibrated so that:
#   - Well-labelled system-design diagrams (concise UML) score ~90–98
#   - Dense/verbose system-design diagrams score ~60–75
#   - Paragraph-heavy roadmap diagrams score proportionally lower
#
# Key calibration insight: p90 and mean penalties are *anchored* — labels
# below the threshold incur zero penalty, so a diagram whose longest labels
# still fall within normal UML conventions is not penalised.

_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "system_design": {
        "mean_threshold": 25.0,        # chars — mean below this → no mean penalty
        "mean_weight": 1.2,
        "mean_cap": 30.0,
        "p90_threshold": 40.0,         # chars — p90 below this → no p90 penalty
        "p90_weight": 0.15,
        "p90_cap": 15.0,
        "verbose_threshold_chars": 40,
        "verbose_penalty": 20.0,       # score loss = verbose_ratio × penalty
        "paragraph_threshold_words": 8,
        "paragraph_penalty": 20.0,
        "density_threshold": 5.0,      # labels/box — below this → no density penalty
        "density_weight": 10.0,
        "density_cap": 40.0,
    },
    "timeline_roadmap": {
        "mean_threshold": 35.0,
        "mean_weight": 1.0,
        "mean_cap": 25.0,
        "p90_threshold": 55.0,
        "p90_weight": 0.10,
        "p90_cap": 10.0,
        "verbose_threshold_chars": 55,
        "verbose_penalty": 15.0,
        "paragraph_threshold_words": 12,
        "paragraph_penalty": 15.0,
        "density_threshold": 6.0,
        "density_weight": 6.0,
        "density_cap": 25.0,
    },
}

_DEFAULT_DIAGRAM_TYPE = "system_design"


def _count_top_level_boxes(shapes: List[Dict[str, Any]]) -> int:
    """
    Count box-like contours at the top level of the RETR_TREE hierarchy.

    Top-level means parent_contour_index is -1 (no parent) or 0 (parent is
    the skipped outermost border contour). Filters for rectangular, non-thin
    shapes to exclude arrows and separator lines.
    """
    boxes = [
        s for s in shapes
        if s.get("parent_contour_index", -1) in (-1, 0)
        and s.get("rectangularity", 0.0) > 0.7
        and s.get("aspect_ratio", 999.0) < 8.0
    ]
    return len(boxes)


def compute_brevity_score(
    labels: List[Dict[str, Any]],
    shapes: List[Dict[str, Any]],
    diagram_type: str = "system_design",
) -> Dict[str, Any]:
    """
    Compute Brevity Quality Score (BQS) for a diagram's label set.

    Args:
        labels:       Pipeline label dicts (each must have a ``"text"`` key).
        shapes:       Shape dicts from ``run_shape_detection`` (used for
                      labels-per-box density calculation).
        diagram_type: ``"system_design"`` or ``"timeline_roadmap"``.

    Returns:
        Dict with keys: brevity_quality_score, mean_chars, median_chars,
        p90_chars, skewness, verbose_ratio, paragraph_ratio,
        labels_per_box, skipped_labels, low_confidence.
    """
    T = _THRESHOLDS.get(diagram_type)
    if T is None:
        logger.warning(
            "Unknown diagram_type %r; falling back to %r.", diagram_type, _DEFAULT_DIAGRAM_TYPE
        )
        T = _THRESHOLDS[_DEFAULT_DIAGRAM_TYPE]

    # ── 1. Extract and validate text strings ─────────────────────────────────
    valid_texts: List[str] = []
    skipped = 0
    for item in labels:
        if not isinstance(item, dict):
            skipped += 1
            continue
        text = item.get("text")
        if not isinstance(text, str):
            skipped += 1
            continue
        stripped = text.strip()
        if not stripped:
            skipped += 1
            continue
        valid_texts.append(stripped)

    if skipped:
        logger.debug("compute_brevity_score: skipped %d invalid/empty labels.", skipped)

    if not valid_texts:
        logger.warning("compute_brevity_score: no valid labels; returning None score.")
        return {
            "brevity_quality_score": None,
            "mean_chars": None,
            "median_chars": None,
            "p90_chars": None,
            "skewness": None,
            "verbose_ratio": None,
            "paragraph_ratio": None,
            "labels_per_box": None,
            "skipped_labels": skipped,
            "low_confidence": True,
        }

    low_confidence = len(valid_texts) == 1

    # ── 2. Character-count distribution ──────────────────────────────────────
    char_counts = np.array([len(t) for t in valid_texts], dtype=float)
    n = len(char_counts)

    mean_chars = float(np.mean(char_counts))
    median_chars = float(np.median(char_counts))
    p90_chars = float(np.percentile(char_counts, 90))

    std = float(np.std(char_counts))
    # Pearson's second skewness coefficient; 0 when std is zero
    skewness = (mean_chars - median_chars) / std if std > 0 else 0.0

    # ── 3. Verbose and paragraph ratios ──────────────────────────────────────
    verbose_ratio = float(np.mean(char_counts > T["verbose_threshold_chars"]))
    word_counts = np.array([len(t.split()) for t in valid_texts], dtype=float)
    paragraph_ratio = float(np.mean(word_counts > T["paragraph_threshold_words"]))

    # ── 4. Box density (labels per visible class/container box) ───────────────
    labels_per_box: Optional[float] = None
    if shapes:
        box_count = _count_top_level_boxes(shapes)
        if box_count == 0:
            # Fallback: assume ~3 contours per box (outer rect + 2 section dividers)
            box_count = max(1, len(shapes) // 3)
        labels_per_box = n / box_count

    # ── 5. Scoring formula (start at 100, apply anchored penalties) ───────────
    score = 100.0

    score -= min(T["mean_cap"], max(0.0, (mean_chars - T["mean_threshold"]) * T["mean_weight"]))
    score -= min(T["p90_cap"], max(0.0, (p90_chars - T["p90_threshold"]) * T["p90_weight"]))
    score -= verbose_ratio * T["verbose_penalty"]
    score -= paragraph_ratio * T["paragraph_penalty"]

    if labels_per_box is not None:
        score -= min(
            T["density_cap"],
            max(0.0, (labels_per_box - T["density_threshold"]) * T["density_weight"]),
        )

    score = max(0.0, score)

    return {
        "brevity_quality_score": round(score, 2),
        "mean_chars": round(mean_chars, 2),
        "median_chars": median_chars,
        "p90_chars": p90_chars,
        "skewness": round(skewness, 4),
        "verbose_ratio": round(verbose_ratio, 4),
        "paragraph_ratio": round(paragraph_ratio, 4),
        "labels_per_box": round(labels_per_box, 2) if labels_per_box is not None else None,
        "skipped_labels": skipped,
        "low_confidence": low_confidence,
    }
