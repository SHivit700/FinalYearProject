#!/usr/bin/env python3
"""
Container utilization detection: 4-stage pipeline for underutilized (empty) containers.

Stage 1  – build candidates from enriched shape data.
Stage 2  – reject obvious false boxes (noise, border fragments, outer frames, duplicates).
Stage 3  – compute content signals per valid candidate (OCR, blank fraction, nesting, connectors).
Stage 4  – score empty_container_confidence, derive is_grouping_container and is_empty_container_final.

Returns image-level metrics, per-box debug records, and an annotated overlay PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .container_utilization_score import compute_container_utilization_score

# ---------------------------------------------------------------------------
# Centralised tunable thresholds
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, Any] = {
    # Stage 2 – candidate filtering
    "min_area_fraction": 0.002,       # box area / image area; below → rejected
    "min_dimension_px": 25,           # both w and h must exceed this
    "max_aspect_ratio": 15.0,          # max(w,h)/min(w,h); above → line-like
    "min_rectangularity": 0.65,       # contour_area / bbox_area; below → blob
    "max_image_fraction_outer": 0.80, # area_fraction above this → outer frame
    "outer_frame_border_fraction": 0.40, # touches_border + area > this → outer frame even if parent
    "border_margin_px": 3,            # px from edge counts as "touching border"
    "iou_duplicate_threshold": 0.85,  # IoU above this → near-duplicate; keep larger
    # Stage 3 – content signals
    "ocr_overlap_threshold": 0.50,    # fraction of OCR bbox inside box to count text
    "interior_shrink_px": 4,          # inset box boundary before sampling interior
    "blank_pixel_threshold": 240,     # grayscale ≥ this → blank (white / near-white)
    "interior_blank_high": 0.92,      # > this → very blank (+0.50 to confidence)
    "interior_blank_med": 0.75,       # > this → moderately blank (+0.25)
    "connector_fg_threshold": 100,    # grayscale < this → dark foreground pixel
    "connector_strip_margin": 5,      # px-wide strip outside bbox to sample connectors
    "connector_density_cutoff": 0.12, # fg fraction in strip above this → connector present
    # Stage 4 – decisions
    "empty_container_confidence_threshold": 0.55,
    "grouping_min_children": 2,       # ≥ this many valid children → grouping container
    # Reporting
    "large_box_area_fraction": 0.05,  # area_fraction ≥ this → "large" empty container
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rect_iou(r1: tuple, r2: tuple) -> float:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _ocr_overlap_fraction(label: dict, bx: int, by: int, bw: int, bh: int) -> float:
    """Fraction of the OCR label's bounding area that falls inside (bx,by,bw,bh)."""
    bbox = label["bbox"]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    lx1, lx2 = min(xs), max(xs)
    ly1, ly2 = min(ys), max(ys)
    label_area = max((lx2 - lx1) * (ly2 - ly1), 1e-6)
    ix = max(0.0, min(float(lx2), float(bx + bw)) - max(float(lx1), float(bx)))
    iy = max(0.0, min(float(ly2), float(by + bh)) - max(float(ly1), float(by)))
    return (ix * iy) / label_area


def _interior_blank_fraction(
    gray: np.ndarray,
    bx: int, by: int, bw: int, bh: int,
    shrink: int,
    blank_thr: int,
) -> float:
    """Fraction of interior pixels that are blank (≥ blank_thr in grayscale)."""
    x1 = max(0, bx + shrink)
    y1 = max(0, by + shrink)
    x2 = min(gray.shape[1], bx + bw - shrink)
    y2 = min(gray.shape[0], by + bh - shrink)
    if x2 <= x1 or y2 <= y1:
        return 1.0  # interior too small to measure; treat as blank
    crop = gray[y1:y2, x1:x2]
    return float(np.mean(crop >= blank_thr))


def _connector_density(
    gray: np.ndarray,
    bx: int, by: int, bw: int, bh: int,
    margin: int,
    fg_thr: int,
) -> float:
    """
    Maximum dark-pixel density across four thin strips just outside the bounding rect.
    High density → a line/arrow is approaching the box.
    """
    img_h, img_w = gray.shape
    strips = []

    def _sample(sy1, sy2, sx1, sx2):
        sy1 = max(0, sy1); sy2 = min(img_h, sy2)
        sx1 = max(0, sx1); sx2 = min(img_w, sx2)
        if sy2 > sy1 and sx2 > sx1:
            strips.append(float(np.mean(gray[sy1:sy2, sx1:sx2] < fg_thr)))

    _sample(by - margin, by, bx, bx + bw)           # top
    _sample(by + bh, by + bh + margin, bx, bx + bw) # bottom
    _sample(by, by + bh, bx - margin, bx)            # left
    _sample(by, by + bh, bx + bw, bx + bw + margin) # right

    return max(strips) if strips else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_container_utilization_metrics(
    bgr_image: np.ndarray,
    labels: list[dict],
    shapes: list[dict],
    image_shape: tuple,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Run the 4-stage container utilization (empty-container) detection pipeline.

    Parameters
    ----------
    bgr_image   : original BGR image (as returned by cv2.imread / detect_labels).
    labels      : OCR label list from run_label_detection.
    shapes      : enriched shape list from run_shape_detection (must include
                  x, y, w, h, aspect_ratio, rectangularity, touches_border,
                  contour_index, parent_contour_index fields).
    image_shape : (H, W, C) tuple.
    output_path : if given, write annotated overlay PNG here.

    Returns
    -------
    {
        "image_metrics": {...},   # scalar metrics for the quality model
        "box_details":  [...],    # per-box debug records
        "overlay_image": ndarray | None,
    }
    """
    T = THRESHOLDS
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    img_area = max(img_h * img_w, 1)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------------------------
    # Stage 1 – build candidate list from shapes
    # -----------------------------------------------------------------------
    candidates: list[dict] = []
    for s in shapes:
        # Prefer enriched fields; fall back to computing from contour.
        if "x" in s:
            x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        else:
            x, y, w, h = cv2.boundingRect(s["contour"])

        area = float(s["area"])
        aspect_ratio = s.get("aspect_ratio", max(w, h) / max(min(w, h), 1))
        rectangularity = s.get("rectangularity", area / max(w * h, 1))
        touches_border = s.get("touches_border", False)
        contour_index = s.get("contour_index", -1)
        parent_contour_index = s.get("parent_contour_index", -1)
        approx_vertex_count = s.get("approx_vertex_count", 4)

        candidates.append({
            "x": x, "y": y, "w": w, "h": h,
            "area": area,
            "area_fraction": area / img_area,
            "aspect_ratio": aspect_ratio,
            "rectangularity": rectangularity,
            "touches_border": touches_border,
            "approx_vertex_count": approx_vertex_count,
            "contour_index": contour_index,
            "parent_contour_index": parent_contour_index,
        })

    # -----------------------------------------------------------------------
    # Stage 2 – reject false-box candidates
    # -----------------------------------------------------------------------
    # Pre-compute which contour indices are parents of at least one other shape.
    # A box that has children clearly contains real content; it must not be
    # discarded due to position-based heuristics (border proximity, outer-frame
    # size) that were designed to catch empty noise rectangles.
    parent_ci: set[int] = {
        c["parent_contour_index"]
        for c in candidates
        if c["parent_contour_index"] >= 0
    }
    has_children: set[int] = {
        c["contour_index"]
        for c in candidates
        if c["contour_index"] in parent_ci
    }

    for c in candidates:
        reasons: list[str] = []
        is_parent = c["contour_index"] in has_children

        # Absolute noise filters — apply to every candidate regardless.
        if c["area_fraction"] < T["min_area_fraction"]:
            reasons.append("too_small_area")
        if c["w"] < T["min_dimension_px"] or c["h"] < T["min_dimension_px"]:
            reasons.append("too_thin")

        # Outer-frame catch: a box that both touches the border AND covers a large
        # fraction of the image is almost certainly the background rectangle, not a
        # real container — reject it even if it has children in the hierarchy.
        if c["touches_border"] and c["area_fraction"] > T["outer_frame_border_fraction"]:
            reasons.append("outer_frame_background")

        if not is_parent:
            # Heuristic filters — only applied to leaf boxes (no children).
            # Parent/container boxes can legitimately have low rectangularity due to
            # merged interior regions; those checks must not reject them.
            if c["aspect_ratio"] > T["max_aspect_ratio"]:
                reasons.append("extreme_aspect_ratio")
            if c["touches_border"]:
                reasons.append("touches_border")
            if c["rectangularity"] < T["min_rectangularity"]:
                reasons.append("poor_rectangularity")
            if c["area_fraction"] > T["max_image_fraction_outer"]:
                reasons.append("outer_frame")

        c["false_box_reasons"] = reasons
        c["is_false_box_rejected"] = bool(reasons)
        c["is_valid_box_candidate"] = not c["is_false_box_rejected"]

    # Remove near-duplicate boxes (IoU ≥ threshold); keep the larger one.
    valid_idx = [i for i, c in enumerate(candidates) if c["is_valid_box_candidate"]]
    to_remove: set[int] = set()
    for ii in range(len(valid_idx)):
        a = valid_idx[ii]
        if a in to_remove:
            continue
        for jj in range(ii + 1, len(valid_idx)):
            b = valid_idx[jj]
            if b in to_remove:
                continue
            ca, cb = candidates[a], candidates[b]
            iou = _rect_iou(
                (ca["x"], ca["y"], ca["w"], ca["h"]),
                (cb["x"], cb["y"], cb["w"], cb["h"]),
            )
            if iou >= T["iou_duplicate_threshold"]:
                discard = b if ca["area"] >= cb["area"] else a
                to_remove.add(discard)

    for idx in to_remove:
        candidates[idx]["is_valid_box_candidate"] = False
        candidates[idx]["is_false_box_rejected"] = True
        candidates[idx]["false_box_reasons"].append("near_duplicate")

    valid = [c for c in candidates if c["is_valid_box_candidate"]]

    # Build a lookup: contour_index → candidate (for child counting).
    ci_to_cand = {c["contour_index"]: c for c in candidates if c["contour_index"] >= 0}

    def _valid_children(parent: dict) -> list[dict]:
        """Valid candidates whose parent_contour_index == parent's contour_index."""
        ci = parent["contour_index"]
        if ci < 0:
            return []
        return [
            c for c in valid
            if c["parent_contour_index"] == ci
        ]

    # -----------------------------------------------------------------------
    # Stage 3 – content signals
    # -----------------------------------------------------------------------
    for c in valid:
        bx, by, bw, bh = c["x"], c["y"], c["w"], c["h"]

        # OCR text overlap
        ocr_inside = [
            lbl for lbl in labels
            if _ocr_overlap_fraction(lbl, bx, by, bw, bh) >= T["ocr_overlap_threshold"]
        ]
        c["ocr_text_count_inside"] = len(ocr_inside)
        ocr_area_sum = sum(lbl["width"] * lbl["height"] for lbl in ocr_inside)
        c["ocr_text_area_fraction_inside"] = ocr_area_sum / max(bw * bh, 1)

        # Interior blank fraction
        c["interior_blank_fraction"] = _interior_blank_fraction(
            gray, bx, by, bw, bh,
            shrink=T["interior_shrink_px"],
            blank_thr=T["blank_pixel_threshold"],
        )

        # Nested shapes from hierarchy
        children = _valid_children(c)
        c["num_internal_shapes"] = len(children)
        c["has_nested_box_inside"] = len(children) > 0

        # Internal lines: children that are elongated (likely compartment dividers)
        c["num_internal_lines"] = sum(1 for ch in children if ch["aspect_ratio"] > 4.0)

        # Connector heuristic: dark pixels in strips just outside the box boundary
        density = _connector_density(
            gray, bx, by, bw, bh,
            margin=T["connector_strip_margin"],
            fg_thr=T["connector_fg_threshold"],
        )
        c["has_connector_endpoint_near_or_inside"] = density > T["connector_density_cutoff"]
        c["max_connector_density"] = density

    # -----------------------------------------------------------------------
    # Stage 4 – empty_container_confidence and final decisions
    # -----------------------------------------------------------------------
    for c in valid:
        score = 0.0

        # Positive signals
        blank = c.get("interior_blank_fraction", 0.5)
        if blank >= T["interior_blank_high"]:
            score += 0.50
        elif blank >= T["interior_blank_med"]:
            score += 0.25

        ocr_count = c.get("ocr_text_count_inside", 0)
        if ocr_count == 0:
            score += 0.30
        elif c.get("ocr_text_area_fraction_inside", 0.0) < 0.05:
            score += 0.10   # negligible text coverage

        if c.get("rectangularity", 0.0) > 0.85:
            score += 0.10   # well-defined rectangle

        # Negative signals
        if c.get("has_nested_box_inside", False):
            score -= 0.35
        if c.get("num_internal_lines", 0) > 0:
            score -= 0.20
        if c.get("has_connector_endpoint_near_or_inside", False):
            score -= 0.15
        if ocr_count > 0:
            score -= 0.20 * min(1.0, ocr_count / 3.0)

        c["empty_container_confidence"] = max(0.0, min(1.0, score))

        # Grouping container: valid box enclosing multiple meaningful children
        c["is_grouping_container"] = (
            c.get("num_internal_shapes", 0) >= T["grouping_min_children"]
        )

        # Final empty decision
        c["is_empty_container_final"] = (
            c["empty_container_confidence"] >= T["empty_container_confidence_threshold"]
            and not c["is_grouping_container"]
        )

        # Borderline / uncertain flag
        lo = T["empty_container_confidence_threshold"] - 0.10
        hi = T["empty_container_confidence_threshold"] + 0.10
        c["is_borderline"] = (
            lo <= c["empty_container_confidence"] <= hi
            and not c["is_empty_container_final"]
        )

    # -----------------------------------------------------------------------
    # Image-level metrics
    # -----------------------------------------------------------------------
    all_rejected = [c for c in candidates if c["is_false_box_rejected"]]
    all_valid    = [c for c in candidates if c["is_valid_box_candidate"]]
    all_empty    = [c for c in all_valid  if c.get("is_empty_container_final", False)]
    all_group    = [c for c in all_valid  if c.get("is_grouping_container", False)]

    large_thr  = T["large_box_area_fraction"]
    large_empty = [c for c in all_empty if c["area_fraction"] >= large_thr]
    small_empty = [c for c in all_empty if c["area_fraction"] <  large_thr]

    image_metrics: dict[str, Any] = {
        "valid_box_count":          len(all_valid),
        "false_box_rejection_count": len(all_rejected),
        "grouping_container_count": len(all_group),
        "empty_container_count":          len(all_empty),
        "empty_container_ratio":          len(all_empty) / max(len(all_valid), 1),
        "empty_container_area_sum":       sum(c["area"] for c in all_empty),
        "empty_container_area_fraction":  sum(c["area_fraction"] for c in all_empty),
        "large_empty_container_count":    len(large_empty),
        "small_empty_container_count":    len(small_empty),
        "max_empty_container_confidence": max(
            (c["empty_container_confidence"] for c in all_valid), default=0.0
        ),
    }

    image_metrics["container_utilization_score"] = compute_container_utilization_score(
        valid_box_count=image_metrics["valid_box_count"],
        false_box_rejection_count=image_metrics["false_box_rejection_count"],
        grouping_container_count=image_metrics["grouping_container_count"],
        empty_container_count=image_metrics["empty_container_count"],
        empty_container_ratio=image_metrics["empty_container_ratio"],
        empty_container_area_fraction=image_metrics["empty_container_area_fraction"],
        max_empty_container_confidence=image_metrics["max_empty_container_confidence"],
    )

    # -----------------------------------------------------------------------
    # Per-box debug records
    # -----------------------------------------------------------------------
    box_details = []
    for c in candidates:
        box_details.append({
            # Geometry
            "x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"],
            "area": c["area"],
            "area_fraction": c["area_fraction"],
            "aspect_ratio": c["aspect_ratio"],
            "rectangularity": c.get("rectangularity", 0.0),
            "touches_border": c.get("touches_border", False),
            # Validity
            "is_valid_box_candidate":  c.get("is_valid_box_candidate", False),
            "is_false_box_rejected":   c.get("is_false_box_rejected", True),
            "false_box_reasons":       c.get("false_box_reasons", []),
            # Content signals
            "ocr_text_count_inside":            c.get("ocr_text_count_inside", 0),
            "ocr_text_area_fraction_inside":     c.get("ocr_text_area_fraction_inside", 0.0),
            "interior_blank_fraction":           c.get("interior_blank_fraction", 0.0),
            "num_internal_lines":                c.get("num_internal_lines", 0),
            "num_internal_shapes":               c.get("num_internal_shapes", 0),
            "has_nested_box_inside":             c.get("has_nested_box_inside", False),
            "has_connector_endpoint_near_or_inside": c.get("has_connector_endpoint_near_or_inside", False),
            "max_connector_density":             c.get("max_connector_density", 0.0),
            # Decision
            "empty_container_confidence": c.get("empty_container_confidence", 0.0),
            "is_grouping_container":  c.get("is_grouping_container", False),
            "is_empty_container_final": c.get("is_empty_container_final", False),
            "is_borderline":          c.get("is_borderline", False),
        })

    # -----------------------------------------------------------------------
    # Visual debug overlay
    # -----------------------------------------------------------------------
    overlay_image = None
    if output_path is not None:
        # BGR colours matching the spec
        C_VALID    = (0, 200, 0)     # green   – valid, non-empty
        C_EMPTY    = (0, 0, 220)     # red     – final empty (unutilized) container
        C_REJECTED = (0, 165, 255)   # orange  – rejected false box
        C_GROUP    = (255, 80, 0)    # blue    – grouping container
        C_BORDER   = (200, 0, 200)   # purple  – borderline / uncertain
        THICK = 2
        FONT  = cv2.FONT_HERSHEY_SIMPLEX
        FS    = 0.42
        FT    = 1

        overlay = bgr_image.copy()
        for c in candidates:
            bx, by, bw, bh = c["x"], c["y"], c["w"], c["h"]

            if c.get("is_false_box_rejected", True):
                color = C_REJECTED
                tag   = "FALSE_BOX"
            elif c.get("is_empty_container_final", False):
                color = C_EMPTY
                tag   = f"EMPTY {c['empty_container_confidence']:.2f}"
            elif c.get("is_grouping_container", False):
                color = C_GROUP
                tag   = "GROUP_CONTAINER"
            elif c.get("is_borderline", False):
                color = C_BORDER
                tag   = f"? {c['empty_container_confidence']:.2f}"
            else:
                color = C_VALID
                tag   = ""

            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, THICK)

            if tag:
                (tw, th), _ = cv2.getTextSize(tag, FONT, FS, FT)
                lx = bx
                ly = max(by - 4, th + 6)
                cv2.rectangle(overlay, (lx, ly - th - 4), (lx + tw + 4, ly + 2), color, -1)
                cv2.putText(
                    overlay, tag, (lx + 2, ly - 2),
                    FONT, FS, (255, 255, 255), FT, cv2.LINE_AA,
                )

        overlay_image = overlay
        out = Path(output_path).with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), overlay_image)

    return {
        "image_metrics": image_metrics,
        "box_details":   box_details,
        "overlay_image": overlay_image,
    }
