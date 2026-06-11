"""
Suggestion Engine — rule-based + LLM synthesis for diagram quality analysis.

Public API:
    generate_suggestions(features, diagram_type, image_path, use_llm, session) -> dict
    format_text_report(suggestions_result) -> str
    new_session(diagram_type) -> dict
    load_session(path) -> dict
    save_session(session, path) -> None
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import threshold_manager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "ok": 2}

# ---------------------------------------------------------------------------
# 10×10 spatial region vocabulary (10% boundary grid + spanning aliases)
# Used by locate_issues_with_llm() to map LLM-returned names to % coordinates.
# Cells are named r{row}-c{col}, 1-indexed from the top-left corner.
# ---------------------------------------------------------------------------

_CELL_SIZE = 10.0

REGION_COORDS: dict[str, dict] = {
    # 100 individual cells
    **{
        f"r{r}-c{c}": {"x": (c - 1) * 10.0, "y": (r - 1) * 10.0, "width": 10.0, "height": 10.0}
        for r in range(1, 11)
        for c in range(1, 11)
    },
    # Row spanning aliases (full width, 10% tall each)
    **{f"row-{r}": {"x": 0.0, "y": (r - 1) * 10.0, "width": 100.0, "height": 10.0} for r in range(1, 11)},
    # Column spanning aliases (10% wide, full height each)
    **{f"col-{c}": {"x": (c - 1) * 10.0, "y": 0.0, "width": 10.0, "height": 100.0} for c in range(1, 11)},
    # Half-diagram aliases
    "top-half":    {"x": 0.0,  "y":  0.0, "width": 100.0, "height": 50.0},
    "bottom-half": {"x": 0.0,  "y": 50.0, "width": 100.0, "height": 50.0},
    "left-half":   {"x": 0.0,  "y":  0.0, "width": 50.0,  "height": 100.0},
    "right-half":  {"x": 50.0, "y":  0.0, "width": 50.0,  "height": 100.0},
    "full":        {"x": 0.0,  "y":  0.0, "width": 100.0, "height": 100.0},
}

_VALID_REGIONS = frozenset(REGION_COORDS.keys())


def _parse_regions(raw: list[str]) -> list[dict]:
    """Convert a list of raw LLM region name strings to coordinate dicts, dropping unknowns."""
    result = []
    for name in raw:
        key = name.strip().lower().replace(" ", "-").replace("_", "-")
        if key in _VALID_REGIONS:
            result.append({"region": key, **REGION_COORDS[key]})
    return result


_METRIC_DISPLAY_NAMES = {
    "label_readability": "Label Readability",
    "label_area": "Label Area Ratio",
    "overlap_metrics": "Label Overlap",
    "edge_clearance": "Edge Clearance",
    "font_hierarchy": "Font Hierarchy",
    "container_utilization": "Container Utilization",
    "isolated_boxes": "Isolated Boxes",
    "brevity": "Label Brevity",
    "whitespace_distribution": "Whitespace Distribution",
    "color_harmony": "Color Harmony",
    "label_contrast": "Label Contrast Quality",
    "cognitive_chunk_density": "Cognitive Chunk Density",
    "orientation_consistency": "Orientation Consistency",
}

_ACTIVE_METRICS = list(_METRIC_DISPLAY_NAMES.keys())

_SESSIONS_DIR = _PROJECT_ROOT / "src" / "data" / "sessions"


# ---------------------------------------------------------------------------
# Score extraction — derive a 0-100 value for every metric
# ---------------------------------------------------------------------------

def _get_score(features: dict, metric: str) -> float | None:
    """
    Extract or synthesise a 0-100 score for composite score computation.
    Returns None when the metric is unavailable or produced no result.
    """
    f = features.get(metric)
    if f is None:
        return None

    if metric == "label_readability":
        status = f.get("readability_status", "") if isinstance(f, dict) else ""
        if status == "high readability":
            return 100.0
        if status == "low readability":
            return 65.0
        return None

    if metric == "label_area":
        try:
            cat = f.category if hasattr(f, "category") else f.get("category")
        except Exception:
            return None
        if cat == "balanced":
            return 100.0
        if cat in ("sparse", "cluttered"):
            return 70.0
        return None

    if metric == "overlap_metrics":
        if not isinstance(f, dict):
            return None
        verdict = f.get("spacing_verdict", "")
        if verdict == "well_spaced":
            return 100.0
        if verdict == "moderate_overlap":
            return 70.0
        if verdict == "crowded labels":
            return 30.0
        return None

    if metric == "edge_clearance":
        if not isinstance(f, dict):
            return None
        lf = f.get("labels_fraction_violating") or 0.0
        sf = f.get("shapes_fraction_violating") or 0.0
        return max(0.0, 100.0 - max(lf, sf) * 200.0)

    if metric == "font_hierarchy":
        if not isinstance(f, dict):
            return None
        fsb = f.get("font_score_breakdown")
        return fsb.get("font_score") if fsb else None

    if metric == "container_utilization":
        if not isinstance(f, dict):
            return None
        return f.get("container_utilization_score")

    if metric == "isolated_boxes":
        if not isinstance(f, dict):
            return None
        return f.get("isolated_box_score")

    if metric == "brevity":
        if not isinstance(f, dict):
            return None
        return f.get("brevity_quality_score")

    if metric == "whitespace_distribution":
        if not isinstance(f, dict):
            return None
        return f.get("whitespace_distribution_score")

    if metric == "color_harmony":
        if not isinstance(f, dict):
            return None
        return f.get("color_harmony_score")

    if metric == "label_contrast":
        if not isinstance(f, dict):
            return None
        return f.get("label_contrast_score")

    if metric == "cognitive_chunk_density":
        if not isinstance(f, dict):
            return None
        return f.get("cognitive_chunk_score")

    if metric == "orientation_consistency":
        if not isinstance(f, dict):
            return None
        return f.get("orientation_consistency_score")

    return None


# ---------------------------------------------------------------------------
# Step 1 — Rule-based suggestion generation
# ---------------------------------------------------------------------------

def _extract_label(detail: str) -> str | None:
    """Extract label text from a location detail string like text='Foo' conf=0.45."""
    m = re.search(r"text=(['\"])(.+?)\1", detail)
    return m.group(2) if m else None


def _names_from_locs(locs: list[dict], limit: int = 5) -> list[str]:
    """Return unique element names extracted from location detail strings."""
    seen: list[str] = []
    for loc in locs:
        name = _extract_label(loc.get("detail", ""))
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= limit:
            break
    return seen


def _loc_label(loc: dict, img_shape: tuple | None = None) -> str:
    """Return a human-readable label with quadrant position for a single location."""
    name = _extract_label(loc.get("detail", ""))
    label_str = repr(name) if name else "element"
    if img_shape:
        H, W = img_shape[0], img_shape[1]
        if H > 0 and W > 0:
            cx_pct = ((loc.get("x1", 0) + loc.get("x2", 0)) / 2) / W * 100
            cy_pct = ((loc.get("y1", 0) + loc.get("y2", 0)) / 2) / H * 100
            h = "left" if cx_pct < 33 else ("center" if cx_pct < 66 else "right")
            v = "top" if cy_pct < 33 else ("middle" if cy_pct < 66 else "bottom")
            if h == "center" and v == "middle":
                quad = "center"
            elif h == "center":
                quad = f"{v} area"
            elif v == "middle":
                quad = f"{h} side"
            else:
                quad = f"{v}-{h}"
            return f"{label_str} ({quad})"
    return label_str


def _build_spatial_issue(base: str, locs: list[dict],
                         img_shape: tuple | None = None, limit: int = 5) -> str:
    """Append a spatially-qualified element list to a base issue string."""
    if not locs:
        return base
    parts: list[str] = []
    seen: set[str] = set()
    for loc in locs[:limit]:
        label = _loc_label(loc, img_shape)
        if label not in seen:
            parts.append(label)
            seen.add(label)
    return f"{base}: {', '.join(parts)}" if parts else base


def _find_parent_shape(cx: int, cy: int, shapes: list) -> dict | None:
    """Return the smallest box-like shape that contains point (cx, cy), or None."""
    containing = [
        s for s in shapes
        if s.get("x", 0) <= cx <= s.get("x", 0) + s.get("w", 0)
        and s.get("y", 0) <= cy <= s.get("y", 0) + s.get("h", 0)
        and s.get("w", 0) * s.get("h", 0) >= 4000
        and s.get("rectangularity", 0.0) > 0.5
        and s.get("aspect_ratio", 999.0) < 8.0
        and not s.get("touches_border", False)
    ]
    if not containing:
        return None
    return min(containing, key=lambda s: s["w"] * s["h"])


def _collect_brevity_candidates(verbose_labels: list[dict], shapes: list) -> list[dict]:
    """Map verbose labels to their parent shapes, deduplicating by shape."""
    seen: set[tuple] = set()
    candidates: list[dict] = []
    for p in verbose_labels:
        if not (p.get("x1") is not None and p["x1"] < p["x2"] and p["y1"] < p["y2"]):
            continue
        cx = (p["x1"] + p["x2"]) // 2
        cy = (p["y1"] + p["y2"]) // 2
        parent = _find_parent_shape(cx, cy, shapes)
        detail = f"text={p.get('text', '')!r} chars={p.get('char_count', '?')}"
        if parent:
            key = (parent["x"], parent["y"], parent["w"], parent["h"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "x": parent["x"], "y": parent["y"],
                "w": parent["w"], "h": parent["h"],
                "detail": detail,
            })
        else:
            candidates.append({
                "x": p["x1"], "y": p["y1"],
                "w": p["x2"] - p["x1"], "h": p["y2"] - p["y1"],
                "detail": detail,
            })
    return candidates


def _find_dense_boxes(shapes: list, per_label_info: list, density_threshold: float) -> list[dict]:
    """Return bounding rects of top-level box shapes containing more labels than the threshold."""
    box_shapes = [
        s for s in shapes
        if s.get("parent_contour_index", -1) in (-1, 0)
        and s.get("rectangularity", 0.0) > 0.7
        and s.get("aspect_ratio", 999.0) < 8.0
        and s.get("w", 0) * s.get("h", 0) >= 4000
    ]
    centroids = [
        ((lbl["x1"] + lbl["x2"]) // 2, (lbl["y1"] + lbl["y2"]) // 2)
        for lbl in per_label_info
        if lbl.get("x1") is not None
    ]
    dense = []
    for s in box_shapes:
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        count = sum(1 for cx, cy in centroids if x <= cx <= x + w and y <= cy <= y + h)
        if count > density_threshold:
            dense.append({"x": x, "y": y, "w": w, "h": h, "label_count": count})
    return dense


def generate_rule_based_suggestions(
    features: dict,
    diagram_type: str = "system_design",
    permanently_dismissed: list[str] | None = None,
    img_shape: tuple | None = None,
) -> list[dict]:
    """
    Analyse the features dict and return a list of Suggestion dicts sorted
    critical-first. Dismissed metrics are forced to severity "ok".
    """
    dismissed = set(permanently_dismissed or [])
    suggestions: list[dict] = []

    def _add(
        metric: str,
        severity: str,
        score: float | None,
        issue: str,
        locations: list[dict],
        recommendation: str,
    ) -> None:
        if metric in dismissed:
            severity = "ok"
        suggestions.append({
            "metric": metric,
            "severity": severity,
            "score": round(score, 2) if score is not None else None,
            "issue": issue,
            "locations": locations,
            "recommendation": recommendation,
        })

    # ── label_readability ─────────────────────────────────────────────────
    lr = features.get("label_readability")
    if isinstance(lr, dict):
        score = _get_score(features, "label_readability")
        status = lr.get("readability_status", "")
        if status == "low readability":
            locs: list[dict] = []
            for det in lr.get("labels_below_threshold_details", []):
                bbox = det.get("bbox", [])
                if bbox:
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    locs.append({
                        "x1": int(min(xs)), "y1": int(min(ys)),
                        "x2": int(max(xs)), "y2": int(max(ys)),
                        "detail": f"text={det.get('text', '')!r} conf={det.get('confidence', 0):.3f}",
                    })
            _lr_t = threshold_manager.get_thresholds("label_readability")
            _lr_crit = _lr_t.get("critical_threshold")
            _lr_warn = _lr_t.get("warning_threshold")
            if score is not None and _lr_crit is not None and score < _lr_crit:
                lr_sev = "critical"
            elif score is not None and _lr_warn is not None and score < _lr_warn:
                lr_sev = "warning"
            else:
                lr_sev = "warning"
            _lr_raw_issue = (
                f"Low OCR confidence on {lr.get('labels_below_threshold', 0)} label(s) "
                f"(fraction_below={lr.get('fraction_labels_below_threshold', 0):.2f}). "
                "Text may be too small, blurry, or have poor contrast."
            )
            _add(
                "label_readability", lr_sev, score,
                _build_spatial_issue(_lr_raw_issue, locs, img_shape),
                locs,
                "Increase font size, sharpen text rendering, or improve contrast between text and background.",
            )
        else:
            _add("label_readability", "ok", score, "Label readability is high.", [], "")

    # ── label_area ────────────────────────────────────────────────────────
    la = features.get("label_area")
    if la is not None:
        score = _get_score(features, "label_area")
        try:
            cat = la.category if hasattr(la, "category") else la.get("category", "balanced")
            ratio = la.ratio if hasattr(la, "ratio") else la.get("ratio", 0.0)
        except Exception:
            cat, ratio = "balanced", 0.0
        if cat in ("sparse", "cluttered"):
            _la_t = threshold_manager.get_thresholds("label_area")
            _la_crit = _la_t.get("critical_threshold")
            _la_warn = _la_t.get("warning_threshold")
            if score is not None and _la_crit is not None and score < _la_crit:
                la_sev = "critical"
            elif score is not None and _la_warn is not None and score < _la_warn:
                la_sev = "warning"
            else:
                la_sev = "warning"
            if img_shape is not None:
                _H, _W = img_shape[0], img_shape[1]
                la_locs: list[dict] = [{"x1": 0, "y1": 0, "x2": _W, "y2": _H, "detail": f"coverage is {cat}"}]
            else:
                la_locs = []
            _add(
                "label_area", la_sev, score,
                f"Label area ratio is {cat} (ratio={ratio:.3f}). "
                + ("Labels cover too little of the diagram — many areas lack context."
                   if cat == "sparse" else
                   "Labels cover too much of the diagram — the layout feels cramped."),
                la_locs,
                ("Add more labels or increase font size to improve coverage."
                 if cat == "sparse" else
                 "Reduce label density, shorten text, or increase diagram size."),
            )
        else:
            _add("label_area", "ok", score, "Label area ratio is balanced.", [], "")

    # ── overlap_metrics ───────────────────────────────────────────────────
    om = features.get("overlap_metrics")
    if isinstance(om, dict):
        score = _get_score(features, "overlap_metrics")
        verdict = om.get("spacing_verdict", "")
        fclose = om.get("fraction_labels_too_close", 0.0)
        any_iou = om.get("fraction_pairs_any_iou", 0.0)
        if verdict in ("crowded labels", "moderate_overlap"):
            _om_t = threshold_manager.get_thresholds("overlap_metrics")
            _om_crit = _om_t.get("critical_threshold")
            _om_warn = _om_t.get("warning_threshold")
            if score is not None and _om_crit is not None and score < _om_crit:
                om_sev = "critical"
            elif score is not None and _om_warn is not None and score < _om_warn:
                om_sev = "warning"
            else:
                om_sev = "warning"
            crowded_locs = om.get("per_label_info", [])
            if verdict == "crowded labels":
                _add(
                    "overlap_metrics", om_sev, score,
                    f"Labels are crowded: {fclose:.0%} of labels are too close to a neighbour "
                    f"({any_iou:.2%} of label pairs physically overlap).",
                    crowded_locs,
                    "Spread elements further apart, reduce label count, or increase diagram canvas size.",
                )
            else:
                _add(
                    "overlap_metrics", om_sev, score,
                    f"Moderate label crowding: {fclose:.0%} of labels are too close to a neighbour.",
                    crowded_locs,
                    "Review crowded areas and consider increasing spacing between nearby labels.",
                )
        else:
            _add("overlap_metrics", "ok", score, "Labels are well-spaced.", [], "")

    # ── edge_clearance ────────────────────────────────────────────────────
    ec = features.get("edge_clearance")
    if isinstance(ec, dict):
        score = _get_score(features, "edge_clearance")
        lf = ec.get("labels_fraction_violating") or 0.0
        sf = ec.get("shapes_fraction_violating") or 0.0
        if lf > 0 or sf > 0:
            parts: list[str] = []
            if lf > 0:
                parts.append(f"{lf:.0%} of labels")
            if sf > 0:
                parts.append(f"{sf:.0%} of shapes")
            # Visualise the margin zone as a border frame (4 thin strips along each
            # edge) rather than the full bounding boxes of violating elements.
            # Full bounding boxes of large shapes span most of the canvas and appear
            # as a highlight "in the middle of the screen", which is confusing.
            ec_locs: list[dict] = []
            if img_shape:
                H_img, W_img = img_shape[0], img_shape[1]
                mf = ec.get("margin_fraction") or 0.05
                margin_px = max(1, int(mf * min(W_img, H_img)))
                ec_locs = [
                    {"x1": 0, "y1": 0,
                     "x2": W_img, "y2": margin_px,
                     "detail": "top border margin — elements must stay below this zone"},
                    {"x1": 0, "y1": H_img - margin_px,
                     "x2": W_img, "y2": H_img,
                     "detail": "bottom border margin — elements must stay above this zone"},
                    {"x1": 0, "y1": 0,
                     "x2": margin_px, "y2": H_img,
                     "detail": "left border margin — elements must stay right of this zone"},
                    {"x1": W_img - margin_px, "y1": 0,
                     "x2": W_img, "y2": H_img,
                     "detail": "right border margin — elements must stay left of this zone"},
                ]
            _ec_t = threshold_manager.get_thresholds("edge_clearance")
            _ec_crit = _ec_t.get("critical_threshold")
            _ec_warn = _ec_t.get("warning_threshold")
            if score is not None and _ec_crit is not None and score < _ec_crit:
                ec_sev = "critical"
            elif score is not None and _ec_warn is not None and score < _ec_warn:
                ec_sev = "warning"
            else:
                ec_sev = "warning"
            _add(
                "edge_clearance", ec_sev, score,
                f"{' and '.join(parts)} are too close to the diagram boundary, risking clipping.",
                ec_locs,
                "Move elements inward to maintain a clear margin around all diagram edges.",
            )
        else:
            _add("edge_clearance", "ok", score, "All elements respect edge clearance margins.", [], "")

    # ── font_hierarchy ────────────────────────────────────────────────────
    fh = features.get("font_hierarchy")
    if isinstance(fh, dict):
        fsb = fh.get("font_score_breakdown")
        font_score = fsb.get("font_score") if fsb else None
        _fh_t = threshold_manager.get_thresholds("font_hierarchy")
        _fh_warn = _fh_t.get("warning_threshold")
        if font_score is not None and _fh_warn is not None and font_score < _fh_warn:
            _add(
                "font_hierarchy", "warning", font_score,
                f"Font hierarchy score is {font_score:.1f}/100 — insufficient size contrast between label levels.",
                [],
                "Use distinctly different font sizes for titles, section headers, and body text.",
            )
        else:
            _add("font_hierarchy", "ok", font_score, "Font hierarchy is adequate.", [], "")

    # ── container_utilization ─────────────────────────────────────────────
    cu = features.get("container_utilization")
    if isinstance(cu, dict):
        cu_score = cu.get("container_utilization_score")
        if cu_score is not None:
            cu_locs = [
                {
                    "x1": b["x"], "y1": b["y"],
                    "x2": b["x"] + b["w"], "y2": b["y"] + b["h"],
                    "detail": "empty container — under-utilized",
                }
                for b in cu.get("empty_container_boxes", [])
            ]
            _cu_t = threshold_manager.get_thresholds("container_utilization")
            _cu_crit = _cu_t.get("critical_threshold")
            _cu_warn = _cu_t.get("warning_threshold")
            if _cu_crit is not None and cu_score < _cu_crit:
                _add(
                    "container_utilization", "critical", cu_score,
                    _build_spatial_issue(
                        f"Container utilization is very low ({cu_score:.1f}/100). Boxes are significantly under-used.",
                        cu_locs, img_shape,
                    ),
                    cu_locs,
                    "Add more content to containers, merge near-empty boxes, or remove unused containers.",
                )
            elif _cu_warn is not None and cu_score < _cu_warn:
                _add(
                    "container_utilization", "warning", cu_score,
                    _build_spatial_issue(
                        f"Container utilization is below optimal ({cu_score:.1f}/100). Some regions are sparse.",
                        cu_locs, img_shape,
                    ),
                    cu_locs,
                    "Balance content across containers or consolidate sparse sections.",
                )
            else:
                _add("container_utilization", "ok", cu_score, "Container utilization is good.", [], "")
        else:
            _add("container_utilization", "ok", None, "Container utilization could not be measured.", [], "")

    # ── isolated_boxes ────────────────────────────────────────────────────
    ib = features.get("isolated_boxes")
    if isinstance(ib, dict):
        ib_score = ib.get("isolated_box_score")
        island_count = ib.get("island_count", 0)
        if ib_score is not None:
            img_area = (img_shape[0] * img_shape[1]) if img_shape else None
            locs = [
                {
                    "x1": b["x"], "y1": b["y"],
                    "x2": b["x"] + b["w"], "y2": b["y"] + b["h"],
                    "detail": "isolated box — no connector lines detected",
                }
                for b in ib.get("island_boxes", [])
                if not (img_area and (b["w"] * b["h"]) > 0.15 * img_area)
            ]
            _ib_t = threshold_manager.get_thresholds("isolated_boxes")
            _ib_crit = _ib_t.get("critical_threshold")
            _ib_warn = _ib_t.get("warning_threshold")
            if _ib_crit is not None and ib_score < _ib_crit:
                _add(
                    "isolated_boxes", "critical", ib_score,
                    _build_spatial_issue(
                        f"{island_count} box(es) have no connector lines (score={ib_score}/100). "
                        "Isolated nodes break the diagram's structural flow.",
                        locs, img_shape,
                    ),
                    locs,
                    "Draw connector arrows or lines between isolated elements and the rest of the diagram.",
                )
            elif _ib_warn is not None and ib_score < _ib_warn:
                _add(
                    "isolated_boxes", "warning", ib_score,
                    _build_spatial_issue(
                        f"{island_count} box(es) appear disconnected from the rest of the diagram.",
                        locs, img_shape,
                    ),
                    locs,
                    "Connect isolated boxes with appropriate arrows or relationship lines.",
                )
            else:
                _add("isolated_boxes", "ok", ib_score, "All major boxes are connected.", [], "")
        else:
            _add("isolated_boxes", "ok", None, "Isolated box analysis yielded no result.", [], "")

    # ── brevity ───────────────────────────────────────────────────────────
    brev = features.get("brevity")
    if isinstance(brev, dict):
        brev_score = brev.get("brevity_quality_score")
        if brev_score is not None:
            all_label_info = brev.get("per_label_info", [])
            verbose_labels = [p for p in all_label_info if p.get("violates_brevity")]
            _brev_shapes = features.get("shapes", [])
            _brev_th = brev.get("thresholds_used", {})

            # Source 1: verbose individual labels → their parent shapes.
            confirmed = brev.get("confirmed_verbose_shapes")
            if confirmed is not None:
                verbose_source = confirmed
            else:
                verbose_source = _collect_brevity_candidates(verbose_labels, _brev_shapes)

            # Source 2: boxes that are too dense (too many labels crammed in).
            density_threshold = _brev_th.get("density_threshold", 5.0)
            dense_boxes = _find_dense_boxes(_brev_shapes, all_label_info, density_threshold)

            # Merge both sources, deduplicating by bounding rect.
            seen_rects: set[tuple] = set()
            locs: list[dict] = []
            for s in verbose_source:
                key = (s["x"], s["y"], s["w"], s["h"])
                if key not in seen_rects:
                    seen_rects.add(key)
                    locs.append({
                        "x1": s["x"], "y1": s["y"],
                        "x2": s["x"] + s["w"], "y2": s["y"] + s["h"],
                        "detail": s.get("detail", ""),
                    })
            for b in dense_boxes:
                key = (b["x"], b["y"], b["w"], b["h"])
                if key not in seen_rects:
                    seen_rects.add(key)
                    locs.append({
                        "x1": b["x"], "y1": b["y"],
                        "x2": b["x"] + b["w"], "y2": b["y"] + b["h"],
                        "detail": f"{b['label_count']} labels in box (density issue)",
                    })

            _brev_t = threshold_manager.get_thresholds("brevity")
            _brev_crit = _brev_t.get("critical_threshold")
            _brev_warn = _brev_t.get("warning_threshold")
            n_verbose = len(verbose_labels)
            n_dense = len(dense_boxes)
            if _brev_crit is not None and brev_score < _brev_crit:
                parts = []
                if n_verbose:
                    parts.append(f"{n_verbose} verbose label(s)")
                if n_dense:
                    parts.append(f"{n_dense} overcrowded box(es)")
                detail_str = f" ({', '.join(parts)} detected)" if parts else ""
                _add(
                    "brevity", "critical", brev_score,
                    _build_spatial_issue(
                        f"Many labels are excessively long or dense "
                        f"(score={brev_score:.1f}/100{detail_str}).",
                        locs, img_shape,
                    ),
                    locs,
                    "Shorten labels to concise identifiers. Move detail to tooltips or a legend.",
                )
            elif _brev_warn is not None and brev_score < _brev_warn:
                parts = []
                if n_verbose:
                    parts.append(f"{n_verbose} verbose label(s)")
                if n_dense:
                    parts.append(f"{n_dense} overcrowded box(es)")
                detail_str = f" ({', '.join(parts)} detected)" if parts else ""
                _add(
                    "brevity", "warning", brev_score,
                    _build_spatial_issue(
                        f"Some labels exceed recommended length or density "
                        f"(score={brev_score:.1f}/100{detail_str}).",
                        locs, img_shape,
                    ),
                    locs,
                    "Trim verbose labels. Aim for ≤40 characters and ≤5 labels per box.",
                )
            else:
                _add("brevity", "ok", brev_score, "Labels are concise.", [], "")
        else:
            _add("brevity", "ok", None, "Brevity score unavailable (no valid labels).", [], "")

    # ── whitespace_distribution ───────────────────────────────────────────
    wds = features.get("whitespace_distribution")
    if isinstance(wds, dict):
        wds_score = wds.get("whitespace_distribution_score")
        if wds_score is not None:
            cov = wds.get("density_cov", 0.0)
            _wds_t = threshold_manager.get_thresholds("whitespace_distribution")
            _wds_crit = _wds_t.get("critical_threshold")
            _wds_warn = _wds_t.get("warning_threshold")
            if _wds_crit is not None and wds_score < _wds_crit:
                _add(
                    "whitespace_distribution", "critical", wds_score,
                    f"Whitespace is very unevenly distributed (score={wds_score:.1f}/100, CoV={cov:.2f}). "
                    "Dense clusters sit next to empty regions.",
                    [],
                    "Redistribute elements to create consistent spacing across the diagram.",
                )
            elif _wds_warn is not None and wds_score < _wds_warn:
                _add(
                    "whitespace_distribution", "warning", wds_score,
                    f"Whitespace distribution is uneven (score={wds_score:.1f}/100, CoV={cov:.2f}).",
                    [],
                    "Spread elements more evenly to improve visual balance.",
                )
            else:
                _add("whitespace_distribution", "ok", wds_score, "Whitespace is well-distributed.", [], "")
        else:
            _add("whitespace_distribution", "ok", None, "Whitespace score unavailable.", [], "")

    # ── color_harmony ─────────────────────────────────────────────────────
    chs = features.get("color_harmony")
    if isinstance(chs, dict):
        ch_score = chs.get("color_harmony_score")
        if ch_score is not None:
            palette_size = chs.get("palette_size", 0)
            palette = chs.get("dominant_colors_hex", [])
            detail = f"palette_size={palette_size}, colors={palette[:5]}"
            _ch_t = threshold_manager.get_thresholds("color_harmony")
            _ch_crit = _ch_t.get("critical_threshold")
            _ch_warn = _ch_t.get("warning_threshold")
            if _ch_crit is not None and ch_score < _ch_crit:
                _add(
                    "color_harmony", "critical", ch_score,
                    f"Color harmony is poor (score={ch_score:.1f}/100, {detail}).",
                    [],
                    "Reduce distinct colors and choose a harmonious palette (analogous or complementary scheme).",
                )
                suggestions[-1]["palette_colors"] = palette
            elif _ch_warn is not None and ch_score < _ch_warn:
                _add(
                    "color_harmony", "warning", ch_score,
                    f"Color harmony could be improved (score={ch_score:.1f}/100, {detail}).",
                    [],
                    "Limit palette to 3-5 well-chosen colors and apply them consistently.",
                )
                suggestions[-1]["palette_colors"] = palette
            else:
                _add("color_harmony", "ok", ch_score, "Color harmony is good.", [], "")
                suggestions[-1]["palette_colors"] = palette
        else:
            _add("color_harmony", "ok", None, "Color harmony score unavailable.", [], "")

    # ── label_contrast ────────────────────────────────────────────────────
    lcq = features.get("label_contrast")
    if isinstance(lcq, dict):
        lc_score = lcq.get("label_contrast_score")
        if lc_score is not None:
            low_contrast = [
                p for p in lcq.get("per_label_info", [])
                if p.get("delta_L", 50.0) < 40.0
            ]
            locs = [
                {
                    "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
                    "detail": (
                        f"text={p.get('text', '')!r} ΔL={p.get('delta_L', 0):.1f}"
                        if p.get("text") else f"ΔL={p.get('delta_L', 0):.1f}"
                    ),
                }
                for p in low_contrast
                if p.get("x1") is not None
            ]
            _lc_t = threshold_manager.get_thresholds("label_contrast")
            _lc_warn = _lc_t.get("warning_threshold")
            if _lc_warn is not None and lc_score < _lc_warn:
                _add(
                    "label_contrast", "warning", lc_score,
                    _build_spatial_issue(
                        f"Label contrast score is {lc_score:.1f}/100. "
                        f"{len(low_contrast)} label(s) have ΔL outside the optimal 40–80 range "
                        f"(mean_ΔL={lcq.get('mean_delta_L', 0):.1f}).",
                        locs, img_shape,
                    ),
                    locs,
                    "Adjust text or background color so ΔL (CIE Lab luminance difference) is between 40 and 80 for all labels.",
                )
            else:
                _add("label_contrast", "ok", lc_score, "Label contrast is within the optimal range.", [], "")
        else:
            _add("label_contrast", "ok", None, "Label contrast score unavailable.", [], "")

    # ── cognitive_chunk_density ───────────────────────────────────────────
    ccd = features.get("cognitive_chunk_density")
    if isinstance(ccd, dict):
        cc_score = ccd.get("cognitive_chunk_score")
        if cc_score is not None:
            chunks = ccd.get("effective_chunks", "?")
            _cc_t = threshold_manager.get_thresholds("cognitive_chunk_density")
            _cc_crit = _cc_t.get("critical_threshold")
            _cc_warn = _cc_t.get("warning_threshold")
            if _cc_crit is not None and cc_score < _cc_crit:
                _add(
                    "cognitive_chunk_density", "critical", cc_score,
                    f"Cognitive chunk density is very high (score={cc_score:.1f}/100, effective_chunks={chunks}). "
                    "Viewers may struggle to parse so many distinct visual groups.",
                    [],
                    "Break the diagram into sub-diagrams, group related elements, or reduce total element count.",
                )
            elif _cc_warn is not None and cc_score < _cc_warn:
                _add(
                    "cognitive_chunk_density", "warning", cc_score,
                    f"Cognitive load is elevated (score={cc_score:.1f}/100, effective_chunks={chunks}).",
                    [],
                    "Consider grouping closely related elements into fewer visual clusters.",
                )
            else:
                _add("cognitive_chunk_density", "ok", cc_score, "Cognitive chunk density is manageable.", [], "")
        else:
            _add("cognitive_chunk_density", "ok", None, "Cognitive chunk score unavailable.", [], "")
        # Attach centroids for frontend badge rendering (safe for all branches above).
        if suggestions and suggestions[-1]["metric"] == "cognitive_chunk_density":
            suggestions[-1]["chunk_centroids"] = ccd.get("chunk_centroids", [])

    # ── orientation_consistency ────────────────────────────────────────────
    oc = features.get("orientation_consistency")
    if isinstance(oc, dict):
        oc_score = oc.get("orientation_consistency_score")
        if oc_score is not None:
            dom_deg = oc.get("dominant_orientation_deg", 0.0)
            frac = oc.get("consistent_label_fraction", 0.0)
            oc_locs: list[dict] = []
            for lbl in oc.get("per_label_info", []):
                if not lbl.get("consistent", True):
                    oc_locs.append({
                        "x1": int(lbl["x1"]),
                        "y1": int(lbl["y1"]),
                        "x2": int(lbl["x2"]),
                        "y2": int(lbl["y2"]),
                        "detail": f"text={lbl.get('text', '')!r} angle={lbl.get('angle_deg', 0):.1f}°",
                    })
            _oc_t = threshold_manager.get_thresholds("orientation_consistency")
            _oc_crit = _oc_t.get("critical_threshold")
            _oc_warn = _oc_t.get("warning_threshold")
            if _oc_crit is not None and oc_score < _oc_crit:
                _add(
                    "orientation_consistency", "critical", oc_score,
                    f"Label orientation is highly inconsistent (score={oc_score:.1f}/100). "
                    f"Only {frac:.0%} of labels follow the dominant orientation ({dom_deg:.1f}°).",
                    oc_locs,
                    "Align all labels to a single orientation (preferably 0° horizontal). Avoid mixing horizontal and rotated text.",
                )
            elif _oc_warn is not None and oc_score < _oc_warn:
                _add(
                    "orientation_consistency", "warning", oc_score,
                    f"Some labels deviate from the dominant orientation (score={oc_score:.1f}/100, "
                    f"dominant={dom_deg:.1f}°, consistent={frac:.0%}).",
                    oc_locs,
                    "Standardise label orientation throughout the diagram.",
                )
            else:
                _add("orientation_consistency", "ok", oc_score, "Label orientation is consistent.", [], "")
        else:
            _add("orientation_consistency", "ok", None, "Orientation consistency score unavailable.", [], "")

    suggestions.sort(key=lambda s: _SEVERITY_ORDER.get(s["severity"], 3))
    return suggestions


# ---------------------------------------------------------------------------
# Composite score calculation
# ---------------------------------------------------------------------------

def _compute_composite_score(
    features: dict,
    permanently_dismissed: list[str],
) -> float | None:
    """Single composite score; dismissed metrics excluded from the average."""
    dismissed = set(permanently_dismissed)
    scores: list[float] = []
    for metric in _ACTIVE_METRICS:
        if metric in dismissed:
            continue
        score = _get_score(features, metric)
        if score is None:
            continue
        scores.append(score)
    return round(sum(scores) / len(scores), 2) if scores else None


# ---------------------------------------------------------------------------
# Step 2 — Session helpers
# ---------------------------------------------------------------------------

def new_session(diagram_type: str = "system_design") -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "session_id": f"session_{ts}",
        "diagram_type": diagram_type,
        "permanently_dismissed": [],
        "diagram_versions": [],
        "chat_history": [],
    }


def load_session(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, default=str)


def _default_session_path(image_path: str | None) -> str:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(image_path).stem if image_path else "diagram"
    return str(_SESSIONS_DIR / f"{stem}_{ts}.json")


# ---------------------------------------------------------------------------
# Step 3 — LLM synthesis
# ---------------------------------------------------------------------------

def _session_history_summary(session: dict) -> str:
    versions = session.get("diagram_versions", [])
    if not versions:
        return ""
    lines = ["Prior diagram versions (most recent first):"]
    for v in reversed(versions[-3:]):
        ver_num = v.get("version", "?")
        score = v.get("composite_score", "?")
        lines.append(f"  Version {ver_num}: composite_score={score}")
    return "\n".join(lines)


def synthesize_with_llm(
    suggestions: list[dict],
    features: dict,
    diagram_type: str,
    session: dict | None = None,
    composite_score: float | None = None,
    composite_score_raw: float | None = None,
    diagram_path: str | None = None,
) -> dict | None:
    """
    Call OpenAI to produce a prioritised natural-language report.
    When diagram_path is provided, the diagram image is sent via vision API so the
    LLM can reference specific visual elements by name and location.
    Returns None (without raising) on any failure.
    """
    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env")
        except ImportError:
            pass

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("LLM synthesis skipped: OPENAI_API_KEY not set.")
            return None

        client = OpenAI(api_key=api_key)

        # Try to load diagram image for vision API
        image_b64: str | None = None
        img_width: int = 0
        img_height: int = 0
        if diagram_path:
            try:
                import base64
                img_bytes = Path(diagram_path).read_bytes()
                image_b64 = base64.b64encode(img_bytes).decode()
                # Get image dimensions for coordinate normalization
                try:
                    import cv2
                    import numpy as np
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                except Exception:
                    pass
            except Exception as exc:
                logger.debug("Could not load diagram image for vision: %s", exc)

        # Choose model: prefer vision-capable gpt-4o-mini when image available
        if image_b64:
            model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        else:
            model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        dismissed = session.get("permanently_dismissed", []) if session else []
        non_ok = [s for s in suggestions if s["severity"] != "ok"]

        issues_text = ""
        for s in non_ok:
            locations = s.get("locations", [])
            loc_count = len(locations)
            loc_str = f" ({loc_count} location(s) flagged)" if loc_count else ""
            display = _METRIC_DISPLAY_NAMES.get(s["metric"], s["metric"])
            issues_text += (
                f"- [{s['severity'].upper()}] {display} (score={s['score']}){loc_str}\n"
                f"  Issue: {s['issue']}\n"
                f"  Fix: {s['recommendation']}\n"
            )
            # Append element names so the LLM can reference them in its "where" field
            element_labels = _names_from_locs(locations)
            if element_labels:
                issues_text += f"  Elements affected: {', '.join(repr(n) for n in element_labels)}\n"
            # Append spatial context using normalized coordinates when image dims are known
            if img_width and img_height and locations:
                spatial_hints = []
                for loc in locations[:5]:
                    px1, py1 = loc.get("x1", 0), loc.get("y1", 0)
                    px2, py2 = loc.get("x2", px1), loc.get("y2", py1)
                    cx_pct = ((px1 + px2) / 2) / img_width * 100
                    cy_pct = ((py1 + py2) / 2) / img_height * 100
                    h = "left" if cx_pct < 33 else ("center" if cx_pct < 66 else "right")
                    v = "top" if cy_pct < 33 else ("middle" if cy_pct < 66 else "bottom")
                    quad = "center" if (h == "center" and v == "middle") else (
                        f"{v} area" if h == "center" else (f"{h} side" if v == "middle" else f"{v}-{h}")
                    )
                    name = _extract_label(loc.get("detail", ""))
                    hint = f"'{name}' at {quad}" if name else f"element at {quad}"
                    spatial_hints.append(hint)
                if spatial_hints:
                    issues_text += f"  Locations: {'; '.join(spatial_hints)}\n"

        history_str = _session_history_summary(session) if session else ""

        score_line = (
            f"Composite quality score (active metrics, dismissed excluded): {composite_score}\n"
            + (f"Raw score (all metrics, for reference): {composite_score_raw}\n" if composite_score_raw is not None else "")
        )
        vision_note = (
            "\nYou are also provided the diagram image. Use it to visually identify and "
            "reference specific elements by name and location in your 'where' fields.\n"
            if image_b64 else ""
        )
        prompt = (
            f"You are an expert diagram quality analyst reviewing a {diagram_type} diagram.{vision_note}\n\n"
            "Automated metric evaluation has produced the following findings:\n\n"
            f"{issues_text or 'No significant issues detected — all metrics passed.'}\n\n"
            f"{score_line}\n"
            + (f"Permanently dismissed metrics (user accepts these as-is): {', '.join(dismissed)}\n\n"
               if dismissed else "")
            + (f"{history_str}\n\n" if history_str else "")
            + "Provide a structured analysis in JSON with EXACTLY this format:\n"
            '{\n'
            '  "overall_summary": "2-3 sentence executive summary of diagram quality",\n'
            '  "priority_issues": [\n'
            '    {"rank": 1, "metric": "metric_key", "severity": "critical|warning", '
            '"what": "concise description", "where": "specific element name and location in the diagram", '
            '"how_to_fix": "specific actionable fix"}\n'
            '  ],\n'
            '  "positive_aspects": ["what the diagram does well"],\n'
            '  "progress_vs_last_turn": "comment on improvement since last turn, '
            'or \\"First analysis\\" if no prior turns"\n'
            "}\n\n"
            "Only include non-ok issues in priority_issues. Be concise and actionable. "
            "Do not repeat advice the user has already dismissed. "
            "In 'where', name the specific element (e.g. 'the API Gateway node in the top-left') rather than a generic region."
        )

        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_b64:
            suffix = Path(diagram_path).suffix.lower().lstrip(".") if diagram_path else "png"
            mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(suffix, suffix) or "png"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{mime};base64,{image_b64}",
                    "detail": "low",
                },
            })

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)

    except Exception as exc:
        logger.warning("LLM synthesis failed (%s): falling back to rule-based output.", exc)
        return None


def locate_issues_with_llm(
    image_path: str | None,
    suggestions: list[dict],
    img_shape: tuple | None = None,
) -> dict[str, list[dict]]:
    """
    One collective LLM vision call for metrics that have no pixel-level locations.

    Returns a dict mapping metric key → list of region coordinate dicts.
    Returns {} on any failure or when there is nothing to locate.
    """
    # Metrics whose problems are inherently diagram-wide (no meaningful spatial location).
    # Sending these to the LLM always produces "full" or all-cell responses, which is
    # useless as an overlay, so we exclude them entirely.
    _GLOBAL_METRICS = {"color_harmony", "font_hierarchy", "label_area"}

    # If the LLM's returned regions for a metric cumulatively cover more than this
    # fraction of the image area, the result is too coarse to be useful and is discarded.
    _MAX_COVERAGE_PCT = 40.0

    try:
        targets = [
            s for s in suggestions
            if s.get("severity") != "ok"
            and not s.get("locations")
            and s["metric"] not in _GLOBAL_METRICS
        ]
        if not targets or not image_path:
            return {}

        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env")
        except ImportError:
            pass

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("locate_issues_with_llm skipped: OPENAI_API_KEY not set.")
            return {}

        import base64
        img_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(img_bytes).decode()
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(suffix, suffix) or "png"

        model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)

        target_keys = {s["metric"] for s in targets}

        issues_block = ""
        for s in targets:
            issue_snippet = (s.get("issue") or "")[:120]
            issues_block += (
                f"- {s['metric']} [{s['severity'].upper()}, score={s['score']}]: "
                f"{issue_snippet}\n"
            )

        system_prompt = (
            "You are a spatial analysis assistant for diagram quality review.\n\n"
            "The diagram is divided into a 10×10 grid. Rows 1-10 run top-to-bottom; "
            "columns 1-10 run left-to-right. Each cell covers 10% of the image per dimension.\n"
            "  r1-c1 = top-left corner   r1-c10 = top-right corner\n"
            "  r10-c1 = bottom-left      r10-c10 = bottom-right\n\n"
            "Cell names: \"r{row}-c{col}\" — e.g. \"r3-c7\", \"r5-c5\" (centre).\n\n"
            "Spanning aliases (prefer these over listing many cells):\n"
            "  \"row-1\" … \"row-10\"   full-width horizontal band\n"
            "  \"col-1\" … \"col-10\"   full-height vertical band\n"
            "  \"top-half\", \"bottom-half\", \"left-half\", \"right-half\"\n\n"
            "RULES:\n"
            "1. Return at most 5 entries per metric.\n"
            "2. Pick only the regions where the problem is MOST concentrated.\n"
            "3. If the issue has no clear spatial location (it is spread uniformly), "
            "return [] (empty array) for that metric.\n"
            "4. Do NOT return \"full\" — if you would return full, return [] instead.\n\n"
            "Return ONLY a JSON object mapping metric_key to an array of region names (or []).\n"
            "Do not include metrics not listed. Do not invent metric names.\n"
            "Example: {\"whitespace_distribution\": [\"r2-c8\", \"r3-c8\"], "
            "\"cognitive_chunk_density\": [\"row-3\", \"row-4\"]}"
        )

        user_content: list[dict] = [
            {
                "type": "text",
                "text": f"Diagram quality issues to locate:\n\n{issues_block}\n\n"
                        "Examine the diagram and return the JSON.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{mime};base64,{image_b64}",
                    "detail": "low",
                },
            },
        ]

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = json.loads(response.choices[0].message.content)
        result: dict[str, list[dict]] = {}
        for key, regions in raw.items():
            if key not in target_keys:
                continue
            if not isinstance(regions, list) or not regions:
                continue
            parsed = _parse_regions([str(r) for r in regions[:5]])
            if not parsed:
                continue
            # Discard if cumulative area coverage is too large to be informative.
            # Each region's coverage = (width * height) / 100  percent of the image.
            total_coverage = sum(r["width"] * r["height"] / 100.0 for r in parsed)
            if total_coverage > _MAX_COVERAGE_PCT:
                logger.debug(
                    "locate_issues_with_llm: dropping %s — coverage %.1f%% exceeds limit",
                    key, total_coverage,
                )
                continue
            result[key] = parsed

        return result

    except Exception as exc:
        logger.warning("locate_issues_with_llm failed (%s): returning empty regions.", exc)
        return {}


# ---------------------------------------------------------------------------
# Container utilization — LLM visual confirmation of ambiguous candidates
# ---------------------------------------------------------------------------

def _confirm_containers_with_llm(
    image_path: str,
    candidate_boxes: list[dict],
) -> list[dict]:
    """
    Use LLM vision to filter tier-2 candidate boxes down to genuinely
    under-utilised containers.

    Draws numbered bounding boxes on the diagram image (in-memory) and asks
    the vision model to distinguish diagram nodes (expected blank interior)
    from sections/containers that truly lack content.

    Returns the subset of ``candidate_boxes`` confirmed as real issues.
    On any failure returns the full candidate list so behaviour degrades
    gracefully to the previous approach.
    """
    if not candidate_boxes:
        return []

    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env")
        except ImportError:
            pass

        import base64
        import cv2
        import numpy as np
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("_confirm_containers_with_llm skipped: OPENAI_API_KEY not set.")
            return candidate_boxes

        img_bytes = Path(image_path).read_bytes()
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return candidate_boxes

        annotated = bgr.copy()
        COLOUR = (0, 0, 220)
        FONT   = cv2.FONT_HERSHEY_SIMPLEX

        for idx, box in enumerate(candidate_boxes, start=1):
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), COLOUR, 2)
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(annotated, (bx, by - th - 6), (bx + tw + 4, by), COLOUR, -1)
            cv2.putText(annotated, label, (bx + 2, by - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        suffix = Path(image_path).suffix.lower().lstrip(".")
        enc_ext = suffix if suffix in ("png", "jpg", "jpeg") else "png"
        _, enc = cv2.imencode(f".{enc_ext}", annotated)
        image_b64 = base64.b64encode(enc.tobytes()).decode()
        mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(suffix, suffix) or "png"

        n = len(candidate_boxes)
        system_prompt = (
            "You are a diagram quality analyst. The image shows a diagram with numbered "
            "red bounding boxes drawn by an automated tool that flagged these regions as "
            "potentially under-utilised containers.\n\n"
            "For each numbered box, decide:\n"
            "  A) DIAGRAM NODE — a labelled entity (e.g. a service, database, user, actor, "
            "component) whose interior is expected to contain only a short label. These are "
            "NOT quality issues.\n"
            "  B) EMPTY CONTAINER — a grouping box, section, or swim-lane that genuinely "
            "lacks content and should have more items inside it. These ARE quality issues.\n\n"
            "Return ONLY valid JSON: {\"confirmed_indices\": [<box numbers that are EMPTY CONTAINERS>]}\n"
            "If none qualify, return {\"confirmed_indices\": []}\n"
            "Do not include explanations."
        )

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"There are {n} numbered box(es) in the image. "
                    "Which are genuinely empty containers (not just diagram nodes with labels)?"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{mime};base64,{image_b64}",
                    "detail": "low",
                },
            },
        ]

        model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = json.loads(response.choices[0].message.content)
        confirmed_indices: list[int] = [
            int(i) for i in raw.get("confirmed_indices", [])
            if isinstance(i, (int, float, str)) and str(i).strip().isdigit()
        ]

        return [
            candidate_boxes[i - 1]
            for i in confirmed_indices
            if 1 <= i <= len(candidate_boxes)
        ]

    except Exception as exc:
        logger.warning("_confirm_containers_with_llm failed (%s): keeping all candidates.", exc)
        return candidate_boxes


# ---------------------------------------------------------------------------
# Brevity — LLM visual confirmation of deterministic shape candidates
# ---------------------------------------------------------------------------

def _confirm_brevity_shapes_with_llm(
    image_path: str,
    candidate_shapes: list[dict],
) -> list[dict]:
    """
    Use LLM vision to filter deterministically-found brevity candidates down to
    genuine diagram elements, removing misdetected shapes like connector lines.

    Draws numbered bounding boxes on the diagram and asks the vision model to
    identify which are real diagram nodes vs detection artifacts.

    Returns the confirmed subset. On any failure returns the full candidate list
    so behaviour degrades gracefully to the deterministic-only approach.
    """
    if not candidate_shapes:
        return []

    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env")
        except ImportError:
            pass

        import base64
        import cv2
        import numpy as np
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("_confirm_brevity_shapes_with_llm skipped: OPENAI_API_KEY not set.")
            return candidate_shapes

        img_bytes = Path(image_path).read_bytes()
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return candidate_shapes

        annotated = bgr.copy()
        COLOUR = (0, 0, 220)
        FONT   = cv2.FONT_HERSHEY_SIMPLEX

        for idx, shape in enumerate(candidate_shapes, start=1):
            bx, by, bw, bh = shape["x"], shape["y"], shape["w"], shape["h"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), COLOUR, 2)
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(annotated, (bx, by - th - 6), (bx + tw + 4, by), COLOUR, -1)
            cv2.putText(annotated, label, (bx + 2, by - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        suffix = Path(image_path).suffix.lower().lstrip(".")
        enc_ext = suffix if suffix in ("png", "jpg", "jpeg") else "png"
        _, enc = cv2.imencode(f".{enc_ext}", annotated)
        image_b64 = base64.b64encode(enc.tobytes()).decode()
        mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(suffix, suffix) or "png"

        n = len(candidate_shapes)
        shape_label_lines = "\n".join(
            f"  Box {idx}: {shape.get('detail', 'no text detected')}"
            for idx, shape in enumerate(candidate_shapes, start=1)
        )
        system_prompt = (
            "You are a diagram quality analyst. The image shows a diagram with numbered "
            "blue bounding boxes. Each box was matched by an automated tool to a label that "
            "may be overly verbose.\n\n"
            "The OCR-detected label text associated with each box is:\n"
            f"{shape_label_lines}\n\n"
            "Confirm a box ONLY if BOTH conditions hold:\n"
            "  1. Looking at the image, the text associated with that box visually appears "
            "INSIDE (or directly on) the numbered box — not on a connector arrow or floating "
            "nearby.\n"
            "  2. The label text shown above is genuinely verbose — a sentence, paragraph, or "
            "bullet list rather than a short concise identifier like 'API Gateway' or 'User'.\n\n"
            "Return ONLY valid JSON: {\"confirmed_indices\": [<box numbers meeting BOTH conditions>]}\n"
            "If none qualify, return {\"confirmed_indices\": []}\n"
            "Do not include explanations."
        )

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"There are {n} numbered box(es) in the image. "
                    "Confirm only boxes whose associated label text is genuinely verbose AND "
                    "visually located inside the numbered box."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{mime};base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]

        model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = json.loads(response.choices[0].message.content)
        confirmed_indices: list[int] = [
            int(i) for i in raw.get("confirmed_indices", [])
            if isinstance(i, (int, float, str)) and str(i).strip().isdigit()
        ]

        return [
            candidate_shapes[i - 1]
            for i in confirmed_indices
            if 1 <= i <= len(candidate_shapes)
        ]

    except Exception as exc:
        logger.warning("_confirm_brevity_shapes_with_llm failed (%s): keeping all candidates.", exc)
        return candidate_shapes


# ---------------------------------------------------------------------------
# Conversational chat helper
# ---------------------------------------------------------------------------

def chat_with_llm(
    user_message: str,
    session: dict,
    current_suggestions: list[dict],
    current_composite_score: float | None,
    diagram_type: str,
) -> dict:
    """
    Send one conversational turn to the LLM and return a structured response.

    Returns:
        {
            "reply": str,
            "action": {"type": "dismiss"|"restore"|"none", "metric": str|None}
        }

    Falls back to a rule-based reply with action=none on any failure.
    """
    _FALLBACK_ACTION: dict[str, Any] = {"type": "none", "metric": None}

    def _fallback() -> dict:
        non_ok = [s for s in current_suggestions if s["severity"] != "ok"]
        if non_ok:
            top = non_ok[0]
            display = _METRIC_DISPLAY_NAMES.get(top["metric"], top["metric"])
            reply = (
                f"The most pressing issue is {display} "
                f"(score={top['score']}): {top['issue']}"
            )
        else:
            reply = "No significant issues remain — all metrics are passing."
        return {"reply": reply, "action": _FALLBACK_ACTION}

    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env")
        except ImportError:
            pass

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("chat_with_llm skipped: OPENAI_API_KEY not set.")
            return _fallback()

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=api_key)

        dismissed = session.get("permanently_dismissed", [])
        versions = session.get("diagram_versions", [])

        score_trend = (
            " → ".join(str(v.get("composite_score", "?")) for v in versions)
            if versions else "No versions yet"
        )

        metric_lookup = "\n".join(
            f'  "{display}" → "{key}"'
            for key, display in _METRIC_DISPLAY_NAMES.items()
        )

        non_ok = [s for s in current_suggestions if s["severity"] != "ok"]
        issues_text = "\n".join(
            f"- [{s['severity'].upper()}] {_METRIC_DISPLAY_NAMES.get(s['metric'], s['metric'])} "
            f"(score={s['score']}): {s['issue']}"
            for s in non_ok
        ) or "No significant issues — all metrics passing."

        dismissed_str = ", ".join(dismissed) if dismissed else "none"
        score_str = f"{current_composite_score:.1f}" if current_composite_score is not None else "N/A"

        system_message = (
            f"You are a diagram quality analyst helping a user improve their {diagram_type} diagram iteratively.\n\n"
            "Metric name lookup (display name → internal key):\n"
            f"{metric_lookup}\n\n"
            "Current state:\n"
            f"- Composite score: {score_str} (active metrics only; dismissed metrics excluded)\n"
            f"- Dismissed metrics: {dismissed_str}\n"
            f"- Diagram versions submitted: {len(versions)} | Score trend: {score_trend}\n"
            f"- Current issues:\n{issues_text}\n\n"
            "Instructions:\n"
            "- Respond conversationally and concisely.\n"
            "- When the user wants to dismiss/ignore a metric, emit action type 'dismiss' with its internal key.\n"
            "- When the user wants to restore a previously dismissed metric, emit action type 'restore'.\n"
            "- Otherwise emit action type 'none' with metric null.\n"
            "- Always respond in JSON with exactly this structure:\n"
            '{"reply": "your response", "action": {"type": "dismiss|restore|none", "metric": "internal_key_or_null"}}'
        )

        history = session.get("chat_history", [])[-20:]
        messages: list[dict] = (
            [{"role": "system", "content": system_message}]
            + history
            + [{"role": "user", "content": user_message}]
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        parsed = json.loads(response.choices[0].message.content)

        reply = parsed.get("reply", "")
        action = parsed.get("action", {"type": "none", "metric": None})

        if action.get("metric") not in _METRIC_DISPLAY_NAMES:
            action = _FALLBACK_ACTION

        return {"reply": reply or "I'm not sure how to respond to that.", "action": action}

    except Exception as exc:
        logger.warning("chat_with_llm failed (%s): falling back to rule-based reply.", exc)
        return _fallback()


# ---------------------------------------------------------------------------
# Step 4 — Public API
# ---------------------------------------------------------------------------

def generate_suggestions(
    features: dict,
    diagram_type: str = "system_design",
    image_path: str | None = None,
    use_llm: bool = True,
    session: dict | None = None,
    img_shape: tuple | None = None,
) -> dict:
    """
    Main entry point. Runs rule-based analysis, optionally calls LLM synthesis,
    and returns a result dict ready for display and session storage.

    Returns:
        rule_based          — Suggestion list, sorted critical-first
        llm_report          — Structured LLM output, or None if skipped/failed
        critical_count      — Number of critical suggestions
        warning_count       — Number of warning suggestions
        composite_score     — Effective score (dismissed metrics count as 100)
        composite_score_raw — Raw equal-weight score across all metrics
    """
    dismissed = session.get("permanently_dismissed", []) if session else []

    # Resolve tier-2 container candidates via LLM before the rule-based pass so
    # that the suggestion engine sees only confirmed empty containers.
    if use_llm and image_path:
        cu = features.get("container_utilization")
        if isinstance(cu, dict):
            candidates = cu.get("empty_container_candidate_boxes", [])
            if candidates:
                confirmed = _confirm_containers_with_llm(image_path, candidates)
                tier1 = cu.get("empty_container_boxes", [])
                final_boxes = tier1 + confirmed
                cu["empty_container_boxes"] = final_boxes
                # When LLM reviewed all candidates and confirmed none are genuine issues,
                # reset the score so the metric correctly shows OK rather than warning
                # with an empty overlay (the pre-LLM score still counted those boxes).
                if not final_boxes:
                    cu["container_utilization_score"] = 100

    # Resolve brevity shape candidates via LLM — filters out misdetected shapes
    # (connector lines, arrows) before the rule-based pass builds flaggedLocations.
    if use_llm and image_path:
        brev = features.get("brevity")
        if isinstance(brev, dict) and brev.get("brevity_quality_score") is not None:
            verbose_labels = [p for p in brev.get("per_label_info", []) if p.get("violates_brevity")]
            if verbose_labels:
                candidates = _collect_brevity_candidates(verbose_labels, features.get("shapes", []))
                if candidates:
                    brev["confirmed_verbose_shapes"] = _confirm_brevity_shapes_with_llm(
                        image_path, candidates
                    )

    rule_based = generate_rule_based_suggestions(features, diagram_type, dismissed, img_shape=img_shape)
    composite_score = _compute_composite_score(features, dismissed)

    llm_regions: dict[str, list[dict]] = {}
    llm_report = None
    if use_llm:
        if image_path:
            llm_regions = locate_issues_with_llm(image_path, rule_based, img_shape=img_shape)
        # Assign "full" overlay for diagram-wide metrics excluded from the LLM spatial call
        _FULL_REGION = {"region": "full", "x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0}
        for s in rule_based:
            if s["metric"] in {"color_harmony", "font_hierarchy"} and s["severity"] != "ok":
                llm_regions.setdefault(s["metric"], [_FULL_REGION])
        llm_report = synthesize_with_llm(
            rule_based,
            features,
            diagram_type,
            session,
            composite_score=composite_score,
            diagram_path=image_path,
        )

    return {
        "rule_based": rule_based,
        "llm_report": llm_report,
        "llm_regions": llm_regions,
        "critical_count": sum(1 for s in rule_based if s["severity"] == "critical"),
        "warning_count": sum(1 for s in rule_based if s["severity"] == "warning"),
        "composite_score": composite_score,
    }


def format_text_report(suggestions_result: dict) -> str:
    """Return a human-readable markdown string summarising the analysis."""
    lines: list[str] = []
    rule_based = suggestions_result.get("rule_based", [])
    llm = suggestions_result.get("llm_report")

    if llm:
        lines.append("## Diagram Quality Analysis")
        lines.append("")
        lines.append(llm.get("overall_summary", ""))
        lines.append("")

        positives = llm.get("positive_aspects", [])
        if positives:
            lines.append("**Strengths:**")
            for p in positives:
                lines.append(f"- {p}")
            lines.append("")

        progress = llm.get("progress_vs_last_turn", "")
        if progress and progress != "First analysis":
            lines.append(f"**Progress since last turn:** {progress}")
            lines.append("")

        for issue in llm.get("priority_issues", []):
            sev = issue.get("severity", "").upper()
            metric = issue.get("metric", "")
            display = _METRIC_DISPLAY_NAMES.get(metric, metric)
            lines.append(
                f"{issue.get('rank', '?')}. [{sev}] **{display}**: {issue.get('what', '')}"
            )
            where = issue.get("where", "")
            if where:
                lines.append(f"   *Where:* {where}")
            lines.append(f"   *Fix:* {issue.get('how_to_fix', '')}")
        lines.append("")
    else:
        lines.append("## Diagram Quality Analysis (Rule-Based)")
        lines.append("")
        non_ok = [s for s in rule_based if s["severity"] != "ok"]
        if not non_ok:
            lines.append("No significant issues detected — all metrics passed.")
        else:
            for s in non_ok:
                sev = s["severity"].upper()
                display = _METRIC_DISPLAY_NAMES.get(s["metric"], s["metric"])
                score_str = f" (score={s['score']})" if s["score"] is not None else ""
                lines.append(f"**[{sev}] {display}**{score_str}")
                lines.append(f"  Issue: {s['issue']}")
                lines.append(f"  Fix:   {s['recommendation']}")
                if s.get("locations"):
                    lines.append(f"  Flagged locations: {len(s['locations'])}")
                lines.append("")

    score = suggestions_result.get("composite_score")
    lines.append(f"**Composite Score:** {score if score is not None else 'N/A'}")
    c = suggestions_result.get("critical_count", 0)
    w = suggestions_result.get("warning_count", 0)
    if c or w:
        lines.append(f"**Issues:** {c} critical, {w} warning")

    return "\n".join(lines)
