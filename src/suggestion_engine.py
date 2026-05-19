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

def generate_rule_based_suggestions(
    features: dict,
    diagram_type: str = "system_design",
    permanently_dismissed: list[str] | None = None,
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
            _add(
                "label_readability", lr_sev, score,
                f"Low OCR confidence on {lr.get('labels_below_threshold', 0)} label(s) "
                f"(fraction_below={lr.get('fraction_labels_below_threshold', 0):.2f}). "
                "Text may be too small, blurry, or have poor contrast.",
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
            _add(
                "label_area", la_sev, score,
                f"Label area ratio is {cat} (ratio={ratio:.3f}). "
                + ("Labels cover too little of the diagram — many areas lack context."
                   if cat == "sparse" else
                   "Labels cover too much of the diagram — the layout feels cramped."),
                [],
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
            if verdict == "crowded labels":
                _add(
                    "overlap_metrics", om_sev, score,
                    f"Labels are crowded: {fclose:.0%} of labels are too close to a neighbour "
                    f"({any_iou:.2%} of label pairs physically overlap).",
                    [],
                    "Spread elements further apart, reduce label count, or increase diagram canvas size.",
                )
            else:
                _add(
                    "overlap_metrics", om_sev, score,
                    f"Moderate label crowding: {fclose:.0%} of labels are too close to a neighbour.",
                    [],
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
                [],
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
            _cu_t = threshold_manager.get_thresholds("container_utilization")
            _cu_crit = _cu_t.get("critical_threshold")
            _cu_warn = _cu_t.get("warning_threshold")
            if _cu_crit is not None and cu_score < _cu_crit:
                _add(
                    "container_utilization", "critical", cu_score,
                    f"Container utilization is very low ({cu_score:.1f}/100). Boxes are significantly under-used.",
                    [],
                    "Add more content to containers, merge near-empty boxes, or remove unused containers.",
                )
            elif _cu_warn is not None and cu_score < _cu_warn:
                _add(
                    "container_utilization", "warning", cu_score,
                    f"Container utilization is below optimal ({cu_score:.1f}/100). Some regions are sparse.",
                    [],
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
            locs = [
                {
                    "x1": b["x"], "y1": b["y"],
                    "x2": b["x"] + b["w"], "y2": b["y"] + b["h"],
                    "detail": "isolated box — no connector lines detected",
                }
                for b in ib.get("island_boxes", [])
            ]
            _ib_t = threshold_manager.get_thresholds("isolated_boxes")
            _ib_crit = _ib_t.get("critical_threshold")
            _ib_warn = _ib_t.get("warning_threshold")
            if _ib_crit is not None and ib_score < _ib_crit:
                _add(
                    "isolated_boxes", "critical", ib_score,
                    f"{island_count} box(es) have no connector lines (score={ib_score}/100). "
                    "Isolated nodes break the diagram's structural flow.",
                    locs,
                    "Draw connector arrows or lines between isolated elements and the rest of the diagram.",
                )
            elif _ib_warn is not None and ib_score < _ib_warn:
                _add(
                    "isolated_boxes", "warning", ib_score,
                    f"{island_count} box(es) appear disconnected from the rest of the diagram.",
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
            verbose_labels = [p for p in brev.get("per_label_info", []) if p.get("violates_brevity")]
            locs = [
                {
                    "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
                    "detail": f"text={p.get('text', '')!r} chars={p.get('char_count', '?')}",
                }
                for p in verbose_labels
                if p.get("x1") is not None
            ]
            _brev_t = threshold_manager.get_thresholds("brevity")
            _brev_crit = _brev_t.get("critical_threshold")
            _brev_warn = _brev_t.get("warning_threshold")
            if _brev_crit is not None and brev_score < _brev_crit:
                _add(
                    "brevity", "critical", brev_score,
                    f"Many labels are excessively long (score={brev_score:.1f}/100, "
                    f"{len(verbose_labels)} verbose label(s) detected).",
                    locs,
                    "Shorten labels to concise identifiers. Move detail to tooltips or a legend.",
                )
            elif _brev_warn is not None and brev_score < _brev_warn:
                _add(
                    "brevity", "warning", brev_score,
                    f"Some labels exceed recommended length (score={brev_score:.1f}/100).",
                    locs,
                    "Trim verbose labels. Aim for ≤40 characters per label in system design diagrams.",
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
            elif _ch_warn is not None and ch_score < _ch_warn:
                _add(
                    "color_harmony", "warning", ch_score,
                    f"Color harmony could be improved (score={ch_score:.1f}/100, {detail}).",
                    [],
                    "Limit palette to 3-5 well-chosen colors and apply them consistently.",
                )
            else:
                _add("color_harmony", "ok", ch_score, "Color harmony is good.", [], "")
        else:
            _add("color_harmony", "ok", None, "Color harmony score unavailable.", [], "")

    # ── label_contrast ────────────────────────────────────────────────────
    lcq = features.get("label_contrast")
    if isinstance(lcq, dict):
        lc_score = lcq.get("label_contrast_score")
        if lc_score is not None:
            low_contrast = [
                p for p in lcq.get("per_label_info", [])
                if p.get("delta_L", 50.0) < 40.0 or p.get("delta_L", 50.0) > 80.0
            ]
            locs = [
                {
                    "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
                    "detail": f"ΔL={p.get('delta_L', 0):.1f}",
                }
                for p in low_contrast
                if p.get("x1") is not None
            ]
            _lc_t = threshold_manager.get_thresholds("label_contrast")
            _lc_warn = _lc_t.get("warning_threshold")
            if _lc_warn is not None and lc_score < _lc_warn:
                _add(
                    "label_contrast", "warning", lc_score,
                    f"Label contrast score is {lc_score:.1f}/100. "
                    f"{len(low_contrast)} label(s) have ΔL outside the optimal 40–80 range "
                    f"(mean_ΔL={lcq.get('mean_delta_L', 0):.1f}).",
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

    # ── orientation_consistency ────────────────────────────────────────────
    oc = features.get("orientation_consistency")
    if isinstance(oc, dict):
        oc_score = oc.get("orientation_consistency_score")
        if oc_score is not None:
            dom_deg = oc.get("dominant_orientation_deg", 0.0)
            frac = oc.get("consistent_label_fraction", 0.0)
            _oc_t = threshold_manager.get_thresholds("orientation_consistency")
            _oc_crit = _oc_t.get("critical_threshold")
            _oc_warn = _oc_t.get("warning_threshold")
            if _oc_crit is not None and oc_score < _oc_crit:
                _add(
                    "orientation_consistency", "critical", oc_score,
                    f"Label orientation is highly inconsistent (score={oc_score:.1f}/100). "
                    f"Only {frac:.0%} of labels follow the dominant orientation ({dom_deg:.1f}°).",
                    [],
                    "Align all labels to a single orientation (preferably 0° horizontal). Avoid mixing horizontal and rotated text.",
                )
            elif _oc_warn is not None and oc_score < _oc_warn:
                _add(
                    "orientation_consistency", "warning", oc_score,
                    f"Some labels deviate from the dominant orientation (score={oc_score:.1f}/100, "
                    f"dominant={dom_deg:.1f}°, consistent={frac:.0%}).",
                    [],
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
) -> dict | None:
    """
    Call OpenAI to produce a prioritised natural-language report.
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

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=api_key)

        dismissed = session.get("permanently_dismissed", []) if session else []
        non_ok = [s for s in suggestions if s["severity"] != "ok"]

        issues_text = ""
        for s in non_ok:
            loc_count = len(s.get("locations", []))
            loc_str = f" ({loc_count} location(s) flagged)" if loc_count else ""
            display = _METRIC_DISPLAY_NAMES.get(s["metric"], s["metric"])
            issues_text += (
                f"- [{s['severity'].upper()}] {display} (score={s['score']}){loc_str}\n"
                f"  Issue: {s['issue']}\n"
                f"  Fix: {s['recommendation']}\n"
            )

        history_str = _session_history_summary(session) if session else ""

        score_line = (
            f"Composite quality score (active metrics, dismissed excluded): {composite_score}\n"
            + (f"Raw score (all metrics, for reference): {composite_score_raw}\n" if composite_score_raw is not None else "")
        )
        prompt = (
            f"You are an expert diagram quality analyst reviewing a {diagram_type} diagram.\n\n"
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
            '"what": "concise description", "where": "location info or empty string", '
            '"how_to_fix": "specific actionable fix"}\n'
            '  ],\n'
            '  "positive_aspects": ["what the diagram does well"],\n'
            '  "progress_vs_last_turn": "comment on improvement since last turn, '
            'or \\"First analysis\\" if no prior turns"\n'
            "}\n\n"
            "Only include non-ok issues in priority_issues. Be concise and actionable. "
            "Do not repeat advice the user has already dismissed."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)

    except Exception as exc:
        logger.warning("LLM synthesis failed (%s): falling back to rule-based output.", exc)
        return None


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

    rule_based = generate_rule_based_suggestions(features, diagram_type, dismissed)
    composite_score = _compute_composite_score(features, dismissed)

    llm_report = None
    if use_llm:
        llm_report = synthesize_with_llm(
            rule_based,
            features,
            diagram_type,
            session,
            composite_score=composite_score,
        )

    return {
        "rule_based": rule_based,
        "llm_report": llm_report,
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
