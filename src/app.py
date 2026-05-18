"""
Streamlit GUI for the Diagram Analyser.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import streamlit as st

import threshold_manager
from analyze_diagram import _analyze_new_version, _apply_llm_action, _build_version_diff
from chat_router import route_message
from suggestion_engine import (
    _ACTIVE_METRICS,
    _compute_composite_score,
    _default_session_path,
    _METRIC_DISPLAY_NAMES,
    chat_with_llm,
    generate_rule_based_suggestions,
    load_session,
    new_session,
    save_session,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {
    "critical": "#ff5370",
    "warning":  "#ffaa4d",
    "ok":       "#3dd68c",
}

_DIAGRAMS_DIR = _SRC_DIR / "data" / "diagrams"
_UPLOADS_DIR  = _SRC_DIR / "data" / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_DIAGRAM_TYPE_LABELS = {
    "system_design":    "System Design",
    "timeline_roadmap": "Timeline / Roadmap",
}

_DIAGRAM_TYPE_ICONS = {
    "system_design":    "◈",
    "timeline_roadmap": "◷",
}


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

def _inject_global_css() -> None:
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --color-bg-surface:    #1a1d2e;
  --color-bg-elevated:   #252840;
  --color-border:        #2d3250;
  --color-border-strong: #4a4f7a;
  --color-text-primary:  #e8eaf0;
  --color-text-secondary:#9499b8;
  --color-text-muted:    #5c6185;
  --color-accent-blue:   #4f7fff;
  --color-accent-purple: #7c6aff;
  --color-critical:      #ff5370;
  --color-warning:       #ffaa4d;
  --color-ok:            #3dd68c;
  --color-dismissed:     #4a4f7a;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --shadow-card: 0 2px 8px rgba(0,0,0,0.35);
}

* { font-family: 'Inter', system-ui, -apple-system, sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { min-width: 300px !important; max-width: 300px !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 20px; padding-left: 16px; padding-right: 16px; }

.da-sidebar-header {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 0 16px 0;
  border-bottom: 1px solid var(--color-border);
  margin-bottom: 20px;
}
.da-sidebar-logo { font-size: 1.4rem; color: var(--color-accent-blue); line-height: 1; }
.da-sidebar-title { font-size: 1rem; font-weight: 700; color: var(--color-text-primary); letter-spacing: -0.01em; }

/* Session chip */
.da-session-chip {
  background: var(--color-bg-elevated);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 10px 12px; margin-bottom: 12px;
}
.da-session-chip-header {
  display: flex; align-items: center; gap: 8px;
  font-weight: 600; font-size: 0.9rem; color: var(--color-text-primary);
}
.da-session-chip-meta { font-size: 0.78rem; color: var(--color-text-secondary); margin-top: 4px; }
.da-session-chip-icon { color: var(--color-accent-blue); font-size: 1rem; }

/* Session cards */
.da-session-card {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  background: var(--color-bg-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  margin-bottom: 4px;
}
.da-session-monogram {
  width: 36px; height: 36px; min-width: 36px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.75rem; color: white;
  background: var(--color-accent-blue);
}
.da-session-monogram--timeline { background: var(--color-accent-purple); }
.da-session-info { flex: 1; min-width: 0; }
.da-session-name { font-weight: 600; font-size: 0.85rem; color: var(--color-text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.da-session-meta { font-size: 0.73rem; color: var(--color-text-secondary); margin-top: 2px; }

/* Badges */
.da-badge {
  display: inline-block; padding: 2px 7px; border-radius: 4px;
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.04em; line-height: 1.6; vertical-align: middle;
}
.da-badge--critical  { background: rgba(255,83,112,0.18);  color: var(--color-critical); }
.da-badge--warning   { background: rgba(255,170,77,0.18);  color: var(--color-warning); }
.da-badge--ok        { background: rgba(61,214,140,0.18);  color: var(--color-ok); }
.da-badge--dismissed { background: rgba(74,79,122,0.3);    color: var(--color-dismissed); }
.da-badge--na        { background: rgba(74,79,122,0.25);   color: var(--color-text-muted); }
.da-badge--score     { font-size: 0.78rem; font-weight: 700; padding: 3px 8px; border-radius: var(--radius-sm); }
.da-badge--intent    {
  background: rgba(79,127,255,0.12); color: var(--color-accent-blue);
  font-size: 0.68rem; padding: 2px 7px; border-radius: 4px;
  letter-spacing: 0; text-transform: none; font-weight: 500; vertical-align: middle; margin-right: 6px;
}

/* Hero */
.da-score-card {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 20px 16px; background: var(--color-bg-surface);
  border: 1px solid var(--color-border); border-radius: var(--radius-lg); min-height: 140px;
}
.da-score-ring-wrap { position: relative; width: 90px; height: 90px; margin-bottom: 10px; }
.da-score-ring {
  width: 90px; height: 90px; border-radius: 50%;
  background: conic-gradient(var(--clr) calc(var(--pct) * 1%), var(--color-border) 0);
  position: relative;
}
@supports not (background: conic-gradient(red 0)) {
  .da-score-ring { background: var(--clr); }
}
.da-score-ring::before {
  content: ''; position: absolute; inset: 12px;
  background: var(--color-bg-surface); border-radius: 50%;
}
.da-score-value {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; font-weight: 700; color: var(--color-text-primary); z-index: 1;
}
.da-score-label { font-size: 0.78rem; color: var(--color-text-secondary); font-weight: 500; text-align: center; }
.da-delta-up   { color: var(--color-ok);      font-size: 0.8rem; font-weight: 600; }
.da-delta-down { color: var(--color-critical); font-size: 0.8rem; font-weight: 600; }

.da-stat-card {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 20px 12px; background: var(--color-bg-surface);
  border: 1px solid var(--color-border); border-radius: var(--radius-lg);
  min-height: 140px; text-align: center;
}
.da-stat-icon  { font-size: 1.2rem; margin-bottom: 6px; }
.da-stat-value { font-size: 2rem; font-weight: 700; color: var(--color-text-primary); line-height: 1; }
.da-stat-label { font-size: 0.75rem; color: var(--color-text-secondary); margin-top: 6px; }

/* Metric cards */
.da-metric-card {
  background: var(--color-bg-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius-md); padding: 10px 12px; margin-bottom: 10px;
  transition: transform 0.1s ease, border-color 0.1s ease;
}
.da-metric-card:hover { transform: translateY(-1px); border-color: var(--color-border-strong); }
.da-metric-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.da-metric-name { font-size: 0.82rem; font-weight: 600; color: var(--color-text-primary); }
.da-metric-bar-track { height: 6px; background: var(--color-border); border-radius: 3px; overflow: hidden; margin-bottom: 5px; }
.da-metric-fill { height: 100%; border-radius: 3px; width: var(--w, 0%); background: var(--clr, var(--color-border-strong)); transition: width 0.4s ease; }
.da-metric-fill--critical { box-shadow: 0 0 6px rgba(255,83,112,0.5); }
.da-metric-score { font-size: 0.72rem; color: var(--color-text-secondary); }

/* Suggestion cards */
.da-suggestion-card {
  border-left: 4px solid var(--sev-clr, var(--color-border-strong));
  background: var(--color-bg-surface);
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  padding: 10px 12px 8px 12px; margin-bottom: 4px;
}
.da-suggestion-header { display: flex; align-items: center; justify-content: space-between; }
.da-suggestion-name { font-weight: 600; font-size: 0.88rem; color: var(--color-text-primary); }
.da-suggestion-score { font-size: 0.78rem; color: var(--color-text-secondary); font-weight: 500; }
.da-recommendation {
  border-left: 3px solid var(--color-accent-blue);
  background: rgba(79,127,255,0.06);
  padding: 8px 10px; border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  font-size: 0.82rem; color: var(--color-text-secondary); margin: 8px 0 6px 0;
}
.da-recommendation strong { color: var(--color-text-primary); }

/* Pending banner */
.da-banner-pending {
  display: flex; align-items: center;
  background: rgba(255,170,77,0.1); border: 1px solid var(--color-warning);
  border-radius: var(--radius-md); padding: 10px 14px; margin-bottom: 12px;
  font-size: 0.85rem; color: var(--color-text-primary);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--color-border) !important; }
.stTabs [data-baseweb="tab"] { border-radius: var(--radius-sm) var(--radius-sm) 0 0; padding: 8px 16px; font-weight: 500; font-size: 0.88rem; color: var(--color-text-secondary); }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--color-accent-purple) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--color-accent-purple) !important; height: 2px !important; }

/* Buttons */
.stButton button[kind="primary"] {
  background: var(--color-accent-blue) !important; border: none !important;
  border-radius: var(--radius-sm) !important; font-weight: 600 !important;
}
.stButton button[kind="secondary"] {
  background: transparent !important; border: 1px solid var(--color-border-strong) !important;
  color: var(--color-text-secondary) !important; border-radius: var(--radius-sm) !important;
}

/* Empty states */
.da-empty-state {
  text-align: center; padding: 48px 24px;
  background: var(--color-bg-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius-lg); margin: 24px 0;
}
.da-empty-icon  { font-size: 2.5rem; margin-bottom: 12px; }
.da-empty-title { font-size: 1.1rem; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px; }
.da-empty-desc  { font-size: 0.85rem; color: var(--color-text-secondary); }

/* Changes summary */
.da-changes-summary {
  background: var(--color-bg-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius-md); padding: 14px 16px; margin-top: 12px;
}
.da-changes-title { font-weight: 600; font-size: 0.88rem; color: var(--color-text-primary); margin-bottom: 10px; }
.da-changes-item  { font-size: 0.82rem; margin-bottom: 4px; padding: 3px 0; }
.da-changes-item--up   { color: var(--color-ok); }
.da-changes-item--down { color: var(--color-critical); }

/* Welcome cards */
.da-welcome-card {
  background: var(--color-bg-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius-lg); padding: 24px; height: 100%;
}
.da-welcome-icon  { font-size: 1.8rem; margin-bottom: 10px; }
.da-welcome-title { font-size: 1rem; font-weight: 600; color: var(--color-text-primary); margin-bottom: 8px; }
.da-welcome-desc  { font-size: 0.85rem; color: var(--color-text-secondary); line-height: 1.6; }

/* Chat */
.stChatInput textarea { font-family: 'Inter', system-ui, sans-serif !important; border-radius: var(--radius-md) !important; }

/* Misc */
.da-divider { border-top: 1px solid var(--color-border); margin: 12px 0; }
footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "session":                  None,
        "session_path":             None,
        "features":                 {},
        "suggestions_result":       None,
        "severity_overrides":       {},
        "pending_severity_changes": {},
        "last_file_id":             None,
        "last_path_input":          None,
        "suggestion_filter":        "all",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_to_init() -> None:
    st.session_state.update({
        "session":                  None,
        "session_path":             None,
        "features":                 {},
        "suggestions_result":       None,
        "severity_overrides":       {},
        "pending_severity_changes": {},
        "last_file_id":             None,
        "last_path_input":          None,
        "suggestion_filter":        "all",
    })


def _has_session() -> bool:
    return st.session_state["session"] is not None


def _has_analysis() -> bool:
    return bool(
        st.session_state.get("session")
        and st.session_state["session"].get("diagram_versions")
    )


def _live_score() -> float | None:
    features = st.session_state.get("features")
    session  = st.session_state["session"]
    if not features or not session:
        return None
    return _compute_composite_score(features, session["permanently_dismissed"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="da-sidebar-header">'
            '<span class="da-sidebar-logo">⬡</span>'
            '<span class="da-sidebar-title">Diagram Analyser</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if not _has_session():
            _render_session_init()
        else:
            _render_session_info()
            st.markdown('<div class="da-divider"></div>', unsafe_allow_html=True)
            _render_image_uploader()
            st.markdown('<div class="da-divider"></div>', unsafe_allow_html=True)
            _render_download_button()


def _list_local_sessions() -> list[dict]:
    import re
    sessions_dir = _SRC_DIR / "data" / "sessions"
    if not sessions_dir.exists():
        return []
    results = []
    for path in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            versions = data.get("diagram_versions", [])
            last_v   = versions[-1] if versions else {}
            img_path = last_v.get("diagram_path", "")
            stem     = Path(img_path).stem if img_path else data.get("session_id", path.stem)
            friendly = re.sub(r"_\d{8}_\d{6}$", "", stem) or path.stem
            results.append({
                "path":          str(path),
                "session_id":    data.get("session_id", path.stem),
                "friendly_name": friendly,
                "diagram_type":  data.get("diagram_type", "system_design"),
                "version_count": len(versions),
                "last_score":    last_v.get("composite_score"),
                "analyzed_at":   last_v.get("analyzed_at", ""),
                "last_image":    img_path,
                "data":          data,
            })
        except Exception:
            continue
    return results


def _score_badge_cls(score: float | None) -> str:
    if score is None:
        return "da-badge--na"
    if score >= 80:
        return "da-badge--ok"
    if score >= 60:
        return "da-badge--warning"
    return "da-badge--critical"


def _render_session_card(meta: dict) -> None:
    dtype_lbl = _DIAGRAM_TYPE_LABELS.get(meta["diagram_type"], meta["diagram_type"])
    score     = meta["last_score"]
    date_raw  = meta["analyzed_at"]
    date_str  = date_raw[:10] if date_raw else "—"
    monogram  = meta["friendly_name"][:3].upper()
    badge_cls = _score_badge_cls(score)
    score_txt = f"{score:.0f}" if score is not None else "N/A"
    mono_cls  = "da-session-monogram--timeline" if meta["diagram_type"] == "timeline_roadmap" else ""

    st.markdown(
        f'<div class="da-session-card">'
        f'<div class="da-session-monogram {mono_cls}">{monogram}</div>'
        f'<div class="da-session-info">'
        f'<div class="da-session-name">{meta["friendly_name"]}</div>'
        f'<div class="da-session-meta">{dtype_lbl} · {meta["version_count"]}v · {date_str}</div>'
        f'</div>'
        f'<span class="da-badge da-badge--score {badge_cls}">{score_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Resume", key=f"resume_{meta['session_id']}", use_container_width=True):
        data     = meta["data"]
        versions = data.get("diagram_versions", [])
        last_v   = versions[-1] if versions else {}
        st.session_state.update({
            "session":                  data,
            "session_path":             meta["path"],
            "severity_overrides":       {},
            "pending_severity_changes": {},
            "last_file_id":             None,
            "last_path_input":          None,
            "features":                 last_v.get("metric_scores", {}),
            "suggestions_result":       {
                "suggestions":     last_v.get("suggestions", []),
                "composite_score": last_v.get("composite_score"),
            } if last_v else None,
        })
        st.rerun()


def _render_session_init() -> None:
    st.markdown("#### New diagram")
    st.caption("Pick a type, then upload your first image after the session starts.")
    dtype = st.selectbox(
        "Diagram type",
        options=list(_DIAGRAM_TYPE_LABELS.keys()),
        format_func=lambda k: _DIAGRAM_TYPE_LABELS[k],
        key="new_session_dtype",
    )
    if st.button("＋ Start Analysis", use_container_width=True, type="primary"):
        session = new_session(dtype)
        st.session_state.update({
            "session":            session,
            "session_path":       None,
            "severity_overrides": {},
        })
        st.rerun()

    past = _list_local_sessions()
    if past:
        st.markdown('<div class="da-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Past sessions")
        st.caption("Click Resume to continue where you left off.")
        for meta in past:
            _render_session_card(meta)


def _render_session_info() -> None:
    session     = st.session_state["session"]
    dtype_label = _DIAGRAM_TYPE_LABELS.get(session["diagram_type"], session["diagram_type"])
    dtype_icon  = _DIAGRAM_TYPE_ICONS.get(session["diagram_type"], "◈")
    n           = len(session.get("diagram_versions", []))
    dismissed   = session.get("permanently_dismissed", [])
    dim_txt     = f'{len(dismissed)} dismissed metric{"s" if len(dismissed) != 1 else ""}' if dismissed else "No dismissed metrics"

    st.markdown(
        f'<div class="da-session-chip">'
        f'<div class="da-session-chip-header">'
        f'<span class="da-session-chip-icon">{dtype_icon}</span>'
        f'<span>{dtype_label}</span>'
        f'<span class="da-badge da-badge--na" style="margin-left:auto">{n}v</span>'
        f'</div>'
        f'<div class="da-session-chip-meta">{dim_txt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("End session", use_container_width=False):
        _reset_to_init()
        st.rerun()


def _render_image_uploader() -> None:
    session = st.session_state["session"]
    n       = len(session.get("diagram_versions", []))
    label   = "Upload next version" if n > 0 else "Upload your diagram"

    st.markdown(f'<div style="font-size:0.85rem;font-weight:600;color:var(--color-text-primary);margin-bottom:6px">{label}</div>', unsafe_allow_html=True)
    if n > 0:
        st.caption("Each upload is saved as a new version.")

    uploaded = st.file_uploader(
        "Drag-drop or browse (PNG, JPG, JPEG, WEBP, GIF, BMP)",
        type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
        key="img_uploader",
        label_visibility="collapsed",
    )
    path_input = st.text_input(
        "Or enter a local file path",
        placeholder="/path/to/diagram.png",
        key="img_path_input",
    )

    if uploaded and uploaded.file_id != st.session_state.get("last_file_id"):
        st.session_state["last_file_id"] = uploaded.file_id
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix  = Path(uploaded.name).suffix
        stem    = Path(uploaded.name).stem
        saved_p = _UPLOADS_DIR / f"{stem}_{ts}{suffix}"
        saved_p.write_bytes(uploaded.getvalue())
        _run_analysis(str(saved_p))

    elif path_input.strip() and path_input.strip() != st.session_state.get("last_path_input"):
        p = Path(path_input.strip()).expanduser().resolve()
        if p.exists():
            st.session_state["last_path_input"] = path_input.strip()
            _run_analysis(str(p))
        else:
            st.error(f"File not found: {p}")


def _run_analysis(image_path_str: str) -> None:
    session = st.session_state["session"]

    if st.session_state["session_path"] is None:
        st.session_state["session_path"] = _default_session_path(image_path_str)

    with st.status("Analysing your diagram...", expanded=True) as status:
        status.write("Running computer vision metrics...")
        try:
            suggestions_result, features = _analyze_new_version(
                image_path_str,
                session,
                st.session_state["session_path"],
            )
            status.write("Generating AI narrative...")
            status.update(label="Analysis complete!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis failed: {exc}")
            return

    st.session_state["features"]                 = features
    st.session_state["suggestions_result"]       = suggestions_result
    st.session_state["severity_overrides"]       = {}
    st.session_state["pending_severity_changes"] = {}
    st.rerun()


def _render_download_button() -> None:
    session = st.session_state["session"]
    if not session:
        return
    blob = json.dumps(session, indent=2, default=str).encode()
    st.download_button(
        label="⬇ Export JSON",
        data=blob,
        file_name=f"{session['session_id']}.json",
        mime="application/json",
        use_container_width=False,
    )


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _badge(label: str, severity: str) -> str:
    cls_map = {
        "critical":  "da-badge--critical",
        "warning":   "da-badge--warning",
        "ok":        "da-badge--ok",
        "dismissed": "da-badge--dismissed",
    }
    cls = cls_map.get(severity.lower(), "da-badge--na")
    return f"<span class='da-badge {cls}'>{label}</span>"


def _build_metric_card_html(key: str, display: str, score: float | None, severity: str) -> str:
    badge_cls_map = {
        "critical":  "da-badge--critical",
        "warning":   "da-badge--warning",
        "ok":        "da-badge--ok",
        "dismissed": "da-badge--dismissed",
    }
    badge_cls = badge_cls_map.get(severity, "da-badge--na")
    badge_lbl = severity.upper() if severity not in ("n/a",) else "N/A"

    if score is not None:
        pct = f"{int(score)}%"
        clr_map = {
            "critical":  "var(--color-critical)",
            "warning":   "var(--color-warning)",
            "ok":        "var(--color-ok)",
            "dismissed": "var(--color-dismissed)",
        }
        clr      = clr_map.get(severity, "var(--color-border-strong)")
        glow_cls = " da-metric-fill--critical" if severity == "critical" else ""
        bar      = (
            f'<div class="da-metric-bar-track">'
            f'<div class="da-metric-fill{glow_cls}" style="--w:{pct};--clr:{clr}"></div>'
            f'</div>'
            f'<span class="da-metric-score">{score:.1f} / 100</span>'
        )
    else:
        bar = '<span class="da-metric-score" style="color:var(--color-text-muted)">No score</span>'

    return (
        f'<div class="da-metric-card">'
        f'<div class="da-metric-header">'
        f'<span class="da-metric-name">{display}</span>'
        f'<span class="da-badge {badge_cls}">{badge_lbl}</span>'
        f'</div>'
        f'{bar}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Tab 1 — Analysis
# ---------------------------------------------------------------------------

def render_analysis_tab() -> None:
    session  = st.session_state["session"]
    features = st.session_state.get("features", {})

    if not _has_analysis():
        st.markdown(
            '<div class="da-empty-state">'
            '<div class="da-empty-icon">📐</div>'
            '<div class="da-empty-title">No diagram analysed yet</div>'
            '<div class="da-empty-desc">Upload a diagram from the sidebar to see your quality score and improvement suggestions.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    versions = session["diagram_versions"]
    latest   = versions[-1]
    prev     = versions[-2] if len(versions) >= 2 else None

    live_score = _live_score()

    _sr         = st.session_state.get("suggestions_result")
    suggestions = _sr["suggestions"] if _sr and "suggestions" in _sr else latest.get("suggestions", [])
    non_ok      = [s for s in suggestions if s["severity"] != "ok"]
    dismissed   = session.get("permanently_dismissed", [])

    critical_count = sum(1 for s in non_ok if s["severity"] == "critical" and s["metric"] not in dismissed)
    warning_count  = sum(1 for s in non_ok if s["severity"] == "warning"  and s["metric"] not in dismissed)

    prev_score = prev["composite_score"] if prev else None
    delta_val  = (live_score - prev_score) if (live_score is not None and prev_score is not None) else None

    # ── Hero ─────────────────────────────────────────────────────────────────
    col_score, col_crit, col_warn, col_dim = st.columns([2, 1, 1, 1])

    with col_score:
        pct = int(live_score) if live_score is not None else 0
        clr = "var(--color-ok)" if pct >= 80 else ("var(--color-warning)" if pct >= 60 else "var(--color-critical)")
        delta_html = ""
        if delta_val is not None:
            sign     = "▲" if delta_val > 0 else "▼"
            dcls     = "da-delta-up" if delta_val > 0 else "da-delta-down"
            delta_html = f'<span class="{dcls}">{sign} {abs(delta_val):.1f}</span>'
        st.markdown(
            f'<div class="da-score-card">'
            f'<div class="da-score-ring-wrap">'
            f'<div class="da-score-ring" style="--pct:{pct};--clr:{clr}">'
            f'<span class="da-score-value">{pct}</span>'
            f'</div></div>'
            f'<div class="da-score-label">Composite Score {delta_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_crit:
        clr = "var(--color-critical)" if critical_count > 0 else "var(--color-text-primary)"
        st.markdown(
            f'<div class="da-stat-card"><div class="da-stat-icon">◈</div>'
            f'<div class="da-stat-value" style="color:{clr}">{critical_count}</div>'
            f'<div class="da-stat-label">Critical Issues</div></div>',
            unsafe_allow_html=True,
        )

    with col_warn:
        clr = "var(--color-warning)" if warning_count > 0 else "var(--color-text-primary)"
        st.markdown(
            f'<div class="da-stat-card"><div class="da-stat-icon">⚠</div>'
            f'<div class="da-stat-value" style="color:{clr}">{warning_count}</div>'
            f'<div class="da-stat-label">Warnings</div></div>',
            unsafe_allow_html=True,
        )

    with col_dim:
        st.markdown(
            f'<div class="da-stat-card"><div class="da-stat-icon">○</div>'
            f'<div class="da-stat-value">{len(dismissed)}</div>'
            f'<div class="da-stat-label">Dismissed</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ── Diagram + AI report ──────────────────────────────────────────────────
    col_img, col_llm = st.columns([1, 1])
    with col_img:
        st.subheader("Diagram")
        diagram_path = latest.get("diagram_path", "")
        if Path(diagram_path).exists():
            st.image(diagram_path, use_container_width=True)
        else:
            st.caption(f"Image not found: `{diagram_path}`")

    with col_llm:
        st.subheader("AI Analysis")
        llm = latest.get("llm_report")
        if llm:
            st.write(llm.get("overall_summary", ""))

            positives = llm.get("positive_aspects", [])
            if positives:
                with st.expander("Strengths", expanded=True):
                    for item in positives:
                        st.markdown(
                            f'<span class="da-badge da-badge--ok">✓</span>&nbsp; {item}',
                            unsafe_allow_html=True,
                        )

            progress = llm.get("progress_vs_last_turn", "")
            if progress and progress != "First analysis":
                st.markdown(
                    f'<div class="da-recommendation"><strong>Progress since last version:</strong> {progress}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No AI report available (LLM may be offline).")

    st.divider()

    # ── Metric grid ──────────────────────────────────────────────────────────
    _render_metric_grid(latest, session)

    st.divider()

    # ── Suggestion cards ─────────────────────────────────────────────────────
    _render_suggestion_cards(suggestions, session, features, session.get("diagram_type", "system_design"))


def _render_metric_grid(latest: dict, session: dict) -> None:
    st.subheader("Metric Scores")
    metric_scores = latest.get("metric_scores", {})
    dismissed     = set(session.get("permanently_dismissed", []))

    cols = st.columns(3)
    for i, (key, display) in enumerate(_METRIC_DISPLAY_NAMES.items()):
        score        = metric_scores.get(key)
        is_dismissed = key in dismissed

        if is_dismissed:
            severity = "dismissed"
        elif score is None:
            severity = "n/a"
        elif score >= 80:
            severity = "ok"
        elif score >= 60:
            severity = "warning"
        else:
            severity = "critical"

        with cols[i % 3]:
            st.markdown(_build_metric_card_html(key, display, score, severity), unsafe_allow_html=True)


def _render_suggestion_cards(
    suggestions: list[dict],
    session: dict,
    features: dict,
    diagram_type: str = "system_design",
) -> None:
    dismissed_set = set(session.get("permanently_dismissed", []))
    overrides     = st.session_state["severity_overrides"]
    pending       = st.session_state["pending_severity_changes"]

    non_ok  = [s for s in suggestions if s["severity"] != "ok"]
    passing = [s for s in suggestions if s["severity"] == "ok"]

    if not non_ok and not dismissed_set:
        st.success("All metrics are passing — nothing to fix!")
        return

    st.subheader("Suggestions")
    st.caption(
        "Use the severity dropdowns to disagree with a classification. "
        "Click **Apply changes** to teach the system your preferences and refresh the issue counts."
    )

    # ── Pending banner ────────────────────────────────────────────────────────
    if pending:
        n            = len(pending)
        label_plural = "change" if n == 1 else "changes"
        col_msg, col_btn = st.columns([3, 1])
        with col_msg:
            st.markdown(
                f'<div class="da-banner-pending">You have <strong>{n} pending {label_plural}</strong>. '
                f'Apply to update thresholds and refresh severity counts.</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("Apply changes", type="primary", use_container_width=True):
                for metric_key, change in pending.items():
                    if change.get("score") is not None:
                        threshold_manager.update_threshold(
                            metric_key, change["score"], change["old_sev"], change["new_sev"]
                        )
                    overrides[metric_key] = change["new_sev"]

                if features:
                    new_suggestions = generate_rule_based_suggestions(
                        features, diagram_type, session.get("permanently_dismissed", [])
                    )
                    sr = st.session_state.get("suggestions_result") or {}
                    sr["suggestions"] = new_suggestions
                    st.session_state["suggestions_result"] = sr

                st.session_state["pending_severity_changes"] = {}
                st.toast("Thresholds updated and suggestions refreshed.", icon="✓")
                st.rerun()

    # ── Filter pills ──────────────────────────────────────────────────────────
    crit_n = sum(1 for s in non_ok if s["severity"] == "critical" and s["metric"] not in dismissed_set)
    warn_n = sum(1 for s in non_ok if s["severity"] == "warning"  and s["metric"] not in dismissed_set)
    filt   = st.session_state.get("suggestion_filter", "all")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        if st.button(f"All ({len(non_ok)})", type="primary" if filt == "all" else "secondary", use_container_width=True):
            st.session_state["suggestion_filter"] = "all"; st.rerun()
    with fc2:
        if st.button(f"Critical ({crit_n})", type="primary" if filt == "critical" else "secondary", use_container_width=True):
            st.session_state["suggestion_filter"] = "critical"; st.rerun()
    with fc3:
        if st.button(f"Warning ({warn_n})", type="primary" if filt == "warning" else "secondary", use_container_width=True):
            st.session_state["suggestion_filter"] = "warning"; st.rerun()
    with fc4:
        if st.button(f"Passing ({len(passing)})", type="primary" if filt == "passing" else "secondary", use_container_width=True):
            st.session_state["suggestion_filter"] = "passing"; st.rerun()

    st.write("")

    # ── Passing view ──────────────────────────────────────────────────────────
    if filt == "passing":
        with st.expander(f"Passing metrics ({len(passing)})", expanded=True):
            for s in passing:
                display   = _METRIC_DISPLAY_NAMES.get(s["metric"], s["metric"])
                score_str = f"{s['score']:.1f}" if s.get("score") is not None else "N/A"
                st.markdown(
                    f'<span class="da-badge da-badge--ok">OK</span>&nbsp; <strong>{display}</strong>: {score_str} — {s.get("issue", "")}',
                    unsafe_allow_html=True,
                )
        return

    # ── Non-ok cards ──────────────────────────────────────────────────────────
    visible = non_ok
    if filt == "critical":
        visible = [s for s in non_ok if s["severity"] == "critical" and s["metric"] not in dismissed_set]
    elif filt == "warning":
        visible = [s for s in non_ok if s["severity"] == "warning" and s["metric"] not in dismissed_set]

    for s in visible:
        metric_key    = s["metric"]
        display_name  = _METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
        is_dismissed  = metric_key in dismissed_set
        eff_severity  = overrides.get(metric_key, s["severity"])
        pending_entry = pending.get(metric_key)
        displayed_sev = pending_entry["new_sev"] if pending_entry else eff_severity

        sev_clr_map = {"critical": "var(--color-critical)", "warning": "var(--color-warning)", "ok": "var(--color-ok)"}
        sev_clr     = "var(--color-dismissed)" if is_dismissed else sev_clr_map.get(eff_severity, "var(--color-border-strong)")

        score_str      = f"{s['score']:.1f}" if s.get("score") is not None else "N/A"
        pending_marker = " *(pending)*" if pending_entry else ""

        st.markdown(
            f'<div class="da-suggestion-card" style="--sev-clr:{sev_clr}">'
            f'<div class="da-suggestion-header">'
            f'<span class="da-suggestion-name">{display_name}{pending_marker}</span>'
            f'<span class="da-suggestion-score">{score_str} / 100</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        col_sev, col_action = st.columns([2, 1])

        with col_sev:
            if not is_dismissed:
                new_sev = st.selectbox(
                    "Severity",
                    options=["critical", "warning", "ok"],
                    index=["critical", "warning", "ok"].index(displayed_sev)
                    if displayed_sev in ("critical", "warning", "ok") else 0,
                    key=f"sev_{metric_key}",
                    label_visibility="collapsed",
                )
                if new_sev != eff_severity:
                    if metric_key not in pending:
                        pending[metric_key] = {"score": s.get("score"), "old_sev": eff_severity}
                    pending[metric_key]["new_sev"] = new_sev
                elif metric_key in pending:
                    del pending[metric_key]
            else:
                st.markdown('<span class="da-badge da-badge--dismissed">DISMISSED</span>', unsafe_allow_html=True)

        with col_action:
            if is_dismissed:
                if st.button("Restore", key=f"restore_{metric_key}", use_container_width=True):
                    session["permanently_dismissed"].remove(metric_key)
                    save_session(session, st.session_state["session_path"])
                    st.rerun()
            else:
                if st.button("Dismiss", key=f"dismiss_{metric_key}", use_container_width=True):
                    session["permanently_dismissed"].append(metric_key)
                    save_session(session, st.session_state["session_path"])
                    st.rerun()

        if not is_dismissed:
            st.write(s.get("issue", ""))
            recommendation = s.get("recommendation", "")
            if recommendation:
                st.markdown(
                    f'<div class="da-recommendation"><strong>Recommendation:</strong> {recommendation}</div>',
                    unsafe_allow_html=True,
                )
            locations = s.get("locations", [])
            if locations:
                with st.expander(f"{len(locations)} flagged location(s)"):
                    for loc in locations[:10]:
                        coord  = f"({loc.get('x1','?')},{loc.get('y1','?')})–({loc.get('x2','?')},{loc.get('y2','?')})"
                        detail = loc.get("detail", loc.get("label", ""))
                        st.code(f"{coord}  {detail}")
                    if len(locations) > 10:
                        st.caption(f"… and {len(locations) - 10} more")
        else:
            st.caption("Excluded from composite score. Click Restore to re-enable.")

        st.write("")

    if filt == "all" and passing:
        with st.expander(f"Passing metrics ({len(passing)})"):
            for s in passing:
                display   = _METRIC_DISPLAY_NAMES.get(s["metric"], s["metric"])
                score_str = f"{s['score']:.1f}" if s.get("score") is not None else "N/A"
                st.markdown(
                    f'<span class="da-badge da-badge--ok">OK</span>&nbsp; <strong>{display}</strong>: {score_str} — {s.get("issue", "")}',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab 2 — Version History
# ---------------------------------------------------------------------------

def render_history_tab() -> None:
    import plotly.graph_objects as go

    session  = st.session_state["session"]
    versions = session.get("diagram_versions", []) if session else []

    if len(versions) < 2:
        st.markdown(
            '<div class="da-empty-state">'
            '<div class="da-empty-icon">📈</div>'
            '<div class="da-empty-title">Not enough versions yet</div>'
            '<div class="da-empty-desc">Upload at least two diagram versions to see score trends and side-by-side comparisons.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Score trend ──────────────────────────────────────────────────────────
    st.subheader("Score Trend")
    ver_nums = [v["version"] for v in versions]
    scores   = [v.get("composite_score") or 0 for v in versions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ver_nums, y=scores,
        mode="lines+markers+text",
        text=[f"{s:.1f}" for s in scores],
        textposition="top center",
        textfont=dict(color="#9499b8", size=11),
        line=dict(color="#4f7fff", width=2.5),
        marker=dict(size=8, color="#4f7fff", line=dict(color="#0f1117", width=2)),
        fill="tozeroy",
        fillcolor="rgba(79,127,255,0.08)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui", color="#9499b8", size=12),
        xaxis=dict(title="Version", tickmode="array", tickvals=ver_nums, showgrid=False, linecolor="#2d3250", tickfont=dict(color="#9499b8")),
        yaxis=dict(title="Composite Score", range=[0, 105], showgrid=True, gridcolor="#2d3250", tickfont=dict(color="#9499b8")),
        height=240,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Version selector ─────────────────────────────────────────────────────
    st.subheader("Version Comparison")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        v1_idx = st.selectbox(
            "Compare From",
            options=list(range(len(versions))),
            format_func=lambda i: f"v{versions[i]['version']}  (score {versions[i].get('composite_score', 0):.1f})",
            index=len(versions) - 2,
            key="compare_v1",
        )
    with col_v2:
        v2_idx = st.selectbox(
            "Compare To",
            options=list(range(len(versions))),
            format_func=lambda i: f"v{versions[i]['version']}  (score {versions[i].get('composite_score', 0):.1f})",
            index=len(versions) - 1,
            key="compare_v2",
        )

    v_prev = versions[v1_idx]
    v_curr = versions[v2_idx]

    # ── Side-by-side images ──────────────────────────────────────────────────
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown(f"**Version {v_prev['version']}** — score {v_prev.get('composite_score', 0):.1f}")
        p1 = v_prev.get("diagram_path", "")
        if Path(p1).exists():
            st.image(p1, use_container_width=True)
        else:
            st.caption(f"Image not found: `{p1}`")
    with col_img2:
        st.markdown(f"**Version {v_curr['version']}** — score {v_curr.get('composite_score', 0):.1f}")
        p2 = v_curr.get("diagram_path", "")
        if Path(p2).exists():
            st.image(p2, use_container_width=True)
        else:
            st.caption(f"Image not found: `{p2}`")

    st.divider()

    # ── Metric diff table ────────────────────────────────────────────────────
    st.subheader("Metric Changes")
    prev_scores = v_prev.get("metric_scores", {})
    curr_scores = v_curr.get("metric_scores", {})

    rows = []
    for key, display in _METRIC_DISPLAY_NAMES.items():
        p = prev_scores.get(key)
        c = curr_scores.get(key)
        if p is None and c is None:
            continue
        delta       = (c - p) if (p is not None and c is not None) else None
        significant = delta is not None and abs(delta) > 10
        rows.append({"display": display, "prev": f"{p:.1f}" if p is not None else "N/A",
                     "curr": f"{c:.1f}" if c is not None else "N/A", "delta": delta, "significant": significant})

    def _fmt_delta(d: float | None) -> str:
        if d is None:
            return "—"
        if abs(d) < 0.5:
            return "→ unchanged"
        return f"▲ +{d:.1f}" if d > 0 else f"▼ {d:.1f}"

    df = pd.DataFrame([
        {
            "Metric":               r["display"] + (" ⚠" if r["significant"] else ""),
            f"v{v_prev['version']}": r["prev"],
            f"v{v_curr['version']}": r["curr"],
            "Change":               _fmt_delta(r["delta"]),
        }
        for r in rows
    ])

    def _colour_change(val: str) -> str:
        if str(val).startswith("▲"):
            return "color: #3dd68c"
        if str(val).startswith("▼"):
            return "color: #ff5370"
        return "color: #9499b8"

    styled = df.style.map(_colour_change, subset=["Change"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Significant changes ──────────────────────────────────────────────────
    prev_sev = {s["metric"]: s["severity"] for s in v_prev.get("suggestions", [])}
    curr_sev = {s["metric"]: s["severity"] for s in v_curr.get("suggestions", [])}

    new_criticals   = [_METRIC_DISPLAY_NAMES[m] for m in _METRIC_DISPLAY_NAMES if curr_sev.get(m) == "critical" and prev_sev.get(m) != "critical"]
    fixed_criticals = [_METRIC_DISPLAY_NAMES[m] for m in _METRIC_DISPLAY_NAMES if prev_sev.get(m) == "critical" and curr_sev.get(m) != "critical"]
    big_changes     = [r for r in rows if r["significant"]]

    if new_criticals or fixed_criticals or big_changes:
        items_html = ""
        for name in new_criticals:
            items_html += f'<div class="da-changes-item da-changes-item--down">▼ New critical: {name}</div>'
        for name in fixed_criticals:
            items_html += f'<div class="da-changes-item da-changes-item--up">▲ Fixed critical: {name}</div>'
        for r in big_changes:
            if r["delta"] is None:
                continue
            if r["delta"] > 0:
                items_html += f'<div class="da-changes-item da-changes-item--up">▲ {r["display"]}: +{r["delta"]:.1f} pts ({r["prev"]} → {r["curr"]})</div>'
            else:
                items_html += f'<div class="da-changes-item da-changes-item--down">▼ {r["display"]}: {r["delta"]:.1f} pts ({r["prev"]} → {r["curr"]})</div>'

        st.markdown(
            f'<div class="da-changes-summary"><div class="da-changes-title">Significant Changes</div>{items_html}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 3 — Chat
# ---------------------------------------------------------------------------

def render_chat_tab() -> None:
    session  = st.session_state["session"]
    features = st.session_state.get("features", {})

    if not _has_analysis():
        st.markdown(
            '<div class="da-empty-state">'
            '<div class="da-empty-icon">💬</div>'
            '<div class="da-empty-title">No analysis yet</div>'
            '<div class="da-empty-desc">Upload and analyse a diagram first to start chatting. Switch to the Analysis tab to get started.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    versions      = session["diagram_versions"]
    latest        = versions[-1]
    current_suggs = latest.get("suggestions", [])
    current_score = _live_score()
    diagram_type  = session.get("diagram_type", "system_design")

    # ── Quick-action chips ────────────────────────────────────────────────────
    prefill = None
    chip_cols = st.columns(4)
    chip_defs = [
        ("What should I fix?",  "What should I fix first?"),
        ("Top critical issue",  "What is the most critical issue?"),
        ("Explain scoring",     "How is the overall score calculated?"),
        ("List dismissed",      "Show me all dismissed suggestions"),
    ]
    for col, (label, text) in zip(chip_cols, chip_defs):
        with col:
            if st.button(label, use_container_width=True, key=f"chip_{label}"):
                prefill = text

    st.write("")

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in session.get("chat_history", []):
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    user_input = prefill or st.chat_input("Ask about your diagram…")
    if not user_input:
        return

    with st.chat_message("user"):
        st.write(user_input)

    route  = route_message(user_input)
    intent = route["intent"]
    data   = route["data"]
    reply  = ""

    if intent == "exit":
        reply = "Use the sidebar or close the browser tab to exit."

    elif intent == "help":
        reply = (
            "Available commands:\n"
            "- **what should I fix first?** — top priority issue\n"
            "- **dismiss [metric name]** — exclude a metric from scoring\n"
            "- **restore [metric name]** — re-enable a dismissed metric\n"
            "- **/path/to/image.png** — analyse a new diagram version\n"
            "- Or just ask any question about your diagram."
        )

    elif intent == "new_image":
        with st.spinner("Analysing new version…"):
            try:
                suggestions_result, new_features = _analyze_new_version(
                    data["path"], session, st.session_state["session_path"],
                )
                st.session_state["features"]           = new_features
                st.session_state["suggestions_result"] = suggestions_result
                v_num     = len(session["diagram_versions"])
                new_score = suggestions_result["composite_score"]
                reply = (
                    f"Version {v_num} analysed. Composite score: **{new_score:.1f}**. "
                    "Switch to the Analysis or Version History tab to see the results."
                )
            except Exception as exc:
                reply = f"Failed to analyse image: {exc}"

    elif intent == "priority":
        llm_report = latest.get("llm_report")
        if llm_report and llm_report.get("priority_issues"):
            top     = llm_report["priority_issues"][0]
            display = _METRIC_DISPLAY_NAMES.get(top.get("metric", ""), top.get("metric", ""))
            reply   = (
                f"Your top priority is **{display}** [{top.get('severity', '').upper()}]: "
                f"{top.get('what', '')}. **Fix:** {top.get('how_to_fix', '')}"
            )
        else:
            non_ok = [s for s in current_suggs if s["severity"] != "ok"]
            if non_ok:
                top     = non_ok[0]
                display = _METRIC_DISPLAY_NAMES.get(top["metric"], top["metric"])
                reply   = (
                    f"Your top priority is **{display}** [{top['severity'].upper()}]: "
                    f"{top['issue']}. **Fix:** {top['recommendation']}"
                )
            else:
                reply = "No issues remain — all metrics are passing."

    elif intent == "dismiss":
        metric  = data["metric"]
        display = _METRIC_DISPLAY_NAMES[metric]
        if metric not in session["permanently_dismissed"]:
            session["permanently_dismissed"].append(metric)
            new_score = _compute_composite_score(features, session["permanently_dismissed"])
            save_session(session, st.session_state["session_path"])
            reply = f"**{display}** dismissed. Composite score is now **{new_score:.1f}**."
        else:
            reply = f"**{display}** is already dismissed."

    elif intent == "restore":
        metric  = data["metric"]
        display = _METRIC_DISPLAY_NAMES[metric]
        if metric in session["permanently_dismissed"]:
            session["permanently_dismissed"].remove(metric)
            new_score = _compute_composite_score(features, session["permanently_dismissed"])
            save_session(session, st.session_state["session_path"])
            reply = f"**{display}** restored. Composite score is now **{new_score:.1f}**."
        else:
            reply = f"**{display}** is not currently dismissed."

    else:
        with st.spinner("Thinking…"):
            result = chat_with_llm(user_input, session, current_suggs, current_score, diagram_type)
        reply  = result["reply"]
        action = result.get("action", {"type": "none"})
        if action.get("type") != "none":
            _apply_llm_action(action, session, features)
            save_session(session, st.session_state["session_path"])

    session["chat_history"].append({"role": "user",      "content": user_input})
    session["chat_history"].append({"role": "assistant",  "content": reply})
    save_session(session, st.session_state["session_path"])

    intent_label = intent.replace("_", " ")
    with st.chat_message("assistant"):
        st.markdown(
            f'<span class="da-badge--intent">{intent_label}</span> {reply}',
            unsafe_allow_html=True,
        )

    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Diagram Analyser",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_global_css()
    _init_state()
    render_sidebar()

    if not _has_session():
        st.markdown("## Welcome to Diagram Analyser")
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div class="da-welcome-card">'
                '<div class="da-welcome-icon">🔁</div>'
                '<div class="da-welcome-title">Iterating on a diagram?</div>'
                '<div class="da-welcome-desc">Create a new session and upload your diagram. Each time you upload '
                'a revised image it is saved as a new version, letting you track your score improvements over time.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                '<div class="da-welcome-card">'
                '<div class="da-welcome-icon">📂</div>'
                '<div class="da-welcome-title">Switching to a different diagram?</div>'
                '<div class="da-welcome-desc">Your past sessions are listed in the sidebar automatically — just click '
                '<strong>Resume</strong> to jump back in. No file paths or JSON uploads needed.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        return

    tab1, tab2, tab3 = st.tabs(["Analysis", "Version History", "Chat"])

    with tab1:
        render_analysis_tab()
    with tab2:
        render_history_tab()
    with tab3:
        render_chat_tab()


if __name__ == "__main__":
    main()
