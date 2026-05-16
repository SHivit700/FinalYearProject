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
    "critical": "#dc3545",
    "warning":  "#fd7e14",
    "ok":       "#28a745",
}


_DIAGRAMS_DIR = _SRC_DIR / "data" / "diagrams"
_UPLOADS_DIR  = _SRC_DIR / "data" / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_DIAGRAM_TYPE_LABELS = {
    "system_design":    "System Design",
    "timeline_roadmap": "Timeline / Roadmap",
}


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
        "setup_mode":               "new",   # "new" | "load"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


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
# Sidebar — session management + image upload
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.title("Diagram Analyser")
        st.divider()

        if not _has_session():
            _render_session_init()
        else:
            _render_session_info()
            st.divider()
            _render_image_uploader()
            st.divider()
            _render_download_button()


def _render_session_init() -> None:
    mode = st.radio(
        "Start",
        options=["New session", "Load existing session"],
        key="setup_mode_radio",
        horizontal=True,
    )

    if mode == "New session":
        dtype = st.selectbox(
            "Diagram type",
            options=list(_DIAGRAM_TYPE_LABELS.keys()),
            format_func=lambda k: _DIAGRAM_TYPE_LABELS[k],
        )
        if st.button("Start New Session", use_container_width=True):
            session = new_session(dtype)
            st.session_state["session"]      = session
            st.session_state["session_path"] = None
            st.session_state["severity_overrides"] = {}
            st.rerun()

    else:  # Load existing session
        uploaded_json = st.file_uploader("Upload session JSON", type=["json"])
        path_input    = st.text_input("Or enter session file path")

        if st.button("Load Session", use_container_width=True):
            try:
                if uploaded_json:
                    raw  = json.loads(uploaded_json.read())
                    path = str(_SRC_DIR / "data" / "sessions" / Path(uploaded_json.name).name)
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(raw, f, indent=2)
                    session = raw
                elif path_input.strip():
                    path    = path_input.strip()
                    session = load_session(path)
                else:
                    st.error("Provide a JSON file or a file path.")
                    return

                st.session_state["session"]      = session
                st.session_state["session_path"] = path
                st.session_state["severity_overrides"] = {}
                st.success("Session loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load session: {exc}")


def _render_session_info() -> None:
    session = st.session_state["session"]
    st.caption(f"**Session:** `{session['session_id']}`")
    st.caption(f"**Type:** {_DIAGRAM_TYPE_LABELS.get(session['diagram_type'], session['diagram_type'])}")
    n = len(session.get("diagram_versions", []))
    st.caption(f"**Versions analysed:** {n}")
    dismissed = session.get("permanently_dismissed", [])
    if dismissed:
        names = ", ".join(_METRIC_DISPLAY_NAMES.get(m, m) for m in dismissed)
        st.caption(f"**Dismissed:** {names}")


def _render_image_uploader() -> None:
    session = st.session_state["session"]

    st.markdown("**Upload diagram**")
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
        st.session_state["last_file_id"] = uploaded.file_id  # set before rerun fires
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

    with st.spinner("Analysing diagram… this may take up to a minute"):
        try:
            suggestions_result, features = _analyze_new_version(
                image_path_str,
                session,
                st.session_state["session_path"],
            )
        except Exception as exc:
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
        label="Download session JSON",
        data=blob,
        file_name=f"{session['session_id']}.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Shared badge helper
# ---------------------------------------------------------------------------

def _badge(label: str, bg: str) -> str:
    return (
        f"<span style='background:{bg};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.75rem;font-weight:600'>{label}</span>"
    )


# ---------------------------------------------------------------------------
# Tab 1 — Analysis
# ---------------------------------------------------------------------------

def render_analysis_tab() -> None:
    session  = st.session_state["session"]
    features = st.session_state.get("features", {})

    if not _has_analysis():
        st.info("Upload a diagram from the sidebar to start.")
        return

    versions = session["diagram_versions"]
    latest   = versions[-1]
    prev     = versions[-2] if len(versions) >= 2 else None

    live_score = _live_score()

    # ── Hero banner ──────────────────────────────────────────────────────────
    # Prefer live suggestions_result (regenerated after threshold updates) over
    # the frozen copy stored inside the session JSON.
    _sr = st.session_state.get("suggestions_result")
    suggestions = _sr["suggestions"] if _sr and "suggestions" in _sr else latest.get("suggestions", [])
    non_ok      = [s for s in suggestions if s["severity"] != "ok"]
    dismissed   = session.get("permanently_dismissed", [])

    critical_count = sum(
        1 for s in non_ok
        if s["severity"] == "critical" and s["metric"] not in dismissed
    )
    warning_count = sum(
        1 for s in non_ok
        if s["severity"] == "warning" and s["metric"] not in dismissed
    )

    prev_score = prev["composite_score"] if prev else None
    delta_val  = (live_score - prev_score) if (live_score is not None and prev_score is not None) else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Composite Score",
            f"{live_score:.1f} / 100" if live_score is not None else "N/A",
            delta=f"{delta_val:+.1f}" if delta_val is not None else None,
            delta_color="normal",
        )
    with col2:
        st.metric("Critical Issues", critical_count)
    with col3:
        st.metric("Warnings", warning_count)
    with col4:
        st.metric("Dismissed Metrics", len(dismissed))

    st.divider()

    # ── Image + LLM report ──────────────────────────────────────────────────
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
                        st.success(item)

            progress = llm.get("progress_vs_last_turn", "")
            if progress and progress != "First analysis":
                st.info(f"**Progress since last version:** {progress}")
        else:
            st.caption("No AI report available (LLM may be offline).")

    st.divider()

    # ── Metric score grid ───────────────────────────────────────────────────
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
        score = metric_scores.get(key)
        is_dismissed = key in dismissed

        if is_dismissed:
            bg, label = "#6c757d", "DISMISSED"
        elif score is None:
            bg, label = "#6c757d", "N/A"
        elif score >= 80:
            bg, label = _SEVERITY_COLORS["ok"], "OK"
        elif score >= 60:
            bg, label = _SEVERITY_COLORS["warning"], "WARNING"
        else:
            bg, label = _SEVERITY_COLORS["critical"], "CRITICAL"

        with cols[i % 3]:
            st.markdown(
                f"**{display}** &nbsp; {_badge(label, bg)}",
                unsafe_allow_html=True,
            )
            if score is not None:
                st.progress(int(score), text=f"{score:.1f} / 100")
            else:
                st.caption("No score")
            st.write("")  # spacing


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

    # ── Apply-changes banner ──────────────────────────────────────────────────
    if pending:
        n = len(pending)
        label_plural = "change" if n == 1 else "changes"
        col_msg, col_btn = st.columns([3, 1])
        with col_msg:
            st.info(f"You have **{n} pending {label_plural}**. Apply to update thresholds and refresh severity counts.")
        with col_btn:
            if st.button("Apply changes", type="primary", use_container_width=True):
                for metric_key, change in pending.items():
                    if change.get("score") is not None:
                        threshold_manager.update_threshold(
                            metric_key, change["score"], change["old_sev"], change["new_sev"]
                        )
                    overrides[metric_key] = change["new_sev"]

                # Regenerate suggestions with updated thresholds so severity counts refresh
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

    st.write("")

    for s in non_ok:
        metric_key   = s["metric"]
        display_name = _METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
        is_dismissed = metric_key in dismissed_set
        # eff_severity = applied override (or original), pending overrides just for display
        eff_severity  = overrides.get(metric_key, s["severity"])
        pending_entry = pending.get(metric_key)
        displayed_sev = pending_entry["new_sev"] if pending_entry else eff_severity

        if is_dismissed:
            border_color = "#6c757d"
        else:
            border_color = _SEVERITY_COLORS.get(eff_severity, "#6c757d")

        st.markdown(
            f"<div style='border-left:4px solid {border_color};"
            "padding:8px 12px;margin-bottom:4px;border-radius:0 4px 4px 0;"
            "background:rgba(0,0,0,0.02)'>",
            unsafe_allow_html=True,
        )

        col_name, col_sev, col_action = st.columns([3, 1.5, 1.2])

        with col_name:
            score_str = f"{s['score']:.1f}" if s.get("score") is not None else "N/A"
            pending_marker = " *(pending)*" if pending_entry else ""
            st.markdown(f"**{display_name}** — score: {score_str}{pending_marker}")

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
                # Track pending change without applying yet
                if new_sev != eff_severity:
                    if metric_key not in pending:
                        pending[metric_key] = {"score": s.get("score"), "old_sev": eff_severity}
                    pending[metric_key]["new_sev"] = new_sev
                elif metric_key in pending:
                    # User reverted back to original — drop the pending entry
                    del pending[metric_key]
            else:
                st.markdown(_badge("DISMISSED", "#6c757d"), unsafe_allow_html=True)

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
                st.info(f"**Recommendation:** {recommendation}")

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

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

    if passing:
        with st.expander(f"Passing metrics ({len(passing)})"):
            for s in passing:
                display = _METRIC_DISPLAY_NAMES.get(s["metric"], s["metric"])
                score_str = f"{s['score']:.1f}" if s.get("score") is not None else "N/A"
                st.success(f"**{display}:** {score_str}  —  {s.get('issue', '')}")


# ---------------------------------------------------------------------------
# Tab 2 — Version History
# ---------------------------------------------------------------------------

def render_history_tab() -> None:
    import plotly.graph_objects as go

    session  = st.session_state["session"]
    versions = session.get("diagram_versions", []) if session else []

    if len(versions) < 2:
        st.info("Upload at least two diagram versions to see a comparison.")
        return

    # ── Score trend chart ────────────────────────────────────────────────────
    st.subheader("Score Trend")
    ver_nums = [v["version"] for v in versions]
    scores   = [v.get("composite_score") or 0 for v in versions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ver_nums,
        y=scores,
        mode="lines+markers+text",
        text=[f"{s:.1f}" for s in scores],
        textposition="top center",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=10),
    ))
    fig.update_layout(
        xaxis=dict(title="Version", tickmode="array", tickvals=ver_nums),
        yaxis=dict(title="Composite Score", range=[0, 105]),
        height=280,
        margin=dict(l=40, r=20, t=30, b=40),
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
        delta = (c - p) if (p is not None and c is not None) else None
        significant = delta is not None and abs(delta) > 10
        rows.append({
            "display":     display,
            "prev":        f"{p:.1f}" if p is not None else "N/A",
            "curr":        f"{c:.1f}" if c is not None else "N/A",
            "delta":       delta,
            "significant": significant,
        })

    hdr1, hdr2, hdr3, hdr4 = st.columns([3, 1, 1, 2])
    hdr1.markdown("**Metric**")
    hdr2.markdown(f"**v{v_prev['version']}**")
    hdr3.markdown(f"**v{v_curr['version']}**")
    hdr4.markdown("**Change**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for row in rows:
        delta = row["delta"]
        if delta is None:
            change_html = "<span style='color:#6c757d'>—</span>"
        elif abs(delta) < 0.5:
            change_html = "<span style='color:#6c757d'>→ unchanged</span>"
        elif delta > 0:
            change_html = f"<span style='color:green'>▲ +{delta:.1f}</span>"
        else:
            change_html = f"<span style='color:#dc3545'>▼ {delta:.1f}</span>"

        label = row["display"]
        if row["significant"]:
            label = f"**{label} ⚠**"

        c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
        c1.markdown(label)
        c2.write(row["prev"])
        c3.write(row["curr"])
        c4.markdown(change_html, unsafe_allow_html=True)

    st.divider()

    # ── Significant changes callout ──────────────────────────────────────────
    prev_sev = {s["metric"]: s["severity"] for s in v_prev.get("suggestions", [])}
    curr_sev = {s["metric"]: s["severity"] for s in v_curr.get("suggestions", [])}

    new_criticals   = [_METRIC_DISPLAY_NAMES[m] for m in _METRIC_DISPLAY_NAMES
                       if curr_sev.get(m) == "critical" and prev_sev.get(m) != "critical"]
    fixed_criticals = [_METRIC_DISPLAY_NAMES[m] for m in _METRIC_DISPLAY_NAMES
                       if prev_sev.get(m) == "critical" and curr_sev.get(m) != "critical"]
    big_changes     = [r for r in rows if r["significant"]]

    if new_criticals or fixed_criticals or big_changes:
        st.subheader("Significant Changes")
        if new_criticals:
            st.error(f"New critical issues: {', '.join(new_criticals)}")
        if fixed_criticals:
            st.success(f"Critical issues resolved: {', '.join(fixed_criticals)}")
        for r in big_changes:
            delta = r["delta"]
            if delta is None:
                continue
            if delta > 0:
                st.success(f"{r['display']}: improved by {delta:+.1f} pts ({r['prev']} → {r['curr']})")
            else:
                st.warning(f"{r['display']}: regressed by {delta:.1f} pts ({r['prev']} → {r['curr']})")


# ---------------------------------------------------------------------------
# Tab 3 — Chat
# ---------------------------------------------------------------------------

def render_chat_tab() -> None:
    session  = st.session_state["session"]
    features = st.session_state.get("features", {})

    if not _has_analysis():
        st.info("Upload and analyse a diagram first to start chatting.")
        return

    versions         = session["diagram_versions"]
    latest           = versions[-1]
    current_suggs    = latest.get("suggestions", [])
    current_score    = _live_score()
    diagram_type     = session.get("diagram_type", "system_design")

    # Render history
    for msg in session.get("chat_history", []):
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your diagram…")
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
                    data["path"],
                    session,
                    st.session_state["session_path"],
                )
                st.session_state["features"]           = new_features
                st.session_state["suggestions_result"] = suggestions_result
                v_num      = len(session["diagram_versions"])
                new_score  = suggestions_result["composite_score"]
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
                f"{top.get('what', '')}. "
                f"**Fix:** {top.get('how_to_fix', '')}"
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

    else:  # "chat" or "needs_metric" → LLM
        with st.spinner("Thinking…"):
            result = chat_with_llm(
                user_input,
                session,
                current_suggs,
                current_score,
                diagram_type,
            )
        reply = result["reply"]
        action = result.get("action", {"type": "none"})
        if action.get("type") != "none":
            _apply_llm_action(action, session, features)
            save_session(session, st.session_state["session_path"])

    session["chat_history"].append({"role": "user",      "content": user_input})
    session["chat_history"].append({"role": "assistant",  "content": reply})
    save_session(session, st.session_state["session_path"])

    with st.chat_message("assistant"):
        st.write(reply)

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

    _init_state()
    render_sidebar()

    if not _has_session():
        st.markdown(
            "## Welcome to Diagram Analyser\n\n"
            "Use the sidebar to **start a new session** or **load an existing one**, "
            "then upload your diagram to get started."
        )
        return

    tab_labels = ["Analysis", "Version History", "Chat"]
    tab1, tab2, tab3 = st.tabs(tab_labels)

    with tab1:
        render_analysis_tab()
    with tab2:
        render_history_tab()
    with tab3:
        render_chat_tab()


if __name__ == "__main__":
    main()
