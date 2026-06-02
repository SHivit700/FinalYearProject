"""
FastAPI server exposing the Python analysis pipeline as REST endpoints.

Run with (from project root):
    uvicorn src.api:app --reload --port 8000

Or (from src/ directory):
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analyze_diagram import _analyze_new_version, _apply_llm_action
from chat_router import route_message
from suggestion_engine import (
    _SESSIONS_DIR,
    chat_with_llm,
    load_session,
    new_session,
    save_session,
)
import threshold_manager

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Diagram Analyser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UPLOADS_DIR = _SRC_DIR / "data" / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_analysis_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Metric mapping
# ---------------------------------------------------------------------------

PYTHON_KEY_TO_REACT_NAME: dict[str, str] = {
    "label_readability":       "Label Readability",
    "label_area":              "Label Area",
    "overlap_metrics":         "Overlap (Crowding)",
    "edge_clearance":          "Edge Clearance",
    "font_hierarchy":          "Font Hierarchy",
    "container_utilization":   "Container Utilisation",
    "isolated_boxes":          "Isolated Boxes",
    "brevity":                 "Brevity",
    "whitespace_distribution": "Whitespace Distribution",
    "color_harmony":           "Color Harmony",
    "label_contrast":          "Label Contrast",
    "cognitive_chunk_density": "Cognitive Chunk Density",
    "orientation_consistency": "Orientation Consistency",
}

REACT_NAME_TO_PYTHON_KEY: dict[str, str] = {v: k for k, v in PYTHON_KEY_TO_REACT_NAME.items()}


def _map_severity_to_react(python_sev: str) -> str:
    return "pass" if python_sev == "ok" else python_sev


def _map_severity_to_python(react_sev: str) -> str:
    return "ok" if react_sev == "pass" else react_sev


def _convert_locations(
    locations: list[dict], image_shape: list | tuple
) -> list[dict]:
    H, W = image_shape[0], image_shape[1]
    if H == 0 or W == 0:
        return []
    result = []
    for loc in locations:
        x1 = loc.get("x1", 0)
        y1 = loc.get("y1", 0)
        x2 = loc.get("x2", x1)
        y2 = loc.get("y2", y1)
        result.append({
            "x":      round(x1 / W * 100, 2),
            "y":      round(y1 / H * 100, 2),
            "width":  round((x2 - x1) / W * 100, 2),
            "height": round((y2 - y1) / H * 100, 2),
        })
    return result


def _build_llm_analysis_index(llm_report: dict) -> dict[str, dict]:
    """Build a lookup from Python metric key -> {where, howToFix} from priority_issues."""
    index: dict[str, dict] = {}
    for issue in (llm_report or {}).get("priority_issues", []):
        py_key = issue.get("metric", "")
        where = issue.get("where", "")
        how_to_fix = issue.get("how_to_fix", "")
        if py_key and (where or how_to_fix):
            index[py_key] = {"where": where, "howToFix": how_to_fix}
    return index


def _suggestion_to_metric_result(
    suggestion: dict,
    image_shape: list | tuple,
    dismissed_py_keys: list[str],
    llm_analysis_index: dict[str, dict] | None = None,
    llm_regions: dict[str, list[dict]] | None = None,
) -> dict:
    py_key = suggestion["metric"]
    react_name = PYTHON_KEY_TO_REACT_NAME.get(py_key, py_key)
    score = suggestion.get("score")
    result: dict = {
        "name":             react_name,
        "score":            round(score) if score is not None else 0,
        "severity":         _map_severity_to_react(suggestion.get("severity", "ok")),
        "description":      suggestion.get("issue", ""),
        "recommendation":   suggestion.get("recommendation", ""),
        "flaggedLocations": _convert_locations(suggestion.get("locations", []), image_shape),
        "llmRegions":       (llm_regions or {}).get(py_key, []),
        "isDismissed":      py_key in dismissed_py_keys,
    }
    if llm_analysis_index:
        llm_data = llm_analysis_index.get(py_key)
        if llm_data:
            result["llmAnalysis"] = llm_data
    if py_key == "whitespace_distribution" and not result["flaggedLocations"]:
        result["flaggedLocations"] = [{"x": 0, "y": 0, "width": 100, "height": 100}]
    if suggestion.get("palette_colors"):
        result["paletteColors"] = suggestion["palette_colors"]
    if py_key == "cognitive_chunk_density" and suggestion.get("chunk_centroids"):
        H, W = image_shape[0], image_shape[1]
        result["chunkCentroids"] = [
            {
                "cx": round(c["cx"] / W * 100, 2),
                "cy": round(c["cy"] / H * 100, 2),
                "displayLabel": c["displayLabel"],
            }
            for c in suggestion["chunk_centroids"]
        ]
    return result


def _load_image_as_base64(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return ""
        data = p.read_bytes()
        suffix = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "gif": "gif", "webp": "webp"}.get(suffix, "png")
        return f"data:image/{mime};base64," + base64.b64encode(data).decode()
    except Exception:
        return ""


def _version_to_analysis_result(version: dict, session: dict) -> dict:
    dismissed_py = session.get("permanently_dismissed", [])
    image_shape = version.get("_image_shape", [1000, 1000, 3])

    llm = version.get("llm_report") or {}
    llm_analysis_index = _build_llm_analysis_index(llm)
    llm_regions_map: dict = version.get("llm_regions") or {}

    metrics = [
        _suggestion_to_metric_result(s, image_shape, dismissed_py, llm_analysis_index, llm_regions_map)
        for s in version.get("suggestions", [])
    ]

    active = [m for m in metrics if not m["isDismissed"]]
    critical_count = sum(1 for m in active if m["severity"] == "critical")
    warning_count  = sum(1 for m in active if m["severity"] == "warning")

    ai_narrative = llm.get("overall_summary") or None

    return {
        "version":        version["version"],
        "timestamp":      version.get("analyzed_at", datetime.now().isoformat()),
        "imagePath":      Path(version.get("diagram_path", "")).name,
        "imageData":      _load_image_as_base64(version.get("diagram_path", "")),
        "compositeScore": round(version.get("composite_score") or 0),
        "metrics":        metrics,
        "criticalCount":  critical_count,
        "warningCount":   warning_count,
        "aiNarrative":    ai_narrative,
    }


def _python_to_react_diagram_type(py_type: str) -> str:
    return py_type.replace("_", "-")


def _react_to_python_diagram_type(react_type: str) -> str:
    return react_type.replace("-", "_")


def _python_session_to_react(session: dict) -> dict:
    return {
        "id":               session["session_id"],
        "name":             session.get("name", session["session_id"]),
        "diagramType":      _python_to_react_diagram_type(session["diagram_type"]),
        "createdAt":        session.get("created_at", datetime.now().isoformat()),
        "updatedAt":        session.get("updated_at", datetime.now().isoformat()),
        "versions":         [_version_to_analysis_result(v, session)
                             for v in session.get("diagram_versions", [])],
        "dismissedMetrics": [PYTHON_KEY_TO_REACT_NAME.get(k, k)
                             for k in session.get("permanently_dismissed", [])],
        "customThresholds": {},
    }


def _find_session_path(session_id: str) -> Path | None:
    # Fast path: API-created sessions use session_id as the filename stem
    fast = _SESSIONS_DIR / f"{session_id}.json"
    if fast.exists():
        return fast
    # Slow path: scan Streamlit-created session files
    for p in _SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("session_id") == session_id:
                return p
        except Exception:
            continue
    return None


def _load_session_by_id(session_id: str) -> tuple[dict, Path]:
    path = _find_session_path(session_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return load_session(str(path)), path


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: str
    diagramType: str


class UpdateSessionRequest(BaseModel):
    dismissedMetrics: list[str] = []
    customThresholds: dict = {}


class ChatRequest(BaseModel):
    message: str


class SeverityUpdateRequest(BaseModel):
    oldSeverity: str
    newSeverity: str
    currentScore: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions() -> list[dict]:
    result = []
    for p in sorted(_SESSIONS_DIR.glob("*.json")):
        try:
            session = json.loads(p.read_text(encoding="utf-8"))
            result.append(_python_session_to_react(session))
        except Exception:
            continue
    return result


@app.post("/api/sessions")
async def create_session(body: CreateSessionRequest) -> dict:
    py_type = _react_to_python_diagram_type(body.diagramType)
    session = new_session(py_type)
    session["name"] = body.name
    now = datetime.now().isoformat()
    session["created_at"] = now
    session["updated_at"] = now
    session_path = _SESSIONS_DIR / f"{session['session_id']}.json"
    save_session(session, str(session_path))
    return _python_session_to_react(session)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session, _ = _load_session_by_id(session_id)
    return _python_session_to_react(session)


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSessionRequest) -> dict:
    session, path = _load_session_by_id(session_id)
    session["permanently_dismissed"] = [
        REACT_NAME_TO_PYTHON_KEY.get(name, name)
        for name in body.dismissedMetrics
    ]
    session["updated_at"] = datetime.now().isoformat()
    save_session(session, str(path))
    return _python_session_to_react(session)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    path = _find_session_path(session_id)
    if path and path.exists():
        path.unlink()
    return {"ok": True}


@app.post("/api/sessions/{session_id}/analyze")
async def analyze(session_id: str, file: UploadFile = File(...)) -> dict:
    session, session_path = _load_session_by_id(session_id)

    # Save uploaded file to disk
    safe_name = f"{uuid.uuid4().hex}_{file.filename or 'image.png'}"
    image_path = _UPLOADS_DIR / safe_name
    image_path.write_bytes(await file.read())

    # Run analysis in thread pool — EasyOCR/PyTorch blocks the event loop
    async with _analysis_lock:
        loop = asyncio.get_event_loop()
        try:
            suggestions_result, features = await loop.run_in_executor(
                None,
                partial(_analyze_new_version, str(image_path), session, str(session_path)),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # Reload session — _analyze_new_version saved it with the new version appended
    session, reloaded_path = _load_session_by_id(session_id)
    version_record = session["diagram_versions"][-1]

    # Persist image_shape so historical coordinate normalisation stays accurate
    image_shape = features.get("image_shape", (1000, 1000, 3))
    if isinstance(image_shape, tuple):
        image_shape = list(image_shape)
    version_record["_image_shape"] = image_shape
    session["updated_at"] = datetime.now().isoformat()
    save_session(session, str(reloaded_path))

    return _version_to_analysis_result(version_record, session)


@app.post("/api/sessions/{session_id}/chat")
async def chat(session_id: str, body: ChatRequest) -> dict:
    session, path = _load_session_by_id(session_id)

    route = route_message(body.message)
    intent = route.get("intent", "chat")
    data   = route.get("data", {})

    versions = session.get("diagram_versions", [])
    current_suggestions = versions[-1].get("suggestions", []) if versions else []
    composite_score     = versions[-1].get("composite_score") if versions else None

    action: dict[str, Any] = {"type": "none", "metric": None}

    if intent in ("exit", "help"):
        reply = (
            "I can help you understand your diagram quality scores, explain specific "
            "metrics, or guide you through improvements. What would you like to know?"
        )
    elif intent == "priority":
        non_ok = [s for s in current_suggestions if s.get("severity") != "ok"]
        if non_ok:
            top = non_ok[0]
            display = PYTHON_KEY_TO_REACT_NAME.get(top["metric"], top["metric"])
            reply = (
                f"The top priority is **{display}** "
                f"(score={top['score']}): {top['issue']} "
                f"— {top['recommendation']}"
            )
        else:
            reply = "All metrics are passing — no critical issues to address right now."
    elif intent == "dismiss":
        metric_key = data.get("metric")
        dismissed = session.setdefault("permanently_dismissed", [])
        if metric_key and metric_key not in dismissed:
            dismissed.append(metric_key)
        display = PYTHON_KEY_TO_REACT_NAME.get(metric_key, metric_key) if metric_key else "that metric"
        reply = f"Dismissed **{display}**. It will no longer affect your composite score."
        action = {"type": "dismiss", "metric": metric_key}
    elif intent == "restore":
        metric_key = data.get("metric")
        dismissed = session.setdefault("permanently_dismissed", [])
        if metric_key and metric_key in dismissed:
            dismissed.remove(metric_key)
        display = PYTHON_KEY_TO_REACT_NAME.get(metric_key, metric_key) if metric_key else "that metric"
        reply = f"Restored **{display}**. It is now included in your composite score again."
        action = {"type": "restore", "metric": metric_key}
    elif intent == "needs_metric":
        action_word = data.get("action", "dismiss")
        reply = (
            f"I understand you want to {action_word} a metric, but I couldn't identify which one. "
            "Please name the metric — for example: 'ignore Label Contrast' or 'restore Brevity'."
        )
    else:
        result = chat_with_llm(
            body.message, session, current_suggestions, composite_score, session["diagram_type"]
        )
        reply  = result["reply"]
        action = result.get("action", {"type": "none", "metric": None})
        _apply_llm_action(action, session, {})

    session.setdefault("chat_history", []).extend([
        {"role": "user",      "content": body.message},
        {"role": "assistant", "content": reply},
    ])
    session["updated_at"] = datetime.now().isoformat()
    save_session(session, str(path))

    return {"reply": reply, "action": action}


@app.patch("/api/sessions/{session_id}/metrics/{metric_key}/severity")
async def update_metric_severity(
    session_id: str,
    metric_key: str,
    body: SeverityUpdateRequest,
) -> dict:
    _load_session_by_id(session_id)  # verify session exists
    # threshold_manager uses Python severity strings ("ok", not "pass")
    old_sev = _map_severity_to_python(body.oldSeverity)
    new_sev = _map_severity_to_python(body.newSeverity)
    threshold_manager.update_threshold(metric_key, body.currentScore, old_sev, new_sev)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static file serving — production mode only (requires pnpm build first)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = _SRC_DIR.parent / "Diagram Analyser App" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="static")
