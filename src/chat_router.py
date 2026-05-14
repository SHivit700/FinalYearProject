"""
Message router for the chat loop.

Turns a raw user string into a typed intent so that well-defined requests
(exit, help, new image, dismiss/restore a metric, priority question) are
handled deterministically without touching the LLM.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from suggestion_engine import _METRIC_DISPLAY_NAMES

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}

_EXIT_WORDS  = {"exit", "quit", "q", "bye"}
_HELP_WORDS  = {"help", "?", "h", "commands"}
_DISMISS_KWS = {
    "ignore", "dismiss", "don't care", "do not care", "skip",
    "exclude", "remove", "not interested",
}
_RESTORE_KWS = {
    "restore", "add back", "re-enable", "undo",
    "include again", "care about again", "bring back",
}
_PRIORITY_KWS = {
    "fix first", "start with", "top priority", "highest priority",
    "most important", "worst issue", "biggest problem", "most critical",
    "where do i start", "what to fix", "what should i fix",
    "what's the priority", "top issue", "number one issue",
}

_PATH_RE = re.compile(
    r'(?:^|\s)((?:[~/.]|[A-Za-z]:)?[\w/\\.\-]+\.(?:png|jpg|jpeg|gif|bmp|webp|svg))(?:\s|$)',
    re.IGNORECASE,
)

HELP_TEXT = """
Available commands:
  help, ?                  — Show this message
  exit, quit, q            — End session and save

Submit a new diagram version (any of these work):
  v2.png
  /absolute/path/to/v2.png
  ~/diagrams/v2.png
  I've fixed the issues, here is my updated diagram: v2.png

Manage metrics (dismiss excludes from score; restore adds it back):
  ignore label contrast
  don't care about cognitive chunk density
  restore label contrast

Ask anything else:
  what should I fix first?
  what improved since version 1?
  explain the whitespace issue
"""


def _resolve_path(candidate: str) -> str | None:
    """Return absolute path if candidate is an existing image file, else None."""
    p = Path(candidate.strip()).expanduser()
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        return None
    if p.exists():
        return str(p.resolve())
    rel = Path.cwd() / p
    if rel.exists():
        return str(rel.resolve())
    return None


def _try_resolve_image_path(raw: str) -> str | None:
    """Try the whole message as a bare path, then scan for a path-like substring."""
    resolved = _resolve_path(raw.strip())
    if resolved:
        return resolved
    for m in _PATH_RE.finditer(raw):
        resolved = _resolve_path(m.group(1))
        if resolved:
            return resolved
    return None


def _match_metric(raw: str) -> str | None:
    """
    Match a metric key from user text.
    Tries: full display name → underscore key as spaces → longest significant word (>4 chars).
    Returns None if nothing matches.
    """
    lower = raw.lower()
    for key, display in _METRIC_DISPLAY_NAMES.items():
        if display.lower() in lower:
            return key
    for key in _METRIC_DISPLAY_NAMES:
        if key.replace("_", " ") in lower:
            return key
    best_key, best_len = None, 0
    for key, display in _METRIC_DISPLAY_NAMES.items():
        for word in display.lower().split():
            if len(word) > 4 and word in lower and len(word) > best_len:
                best_key, best_len = key, len(word)
    return best_key


def route_message(raw: str) -> dict:
    """
    Return {"intent": str, "data": dict}.

    Intents:
        "exit"         — user wants to quit
        "help"         — user wants to see commands
        "new_image"    — user submitted a new diagram path  (data: {"path": str})
        "priority"     — user wants to know the top issue
        "dismiss"      — dismiss a metric             (data: {"metric": str})
        "restore"      — restore a dismissed metric   (data: {"metric": str})
        "needs_metric" — dismiss/restore keyword but metric unclear
                         (data: {"action": "dismiss"|"restore"})
        "chat"         — everything else → send to LLM
    """
    lower = raw.lower().strip()

    if lower in _EXIT_WORDS:
        return {"intent": "exit", "data": {}}

    if lower in _HELP_WORDS:
        return {"intent": "help", "data": {}}

    path = _try_resolve_image_path(raw)
    if path:
        return {"intent": "new_image", "data": {"path": path}}

    if any(kw in lower for kw in _PRIORITY_KWS):
        return {"intent": "priority", "data": {}}

    if any(kw in lower for kw in _DISMISS_KWS):
        metric = _match_metric(raw)
        if metric:
            return {"intent": "dismiss", "data": {"metric": metric}}
        return {"intent": "needs_metric", "data": {"action": "dismiss"}}

    if any(kw in lower for kw in _RESTORE_KWS):
        metric = _match_metric(raw)
        if metric:
            return {"intent": "restore", "data": {"metric": metric}}
        return {"intent": "needs_metric", "data": {"action": "restore"}}

    return {"intent": "chat", "data": {}}
