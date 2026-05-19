"""
Dynamic threshold manager.

Thresholds for score-based severity metrics are stored in data/thresholds.csv
and updated whenever the user reclassifies a metric's severity in the UI.

Learning rule (exponential step toward the score that triggered the disagreement):
    new_threshold = old_threshold + learning_rate * (current_score - old_threshold)

Direction is natural: downgrade (critical→ok) moves threshold down; upgrade moves it up.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Optional

_CSV_PATH = Path(__file__).parent / "data" / "thresholds.csv"
_FIELDNAMES = [
    "metric",
    "critical_threshold",
    "warning_threshold",
    "learning_rate",
    "update_count",
]

# Original hardcoded values — used as defaults and for reset
_DEFAULTS: dict[str, dict] = {
    "label_readability":       {"critical_threshold": 70.0, "warning_threshold": 90.0,  "learning_rate": 0.10},
    "label_area":              {"critical_threshold": 50.0, "warning_threshold": 80.0,  "learning_rate": 0.10},
    "overlap_metrics":         {"critical_threshold": 50.0, "warning_threshold": 80.0,  "learning_rate": 0.10},
    "edge_clearance":          {"critical_threshold": 50.0, "warning_threshold": 80.0,  "learning_rate": 0.10},
    "font_hierarchy":          {"critical_threshold": None, "warning_threshold": 80.0,  "learning_rate": 0.10},
    "container_utilization":   {"critical_threshold": 65.0, "warning_threshold": 90.0,  "learning_rate": 0.10},
    "isolated_boxes":          {"critical_threshold": 60.0, "warning_threshold": 70.0,  "learning_rate": 0.10},
    "brevity":                 {"critical_threshold": 60.0, "warning_threshold": 70.0,  "learning_rate": 0.10},
    "whitespace_distribution": {"critical_threshold": 50.0, "warning_threshold": 80.0,  "learning_rate": 0.10},
    "color_harmony":           {"critical_threshold": 40.0, "warning_threshold": 60.0,  "learning_rate": 0.10},
    "label_contrast":          {"critical_threshold": None, "warning_threshold": 90.0,  "learning_rate": 0.10},
    "cognitive_chunk_density": {"critical_threshold": 50.0, "warning_threshold": 90.0,  "learning_rate": 0.10},
    "orientation_consistency": {"critical_threshold": 50.0, "warning_threshold": 80.0,  "learning_rate": 0.10},
}

_SEVERITY_RANK = {"ok": 0, "warning": 1, "critical": 2}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _parse_float(value: str) -> Optional[float]:
    """Return float or None for empty/missing values."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_thresholds() -> dict[str, dict]:
    """Read thresholds.csv and return a dict keyed by metric name."""
    if not _CSV_PATH.exists():
        reset_thresholds()

    result: dict[str, dict] = {}
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"].strip()
            result[metric] = {
                "critical_threshold": _parse_float(row.get("critical_threshold")),
                "warning_threshold":  _parse_float(row.get("warning_threshold")),
                "learning_rate":      _parse_float(row.get("learning_rate")) or 0.10,
                "update_count":       int(row.get("update_count", 0) or 0),
            }
    return result


def get_thresholds(metric: str) -> dict:
    """Return threshold dict for a single metric (reads CSV each call)."""
    thresholds = load_thresholds()
    if metric in thresholds:
        return thresholds[metric]
    # Fallback to defaults if metric not in CSV
    return dict(_DEFAULTS.get(metric, {"critical_threshold": None, "warning_threshold": None, "learning_rate": 0.10}))


def _save_thresholds(thresholds: dict[str, dict]) -> None:
    """Write thresholds dict to CSV atomically via temp file."""
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(_CSV_PATH.parent), prefix=".thresholds_tmp_", suffix=".csv"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            for metric, t in thresholds.items():
                writer.writerow({
                    "metric":             metric,
                    "critical_threshold": "" if t.get("critical_threshold") is None else f"{t['critical_threshold']:.4f}",
                    "warning_threshold":  "" if t.get("warning_threshold")  is None else f"{t['warning_threshold']:.4f}",
                    "learning_rate":      f"{t.get('learning_rate', 0.10):.2f}",
                    "update_count":       t.get("update_count", 0),
                })
        os.replace(tmp_path, str(_CSV_PATH))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def reset_thresholds() -> None:
    """Overwrite CSV with the original hardcoded defaults."""
    defaults = {}
    for metric, d in _DEFAULTS.items():
        defaults[metric] = {
            "critical_threshold": d["critical_threshold"],
            "warning_threshold":  d["warning_threshold"],
            "learning_rate":      d["learning_rate"],
            "update_count":       0,
        }
    _save_thresholds(defaults)


# ---------------------------------------------------------------------------
# Learning logic
# ---------------------------------------------------------------------------

def _clamp_and_validate(t: dict) -> dict:
    """Ensure both thresholds are in [0, 100] and critical < warning."""
    crit = t.get("critical_threshold")
    warn = t.get("warning_threshold")

    if crit is not None:
        crit = max(0.0, min(100.0, crit))
    if warn is not None:
        warn = max(0.0, min(100.0, warn))

    # Maintain invariant: critical < warning (with a minimum gap of 1 point)
    if crit is not None and warn is not None and crit >= warn:
        mid = (crit + warn) / 2.0
        crit = max(0.0, mid - 0.5)
        warn = min(100.0, mid + 0.5)

    t = dict(t)
    t["critical_threshold"] = crit
    t["warning_threshold"] = warn
    return t


def _apply_learning(t: dict, score: float, old_sev: str, new_sev: str) -> dict:
    """
    Return an updated threshold dict by applying one learning step.

    Which thresholds to move:
        critical → warning  : critical_threshold only
        critical → ok       : both
        warning  → ok       : warning_threshold only
        ok       → warning  : warning_threshold only
        ok       → critical : both
        warning  → critical : critical_threshold only
    """
    old_rank = _SEVERITY_RANK.get(old_sev, -1)
    new_rank = _SEVERITY_RANK.get(new_sev, -1)
    if old_rank == new_rank or old_rank < 0 or new_rank < 0:
        return t  # no-op

    t = dict(t)
    lr = t.get("learning_rate", 0.10)

    adjust_critical = (
        (old_sev == "critical" and new_sev in ("warning", "ok")) or
        (new_sev == "critical" and old_sev in ("warning", "ok"))
    )
    adjust_warning = (
        (old_sev in ("warning", "ok") and new_sev in ("warning", "ok") and old_rank != new_rank) or
        (old_sev == "critical" and new_sev == "ok") or
        (old_sev == "ok" and new_sev == "critical")
    )

    if adjust_critical and t.get("critical_threshold") is not None:
        t["critical_threshold"] = t["critical_threshold"] + lr * (score - t["critical_threshold"])

    if adjust_warning and t.get("warning_threshold") is not None:
        t["warning_threshold"] = t["warning_threshold"] + lr * (score - t["warning_threshold"])

    return _clamp_and_validate(t)


def update_threshold(
    metric: str,
    current_score: float,
    old_severity: str,
    new_severity: str,
) -> dict:
    """
    Apply one learning step for *metric*, persist to CSV, and return the
    updated threshold dict for that metric.
    """
    thresholds = load_thresholds()
    if metric not in thresholds:
        # Initialise from defaults if metric somehow missing from CSV
        thresholds[metric] = dict(
            _DEFAULTS.get(metric, {"critical_threshold": None, "warning_threshold": None, "learning_rate": 0.10})
        )
        thresholds[metric].setdefault("update_count", 0)

    t = thresholds[metric]
    t = _apply_learning(t, current_score, old_severity, new_severity)
    t["update_count"] = int(t.get("update_count", 0)) + 1
    thresholds[metric] = t
    _save_thresholds(thresholds)
    return t
