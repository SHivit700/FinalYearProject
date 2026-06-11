"""
CLI entry point for the Suggestion Engine.

Usage:
    python src/analyze_diagram.py <image_path> [options]

Options:
    --diagram-type   system_design | timeline_roadmap  (default: system_design)
    --session        Path to existing session JSON to continue from a prior run
    --interactive    Prompt user to respond to each suggestion before saving session

Non-interactive mode (default):
    Prints the full report, saves the session, and exits. Re-run with
    --session <file> on the next iteration to carry conversation history forward.

Interactive mode:
    After each non-ok suggestion the user is prompted to:
        dismiss   — permanently ignore this metric going forward
        accept    — acknowledge and note it has been addressed
        note <text> — record a comment (metric stays active)
        skip      — record no decision for this turn
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from chat_router import HELP_TEXT, _match_metric, route_message
from feature_detection import extract_features_for_image
from suggestion_engine import (
    _ACTIVE_METRICS,
    _compute_composite_score,
    _default_session_path,
    _get_score,
    _METRIC_DISPLAY_NAMES,
    chat_with_llm,
    format_text_report,
    generate_suggestions,
    load_session,
    new_session,
    save_session,
)

_SEP = "─" * 60


def _print_sep() -> None:
    print(_SEP)


def _prompt_user(metric: str, suggestion: dict) -> tuple[str, str]:
    """
    Show the suggestion and prompt for a decision.
    Returns (decision, user_note).
    """
    sev = suggestion["severity"].upper()
    display = _METRIC_DISPLAY_NAMES.get(metric, metric)
    score_str = f" (score={suggestion['score']})" if suggestion["score"] is not None else ""
    print(f"\n  {sev}  {display}{score_str}")
    print(f"  {suggestion['issue']}")
    print(f"  → {suggestion['recommendation']}")
    locs = suggestion.get("locations", [])
    if locs:
        print(f"  Flagged locations: {len(locs)}")
        for loc in locs[:3]:
            coord = f"({loc['x1']},{loc['y1']})–({loc['x2']},{loc['y2']})"
            detail = loc.get("detail", "")
            print(f"    {coord}  {detail}")
        if len(locs) > 3:
            print(f"    … and {len(locs) - 3} more")

    while True:
        raw = input("\n  [dismiss / accept / note <text> / skip]: ").strip()
        if not raw:
            continue
        low = raw.lower()
        if low == "dismiss":
            return "dismissed", ""
        if low == "accept":
            return "accepted", ""
        if low == "skip":
            return "skipped", ""
        if low.startswith("note "):
            return "noted", raw[5:].strip()
        print("  Unrecognised input. Type dismiss, accept, note <text>, or skip.")


def _interactive_loop(
    suggestions: list[dict],
    session: dict,
    turn_record: dict,
) -> None:
    """Walk the user through each non-ok suggestion and record their decisions."""
    non_ok = [s for s in suggestions if s["severity"] != "ok"]
    if not non_ok:
        print("\n  No significant issues — nothing to respond to.")
        return

    print(f"\n{_SEP}")
    print(f"  Responding to {len(non_ok)} issue(s)  "
          f"(type 'dismiss' to permanently ignore a metric)")
    _print_sep()

    user_responses: dict = turn_record.setdefault("user_responses", {})
    dismissed_now: list[str] = []

    for s in non_ok:
        metric = s["metric"]
        decision, note = _prompt_user(metric, s)
        user_responses[metric] = {"decision": decision, "user_note": note}
        if decision == "dismissed":
            dismissed_now.append(metric)
            print(f"  ✓ {metric} will be ignored in future turns.")

    for m in dismissed_now:
        if m not in session["permanently_dismissed"]:
            session["permanently_dismissed"].append(m)


def _build_version_diff(prev_version: dict, curr_version: dict) -> str:
    """Compare metric_scores between two versions and return a formatted string."""
    prev_scores = prev_version.get("metric_scores", {})
    curr_scores = curr_version.get("metric_scores", {})
    prev_score = prev_version.get("composite_score")
    curr_score = curr_version.get("composite_score")

    if prev_score is not None and curr_score is not None:
        delta = curr_score - prev_score
        sign = "+" if delta >= 0 else ""
        score_delta = f"{prev_score} → {curr_score} ({sign}{delta:.1f})"
    else:
        score_delta = f"{prev_score} → {curr_score}"

    lines = [f"## What Changed Since Version {prev_version.get('version', '?')} ({score_delta})"]

    for metric in _ACTIVE_METRICS:
        p = prev_scores.get(metric)
        c = curr_scores.get(metric)
        if p is None or c is None:
            continue
        delta = c - p
        if abs(delta) < 0.5:
            symbol, label = "→", "unchanged"
        elif delta > 0:
            symbol, label = "✓", f"improved: {p:.0f} → {c:.0f}"
        else:
            symbol, label = "✗", f"regressed: {p:.0f} → {c:.0f}"
        display = _METRIC_DISPLAY_NAMES.get(metric, metric)
        lines.append(f"  {symbol} {display}: {label}")

    return "\n".join(lines)


def _apply_llm_action(action: dict, session: dict, features: dict) -> None:
    """Apply a dismiss/restore action returned by the LLM to the session state."""
    action_type = action.get("type", "none")
    metric = action.get("metric")
    if action_type == "dismiss" and metric and metric in _METRIC_DISPLAY_NAMES:
        if metric not in session["permanently_dismissed"]:
            session["permanently_dismissed"].append(metric)
    elif action_type == "restore" and metric and metric in _METRIC_DISPLAY_NAMES:
        if metric in session["permanently_dismissed"]:
            session["permanently_dismissed"].remove(metric)


def _analyze_new_version(
    image_path_str: str,
    session: dict,
    session_path: str,
) -> tuple[dict, dict]:
    """Run the full analysis pipeline on a new image within the existing session."""
    image_path = Path(image_path_str)
    version_number = len(session.get("diagram_versions", [])) + 1
    print(f"\n[Version {version_number}] Analysing {image_path.name}...")

    _t0 = time.perf_counter()
    result = extract_features_for_image(str(image_path), diagram_type=session["diagram_type"])
    _t_extraction = time.perf_counter() - _t0
    features = result["features"]
    _stage_timings = result.get("timings", {})

    _raw_shape = features.get("image_shape", (1000, 1000, 3))
    _img_shape_2d = (_raw_shape[0], _raw_shape[1])

    _t0 = time.perf_counter()
    suggestions_result = generate_suggestions(
        features,
        diagram_type=session["diagram_type"],
        image_path=str(image_path),
        use_llm=True,
        session=session,
        img_shape=_img_shape_2d,
    )
    _t_llm = time.perf_counter() - _t0

    suggestions_result["timings"] = {
        **_stage_timings,
        "stage4_llm_s": round(_t_llm, 3),
        "total_no_llm_s": round(_t_extraction, 3),
        "total_with_llm_s": round(_t_extraction + _t_llm, 3),
    }

    metric_scores = {m: _get_score(features, m) for m in _ACTIVE_METRICS}
    version_record = {
        "version": version_number,
        "diagram_path": str(image_path),
        "analyzed_at": datetime.now().isoformat(),
        "composite_score": suggestions_result["composite_score"],
        "metric_scores": {k: v for k, v in metric_scores.items() if v is not None},
        "suggestions": suggestions_result["rule_based"],
        "llm_report": suggestions_result["llm_report"],
        "llm_regions": suggestions_result.get("llm_regions", {}),
    }
    session["diagram_versions"].append(version_record)
    save_session(session, session_path)
    return suggestions_result, features


def _chat_loop(
    suggestions_result: dict,
    session: dict,
    session_path: str,
    features: dict,
) -> None:
    """Conversational chat loop over an already-analysed diagram version."""
    print()
    print(format_text_report(suggestions_result))
    _print_sep()

    versions = session.get("diagram_versions", [])
    if len(versions) >= 2:
        print(_build_version_diff(versions[-2], versions[-1]))
        _print_sep()

    current_suggestions = suggestions_result["rule_based"]
    current_composite_score = suggestions_result["composite_score"]

    print("\nType your message (or 'exit' to quit, 'help' for commands):")
    while True:
        try:
            raw = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        route = route_message(raw)
        intent = route["intent"]
        data = route["data"]
        reply = ""

        if intent == "exit":
            break

        elif intent == "help":
            print(HELP_TEXT)
            continue

        elif intent == "new_image":
            try:
                suggestions_result, features = _analyze_new_version(data["path"], session, session_path)
            except Exception as exc:
                print(f"\nFailed to analyse image: {exc}")
                continue
            current_suggestions = suggestions_result["rule_based"]
            current_composite_score = suggestions_result["composite_score"]
            versions = session.get("diagram_versions", [])
            print()
            print(format_text_report(suggestions_result))
            _print_sep()
            if len(versions) >= 2:
                print(_build_version_diff(versions[-2], versions[-1]))
                _print_sep()
            reply = f"Version {len(versions)} analysed. Composite score: {current_composite_score}."
            session["chat_history"] += [
                {"role": "user", "content": f"[submitted new diagram: {data['path']}]"},
                {"role": "assistant", "content": reply},
            ]
            save_session(session, session_path)
            continue

        elif intent == "priority":
            non_ok = [s for s in current_suggestions if s["severity"] != "ok"]
            llm_report = (
                session["diagram_versions"][-1].get("llm_report")
                if session.get("diagram_versions") else None
            )
            if llm_report and llm_report.get("priority_issues"):
                top = llm_report["priority_issues"][0]
                display = _METRIC_DISPLAY_NAMES.get(top.get("metric", ""), top.get("metric", ""))
                reply = (
                    f"Your top priority is **{display}** [{top.get('severity', '').upper()}]: "
                    f"{top.get('what', '')}. "
                    f"Fix: {top.get('how_to_fix', '')}"
                )
            elif non_ok:
                top = non_ok[0]
                display = _METRIC_DISPLAY_NAMES.get(top["metric"], top["metric"])
                score_str = f" (score={top['score']})" if top["score"] is not None else ""
                reply = (
                    f"Your top priority is **{display}**{score_str} [{top['severity'].upper()}]: "
                    f"{top['issue']}. Fix: {top['recommendation']}"
                )
            else:
                reply = "No issues remain — all metrics are passing."
            print(f"\nAssistant: {reply}")

        elif intent == "dismiss":
            metric = data["metric"]
            display = _METRIC_DISPLAY_NAMES[metric]
            if metric not in session["permanently_dismissed"]:
                session["permanently_dismissed"].append(metric)
                current_composite_score = _compute_composite_score(features, session["permanently_dismissed"])
                reply = f"{display} dismissed. Composite score is now {current_composite_score}."
            else:
                reply = f"{display} is already dismissed."
            print(f"\nAssistant: {reply}")

        elif intent == "restore":
            metric = data["metric"]
            display = _METRIC_DISPLAY_NAMES[metric]
            if metric in session["permanently_dismissed"]:
                session["permanently_dismissed"].remove(metric)
                current_composite_score = _compute_composite_score(features, session["permanently_dismissed"])
                reply = f"{display} restored. Composite score is now {current_composite_score}."
            else:
                reply = f"{display} is not currently dismissed."
            print(f"\nAssistant: {reply}")

        elif intent == "needs_metric":
            action = data["action"]
            dismissed = session.get("permanently_dismissed", [])
            if action == "restore" and not dismissed:
                reply = "No metrics are currently dismissed."
                print(f"\nAssistant: {reply}")
            else:
                candidates = dismissed if action == "restore" else list(_METRIC_DISPLAY_NAMES.keys())
                print(f"\nAssistant: Which metric would you like to {action}?")
                print("  " + ", ".join(candidates))
                try:
                    clarification = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                metric = _match_metric(clarification)
                if metric:
                    display = _METRIC_DISPLAY_NAMES[metric]
                    if action == "dismiss":
                        if metric not in session["permanently_dismissed"]:
                            session["permanently_dismissed"].append(metric)
                            current_composite_score = _compute_composite_score(features, session["permanently_dismissed"])
                            reply = f"{display} dismissed. Composite score is now {current_composite_score}."
                        else:
                            reply = f"{display} is already dismissed."
                    else:
                        if metric in session["permanently_dismissed"]:
                            session["permanently_dismissed"].remove(metric)
                            current_composite_score = _compute_composite_score(features, session["permanently_dismissed"])
                            reply = f"{display} restored. Composite score is now {current_composite_score}."
                        else:
                            reply = f"{display} is not currently dismissed."
                    raw = clarification
                    print(f"\nAssistant: {reply}")
                else:
                    combined = f"{raw}\n{clarification}"
                    result = chat_with_llm(combined, session, current_suggestions, current_composite_score, session["diagram_type"])
                    reply = result["reply"]
                    _apply_llm_action(result["action"], session, features)
                    current_composite_score = _compute_composite_score(features, session["permanently_dismissed"])
                    print(f"\nAssistant: {reply}")

        else:  # "chat" → LLM
            result = chat_with_llm(raw, session, current_suggestions, current_composite_score, session["diagram_type"])
            reply = result["reply"]
            _apply_llm_action(result["action"], session, features)
            current_composite_score = _compute_composite_score(features, session["permanently_dismissed"])
            print(f"\nAssistant: {reply}")

        session["chat_history"] += [{"role": "user", "content": raw}, {"role": "assistant", "content": reply}]
        save_session(session, session_path)

    print(f"Session saved → {session_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a diagram image and produce quality improvement suggestions.",
    )
    parser.add_argument("image_path", help="Path to the diagram image.")
    parser.add_argument(
        "--diagram-type",
        default="system_design",
        choices=["system_design", "timeline_roadmap"],
        help="Diagram type (default: system_design).",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="PATH",
        help="Path to an existing session JSON to continue from.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for user responses after each suggestion.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch conversational chat loop after initial analysis.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if args.session:
        session_path = args.session
        session = load_session(session_path)
    else:
        session = new_session(args.diagram_type)
        session_path = _default_session_path(str(image_path))
        print(f"New session: {session['session_id']}")

    version_number = len(session.get("diagram_versions", [])) + 1
    if args.session:
        print(f"Continuing session '{session['session_id']}' (version {version_number})")
    _print_sep()
    print(f"[Version {version_number}] Analysing diagram: {image_path.name}")
    _print_sep()

    try:
        result = extract_features_for_image(
            str(image_path),
            diagram_type=session["diagram_type"],
        )
    except Exception as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    features = result["features"]

    image_shape_raw = features.get("image_shape", (1000, 1000, 3))
    image_shape_2d = (image_shape_raw[0], image_shape_raw[1])

    suggestions_result = generate_suggestions(
        features,
        diagram_type=session["diagram_type"],
        image_path=str(image_path),
        use_llm=True,
        session=session,
        img_shape=image_shape_2d,
    )

    metric_scores = {
        metric: _get_score(features, metric)
        for metric in _ACTIVE_METRICS
    }
    version_record: dict = {
        "version": version_number,
        "diagram_path": str(image_path),
        "analyzed_at": datetime.now().isoformat(),
        "composite_score": suggestions_result["composite_score"],
        "metric_scores": {k: v for k, v in metric_scores.items() if v is not None},
        "suggestions": suggestions_result["rule_based"],
        "llm_report": suggestions_result["llm_report"],
        "llm_regions": suggestions_result.get("llm_regions", {}),
    }
    session.setdefault("diagram_versions", []).append(version_record)
    session.setdefault("chat_history", [])

    if args.chat:
        save_session(session, session_path)
        _chat_loop(suggestions_result, session, session_path, features)
        return

    print()
    print(format_text_report(suggestions_result))
    _print_sep()

    if args.interactive:
        _interactive_loop(suggestions_result["rule_based"], session, version_record)

    score = suggestions_result["composite_score"]
    print(f"\nComposite score: {score if score is not None else 'N/A'}")

    if session["permanently_dismissed"]:
        print(f"Permanently dismissed metrics: {', '.join(session['permanently_dismissed'])}")

    save_session(session, session_path)
    print(f"Session saved → {session_path}")


if __name__ == "__main__":
    main()
