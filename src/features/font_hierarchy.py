"""Font-tier clustering from label bbox heights and diagram-level consistency metrics."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

SymbolSnapMode = Literal["majority", "nearest_height", "largest"]

DEFAULT_SYMBOL_SNAP_EXTRA_TOKENS: Tuple[str, ...] = ()
# Max relative step between sorted unique heights to merge into one tier (larger → fewer tiers).
DEFAULT_MERGE_RELATIVE_THRESHOLD: float = 0.16
# Minimum share on the dominant tier before chain absorption runs.
DEFAULT_FONT_CHAIN_ABSORB_MIN_DOMINANT_FRACTION: float = 0.96

DEFAULT_SYMBOL_SNAP_MAX_CHARS: int = 3
DEFAULT_SYMBOL_SNAP_MODE: SymbolSnapMode = "majority"
DEFAULT_SYMBOL_PROPAGATION_MAX_ITERS: int = 8
DEFAULT_SYMBOL_PROPAGATION_MAX_DISTANCE_PX: Optional[float] = None
DEFAULT_FONT_CHAIN_ABSORB_ENABLED: bool = True
DEFAULT_FONT_CHAIN_ABSORB_LINK_MAX_PX: Optional[float] = None
DEFAULT_FONT_CHAIN_ABSORB_TITLE_HEIGHT_RATIO: float = 1.38

DEFAULT_FONT_SCORE_WEIGHT_LEVELS: float = 0.4
DEFAULT_FONT_SCORE_WEIGHT_CONSISTENCY: float = 0.4
DEFAULT_FONT_SCORE_WEIGHT_SPREAD: float = 0.2
DEFAULT_FONT_SCORE_SPREAD_LOW: float = 0.3
DEFAULT_FONT_SCORE_SPREAD_HIGH: float = 0.6


def compute_font_hierarchy_score(
    num_levels: int,
    dominant_fraction: float,
    bbox_heights: Sequence[float],
) -> Dict[str, Any]:
    """Map tier count, dominant-tier share, and height spread to a 0–100 score."""
    L = int(num_levels)
    p = float(dominant_fraction)
    n = len(bbox_heights)
    hs = [float(h) for h in bbox_heights if float(h) > 0]
    if not hs:
        h_min = h_max = 0.0
        r = 0.0
    else:
        h_min = float(min(hs))
        h_max = float(max(hs))
        r = (h_max - h_min) / h_max if h_max > 0 else 0.0

    if L <= 0:
        s_l = 50.0
    elif L == 1:
        s_l = 70.0
    elif L <= 3:
        s_l = 100.0
    elif L == 4:
        s_l = 70.0
    else:
        s_l = 50.0

    s_c = 100.0 * p

    lo = float(DEFAULT_FONT_SCORE_SPREAD_LOW)
    hi = float(DEFAULT_FONT_SCORE_SPREAD_HIGH)
    if r < lo:
        s_s = 100.0 - 200.0 * (lo - r) ** 2
    elif r > hi:
        s_s = 100.0 - 200.0 * (r - hi) ** 2
    else:
        s_s = 100.0
    s_s = float(np.clip(s_s, 0.0, 100.0))

    w_l = float(DEFAULT_FONT_SCORE_WEIGHT_LEVELS)
    w_c = float(DEFAULT_FONT_SCORE_WEIGHT_CONSISTENCY)
    w_sp = float(DEFAULT_FONT_SCORE_WEIGHT_SPREAD)
    font_score = w_l * s_l + w_c * s_c + w_sp * s_s

    return {
        "font_score": float(np.clip(font_score, 0.0, 100.0)),
        "S_L": s_l,
        "S_C": s_c,
        "S_S": s_s,
        "L": L,
        "p": p,
        "n_labels": n,
        "h_min_px": round(h_min, 4),
        "h_max_px": round(h_max, 4),
        "r_spread": round(r, 6),
        "weights": {
            "levels": w_l,
            "consistency": w_c,
            "spread": w_sp,
        },
        "spread_band": {"low": lo, "high": hi},
    }


def _centre_and_size(label: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = label.get("bbox")
    if not bbox:
        h = float(label.get("height") or 0.0)
        w = float(label.get("width") or 0.0)
        if h <= 0:
            return None
        return 0.0, 0.0, w, h
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    if h <= 0:
        return None
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy, w, h


def _level_names(n: int) -> List[str]:
    base = ["small", "medium", "large", "xlarge", "xxlarge"]
    if n <= len(base):
        return base[:n]
    return base + [f"level_{i + 1}" for i in range(len(base), n)]


def _normalize_symbol_snap_extra(
    tokens: Optional[Sequence[str]],
) -> frozenset[str]:
    merged: List[str] = list(DEFAULT_SYMBOL_SNAP_EXTRA_TOKENS)
    if tokens is not None:
        merged.extend(tokens)
    out: set[str] = set()
    for s in merged:
        c = _text_compact_no_space(str(s))
        if c:
            out.add(c)
    return frozenset(out)


def _text_compact_no_space(text: str) -> str:
    """Collapse whitespace so tokens like ' = ' and '=' match."""
    return "".join((text or "").split())


def _is_symbol_snap_label(
    text: str,
    max_chars: int,
    extra_tokens: frozenset[str],
) -> bool:
    compact = _text_compact_no_space(text)
    if compact in extra_tokens:
        return True
    return _is_symbol_only_label(text, max_chars)


def _is_symbol_only_label(text: str, max_chars: int) -> bool:
    """True for very short operator-like text where bbox height is a weak size cue."""
    if max_chars <= 0:
        return False
    compact = _text_compact_no_space(text)
    if not compact or len(compact) > max_chars:
        return False
    return not any(c.isalnum() for c in compact)


def _cluster_heights_by_gap(
    heights: List[float],
    merge_relative_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster sorted unique heights by relative gap; ids per label and sorted tier centres."""
    arr = np.array(heights, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    uniq = np.unique(arr)
    if uniq.size == 1:
        return np.zeros(arr.size, dtype=int), np.array([float(uniq[0])])

    tiers: List[List[float]] = [[float(uniq[0])]]
    for x in uniq[1:]:
        prev = tiers[-1][-1]
        rel = (float(x) - prev) / max(prev, 1e-6)
        if rel <= merge_relative_threshold:
            tiers[-1].append(float(x))
        else:
            tiers.append([float(x)])

    centres = np.array([sum(t) / len(t) for t in tiers], dtype=float)
    order = np.argsort(centres)
    centres_sorted = centres[order]

    remap = {int(old): int(rank) for rank, old in enumerate(order)}

    tier_for_uniq_idx = np.zeros(len(uniq), dtype=int)
    for ti, tier in enumerate(tiers):
        for v in tier:
            matches = np.nonzero(np.isclose(uniq, v, rtol=0.0, atol=1e-6))[0]
            for mi in matches:
                tier_for_uniq_idx[int(mi)] = ti

    cluster_ids = np.zeros(arr.size, dtype=int)
    for i in range(arr.size):
        matches = np.nonzero(np.isclose(uniq, arr[i], rtol=0.0, atol=1e-6))[0]
        if matches.size:
            uidx = int(matches[0])
        else:
            uidx = int(np.argmin(np.abs(uniq - arr[i])))
        cluster_ids[i] = remap[int(tier_for_uniq_idx[uidx])]
    return cluster_ids, centres_sorted


def _propagate_symbol_tiers_to_max_neighbor(
    tier_ids: List[int],
    snap_effective: List[bool],
    centres_row: List[Tuple[float, float]],
    max_dist_px: float,
    max_iters: int,
) -> Tuple[List[int], int, List[bool]]:
    """Iteratively pull symbol tiers up to the max tier among neighbours within range."""
    n = len(tier_ids)
    before = list(tier_ids)
    changed_row = [False] * n
    if n == 0 or max_iters <= 0 or max_dist_px <= 0:
        return list(tier_ids), 0, changed_row

    tiers = list(tier_ids)
    cx = np.array([p[0] for p in centres_row], dtype=float)
    cy = np.array([p[1] for p in centres_row], dtype=float)
    d2_thresh = max_dist_px * max_dist_px
    rounds_used = 0

    for _ in range(max_iters):
        new_tiers = tiers[:]
        any_change = False
        for i in range(n):
            if not snap_effective[i]:
                continue
            d2 = (cx - cx[i]) ** 2 + (cy - cy[i]) ** 2
            mask = (d2 <= d2_thresh) & (np.arange(n) != i)
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            neigh_max = int(max(int(tiers[j]) for j in idxs))
            if neigh_max != tiers[i]:
                new_tiers[i] = neigh_max
                any_change = True
        tiers = new_tiers
        if any_change:
            rounds_used += 1
        else:
            break

    for i in range(n):
        if snap_effective[i] and tiers[i] != before[i]:
            changed_row[i] = True
    return tiers, rounds_used, changed_row


def _absorb_to_dominant_via_chain(
    tier_ids: List[int],
    heights: List[float],
    centres_row: List[Tuple[float, float]],
    centres_sorted: np.ndarray,
    *,
    enabled: bool,
    min_dominant_fraction: float,
    link_max_px: float,
    title_height_ratio: float,
) -> Tuple[List[int], int, List[bool]]:
    """Walk from the dominant tier on short links; absorb neighbours unless they exceed a height barrier."""
    n = len(tier_ids)
    k = int(centres_sorted.size)
    absorbed_flag = [False] * n
    if not enabled or n == 0 or k == 0 or link_max_px <= 0:
        return list(tier_ids), 0, absorbed_flag

    arr = np.array(tier_ids, dtype=int)
    counts = np.bincount(arr, minlength=k)
    dominant_id = int(np.argmax(counts))
    dom_frac = float(counts[dominant_id]) / float(n)
    if dom_frac < min_dominant_fraction:
        return list(tier_ids), 0, absorbed_flag

    h_dom = float(centres_sorted[dominant_id])
    barrier_h = h_dom * title_height_ratio

    cx = np.array([p[0] for p in centres_row], dtype=float)
    cy = np.array([p[1] for p in centres_row], dtype=float)
    h_arr = np.array(heights, dtype=float)
    d2_max = link_max_px * link_max_px

    tiers = list(tier_ids)

    q: deque[int] = deque()
    visited = [False] * n
    for i in range(n):
        if tiers[i] == dominant_id:
            visited[i] = True
            q.append(i)

    n_absorbed = 0
    while q:
        i = q.popleft()
        for j in range(n):
            if j == i or visited[j]:
                continue
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy > d2_max:
                continue
            if tiers[j] == dominant_id:
                visited[j] = True
                q.append(j)
                continue
            if float(h_arr[j]) > barrier_h:
                continue
            if tiers[j] != dominant_id:
                tiers[j] = dominant_id
                n_absorbed += 1
                absorbed_flag[j] = True
            visited[j] = True
            q.append(j)

    return tiers, n_absorbed, absorbed_flag


def _level_consistency(level_names: List[str]) -> Dict[str, Any]:
    """Per-tier counts and the dominant tier's share."""
    n = len(level_names)
    if n == 0:
        return {
            "n_labels": 0,
            "dominant_level": None,
            "dominant_fraction": None,
            "by_level": {},
        }
    counts: Dict[str, int] = {}
    for lvl in level_names:
        counts[lvl] = counts.get(lvl, 0) + 1
    dom = max(counts, key=counts.get)
    return {
        "n_labels": n,
        "dominant_level": dom,
        "dominant_fraction": counts[dom] / n,
        "by_level": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def compute_font_hierarchy_metrics(labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cluster bbox heights into tiers; optional symbol snap, propagation, and chain absorption."""
    extra_symbols_compact = _normalize_symbol_snap_extra(None)

    empty: Dict[str, Any] = {
        "level_consistency": _level_consistency([]),
        "font_score_breakdown": None,
    }
    if not labels:
        return empty

    heights: List[float] = []
    symbol_snap_candidate: List[bool] = []
    centres_row: List[Tuple[float, float]] = []

    for label in labels:
        text = (label.get("text") or "").strip()
        geom = _centre_and_size(label)
        if geom is None:
            continue
        cx, cy, _w, h = geom
        heights.append(float(h))
        centres_row.append((cx, cy))
        symbol_snap_candidate.append(
            _is_symbol_snap_label(text, DEFAULT_SYMBOL_SNAP_MAX_CHARS, extra_symbols_compact)
        )

    if not heights:
        return empty

    reliable_heights = [
        h for h, snap in zip(heights, symbol_snap_candidate) if not snap
    ]
    if not reliable_heights:
        reliable_heights = list(heights)
        snap_effective = [False] * len(heights)
    else:
        snap_effective = list(symbol_snap_candidate)

    cluster_ids_rel, centres_sorted = _cluster_heights_by_gap(
        reliable_heights, DEFAULT_MERGE_RELATIVE_THRESHOLD
    )
    names_ranked = _level_names(len(centres_sorted))
    id_to_name = {i: names_ranked[i] for i in range(len(centres_sorted))}

    n_tiers = int(centres_sorted.size)
    if n_tiers == 0:
        snap_majority = 0
        snap_largest = 0
    else:
        counts_rel = np.bincount(cluster_ids_rel.astype(int), minlength=n_tiers)
        snap_majority = int(np.argmax(counts_rel))
        snap_largest = int(np.argmax(centres_sorted))

    ri = 0
    cluster_ids_full: List[int] = []
    for h, snap in zip(heights, snap_effective):
        if snap:
            if n_tiers == 0:
                cluster_ids_full.append(0)
            elif DEFAULT_SYMBOL_SNAP_MODE == "nearest_height":
                cluster_ids_full.append(int(np.argmin(np.abs(centres_sorted - h))))
            elif DEFAULT_SYMBOL_SNAP_MODE == "largest":
                cluster_ids_full.append(snap_largest)
            else:
                cluster_ids_full.append(snap_majority)
        else:
            cluster_ids_full.append(int(cluster_ids_rel[ri]))
            ri += 1
    assert ri == len(cluster_ids_rel)

    if (
        DEFAULT_SYMBOL_PROPAGATION_MAX_ITERS > 0
        and any(snap_effective)
        and len(centres_row) == len(cluster_ids_full)
    ):
        max_dist = DEFAULT_SYMBOL_PROPAGATION_MAX_DISTANCE_PX
        if max_dist is None:
            med = float(np.median(np.array(heights, dtype=float)))
            max_dist = max(48.0, 3.0 * med)
        cluster_ids_full, _, _ = _propagate_symbol_tiers_to_max_neighbor(
            cluster_ids_full,
            snap_effective,
            centres_row,
            max_dist,
            DEFAULT_SYMBOL_PROPAGATION_MAX_ITERS,
        )

    if (
        DEFAULT_FONT_CHAIN_ABSORB_ENABLED
        and len(cluster_ids_full) == len(centres_row) == len(heights)
        and centres_sorted.size > 0
    ):
        link_px = DEFAULT_FONT_CHAIN_ABSORB_LINK_MAX_PX
        if link_px is None:
            med_h = float(np.median(np.array(heights, dtype=float)))
            link_px = max(120.0, 3.5 * med_h)
        cluster_ids_full, _, _ = _absorb_to_dominant_via_chain(
            cluster_ids_full,
            heights,
            centres_row,
            centres_sorted,
            enabled=True,
            min_dominant_fraction=DEFAULT_FONT_CHAIN_ABSORB_MIN_DOMINANT_FRACTION,
            link_max_px=link_px,
            title_height_ratio=DEFAULT_FONT_CHAIN_ABSORB_TITLE_HEIGHT_RATIO,
        )

    cluster_ids_arr = np.array(cluster_ids_full, dtype=int)
    level_names_per_label = [id_to_name[int(c)] for c in cluster_ids_arr]

    counts_by_id = np.bincount(cluster_ids_arr, minlength=len(centres_sorted))
    num_levels = int(np.count_nonzero(counts_by_id))

    level_consistency = _level_consistency(level_names_per_label)

    dom_frac = level_consistency.get("dominant_fraction")
    if dom_frac is None:
        font_score_breakdown = None
    else:
        font_score_breakdown = compute_font_hierarchy_score(
            num_levels,
            float(dom_frac),
            heights,
        )

    return {
        "level_consistency": level_consistency,
        "font_score_breakdown": font_score_breakdown,
    }
