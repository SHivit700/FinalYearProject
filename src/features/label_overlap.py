from typing import Any, Dict, List, Optional, Tuple


THRESHOLDS: dict = {
    # < gap_threshold → labels too close
    "gap_threshold": 0.1,

    # Diagram-level verdict bands, in terms of fraction of labels that are too close
    "fraction_too_close_moderate": 0.6,
    "fraction_too_close_crowded": 0.8,

    # Hard overlap: fraction of label pairs with IoU > 0 — above this → crowded
    "fraction_pairs_any_iou_crowded": 0.02,
}


def _aabb_gap(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    """
    Minimum Euclidean gap between two AABBs.
    Returns 0.0 if they touch or overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0.0, max(ay1, by1) - min(ay2, by2))
    return (gap_x ** 2 + gap_y ** 2) ** 0.5


def _font_normalised_gap(a: Tuple[float, float, float, float],
                         b: Tuple[float, float, float, float]) -> float:
    """
    Gap between two AABBs normalised by the average bbox height of the two
    labels (used as a font-size proxy).
    """
    raw = _aabb_gap(a, b)
    h_a = max(1.0, a[3] - a[1])
    h_b = max(1.0, b[3] - b[1])
    avg_h = (h_a + h_b) / 2.0
    return raw / avg_h


def _pairwise_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """
    Intersection over Union for two AABBs.
    Returns 0.0 if no overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe_aabbs_by_rounded_key(
    aabbs: List[Tuple[float, float, float, float]],
    indices_with_bbox: List[int],
    ndigits: int,
) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """
    Keep the first box for each key tuple(round(coord, ndigits) for coord in box).
    Drops later rows that match the same key.
    """
    seen: set = set()
    out_aabbs: List[Tuple[float, float, float, float]] = []
    out_indices: List[int] = []
    for box, src_idx in zip(aabbs, indices_with_bbox):
        key = tuple(round(c, ndigits) for c in box)
        if key in seen:
            continue
        seen.add(key)
        out_aabbs.append(box)
        out_indices.append(src_idx)
    return out_aabbs, out_indices


def compute_label_overlap_metrics(
    labels: List[Dict[str, Any]],
    image_shape: Tuple[int, ...],
    gap_threshold: float = THRESHOLDS["gap_threshold"],
    *,
    debug_log_zero_gap_pairs: bool = False,
    max_zero_gap_debug_prints: int = 10,
    dedupe_aabbs_round_ndigits: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute font-size-aware proximity metrics between all label pairs.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    n = len(labels)

    aabbs: List[Tuple[float, float, float, float]] = []
    indices_with_bbox: List[int] = []
    for idx, item in enumerate(labels):
        bbox = item.get("bbox")
        if not bbox:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        aabbs.append((min(xs), min(ys), max(xs), max(ys)))
        indices_with_bbox.append(idx)

    if dedupe_aabbs_round_ndigits is not None:
        aabbs, indices_with_bbox = _dedupe_aabbs_by_rounded_key(
            aabbs, indices_with_bbox, dedupe_aabbs_round_ndigits
        )

    _zero = {
        "fraction_labels_too_close": 0.0,
        "mean_normalised_gap": float("inf"),
        "fraction_pairs_too_close": 0.0,
        "fraction_pairs_any_iou": 0.0,
        "spacing_verdict": "well_spaced",
    }
    if n == 0 or height <= 0 or width <= 0 or not aabbs:
        return _zero

    m = len(aabbs)
    normalised_gaps: List[float] = []
    too_close_label_indices: set = set()
    iou_overlap_label_indices: set = set()
    too_close_pairs: int = 0
    total_pairs: int = m * (m - 1) // 2
    iou_overlap_pairs: int = 0
    zero_gap_debug_remaining = max_zero_gap_debug_prints if debug_log_zero_gap_pairs else 0

    for i in range(m):
        for j in range(i + 1, m):
            raw_gap = _aabb_gap(aabbs[i], aabbs[j])
            h_i = max(1.0, aabbs[i][3] - aabbs[i][1])
            h_j = max(1.0, aabbs[j][3] - aabbs[j][1])
            ng = raw_gap / ((h_i + h_j) / 2.0)
            normalised_gaps.append(ng)

            iou = _pairwise_iou(aabbs[i], aabbs[j])
            if zero_gap_debug_remaining > 0 and ng == 0.0:
                dbg_n = max_zero_gap_debug_prints - zero_gap_debug_remaining + 1
                ti = labels[indices_with_bbox[i]].get("text", "")
                tj = labels[indices_with_bbox[j]].get("text", "")
                print(
                    "[overlap-debug] zero-gap pair "
                    f"{dbg_n}/{max_zero_gap_debug_prints} | "
                    f"image_shape=(h={height}, w={width}) | "
                    f"pair_indices=({indices_with_bbox[i]}, {indices_with_bbox[j]}) | "
                    f"text_i={ti!r} text_j={tj!r} | "
                    f"aabb_i={aabbs[i]} aabb_j={aabbs[j]} | "
                    f"raw_gap={raw_gap:.6g} normalised_gap={ng:.6g} iou={iou:.6g}"
                )
                zero_gap_debug_remaining -= 1

            # IoU > 0 → literal overlap
            if iou > 0.0:
                iou_overlap_pairs += 1
                iou_overlap_label_indices.add(i)
                iou_overlap_label_indices.add(j)
                too_close_label_indices.add(i)
                too_close_label_indices.add(j)

            # A pair is "too close" if normalised gap is below threshold:
            if ng < gap_threshold:
                too_close_label_indices.add(i)
                too_close_label_indices.add(j)
                too_close_pairs += 1

    mean_ng = sum(normalised_gaps) / len(normalised_gaps) if normalised_gaps else float("inf")
    min_ng = min(normalised_gaps) if normalised_gaps else float("inf")
    fraction_labels_too_close = len(too_close_label_indices) / n
    fraction_pairs_too_close = too_close_pairs / total_pairs if total_pairs > 0 else 0.0
    fraction_pairs_any_iou = iou_overlap_pairs / total_pairs if total_pairs > 0 else 0.0

    # Diagram-level verdict
    if fraction_labels_too_close < THRESHOLDS["fraction_too_close_moderate"]:
        spacing_verdict = "well_spaced"
    elif fraction_labels_too_close <= THRESHOLDS["fraction_too_close_crowded"]:
        spacing_verdict = "moderate_overlap"
    else:
        spacing_verdict = "crowded labels"

    if fraction_pairs_any_iou > THRESHOLDS["fraction_pairs_any_iou_crowded"]:
        spacing_verdict = "crowded labels"

    # Build per-label info for visualization
    per_label_info: List[Dict[str, Any]] = []
    for aabb_idx in too_close_label_indices:
        src_idx = indices_with_bbox[aabb_idx]
        lbl = labels[src_idx]
        x1, y1, x2, y2 = aabbs[aabb_idx]
        per_label_info.append({
            "bbox": lbl.get("bbox", []),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "text": lbl.get("text", ""),
            "has_iou_overlap": aabb_idx in iou_overlap_label_indices,
        })

    return {
        "fraction_labels_too_close": fraction_labels_too_close,
        "mean_normalised_gap": mean_ng,
        "fraction_pairs_too_close": fraction_pairs_too_close,
        "fraction_pairs_any_iou": fraction_pairs_any_iou,
        "spacing_verdict": spacing_verdict,
        "per_label_info": per_label_info,
    }
