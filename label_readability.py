from typing import Any, Dict, List


def compute_label_readability(
    labels: List[Dict[str, Any]],
    readability_threshold: float = 0.4,
    max_fraction_below_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Returns mean confidence, count of labels strictly below the confidence
    threshold, a list of those labels' details, and an aggregate status.

    Status is low readability if either:
      - mean confidence is below ``readability_threshold``, or
      - strictly more than ``max_fraction_below_threshold`` of
        labels have confidence below ``readability_threshold``.

    Otherwise status is **high readability** (both mean and the low-confidence
    fraction are acceptable).
    """
    thr = max(0.0, min(float(readability_threshold), 1.0))
    frac_limit = max(0.0, min(float(max_fraction_below_threshold), 1.0))

    if not labels:
        return {
            "mean_confidence": 0.0,
            "labels_below_threshold": 0,
            "fraction_labels_below_threshold": 0.0,
            "labels_below_threshold_details": [],
            "readability_status": "low readability",
            "readability_threshold": thr,
            "max_fraction_below_threshold": frac_limit,
        }

    confidences: List[float] = []
    below = 0
    below_details: List[Dict[str, Any]] = []

    for i, item in enumerate(labels):
        try:
            c = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            c = 0.0
        c = max(0.0, min(1.0, c))
        confidences.append(c)
        if c < thr:
            below += 1
            below_details.append(
                {
                    "index": i,
                    "text": item.get("text", ""),
                    "confidence": c,
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "bbox": item.get("bbox"),
                }
            )

    n = len(confidences)
    mean_confidence = sum(confidences) / n
    fraction_below = below / n

    mean_ok = mean_confidence >= thr
    fraction_ok = fraction_below <= frac_limit
    if mean_ok and fraction_ok:
        status = "high readability"
    else:
        status = "low readability"

    return {
        "mean_confidence": mean_confidence,
        "labels_below_threshold": below,
        "fraction_labels_below_threshold": fraction_below,
        "labels_below_threshold_details": below_details,
        "readability_status": status,
        "readability_threshold": thr,
        "max_fraction_below_threshold": frac_limit,
    }
