from typing import Any, Dict, List


def compute_label_readability(
    labels: List[Dict[str, Any]],
    readability_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Returns mean confidence, count of labels strictly below the threshold, a
    list of those labels' details (text, confidence, geometry, index), and an
    aggregate status (readable high / readable low) from the mean.
    """
    thr = max(0.0, min(float(readability_threshold), 1.0))

    if not labels:
        return {
            "mean_confidence": 0.0,
            "labels_below_threshold": 0,
            "labels_below_threshold_details": [],
            "readability_status": "readable low",
            "readability_threshold": thr,
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

    mean_confidence = sum(confidences) / len(confidences)
    if mean_confidence >= thr:
        status = "high readability"
    else:
        status = "low readability"

    return {
        "mean_confidence": mean_confidence,
        "labels_below_threshold": below,
        "labels_below_threshold_details": below_details,
        "readability_status": status,
        "readability_threshold": thr,
    }
