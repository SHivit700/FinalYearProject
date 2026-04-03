from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple


@dataclass(frozen=True)
class LabelArea:
    """
    Label-area feature
    """
    ratio: float
    category: Literal["sparse", "balanced", "cluttered"]


def compute_label_area_ratio(
    labels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    sparse_threshold: float = 0.08,
    cluttered_threshold: float = 0.13,
) -> LabelArea:
    """
    Compute how much of the image is occupied by text labels and categorise it as sparse / balanced / cluttered.

    ratio = total_label_pixels / total_image_pixels

    Heuristic defaults:
        - ratio < sparse_threshold                         -> "sparse"
        - sparse_threshold <= ratio <= cluttered_threshold -> "balanced"
        - ratio > cluttered_threshold                      -> "cluttered"
    """
    height, width = image_shape[:2]
    total_pixels = float(height * width)
    if total_pixels == 0:
        # No canvas
        return LabelArea(ratio=0.0, category="sparse")

    total_label_area = 0.0

    # Calculate the total area of all labels by approximating each label's polygonal bbox by its axis-aligned bounding rectangle.
    for item in labels:
        bbox = item.get("bbox")
        if not bbox:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)
        total_label_area += w * h

    ratio = total_label_area / total_pixels

    if ratio < sparse_threshold:
        category: Literal["sparse", "balanced", "cluttered"] = "sparse"
    elif ratio > cluttered_threshold:
        category = "cluttered"
    else:
        category = "balanced"

    return LabelArea(ratio=ratio, category=category)
