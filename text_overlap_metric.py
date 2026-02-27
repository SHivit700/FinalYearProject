"""
Text overlap ratio metric for graph label quality.

Metric definition:
- Each detected label has a bounding polygon (quadrilateral from OCR).
- Total label area = sum of areas of all label boxes.
- Union area = area of the union of all label polygons (no double-counting).
- Overlapping area = Total label area - Union area.
- Text overlap ratio = Overlapping area / Total label area  (in [0, 1]).

Interpretation:
- 0: no overlap (all labels disjoint).
- 1: complete overlap (all labels occupy the same region).
- Values in (0, 1): fraction of total label area that is covered by more than one label.
"""

from __future__ import annotations


def _bbox_to_polygon(bbox: list[list[float]]) -> "Polygon":
    """Convert EasyOCR-style bbox (list of 4 [x,y] points) to a Shapely polygon."""
    from shapely.geometry import Polygon

    # Ensure closed ring (first point repeated at end if needed)
    coords = [tuple(p) for p in bbox]
    if coords and coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return Polygon(coords)


def compute_text_overlap_ratio(bboxes: list[list[list[float]]]) -> dict[str, float]:
    """
    Compute the text overlap ratio from a list of label bounding boxes.

    Each bbox is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    (e.g. EasyOCR readtext output).

    Returns:
        dict with:
        - overlap_ratio: in [0, 1]; 0 = no overlap, 1 = full overlap.
        - total_label_area: sum of areas of all boxes.
        - union_area: area of union of all boxes.
        - overlapping_area: total_label_area - union_area.
        - fraction_labels_overlapping: proportion of labels that intersect at least one other (0–1).
    """
    if not bboxes:
        return {
            "overlap_ratio": 0.0,
            "total_label_area": 0.0,
            "union_area": 0.0,
            "overlapping_area": 0.0,
            "fraction_labels_overlapping": 0.0,
        }

    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        raise ImportError("shapely is required for text overlap ratio. Install with: pip install shapely")

    polygons = []
    for bbox in bboxes:
        try:
            p = _bbox_to_polygon(bbox)
            if not p.is_valid:
                p = p.buffer(0)  # fix invalid polygon
            if not p.is_empty and p.area > 0:
                polygons.append(p)
        except Exception:
            continue

    if not polygons:
        return {
            "overlap_ratio": 0.0,
            "total_label_area": 0.0,
            "union_area": 0.0,
            "overlapping_area": 0.0,
            "fraction_labels_overlapping": 0.0,
        }

    total_area = sum(p.area for p in polygons)
    union = unary_union(polygons)
    union_area = float(union.area) if union and not union.is_empty else 0.0
    overlapping_area = max(0.0, total_area - union_area)
    overlap_ratio = overlapping_area / total_area if total_area > 0 else 0.0

    # Fraction of labels that overlap at least one other
    n_overlapping = 0
    for i, pi in enumerate(polygons):
        for j, pj in enumerate(polygons):
            if i != j and pi.intersects(pj):
                n_overlapping += 1
                break
    fraction_overlapping = n_overlapping / len(polygons) if polygons else 0.0

    return {
        "overlap_ratio": round(overlap_ratio, 4),
        "total_label_area": round(total_area, 2),
        "union_area": round(union_area, 2),
        "overlapping_area": round(overlapping_area, 2),
        "fraction_labels_overlapping": round(fraction_overlapping, 4),
    }
