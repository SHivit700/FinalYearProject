#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict

from detection.detect_labels import run_label_detection
from detection.detect_shapes import run_shape_detection
from features.edge_margin_ratio import compute_edge_margin_metrics
from features.container_utilization.container_utilization_detection import compute_container_utilization_metrics
from features.font_hierarchy import compute_font_hierarchy_metrics
from features.label_area_ratio import compute_label_area_ratio
from features.label_overlap import compute_label_overlap_metrics
from features.label_readability import compute_label_readability
from features.layout_structure_score import compute_layout_structure_score


def extract_features_for_image(image_path: str, lang: str = "en") -> Dict[str, Any]:
    """OCR and shape detection, then aggregate feature metrics into one dict."""
    image_path = Path(image_path)
    # Previews go next to the diagram tree (e.g. labels_output/, shapes_output/).
    output_path = (
        image_path.parent.parent
        / "labels_output"
        / f"{image_path.stem}_labels.png"
    )

    detection_result = run_label_detection(str(image_path), lang=lang, output_path=output_path)
    labels = detection_result["labels"]
    image_shape = detection_result["image_shape"]
    bgr_image = detection_result["bgr_image"]

    shapes_output_path = (
        image_path.parent.parent
        / "shapes_output"
        / f"shapes_{image_path.stem}.png"
    )
    margin_fraction = 0.05
    shapes, _shape_image_shape, _shape_highlighted = run_shape_detection(
        str(image_path),
        output_path=str(shapes_output_path),
    )

    container_utilization_overlay_path = (
        image_path.parent.parent
        / "shapes_output"
        / f"container_utilization_{image_path.stem}.png"
    )

    features: Dict[str, Any] = {}

    features["label_area"] = compute_label_area_ratio(labels, image_shape)
    features["label_readability"] = compute_label_readability(labels)
    features["overlap_metrics"] = compute_label_overlap_metrics(labels, image_shape)
    features["edge_clearance"] = compute_edge_margin_metrics(
        labels, shapes, image_shape, margin_fraction=margin_fraction
    )
    features["layout_structure"] = compute_layout_structure_score(shapes, image_shape)
    features["font_hierarchy"] = compute_font_hierarchy_metrics(labels)
    _container_utilization = compute_container_utilization_metrics(
        bgr_image,
        labels,
        shapes,
        image_shape,
        output_path=str(container_utilization_overlay_path),
    )
    features["container_utilization"] = {
        "container_utilization_score": _container_utilization["image_metrics"][
            "container_utilization_score"
        ],
    }

    return {
        "image_path": str(image_path),
        "labels": labels,
        "shapes": shapes,
        "image_shape": image_shape,
        "features": features,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract diagram features, save label/shape previews, print metrics.",
    )
    parser.add_argument("image_path", help="Path to input diagram image")
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language codes for EasyOCR (comma-separated)",
    )
    args = parser.parse_args()

    path = Path(args.image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    result = extract_features_for_image(str(path), lang=args.lang)

    print("---------------IMAGE------------------")
    print(f"[ENTRY] Processed image: {result['image_path']}")

    print("---------------LABELS------------------")
    print(f"[ENTRY] Detected {len(result['labels'])} label(s).")

    print("---------------SHAPES------------------")
    print(f"[ENTRY] Detected {len(result['shapes'])} contour(s).")

    feats = result["features"]

    print("---------------LABEL AREA------------------")
    label_area = feats["label_area"]
    print(
        f"[ENTRY] label_area_ratio: {label_area.ratio:.4f} "
        f"({label_area.category})"
    )

    print("---------------EDGE CLEARANCE------------------")
    ec = feats["edge_clearance"]
    print(
        f"[ENTRY] labels_fraction_violating={ec['labels_fraction_violating']:.4f}, "
        f"shapes_fraction_violating={ec['shapes_fraction_violating']:.4f}"
    )

    print("--------------LABEL OVERLAP METRICS------------------")
    overlap_metrics = feats["overlap_metrics"]
    print(f"[ENTRY] spacing_verdict: {overlap_metrics['spacing_verdict']}")
    print(
        f"[ENTRY] fraction_labels_too_close: {overlap_metrics['fraction_labels_too_close']:.4f}"
    )
    print(
        f"[ENTRY] fraction_pairs_too_close: {overlap_metrics['fraction_pairs_too_close']:.4f}"
    )
    print(
        f"[ENTRY] mean_normalised_gap: {overlap_metrics['mean_normalised_gap']:.4f}"
    )
    print(
        f"[ENTRY] fraction_pairs_any_iou: {overlap_metrics['fraction_pairs_any_iou']:.4f}"
    )

    print("---------------LABEL READABILITY------------------")
    label_readability = feats["label_readability"]
    print(
        f"[ENTRY] label_readability: mean_confidence={label_readability['mean_confidence']:.4f} "
        f"(threshold={label_readability['readability_threshold']:.4f}), "
        f"labels_below_threshold={label_readability['labels_below_threshold']}, "
        f"fraction_below={label_readability['fraction_labels_below_threshold']:.4f} "
        f"(max_fraction_ok={label_readability['max_fraction_below_threshold']:.4f}), "
        f"status={label_readability['readability_status']}"
    )
    for row in label_readability["labels_below_threshold_details"]:
        print(
            f"[ENTRY]   low-confidence label #{row['index']}: "
            f"{row['text']!r} conf={row['confidence']:.4f}"
        )

    print("---------------LAYOUT STRUCTURE (ML)------------------")
    layout = feats["layout_structure"]
    print(f"[ENTRY] layout_structure_score: {layout['layout_structure_score']:.4f}")

    print("---------------FONT HIERARCHY------------------")
    fh = feats["font_hierarchy"]
    lc = fh["level_consistency"]
    if lc["n_labels"] == 0:
        print("[ENTRY] level_consistency: (no labels)")
    else:
        print(
            f"[ENTRY] level_consistency: n={lc['n_labels']} "
            f"dominant_level={lc['dominant_level']!r} "
            f"fraction={lc['dominant_fraction']:.3f} by_level={lc['by_level']}"
        )
    fsb = fh.get("font_score_breakdown")
    if fsb is not None:
        print(
            f"[ENTRY] font_score: {fsb['font_score']:.2f} "
            f"(S_L={fsb['S_L']:.1f} S_C={fsb['S_C']:.1f} S_S={fsb['S_S']:.1f}, "
            f"r={fsb['r_spread']:.3f}, h_min={fsb['h_min_px']} h_max={fsb['h_max_px']})"
        )

    print("---------------CONTAINER UTILIZATION------------------")
    cu_score = feats["container_utilization"]["container_utilization_score"]
    print(f"[ENTRY] container_utilization_score: {cu_score:.6f}")

    print("--------------------------------")
