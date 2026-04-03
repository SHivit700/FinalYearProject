#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict

from detection.detect_labels import run_label_detection
from detection.detect_shapes import run_shape_detection
from features.label_area_ratio import compute_label_area_ratio
from features.label_overlap import compute_label_overlap_metrics
from features.label_readability import compute_label_readability


def extract_features_for_image(image_path: str, lang: str = "en") -> Dict[str, Any]:
    """
    Top-level feature extraction entry point.

    Flow:
        1. Run OCR to detect labels.
        2. Feed detections into feature modules.
        3. Aggregate all feature values into a single dict for downstream quality modelling.
    """
    image_path = Path(image_path)
    # Same level as the diagrams folder (e.g. data/labels_output/, data/shapes_output/ next to data/diagrams/).
    output_path = (
        image_path.parent.parent
        / "labels_output"
        / f"{image_path.stem}_labels.png"
    )

    # Step 1: OCR / label detection
    detection_result = run_label_detection(str(image_path), lang=lang, output_path=output_path)
    labels = detection_result["labels"]
    image_shape = detection_result["image_shape"]

    shapes_output_path = (
        image_path.parent.parent
        / "shapes_output"
        / f"shapes_{image_path.stem}.png"
    )
    shapes, _shape_image_shape, _shape_highlighted = run_shape_detection(
        str(image_path),
        output_path=str(shapes_output_path),
    )

    # Step 2: feature modules
    features: Dict[str, Any] = {}

    # 2.1 Label area ratio (label balance & visual complexity)
    features["label_area"] = compute_label_area_ratio(labels, image_shape)

    # 2.2 Label readability (text quality & clarity)
    features["label_readability"] = compute_label_readability(labels)

    # 2.3 Label overlap (font-normalised gap threshold and IoU)
    features["overlap_metrics"] = compute_label_overlap_metrics(labels, image_shape)

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
        description="Global feature extraction entry point for diagram quality."
    )
    parser.add_argument("image_path", help="Path to input diagram image")
    parser.add_argument("--lang", default="en", help="OCR language codes (comma-separated)")
    args = parser.parse_args()

    path = Path(args.image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    result = extract_features_for_image(args.image_path, lang=args.lang)

    print("---------------IMAGE------------------")
    print(f"[ENTRY] Processed image: {result['image_path']}")

    print("---------------LABELS------------------")
    print(f"[ENTRY] Detected {len(result['labels'])} label(s).")

    print("---------------SHAPES------------------")
    shapes = result["shapes"]
    print(f"[ENTRY] Detected {len(shapes)} contour(s).")

    # Log the feature(s)
    print("---------------LABEL AREA------------------")
    feats = result["features"]
    label_area = feats["label_area"]
    print(
        f"[ENTRY] label_area_ratio: {label_area.ratio:.4f} "
        f"({label_area.category})"
    )

    print("--------------LABEL OVERLAP METRICS------------------")
    overlap_metrics = feats["overlap_metrics"]
    print(f"[ENTRY] spacing_verdict: {overlap_metrics['spacing_verdict']}")
    print(f"[ENTRY] fraction_labels_too_close: {overlap_metrics['fraction_labels_too_close']:.4f}")
    print(f"[ENTRY] fraction_pairs_too_close: {overlap_metrics['fraction_pairs_too_close']:.4f}")
    print(f"[ENTRY] mean_normalised_gap: {overlap_metrics['mean_normalised_gap']:.4f}")
    print(f"[ENTRY] fraction_pairs_any_iou: {overlap_metrics['fraction_pairs_any_iou']:.4f}")

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

    print("--------------------------------")