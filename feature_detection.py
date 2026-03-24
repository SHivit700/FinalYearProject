#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict

from detect_labels import run_label_detection
from label_area_ratio import compute_label_area_ratio
from label_overlap import compute_label_overlap_metrics
from label_readability import compute_label_readability


def extract_features_for_image(image_path: str, lang: str = "en") -> Dict[str, Any]:
    """
    Top-level feature extraction entry point.

    Flow:
        1. Run OCR to detect labels.
        2. Feed detections into feature modules.
        3. Aggregate all feature values into a single dict for downstream quality modelling.
    """
    image_path = Path(image_path)
    # Same level as the diagrams folder (e.g. Data/labels_output/ next to Data/Diagrams/).
    output_path = (
        image_path.parent.parent
        / "labels_output"
        / f"{image_path.stem}_labels.png"
    )

    # Step 1: OCR / label detection
    detection_result = run_label_detection(str(image_path), lang=lang, output_path=output_path)
    labels = detection_result["labels"]
    image_shape = detection_result["image_shape"]

    # Step 2: feature modules
    features: Dict[str, Any] = {}

    # 2.1 Label area ratio (label balance & visual complexity)
    features["label_area"] = compute_label_area_ratio(labels, image_shape)

    # 2.2 Label readability (text quality & clarity)
    features["label_readability"] = compute_label_readability(labels)

    # 2.3 Label overlap (text density & visual clutter)
    features["overlap_metrics"] = compute_label_overlap_metrics(labels, image_shape)

    return {
        "image_path": str(image_path),
        "labels": labels,
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

    result = extract_features_for_image(args.image_path, lang=args.lang)

    print("--------------------------------")
    print(f"[ENTRY] Processed image: {result['image_path']}")

    print("--------------------------------")
    print(f"[ENTRY] Detected {len(result['labels'])} label(s).")

    # Log the feature(s)
    print("--------------------------------")
    feats = result["features"]
    label_area = feats["label_area"]
    print(
        f"[ENTRY] label_area_ratio: {label_area.ratio:.4f} "
        f"({label_area.category})"
    )
    overlap_metrics = feats["overlap_metrics"]

    print("--------------------------------")
    print(f"[ENTRY] Text overlap ratio: {overlap_metrics['overlap_ratio']:.4f}")
    print(
        f"[ENTRY] Fraction of labels overlapping: {overlap_metrics['fraction_labels_overlapping']:.4f}"
    )

    print("--------------------------------")
    label_readability = feats["label_readability"]
    print(
        f"[ENTRY] label_readability: mean_confidence={label_readability['mean_confidence']:.4f} "
        f"(threshold={label_readability['readability_threshold']:.4f}), "
        f"labels_below_threshold={label_readability['labels_below_threshold']}, "
        f"status={label_readability['readability_status']}"
    )
    for row in label_readability["labels_below_threshold_details"]:
        print(
            f"[ENTRY]   low-confidence label #{row['index']}: "
            f"{row['text']!r} conf={row['confidence']:.4f}"
        )

    print("--------------------------------")