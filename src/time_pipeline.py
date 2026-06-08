#!/usr/bin/env python3
"""
Timing harness for the four pipeline stages described in evaluation section 4.2.1.

Usage (from project root):
    python src/time_pipeline.py <image_path>
    python src/time_pipeline.py <image_path> --no-llm

Output: a clean table of wall-clock times per stage, ready to paste.
"""
import argparse
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detection.detect_shapes import run_shape_detection
from detection.detect_labels import run_label_detection

from features.edge_margin_ratio import compute_edge_margin_metrics
from features.container_utilization.container_utilization_detection import compute_container_utilization_metrics
from features.font_hierarchy import compute_font_hierarchy_metrics
from features.label_area_ratio import compute_label_area_ratio
from features.label_overlap import compute_label_overlap_metrics
from features.label_readability import compute_label_readability
from features.layout_structure_score import compute_layout_structure_score
from features.isolated_box_detection import compute_isolated_box_metrics
from features.brevity_score import compute_brevity_score
from features.whitespace_distribution import compute_whitespace_distribution_from_diagram
from features.color_harmony import compute_color_harmony_score
from features.cognitive_chunk_density import compute_cognitive_chunk_density_from_diagram
from features.label_contrast_quality import compute_label_contrast_quality
from features.orientation_consistency import compute_orientation_consistency
from features.edge_detection_visualization import compute_edge_detection_visualization

from feature_detection import _augment_labels_with_tilted_shapes
from suggestion_engine import generate_suggestions, new_session


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}s"


def run(image_path: str, use_llm: bool) -> None:
    path = Path(image_path).resolve()
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nImage : {path.name}")
    print(f"LLM   : {'enabled' if use_llm else 'disabled'}")
    print("─" * 50)

    # Stage 1 — Shape detection
    t0 = time.perf_counter()
    shapes, _shape_image_shape, _shape_highlighted = run_shape_detection(str(path))
    t_shape = time.perf_counter() - t0

    # Stage 2 — Label detection (EasyOCR)
    t0 = time.perf_counter()
    detection_result = run_label_detection(str(path), lang="en")
    t_ocr = time.perf_counter() - t0

    labels = detection_result["labels"]
    image_shape = detection_result["image_shape"]
    bgr_image = detection_result["bgr_image"]
    margin_fraction = 0.05

    # Stage 3 — Feature extraction
    t0 = time.perf_counter()

    features: dict = {}
    features["label_area"] = compute_label_area_ratio(labels, image_shape)
    features["label_readability"] = compute_label_readability(labels)
    features["overlap_metrics"] = compute_label_overlap_metrics(labels, image_shape)
    features["edge_clearance"] = compute_edge_margin_metrics(labels, shapes, image_shape, margin_fraction=margin_fraction)
    features["layout_structure"] = compute_layout_structure_score(shapes, image_shape)
    features["font_hierarchy"] = compute_font_hierarchy_metrics(labels)

    _cu = compute_container_utilization_metrics(bgr_image, labels, shapes, image_shape)
    features["container_utilization"] = {
        "container_utilization_score": _cu["image_metrics"]["container_utilization_score"],
        "empty_container_boxes": [
            {"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]}
            for d in _cu.get("box_details", []) if d.get("is_tier1_empty", False)
        ],
        "empty_container_candidate_boxes": [
            {"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]}
            for d in _cu.get("box_details", []) if d.get("is_tier2_candidate", False)
        ],
    }

    _ib = compute_isolated_box_metrics(bgr_image, shapes, image_shape)
    features["isolated_boxes"] = {
        "total_box_count": _ib["total_box_count"],
        "connected_count": _ib["connected_count"],
        "island_count": _ib["island_count"],
        "island_fraction": _ib["island_fraction"],
        "isolated_box_score": _ib["isolated_box_score"],
        "island_boxes": _ib["island_boxes"],
    }

    features["brevity"] = compute_brevity_score(labels, shapes)
    features["whitespace_distribution"] = compute_whitespace_distribution_from_diagram(labels, shapes, image_shape, bgr_image=bgr_image)
    features["color_harmony"] = compute_color_harmony_score(bgr_image, labels)
    features["label_contrast"] = compute_label_contrast_quality(bgr_image, labels)
    features["cognitive_chunk_density"] = compute_cognitive_chunk_density_from_diagram(labels, shapes, image_shape)

    _oc_labels = _augment_labels_with_tilted_shapes(labels, shapes)
    features["orientation_consistency"] = compute_orientation_consistency(_oc_labels)
    features["edge_detection"] = compute_edge_detection_visualization(bgr_image, shapes, image_shape, labels=labels)
    features["image_shape"] = image_shape
    features["shapes"] = shapes

    t_features = time.perf_counter() - t0

    # Stage 4 — LLM synthesis (optional)
    t_llm = None
    if use_llm:
        session = new_session("system_design")
        img_shape_2d = (image_shape[0], image_shape[1])

        t0 = time.perf_counter()
        generate_suggestions(
            features,
            diagram_type="system_design",
            image_path=str(path),
            use_llm=True,
            session=session,
            img_shape=img_shape_2d,
        )
        t_llm = time.perf_counter() - t0

    total_no_llm = t_shape + t_ocr + t_features
    total_with_llm = total_no_llm + (t_llm or 0.0)

    print(f"{'Stage':<35} {'Time':>8}")
    print("─" * 50)
    print(f"{'1. Shape detection (OpenCV)':<35} {_fmt(t_shape):>8}")
    print(f"{'2. Label detection (EasyOCR)':<35} {_fmt(t_ocr):>8}")
    print(f"{'3. Feature extraction (13 metrics)':<35} {_fmt(t_features):>8}")
    if t_llm is not None:
        print(f"{'4. LLM synthesis (gpt-4o-mini)':<35} {_fmt(t_llm):>8}")
    print("─" * 50)
    print(f"{'Total (without LLM)':<35} {_fmt(total_no_llm):>8}")
    if t_llm is not None:
        print(f"{'Total (with LLM)':<35} {_fmt(total_with_llm):>8}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time the four pipeline stages.")
    parser.add_argument("image_path", help="Path to the diagram image to analyse.")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM synthesis stage.")
    args = parser.parse_args()

    run(args.image_path, use_llm=not args.no_llm)
