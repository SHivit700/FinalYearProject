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
from features.isolated_box_detection import compute_isolated_box_metrics
from features.brevity_score import compute_brevity_score
from features.symmetry import compute_symmetry_score_from_diagram
from features.whitespace_distribution import compute_whitespace_distribution_from_diagram
from features.color_harmony import compute_color_harmony_score


def extract_features_for_image(image_path: str, lang: str = "en", diagram_type: str = "system_design") -> Dict[str, Any]:
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

    island_detection_output_path = (
        image_path.parent.parent
        / "island_detection"
        / f"island_detection_{image_path.stem}.png"
    )
    _island_result = compute_isolated_box_metrics(
        bgr_image,
        shapes,
        image_shape,
        output_path=str(island_detection_output_path),
    )
    features["isolated_boxes"] = {
        "total_box_count": _island_result["total_box_count"],
        "connected_count": _island_result["connected_count"],
        "island_count": _island_result["island_count"],
        "island_fraction": _island_result["island_fraction"],
        "isolated_box_score": _island_result["isolated_box_score"],
        "island_boxes": _island_result["island_boxes"],
    }

    features["brevity"] = compute_brevity_score(labels, shapes, diagram_type=diagram_type)

    features["symmetry"] = compute_symmetry_score_from_diagram(
        labels, shapes, image_shape
    )

    features["whitespace_distribution"] = compute_whitespace_distribution_from_diagram(
        labels, shapes, image_shape, bgr_image=bgr_image
    )

    features["color_harmony"] = compute_color_harmony_score(bgr_image, labels)

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
    parser.add_argument(
        "--diagram-type",
        default="system_design",
        choices=["system_design", "timeline_roadmap"],
        help="Diagram type — controls brevity score thresholds",
    )
    args = parser.parse_args()

    path = Path(args.image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    result = extract_features_for_image(str(path), lang=args.lang, diagram_type=args.diagram_type)

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

    print("---------------ISOLATED BOXES------------------")
    ib = feats["isolated_boxes"]
    print(
        f"[ENTRY] isolated_box_score: {ib['isolated_box_score']} "
        f"(islands={ib['island_count']}/{ib['total_box_count']}, "
        f"island_fraction={ib['island_fraction']:.4f})"
    )
    for box in ib["island_boxes"]:
        print(f"[ENTRY]   island box: x={box['x']} y={box['y']} w={box['w']} h={box['h']}")

    print("---------------BREVITY------------------")
    brev = feats["brevity"]
    if brev["brevity_quality_score"] is None:
        print("[ENTRY] brevity_quality_score: N/A (no valid labels)")
    else:
        conf_flag = " [low confidence: single label]" if brev["low_confidence"] else ""
        print(
            f"[ENTRY] brevity_quality_score: {brev['brevity_quality_score']:.2f}{conf_flag}"
        )
        print(
            f"[ENTRY]   mean_chars={brev['mean_chars']} median={brev['median_chars']} "
            f"p90={brev['p90_chars']} skewness={brev['skewness']}"
        )
        print(
            f"[ENTRY]   verbose_ratio={brev['verbose_ratio']:.4f} "
            f"paragraph_ratio={brev['paragraph_ratio']:.4f}"
        )
        lpb = brev["labels_per_box"]
        print(
            f"[ENTRY]   labels_per_box={'N/A' if lpb is None else f'{lpb:.2f}'} "
            f"skipped_labels={brev['skipped_labels']}"
        )

    print("---------------SYMMETRY------------------")
    sym = feats["symmetry"]
    if sym["symmetry_score"] is None:
        print(f"[ENTRY] symmetry_score: N/A (node_count={sym['node_count']})")
    else:
        conf_flag = " [low confidence: all nodes at centroid]" if sym["low_confidence"] else ""
        print(
            f"[ENTRY] symmetry_score: {sym['symmetry_score']:.2f}{conf_flag}"
        )
        print(
            f"[ENTRY]   horizontal={sym['horizontal_symmetry']:.2f} "
            f"vertical={sym['vertical_symmetry']:.2f} "
            f"dominant_axis={sym['dominant_axis']} "
            f"node_count={sym['node_count']}"
        )

    print("---------------WHITESPACE DISTRIBUTION------------------")
    wds = feats["whitespace_distribution"]
    if wds["whitespace_distribution_score"] is None:
        print(f"[ENTRY] whitespace_distribution_score: N/A (low_confidence={wds['low_confidence']})")
    else:
        deg_flag = " [degenerate layout]" if wds["degenerate_layout"] else ""
        print(
            f"[ENTRY] whitespace_distribution_score: {wds['whitespace_distribution_score']:.2f}{deg_flag}"
        )
        print(
            f"[ENTRY]   density_cov={wds['density_cov']:.4f} "
            f"density_mean={wds['density_mean']:.4f} "
            f"density_std={wds['density_std']:.4f} "
            f"empty_cell_ratio={wds['empty_cell_ratio']:.4f} "
            f"grid={wds['grid_resolution']}x{wds['grid_resolution']}"
        )

    print("---------------COLOR HARMONY------------------")
    chs = feats["color_harmony"]
    if chs["color_harmony_score"] is None:
        print("[ENTRY] color_harmony_score: N/A (image read failure)")
    else:
        flags = []
        if chs["monochrome"]:
            flags.append("monochrome")
        if chs["low_confidence"]:
            flags.append("low confidence")
        if chs["text_colors_unavailable"]:
            flags.append("text colors unavailable")
        if chs["contrast_violation"]:
            flags.append("contrast violation")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(
            f"[ENTRY] color_harmony_score: {chs['color_harmony_score']:.2f}{flag_str}"
        )
        print(
            f"[ENTRY]   palette_size={chs['palette_size']} "
            f"(bg={chs['background_palette_size']} text={chs['text_palette_size']}) "
            f"harmony_ratio={chs['harmony_ratio']:.4f} "
            f"palette_penalty={chs['palette_penalty']:.2f} "
            f"skipped_labels={chs['skipped_labels']}"
        )
        print(f"[ENTRY]   dominant_colors_hex={chs['dominant_colors_hex']}")

    print("--------------------------------")
