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
from features.whitespace_distribution import compute_whitespace_distribution_from_diagram
from features.color_harmony import compute_color_harmony_score
from features.cognitive_chunk_density import compute_cognitive_chunk_density_from_diagram
from features.edge_detection_visualization import compute_edge_detection_visualization
from features.label_contrast_quality import compute_label_contrast_quality
from features.label_contrast_quality_visualization import compute_label_contrast_visualization
from features.edge_clearance_visualization import compute_edge_clearance_visualization
from features.label_readability_visualization import compute_label_readability_visualization
from features.label_overlap_visualization import compute_label_overlap_visualization
from features.orientation_consistency import compute_orientation_consistency
from features.orientation_consistency_visualization import compute_orientation_consistency_visualization


def _augment_labels_with_tilted_shapes(labels: list, shapes: list) -> list:
    """Return labels plus synthetic entries for tilted shapes with no OCR coverage.

    A shape is "tilted" when its minAreaRect angle is not near 0° or -90°.
    A shape is "covered" when an existing OCR label sits mostly inside it
    (intersection / label_area > 50 %).  Uncovered tilted shapes get a
    synthetic label whose bbox uses the minAreaRect corners — this gives a
    non-zero dy so _label_angle_deg can compute the true rotation angle.
    """
    import cv2

    _MIN_SHAPE_AREA = 2000   # px² — ignore tiny noise contours
    _TILT_LO, _TILT_HI = -75.0, -15.0   # minAreaRect angles considered "tilted"
    _COVERED_THRESH = 0.50   # fraction of label area that must overlap the shape

    label_aabbs = []
    for lbl in labels:
        bbox = lbl.get("bbox", [])
        if not bbox:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        label_aabbs.append((min(xs), min(ys), max(xs), max(ys)))

    augmented = list(labels)

    for shape in shapes:
        cnt = shape.get("contour")
        if cnt is None:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), angle = rect

        if w * h < _MIN_SHAPE_AREA:
            continue
        if not (_TILT_LO < angle < _TILT_HI):
            continue

        sx, sy, srw, srh = cv2.boundingRect(cnt)
        sx1, sy1, sx2, sy2 = sx, sy, sx + srw, sy + srh

        covered = False
        for lx1, ly1, lx2, ly2 in label_aabbs:
            ix1, iy1 = max(sx1, lx1), max(sy1, ly1)
            ix2, iy2 = min(sx2, lx2), min(sy2, ly2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            label_area = max(1, (lx2 - lx1) * (ly2 - ly1))
            if inter / label_area > _COVERED_THRESH:
                covered = True
                break

        if covered:
            continue

        # Build [TL, TR, BR, BL] from the four minAreaRect corners so that
        # bbox[3]→bbox[2] is the bottom edge with the correct rotation dy.
        box_pts = cv2.boxPoints(rect).tolist()
        box_pts_s = sorted(box_pts, key=lambda p: p[1])
        top = sorted(box_pts_s[:2], key=lambda p: p[0])
        bottom = sorted(box_pts_s[2:], key=lambda p: p[0])
        tl, tr = top
        bl, br = bottom
        bbox = [tl, tr, br, bl]

        augmented.append({
            "bbox": bbox,
            "text": "",
            "confidence": 0.0,
            "width": float(w),
            "height": float(h),
        })
        # Treat the newly added synthetic label as coverage so that inner/outer
        # duplicate contours of the same shape don't both produce an entry.
        label_aabbs.append((sx1, sy1, sx2, sy2))

    return augmented


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

    label_readability_viz_path = (
        image_path.parent.parent
        / "label readability detection"
        / f"label_readability_{image_path.stem}.png"
    )
    compute_label_readability_visualization(
        bgr_image,
        features["label_readability"],
        output_path=str(label_readability_viz_path),
    )

    features["overlap_metrics"] = compute_label_overlap_metrics(labels, image_shape)

    label_overlap_viz_path = (
        image_path.parent.parent
        / "label overlap detection"
        / f"label_overlap_{image_path.stem}.png"
    )
    compute_label_overlap_visualization(
        bgr_image,
        features["overlap_metrics"],
        output_path=str(label_overlap_viz_path),
    )

    features["edge_clearance"] = compute_edge_margin_metrics(
        labels, shapes, image_shape, margin_fraction=margin_fraction
    )

    edge_clearance_viz_path = (
        image_path.parent.parent
        / "edge clearance detection"
        / f"edge_clearance_{image_path.stem}.png"
    )
    compute_edge_clearance_visualization(
        bgr_image,
        features["edge_clearance"],
        output_path=str(edge_clearance_viz_path),
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

    features["whitespace_distribution"] = compute_whitespace_distribution_from_diagram(
        labels, shapes, image_shape, bgr_image=bgr_image
    )

    features["color_harmony"] = compute_color_harmony_score(bgr_image, labels)

    features["label_contrast"] = compute_label_contrast_quality(bgr_image, labels)

    label_contrast_viz_path = (
        image_path.parent.parent
        / "label contrast quality detection"
        / f"label_contrast_{image_path.stem}.png"
    )
    compute_label_contrast_visualization(
        bgr_image,
        features["label_contrast"],
        output_path=str(label_contrast_viz_path),
    )

    features["cognitive_chunk_density"] = compute_cognitive_chunk_density_from_diagram(
        labels, shapes, image_shape
    )

    _oc_labels = _augment_labels_with_tilted_shapes(labels, shapes)
    features["orientation_consistency"] = compute_orientation_consistency(_oc_labels)

    orientation_viz_path = (
        image_path.parent.parent
        / "orientation consistency detection"
        / f"orientation_consistency_{image_path.stem}.png"
    )
    compute_orientation_consistency_visualization(
        bgr_image,
        features["orientation_consistency"],
        output_path=str(orientation_viz_path),
    )

    edge_viz_output_path = (
        image_path.parent.parent
        / "edge_detection"
        / f"edge_detection_{image_path.stem}.png"
    )
    features["edge_detection"] = compute_edge_detection_visualization(
        bgr_image,
        shapes,
        image_shape,
        labels=labels,
        output_path=str(edge_viz_output_path),
    )

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
    # for row in label_readability["labels_below_threshold_details"]:
    #     print(
    #         f"[ENTRY]   low-confidence label #{row['index']}: "
    #         f"{row['text']!r} conf={row['confidence']:.4f}"
        # )

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

    print("---------------LABEL CONTRAST QUALITY------------------")
    lcq = feats["label_contrast"]
    if lcq["label_contrast_score"] is None:
        print("[ENTRY] label_contrast_score: N/A (no labels)")
    else:
        harsh_flag = " [harsh contrast]" if lcq["harsh_contrast_flag"] else ""
        print(
            f"[ENTRY] label_contrast_score: {lcq['label_contrast_score']:.2f}{harsh_flag}"
        )
        print(
            f"[ENTRY]   mean_delta_L={lcq['mean_delta_L']:.2f} "
            f"min_delta_L={lcq['min_delta_L']:.2f} "
            f"std_delta_L={lcq['std_delta_L']:.2f} "
            f"mean_outline_ratio={lcq['mean_outline_ratio']:.4f} "
            f"low_contrast_label_ratio={lcq['low_contrast_label_ratio']:.4f}"
        )
        if lcq["skipped_labels"]:
            print(f"[ENTRY]   skipped_labels={lcq['skipped_labels']}")
        if lcq["skipped_outline_count"]:
            print(f"[ENTRY]   skipped_outline_count={lcq['skipped_outline_count']}")

    print("---------------COGNITIVE CHUNK DENSITY------------------")
    ccd = feats["cognitive_chunk_density"]
    if ccd["cognitive_chunk_score"] is None:
        print(f"[ENTRY] cognitive_chunk_score: N/A (low_confidence={ccd['low_confidence']})")
    else:
        deg_flag = " [degenerate layout]" if ccd["degenerate_layout"] else ""
        print(
            f"[ENTRY] cognitive_chunk_score: {ccd['cognitive_chunk_score']:.2f}{deg_flag}"
        )
        print(
            f"[ENTRY]   effective_chunks={ccd['effective_chunks']} "
            f"(clusters={ccd['cluster_count']} singletons={ccd['singleton_count']}) "
            f"eps_used={ccd['eps_used']:.1f}px"
        )

    print("---------------ORIENTATION CONSISTENCY------------------")
    oc = feats["orientation_consistency"]
    if oc["orientation_consistency_score"] is None:
        lc_flag = " [low confidence]" if oc["low_confidence"] else ""
        print(f"[ENTRY] orientation_consistency_score: N/A{lc_flag} (scored_labels={oc['scored_labels']})")
    else:
        lc_flag = " [low confidence]" if oc["low_confidence"] else ""
        print(
            f"[ENTRY] orientation_consistency_score: {oc['orientation_consistency_score']:.2f}{lc_flag}"
        )
        print(
            f"[ENTRY]   dominant_orientation={oc['dominant_orientation_deg']:.2f}° "
            f"consistent_fraction={oc['consistent_label_fraction']:.4f} "
            f"scored={oc['scored_labels']} skipped={oc['skipped_labels']}"
        )

    print("--------------------------------")
