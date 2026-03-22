#!/usr/bin/env python3
"""
Detect all text labels on an image using OCR, print them, and save a copy
with labels highlighted (bounding boxes) for visual confirmation.
"""

from pathlib import Path


def run_label_detection(image_path, lang="en", output_path=None):
    """
    Parameters
    ----------
    image_path : str or Path
        Input image.
    lang : str
        Comma-separated languages for easyocr (default: "en").
    output_path : str or Path or None
        If not None, draw label overlays, save a PNG to this path (.png suffix), and
        return the same dict. If None, skip overlay drawing and do not write any file;
        ``highlighted_image`` is then a plain copy of the source (OCR-only path).

    Returns
    -------
    dict with keys:
        - "labels": list of dicts with keys "bbox", "text", "confidence"
        - "highlighted_image": numpy array (BGR)
    """
    import cv2
    import numpy as np
    import easyocr

    # Load source; overlay is mutated for highlights while keeping the original for blending.
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    langs = [s.strip() for s in lang.split(",") if s.strip()]
    if not langs:
        langs = ["en"]

    reader = easyocr.Reader(langs, verbose=False)
    results = reader.readtext(str(path))

    # Stable dict shape for pipelines: bbox is 4 corners [[x,y], ...], text stripped, conf as float.
    structured_results = [
        {
            "bbox": bbox,
            "text": (text or "").strip(),
            "confidence": float(conf),
        }
        for (bbox, text, conf) in results
    ]

    if not results:
        highlighted = image.copy()
        if output_path is not None:
            out_path = Path(output_path).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), highlighted)
        return {
            "labels": structured_results,
            "highlighted_image": highlighted,
        }

    if output_path is None:
        # OCR only: no overlays and no PNG
        return {
            "labels": structured_results,
            "highlighted_image": image.copy(),
        }

    box_color = (0, 200, 0)
    fill_color = (200, 255, 200)
    text_bg_color = (0, 200, 0)
    text_color = (255, 255, 255)
    thickness = 2

    overlay = image.copy()
    # Pass 1: filled regions + boxes + text on overlay only.
    for bbox, text, _confidence in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.polylines(overlay, [pts], True, box_color, thickness)
        text_draw = (text or "").strip()
        if text_draw:
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            (tw, th), _ = cv2.getTextSize(text_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                overlay,
                (x_min, y_min - th - 6),
                (x_min + tw + 4, y_min),
                text_bg_color,
                -1,
            )
            cv2.putText(
                overlay,
                text_draw,
                (x_min + 2, y_min - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

    # Pass 2: blend so fills read softly, then redraw sharp outlines and text on top.
    alpha = 0.5
    highlighted = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    for (bbox, text, _) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(highlighted, [pts], True, box_color, thickness)
    for (bbox, text, _) in results:
        text_draw = (text or "").strip()
        if text_draw:
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            (tw, th), _ = cv2.getTextSize(text_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                highlighted,
                (x_min, y_min - th - 6),
                (x_min + tw + 4, y_min),
                text_bg_color,
                -1,
            )
            cv2.putText(
                highlighted,
                text_draw,
                (x_min + 2, y_min - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

    out_path = Path(output_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), highlighted)

    return {
        "labels": structured_results,
        "highlighted_image": highlighted,
    }
