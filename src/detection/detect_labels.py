#!/usr/bin/env python3
"""
OCR-based label detection using EasyOCR with multi-rotation support and IoU NMS.
"""
from pathlib import Path

# Rotation angles (degrees) tried in addition to the default 0° pass.
# Every 45° gives good diagonal coverage at the cost of ~7× extra inference time.
_ROTATION_INFO = list(range(45, 360, 45))


def _nms_labels(results, iou_threshold: float = 0.4):
    """IoU-based NMS: keep highest-confidence detection when bboxes overlap."""
    if not results:
        return results

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    kept = []

    for candidate in sorted_results:
        bbox_c, _, _ = candidate
        xs_c = [p[0] for p in bbox_c]
        ys_c = [p[1] for p in bbox_c]
        ax1, ay1, ax2, ay2 = min(xs_c), min(ys_c), max(xs_c), max(ys_c)

        duplicate = False
        for kept_item in kept:
            bbox_k, _, _ = kept_item
            xs_k = [p[0] for p in bbox_k]
            ys_k = [p[1] for p in bbox_k]
            bx1, by1, bx2, by2 = min(xs_k), min(ys_k), max(xs_k), max(ys_k)

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            iou = inter / (area_a + area_b - inter)
            if iou > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


def run_label_detection(image_path, lang="en", output_path=None):
    """OCR the image; optionally write a highlighted PNG. Returns labels, shape, overlays, and the BGR array."""
    import cv2
    import numpy as np
    import easyocr

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    langs = [s.strip() for s in lang.split(",") if s.strip()]
    if not langs:
        langs = ["en"]

    reader = easyocr.Reader(langs, verbose=False)
    results = reader.readtext(image_path, rotation_info=_ROTATION_INFO)
    results = _nms_labels(results)

    structured_results = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        structured_results.append(
            {
                "bbox": bbox,
                "text": (text or "").strip(),
                "confidence": float(conf),
                "width": max(0.0, float(x_max - x_min)),
                "height": max(0.0, float(y_max - y_min)),
            }
        )

    image_shape = tuple(int(x) for x in image.shape)

    if not results:
        highlighted = image.copy()
        if output_path is not None:
            out_path = Path(output_path).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), highlighted)
        return {
            "labels": structured_results,
            "image_shape": image_shape,
            "highlighted_image": highlighted,
            "bgr_image": image,
        }

    if output_path is None:
        return {
            "labels": structured_results,
            "image_shape": image_shape,
            "highlighted_image": image.copy(),
            "bgr_image": image,
        }

    box_color = (0, 200, 0)
    fill_color = (200, 255, 200)
    text_bg_color = (0, 200, 0)
    text_color = (255, 255, 255)
    thickness = 2

    overlay = image.copy()
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
        "image_shape": image_shape,
        "highlighted_image": highlighted,
        "bgr_image": image,
    }
