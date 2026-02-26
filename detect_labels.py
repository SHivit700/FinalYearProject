#!/usr/bin/env python3
"""
Detect all text labels on an image using OCR, print them, and save a copy
with labels highlighted (bounding boxes) for visual confirmation.
Usage: python3 detect_labels.py <image_path> [--output output.png]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Detect all text labels in an image (OCR) and highlight them in an output PNG."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path for the output PNG (default: labels_output/<input_stem>_labels.png, folder outside Data)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language(s) for OCR, comma-separated (default: en)",
    )
    args = parser.parse_args()

    path = Path(args.image_path)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_file():
        print(f"Error: Not a file: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: opencv-python is required. Install with: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    try:
        import easyocr
    except ImportError:
        print("Error: easyocr is required. Install with: pip install easyocr", file=sys.stderr)
        sys.exit(1)

    # Output path: save to labels_output folder
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "labels_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        out_path = output_dir / f"{path.stem}_labels.png"
    else:
        out_path = Path(args.output)
    out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image (OpenCV for drawing; EasyOCR can use path or numpy array)
    image = cv2.imread(str(path))
    if image is None:
        print(f"Error: Could not load image: {path}", file=sys.stderr)
        sys.exit(1)
    overlay = image.copy()

    # Initialize OCR (first run downloads model)
    langs = [s.strip() for s in args.lang.split(",")]
    print(f"Loading OCR reader (languages: {langs})...")
    reader = easyocr.Reader(langs, verbose=False)

    print(f"Detecting labels in: {path}")
    results = reader.readtext(str(path))

    if not results:
        print("No text labels detected.")
        cv2.imwrite(str(out_path), image)
        print(f"Saved unchanged image to: {out_path}")
        return

    # Colors (BGR): box, fill highlight, text background, text
    box_color = (0, 200, 0)           # green
    fill_color = (200, 255, 200)      # light green
    text_bg_color = (0, 200, 0)
    text_color = (255, 255, 255)
    thickness = 2

    print("\nDetected labels:")
    print("-" * 50)

    for i, (bbox, text, confidence) in enumerate(results, 1):
        # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.polylines(overlay, [pts], True, box_color, thickness)
        # Label text above the box
        text = (text or "").strip()
        print(f"  {i}. {text!r}  (confidence: {confidence:.2%})")
        if text:
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                overlay,
                (x_min, y_min - th - 6),
                (x_min + tw + 4, y_min),
                text_bg_color,
                -1,
            )
            cv2.putText(
                overlay,
                text,
                (x_min + 2, y_min - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

    # Blend overlay with original so highlights are visible but not harsh
    alpha = 0.5
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    # Redraw sharp boxes and text on top
    for (bbox, text, _) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], True, box_color, thickness)
    for (bbox, text, _) in results:
        text = (text or "").strip()
        if text:
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), text_bg_color, -1)
            cv2.putText(result, text, (x_min + 2, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), result)
    print("-" * 50)
    print(f"Total: {len(results)} label(s).")
    print(f"Highlighted image saved to: {out_path}")


if __name__ == "__main__":
    main()
