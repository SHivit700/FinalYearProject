#!/usr/bin/env python3
"""
Starting point for the entire diagram quality model pipeline.

For now this module orchestrates label detection only. Additional feature
extraction modules and the global quality prediction step will be wired in
here later.
"""

from pathlib import Path

from detect_labels import run_label_detection


def process_image_for_labels(image_path: str, lang: str = "en"):
    """
    Top-level entry for processing a single diagram image.

    For now:
    - runs the label detection component
    - logs basic statistics
    - returns the structured label detection result

    Later:
    - this function will also call other feature modules and the global quality model.
    """
    image_path = Path(image_path)
    # Output folder for diagnostic images, colocated with the input image
    output_dir = image_path.parent / "labels_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_image_path = output_dir / f"{image_path.stem}_labels.png"

    result = run_label_detection(
        image_path=str(image_path),
        lang=lang,
        output_path=str(out_image_path),
    )

    labels = result["labels"]
    print(f"[ENTRY] Processed image: {image_path}")
    print(f"[ENTRY] Detected {len(labels)} label(s).")

    for i, label in enumerate(labels, 1):
        print(f"[ENTRY]   {i}. {label['text']!r}  (conf: {label['confidence']:.2%})")

    print(f"[ENTRY] Highlighted label image saved to: {out_image_path}")

    # This return value will later be extended to include other feature families
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Entry point for diagram quality model (currently: label detection)."
    )
    parser.add_argument("image_path", help="Path to input diagram image")
    parser.add_argument("--lang", default="en", help="OCR language codes (comma-separated)")

    args = parser.parse_args()

    # For now this only runs label detection; later it will run the full pipeline.
    process_image_for_labels(args.image_path, lang=args.lang)


if __name__ == "__main__":
    main()
