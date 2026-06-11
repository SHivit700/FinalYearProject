"""
Batch layout-structure annotation script.

Runs GPT-4.1-mini (text-only) on each diagram's detected contour geometry and writes
row/column/layout scores to data/score_annotation.csv for supervised model training.

Usage:
    python src/batch_rate_images.py              # process all images in data/diagrams/
    python src/batch_rate_images.py diagram.png  # process a single image
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2

from detection.detect_shapes import run_shape_detection

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'python-dotenv'. Install with: pip install python-dotenv"
    ) from e

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'openai'. Install with: pip install openai"
    ) from e

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

CSV_PATH = Path("./data/score_annotation.csv")
IMAGE_DIR = Path("./data/diagrams")
TEST_LIMIT = 660
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def extract_detection_context(image_path: Path) -> dict[str, Any]:
    """
    Run contour shape detection only; optional debug image under data/shapes_output
    (same layout as feature_detection).
    """
    s = str(image_path.resolve())
    stem = image_path.stem
    data_root = image_path.parent.parent
    shapes_out = data_root / "shapes_output" / f"shapes_{stem}.png"

    shapes, image_shape, _hl = run_shape_detection(s, output_path=shapes_out)

    return {
        "shapes": shapes,
        "image_shape": image_shape,
    }


def load_existing_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_id",
        "image_path",
        "row_alignment_score",
        "column_alignment_score",
        "layout_structure_score",
        "notes",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)


def list_images(limit: int) -> list[Path]:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    files = [
        p
        for p in sorted(IMAGE_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files[:limit]


def resolve_single_image(arg: str) -> Path:
    """Resolve a user-supplied path or filename to an existing image file."""
    raw = Path(arg).expanduser()
    candidates: list[Path] = [raw]
    if not raw.is_absolute():
        candidates.append(IMAGE_DIR / raw.name)
        candidates.append(IMAGE_DIR / raw)
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTS:
            return resolved
    raise SystemExit(
        f"Image not found or unsupported type: {arg!r}. "
        f"Use a path to a file with extension {sorted(IMAGE_EXTS)} "
        f"or a filename under {IMAGE_DIR}."
    )


def csv_path_for_image(img_path: Path) -> str:
    """Match batch mode: store path relative to cwd when possible."""
    abs_img = img_path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return abs_img.relative_to(cwd).as_posix()
    except ValueError:
        return abs_img.as_posix()


def make_image_id(n: int) -> str:
    return f"img_{n:03d}"


def build_layout_prompt(image_path: Path, image_shape: tuple[int, ...], shapes: list[dict]) -> str:
    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    iw = max(image_width, 1)
    ih = max(image_height, 1)

    kept: list[tuple[int, int, int, int]] = []
    for s in shapes:
        if s.get("type") != "contour":
            continue
        contour = s.get("contour")
        if contour is None:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 40 and h >= 20:
            kept.append((int(x), int(y), int(w), int(h)))

    lines: list[str] = [
        f"[ENTRY] Processed image: {image_path}",
        f"[ENTRY] Image size: width={image_width} height={image_height}",
        f"[ENTRY] Detected {len(kept)} likely box contours.",
    ]
    for i, (x, y, w, h) in enumerate(kept):
        cx = (x + w / 2) / iw
        cy = (y + h / 2) / ih
        nw = w / iw
        nh = h / ih
        lines.append(
            f"[ENTRY] Box #{i}: bbox_xywh=({x},{y},{w},{h}), "
            f"center_norm=({cx:.3f},{cy:.3f}), size_norm=({nw:.3f},{nh:.3f})"
        )

    instructions = """

You are rating one visual feature in a UML / box-and-arrow diagram.

Feature: Hierarchy / Layout Structure.

Judge how well the detected box centres align into a small number of clean horizontal rows and vertical columns.

What to evaluate:
1. Row alignment:
- Boxes that share very similar y positions form clean rows.
- Small vertical jitter within each row is acceptable.
- When several boxes sit noticeably between rows, or rows are sloped/uneven, the row score must go down.

2. Column alignment:
- Boxes that share very similar x positions form clean columns.
- Small horizontal jitter within each column is acceptable.
- When multiple boxes sit between columns or columns are staggered, the column score must go down.

3. Overall layout structure:
- High score only when boxes clearly follow a grid-like or banded structure in BOTH directions.
- Layouts that are mostly row-aligned but only loosely column-aligned should get a MID score, not a very high one.

Important calibration:
- Use the FULL [0,1] range.
- 0.9–1.0: almost perfect rows AND columns; boxes tightly grouped on a small number of bands with minimal outliers.
- 0.7–0.9: generally structured; clear rows and/or columns but with several noticeable offsets or drift.
- 0.4–0.7: mixed; some weak banding but many boxes in between rows/columns or visibly irregular spacing.
- 0.0–0.4: largely unstructured; box centres are scattered with no obvious rows or columns.

Important:
- Use only the contour geometry provided.
- Ignore tiny artifacts if they are unlikely to be real boxes.
- Judge based on likely class/component boxes from bbox size and placement.
- If you are unsure between “high” and “mid”, choose the LOWER bucket.

Return ONLY valid JSON in exactly this format:
{
  "row_alignment_score": 0.84,
  "column_alignment_score": 0.71,
  "layout_structure_score": 0.78,
  "note": "Most boxes align into a few clean rows and columns with only minor positional jitter."
}

Rules:
- All three scores must be floats between 0 and 1.
- layout_structure_score should roughly reflect the weaker of row and column alignment, not just the better one.
- note must be exactly one sentence.
- Do not add extra keys.
- Do not use markdown or code fences.
""".strip()

    return "\n".join(lines) + "\n\n" + instructions


def call_openai_layout_rating(client: OpenAI, image_path: Path) -> tuple[float, float, float, str]:
    detection = extract_detection_context(image_path)
    prompt = build_layout_prompt(
        image_path,
        detection["image_shape"],
        detection["shapes"],
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )

    text = response.output_text.strip()
    data = json.loads(text)
    row_score = max(0.0, min(1.0, float(data["row_alignment_score"])))
    col_score = max(0.0, min(1.0, float(data["column_alignment_score"])))
    layout_score = max(0.0, min(1.0, float(data["layout_structure_score"])))
    note = str(data["note"]).strip()
    return row_score, col_score, layout_score, note


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rate diagrams for hierarchy/layout structure using contour geometry only."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        metavar="FILE",
        help=(
            "Optional: one image to rate. Path to file, or a name under data/diagrams. "
            "If omitted, processes up to TEST_LIMIT images from data/diagrams."
        ),
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY in your environment before running this script.")

    client = OpenAI(api_key=api_key)
    rows = load_existing_rows()
    existing_paths = {row.get("image_path", "") for row in rows}

    if args.image:
        images = [resolve_single_image(args.image)]
    else:
        images = list_images(TEST_LIMIT)
        if not images:
            raise SystemExit(f"No images found in {IMAGE_DIR}")

    next_idx = len(rows) + 1
    for img_path in images:
        rel_path = csv_path_for_image(img_path)
        if rel_path in existing_paths:
            print(f"Skipping existing: {rel_path}")
            continue

        row_score, col_score, layout_score, note = call_openai_layout_rating(client, img_path)
        row = {
            "image_id": make_image_id(next_idx),
            "image_path": rel_path,
            "row_alignment_score": f"{row_score:.4f}",
            "column_alignment_score": f"{col_score:.4f}",
            "layout_structure_score": f"{layout_score:.4f}",
            "notes": note,
        }
        rows.append(row)
        save_rows(rows)
        print(
            f"Saved {row['image_id']}: row={row_score:.4f} col={col_score:.4f} "
            f"layout={layout_score:.4f} - {rel_path}"
        )
        next_idx += 1

    print(f"Done. Updated CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
