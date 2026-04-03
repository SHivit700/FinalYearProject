#!/usr/bin/env python3
"""
Diagram contour detection: grayscale threshold, RETR_TREE contours, area filter,
and random-color overlay. Skips the outermost contour (full frame).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def run_shape_detection(
    image_path: str,
    output_path: str | None = None,
    min_area: float = 600.0,
    threshold: int = 200,
) -> tuple[list[dict[str, Any]], tuple[int, int, int], np.ndarray]:
    """
    Load an image from ``image_path``, find thresholded contours, draw each accepted contour on a copy of the image, and save the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    if image.ndim == 2:
        image_shape = (int(image.shape[0]), int(image.shape[1]), 1)
        gray_image = image
    else:
        image_shape = (
            int(image.shape[0]),
            int(image.shape[1]),
            int(image.shape[2]),
        )
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(
        gray_image, threshold, 255, cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if image.ndim == 2:
        highlighted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        highlighted = image.copy()

    shapes: list[dict[str, Any]] = []

    for i, contour in enumerate(contours):
        if i == 0:
            continue

        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue

        epsilon = 0.09 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 2:
            continue

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        cv2.drawContours(highlighted, [contour], 0, color, 4)

        shapes.append(
            {
                "type": "contour",
                "contour": contour,
                "area": area,
                "approx_vertex_count": int(len(approx)),
            }
        )

    if output_path is None:
        out = (
            Path(__file__).resolve().parent
            / "Data"
            / "shapes_output"
            / f"shapes_{image_path.stem}.png"
        )
    else:
        out = Path(output_path).with_suffix(".png")

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), highlighted)

    return shapes, image_shape, highlighted
