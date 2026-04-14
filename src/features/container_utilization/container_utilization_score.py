"""Single-image container utilization score (0–100) from detection metrics."""

from __future__ import annotations


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def compute_container_utilization_score(
    valid_box_count: int,
    false_box_rejection_count: int,
    grouping_container_count: int,
    empty_container_count: int,
    empty_container_ratio: float,
    empty_container_area_fraction: float,
    max_empty_container_confidence: float,
) -> int:
    eps = 1e-6

    V = max(0, int(valid_box_count))
    F = max(0, int(false_box_rejection_count))
    G = max(0, int(grouping_container_count))
    R = clamp(float(empty_container_ratio), 0.0, 1.0)
    A = max(0.0, float(empty_container_area_fraction))
    C = clamp(float(max_empty_container_confidence), 0.0, 1.0)

    q = F / (V + F + eps)
    g = G / (V + G + eps)

    penalty = (
        0.55 * R
        + 0.20 * min(1.0, 8.0 * A)
        + 0.10 * C
        + 0.35 * q
        - 0.18 * g
    )
    penalty = clamp(penalty, 0.0, 1.0)

    score = 100.0 * (1.0 - penalty)
    return int(round(clamp(score, 0.0, 100.0)))
