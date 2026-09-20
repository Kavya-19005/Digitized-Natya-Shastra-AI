"""Small, NaN-safe geometry helpers.

Image coordinate convention: x grows to the right, y grows DOWNWARDS. Every
helper returns NaN rather than raising when a point is missing or a vector is
degenerate, so the parameter layer never divides by zero.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def valid(*points: np.ndarray) -> bool:
    return all(p is not None and np.all(np.isfinite(p)) for p in points)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    if not valid(a, b):
        return float("nan")
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not valid(a, b):
        return np.array([np.nan, np.nan])
    return (np.asarray(a, float) + np.asarray(b, float)) * 0.5


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at `b` of the chain a-b-c, in [0, 180].

    This is the RAW JOINT ANGLE. Flexion is 180 - raw joint angle; see
    `flexion_from_joint_angle`.
    """
    if not valid(a, b, c):
        return float("nan")
    v1 = np.asarray(a, float) - np.asarray(b, float)
    v2 = np.asarray(c, float) - np.asarray(b, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return float("nan")
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def flexion_from_joint_angle(joint_angle: float) -> float:
    """Convert a raw joint angle to flexion. 0 deg flexion = fully extended."""
    if not np.isfinite(joint_angle):
        return float("nan")
    return 180.0 - joint_angle


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    if not valid(v1, v2):
        return float("nan")
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return float("nan")
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def angle_from_vertical_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Tilt of the segment a->b away from the image vertical, in [0, 90]."""
    if not valid(a, b):
        return float("nan")
    v = np.asarray(b, float) - np.asarray(a, float)
    if np.linalg.norm(v) < EPS:
        return float("nan")
    angle = np.degrees(np.arctan2(abs(v[0]), abs(v[1])))
    return float(angle)


def angle_from_horizontal_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Tilt of the segment a->b away from the image horizontal, in [0, 90]."""
    if not valid(a, b):
        return float("nan")
    v = np.asarray(b, float) - np.asarray(a, float)
    if np.linalg.norm(v) < EPS:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) < EPS:
        return float("nan")
    return float(numerator / denominator)


def abs_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b))


def nan_mean(values) -> float:
    arr = np.asarray([v for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def segment_min_distance(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> float:
    """Shortest distance between the 2D segments p1-p2 and q1-q2."""
    if not valid(p1, p2, q1, q2):
        return float("nan")
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))

    def point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < EPS:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    # segments that properly intersect have distance zero
    def orientation(a, b, c):
        return float(np.cross(b - a, c - a))

    o1, o2 = orientation(p1, p2, q1), orientation(p1, p2, q2)
    o3, o4 = orientation(q1, q2, p1), orientation(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return 0.0

    return min(
        point_to_segment(p1, q1, q2),
        point_to_segment(p2, q1, q2),
        point_to_segment(q1, p1, p2),
        point_to_segment(q2, p1, p2),
    )
