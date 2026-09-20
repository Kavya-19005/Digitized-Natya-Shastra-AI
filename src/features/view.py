"""Camera-viewpoint classification and per-view parameter gating.

The college dataset was shot with four cameras at different angles:

* Cam 5 - frontal
* Cam 6 - profile (side)
* Cam 7 - profile (side)
* Cam 8 - rear

This matters because a single RGB camera cannot measure every parameter from
every angle:

* FRONTAL-PLANE (coronal) parameters -- foot separation, foot turnout, foot-line
  deviation, knee symmetry, knee spread, pelvic tilt, elbow-shoulder alignment,
  the weight-shift proxy -- need a coronal view (front or rear). From a pure
  profile the two legs overlap and the shoulder width collapses, so these values
  become meaningless rather than merely noisy.
* SAGITTAL-PLANE parameters -- knee flexion, trunk inclination, heel lift -- are
  measurable from a profile and are in fact best seen from there.
* MUDRA (finger) parameters need the palm to face the camera, so they cannot be
  measured from the rear view at all.

Instead of silently emitting nonsense, the pipeline classifies the view from the
landmarks themselves and blanks the parameters that view cannot support. Every
result row carries the view class so the report can state exactly which
parameters each camera supports.
"""

from __future__ import annotations

import numpy as np

CORONAL = "CORONAL"
OBLIQUE = "OBLIQUE"
SAGITTAL = "SAGITTAL"
UNKNOWN_VIEW = "UNKNOWN"

FACING_FRONT = "FRONT"
FACING_BACK = "BACK"
FACING_UNKNOWN = ""

# shoulder_width / torso_height. A coronal view shows the full shoulder span;
# in a profile the shoulders project onto each other and the ratio collapses.
CORONAL_MIN_RATIO = 0.42
SAGITTAL_MAX_RATIO = 0.26

# Parameters that require a coronal (front or rear) view.
CORONAL_ONLY_PARAMETERS = (
    "foot_separation_norm",
    "heel_separation_norm",
    "toe_separation_norm",
    "toe_heel_separation_ratio",
    "ankle_separation_px",
    "foot_line_deviation_deg",
    "ankle_line_deviation_deg",
    "left_foot_turnout_deg",
    "right_foot_turnout_deg",
    "mean_foot_turnout_deg",
    "foot_turnout_asymmetry_deg",
    "left_knee_foot_alignment_deg",
    "right_knee_foot_alignment_deg",
    "mean_knee_foot_alignment_deg",
    "knee_spread_norm",
    "knee_over_ankle_spread_ratio",
    "knee_asymmetry_deg",
    "hip_tilt_deg",
    "shoulder_tilt_deg",
    "pelvic_symmetry_deg",
    "com_lateral_offset_norm",
    "pelvis_lateral_offset_norm",
    "weight_shift_proxy",
    "leg_length_ratio_proxy",
    "shin_crossing_angle_deg",
    "shin_crossing_distance_norm",
    "ankle_crossing_signed_norm",
    "knee_crossing_signed_norm",
    "left_ankle_hip_deviation_norm",
    "right_ankle_hip_deviation_norm",
    "left_elbow_shoulder_alignment_norm",
    "right_elbow_shoulder_alignment_norm",
    "mean_elbow_shoulder_alignment_norm",
)


def classify_view(shoulder_width: float, torso_height: float) -> tuple[str, float]:
    """Return (view class, shoulder/torso ratio)."""
    if (
        not np.isfinite(shoulder_width)
        or not np.isfinite(torso_height)
        or torso_height < 1e-6
    ):
        return UNKNOWN_VIEW, float("nan")

    ratio = float(shoulder_width / torso_height)
    if ratio >= CORONAL_MIN_RATIO:
        return CORONAL, ratio
    if ratio <= SAGITTAL_MAX_RATIO:
        return SAGITTAL, ratio
    return OBLIQUE, ratio


def classify_facing(
    view: str, left_shoulder_x: float, right_shoulder_x: float
) -> str:
    """Front vs rear from the left/right shoulder ordering.

    A dancer facing the camera shows their anatomical LEFT shoulder on the image
    RIGHT, so left_x > right_x; seen from behind the order flips. The cue is only
    meaningful for a coronal view, where the shoulders are well separated.

    Measured on the labelled set: 52/52 correct on the frontal camera and 44/52
    on the rear camera (the misses are deep Muzhumandi squats where the shoulders
    nearly overlap).
    """
    if view != CORONAL:
        return FACING_UNKNOWN
    if not np.isfinite(left_shoulder_x) or not np.isfinite(right_shoulder_x):
        return FACING_UNKNOWN
    return FACING_FRONT if left_shoulder_x > right_shoulder_x else FACING_BACK


def gate_by_view(params: dict[str, float], view: str) -> dict[str, float]:
    """Blank the parameters the detected view cannot support."""
    if view in (SAGITTAL, UNKNOWN_VIEW):
        for name in CORONAL_ONLY_PARAMETERS:
            if name in params:
                params[name] = float("nan")
    return params
