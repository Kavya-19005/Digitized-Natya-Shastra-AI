"""Bharatanatyam body parameters computed from the common 67 landmarks.

PARAMETER CONVENTION (fixed for the whole project)
--------------------------------------------------
* `*_knee_joint_angle_deg` is the RAW joint angle hip -> knee -> ankle.
* `*_knee_flexion_deg`     is FLEXION = 180 - raw joint angle.

All published Bharatanatyam ranges in this project are expressed as FLEXION, so
every rule consumes the `*_flexion_deg` fields. A raw joint angle is never
called "flexion" anywhere in the codebase.

All distances are normalised by shoulder width so results are scale- and
camera-distance invariant.
"""

from __future__ import annotations

import numpy as np

from ..landmarks import common as C
from ..landmarks.common import Landmarks
from .geometry import (
    abs_diff,
    angle_between_vectors_deg,
    angle_from_horizontal_deg,
    angle_from_vertical_deg,
    distance,
    flexion_from_joint_angle,
    joint_angle_deg,
    midpoint,
    nan_mean,
    safe_ratio,
    segment_min_distance,
    valid,
)
from .view import classify_facing, classify_view, gate_by_view

NAN = float("nan")


def _pt(lm: Landmarks, idx: int, mask: np.ndarray) -> np.ndarray:
    """Landmark position, or NaN when the point is missing/low confidence."""
    if not mask[idx]:
        return np.array([np.nan, np.nan])
    return lm.xy[idx].astype(float)


def compute_body_parameters(
    lm: Landmarks, score_threshold: float
) -> dict[str, float | str]:
    """Return every body parameter; unmeasurable ones come back as NaN.

    `view_class` and `facing` are strings; every other entry is a float.
    """
    mask = lm.available(score_threshold)
    p = {
        "l_shoulder": _pt(lm, C.LEFT_SHOULDER, mask),
        "r_shoulder": _pt(lm, C.RIGHT_SHOULDER, mask),
        "l_elbow": _pt(lm, C.LEFT_ELBOW, mask),
        "r_elbow": _pt(lm, C.RIGHT_ELBOW, mask),
        "l_hip": _pt(lm, C.LEFT_HIP, mask),
        "r_hip": _pt(lm, C.RIGHT_HIP, mask),
        "l_knee": _pt(lm, C.LEFT_KNEE, mask),
        "r_knee": _pt(lm, C.RIGHT_KNEE, mask),
        "l_ankle": _pt(lm, C.LEFT_ANKLE, mask),
        "r_ankle": _pt(lm, C.RIGHT_ANKLE, mask),
        "l_heel": _pt(lm, C.LEFT_HEEL, mask),
        "r_heel": _pt(lm, C.RIGHT_HEEL, mask),
        "l_toe": _pt(lm, C.LEFT_FOOT_INDEX, mask),
        "r_toe": _pt(lm, C.RIGHT_FOOT_INDEX, mask),
        "nose": _pt(lm, C.NOSE, mask),
    }

    out: dict[str, float | str] = {}

    # --- scale normaliser -------------------------------------------------
    shoulder_width = distance(p["l_shoulder"], p["r_shoulder"])
    hip_width = distance(p["l_hip"], p["r_hip"])
    # fall back to hip width when a shoulder is missing
    scale = shoulder_width if np.isfinite(shoulder_width) else hip_width
    out["shoulder_width_px"] = shoulder_width
    out["hip_width_px"] = hip_width
    out["scale_px"] = scale

    mid_shoulder = midpoint(p["l_shoulder"], p["r_shoulder"])
    mid_hip = midpoint(p["l_hip"], p["r_hip"])
    mid_ankle = midpoint(p["l_ankle"], p["r_ankle"])

    # --- knee flexion (the project's single convention) -------------------
    l_joint = joint_angle_deg(p["l_hip"], p["l_knee"], p["l_ankle"])
    r_joint = joint_angle_deg(p["r_hip"], p["r_knee"], p["r_ankle"])
    out["left_knee_joint_angle_deg"] = l_joint
    out["right_knee_joint_angle_deg"] = r_joint
    out["left_knee_flexion_deg"] = flexion_from_joint_angle(l_joint)
    out["right_knee_flexion_deg"] = flexion_from_joint_angle(r_joint)
    out["mean_knee_flexion_deg"] = nan_mean(
        [out["left_knee_flexion_deg"], out["right_knee_flexion_deg"]]
    )
    out["knee_asymmetry_deg"] = abs_diff(
        out["left_knee_flexion_deg"], out["right_knee_flexion_deg"]
    )

    # --- foot separation and placement ------------------------------------
    ankle_sep = distance(p["l_ankle"], p["r_ankle"])
    heel_sep = distance(p["l_heel"], p["r_heel"])
    toe_sep = distance(p["l_toe"], p["r_toe"])
    out["ankle_separation_px"] = ankle_sep
    out["foot_separation_norm"] = safe_ratio(ankle_sep, scale)
    out["heel_separation_norm"] = safe_ratio(heel_sep, scale)
    out["toe_separation_norm"] = safe_ratio(toe_sep, scale)
    # >> 1 means heels together with toes apart, i.e. a V-shaped stance
    out["toe_heel_separation_ratio"] = safe_ratio(toe_sep, heel_sep)

    # deviation of the heel-to-heel line from the image horizontal: non-zero
    # when one foot is placed forward/backward instead of on a straight line
    out["foot_line_deviation_deg"] = angle_from_horizontal_deg(p["l_heel"], p["r_heel"])
    out["ankle_line_deviation_deg"] = angle_from_horizontal_deg(
        p["l_ankle"], p["r_ankle"]
    )

    # --- foot orientation / turnout ----------------------------------------
    # 0 deg  = foot pointing straight at the camera (no turnout)
    # 90 deg = foot pointing fully sideways (maximum turnout)
    out["left_foot_turnout_deg"] = _turnout_deg(p["l_heel"], p["l_toe"], scale)
    out["right_foot_turnout_deg"] = _turnout_deg(p["r_heel"], p["r_toe"], scale)
    out["left_foot_length_norm"] = safe_ratio(distance(p["l_heel"], p["l_toe"]), scale)
    out["right_foot_length_norm"] = safe_ratio(distance(p["r_heel"], p["r_toe"]), scale)
    out["mean_foot_turnout_deg"] = nan_mean(
        [out["left_foot_turnout_deg"], out["right_foot_turnout_deg"]]
    )
    out["foot_turnout_asymmetry_deg"] = abs_diff(
        out["left_foot_turnout_deg"], out["right_foot_turnout_deg"]
    )

    # --- knee-foot alignment ----------------------------------------------
    out["left_knee_foot_alignment_deg"] = _knee_foot_alignment_deg(
        p["l_knee"], p["l_ankle"], p["l_heel"], p["l_toe"]
    )
    out["right_knee_foot_alignment_deg"] = _knee_foot_alignment_deg(
        p["r_knee"], p["r_ankle"], p["r_heel"], p["r_toe"]
    )
    out["mean_knee_foot_alignment_deg"] = nan_mean(
        [out["left_knee_foot_alignment_deg"], out["right_knee_foot_alignment_deg"]]
    )
    # how far the knees are pushed out sideways relative to the ankles
    out["knee_spread_norm"] = safe_ratio(distance(p["l_knee"], p["r_knee"]), scale)
    out["knee_over_ankle_spread_ratio"] = safe_ratio(
        distance(p["l_knee"], p["r_knee"]), ankle_sep
    )

    # --- trunk and pelvis --------------------------------------------------
    out["trunk_inclination_deg"] = angle_from_vertical_deg(mid_hip, mid_shoulder)
    # Bending forward/backward is a SAGITTAL-plane motion and is therefore only
    # directly visible from a side view. From a coronal view it shows up as the
    # torso foreshortening, which this ratio captures as a weak proxy.
    out["trunk_foreshortening_ratio"] = safe_ratio(
        distance(mid_shoulder, mid_hip), shoulder_width
    )
    out["hip_tilt_deg"] = angle_from_horizontal_deg(p["l_hip"], p["r_hip"])
    out["shoulder_tilt_deg"] = angle_from_horizontal_deg(
        p["l_shoulder"], p["r_shoulder"]
    )
    out["pelvic_symmetry_deg"] = out["hip_tilt_deg"]

    # --- loading / weight-shift PROXY -------------------------------------
    # NOTE: RGB landmarks cannot measure ground reaction force. These are
    # landmark-based proxies only and are named as such.
    out["com_lateral_offset_norm"] = _lateral_offset_norm(mid_shoulder, mid_ankle, scale)
    out["pelvis_lateral_offset_norm"] = _lateral_offset_norm(mid_hip, mid_ankle, scale)
    left_leg = distance(p["l_hip"], p["l_ankle"])
    right_leg = distance(p["r_hip"], p["r_ankle"])
    out["leg_length_ratio_proxy"] = safe_ratio(
        min(left_leg, right_leg) if np.isfinite(left_leg + right_leg) else NAN,
        max(left_leg, right_leg) if np.isfinite(left_leg + right_leg) else NAN,
    )
    out["weight_shift_proxy"] = nan_mean(
        [abs(out["com_lateral_offset_norm"]), abs(out["pelvis_lateral_offset_norm"])]
    )

    # --- heel configuration (Muzhumandi) -----------------------------------
    # positive = heel sits higher in the image than the toe, i.e. heel lifted
    out["left_heel_lift_norm"] = safe_ratio(
        (p["l_toe"][1] - p["l_heel"][1]) if valid(p["l_toe"], p["l_heel"]) else NAN,
        scale,
    )
    out["right_heel_lift_norm"] = safe_ratio(
        (p["r_toe"][1] - p["r_heel"][1]) if valid(p["r_toe"], p["r_heel"]) else NAN,
        scale,
    )
    out["mean_heel_lift_norm"] = nan_mean(
        [out["left_heel_lift_norm"], out["right_heel_lift_norm"]]
    )

    # --- squat depth (helps separate Aramandi from Muzhumandi) -------------
    hip_height_px = (mid_ankle[1] - mid_hip[1]) if valid(mid_ankle, mid_hip) else NAN
    knee_height_px = (
        (mid_ankle[1] - midpoint(p["l_knee"], p["r_knee"])[1])
        if valid(mid_ankle, p["l_knee"], p["r_knee"])
        else NAN
    )
    out["hip_height_norm"] = safe_ratio(hip_height_px, scale)
    out["knee_height_norm"] = safe_ratio(knee_height_px, scale)
    # Normalising by TORSO height instead of shoulder width keeps these usable
    # from a side view, where the shoulder span collapses and any
    # shoulder-normalised value blows up. Posture recognition uses these.
    torso_px = distance(mid_shoulder, mid_hip)
    out["hip_height_torso_norm"] = safe_ratio(hip_height_px, torso_px)
    out["knee_height_torso_norm"] = safe_ratio(knee_height_px, torso_px)

    # --- Swastikapadam: shin crossing --------------------------------------
    out.update(_shin_crossing_parameters(p, scale))

    # --- elbow-shoulder alignment (used by the mudra rules) ----------------
    # --- camera viewpoint -------------------------------------------------
    torso_height = distance(mid_shoulder, mid_hip)
    out["torso_height_px"] = torso_height
    view, ratio = classify_view(shoulder_width, torso_height)
    out["shoulder_torso_ratio"] = ratio
    out["view_class"] = view
    out["facing"] = classify_facing(
        view,
        float(p["l_shoulder"][0]) if valid(p["l_shoulder"]) else NAN,
        float(p["r_shoulder"][0]) if valid(p["r_shoulder"]) else NAN,
    )

    out["left_elbow_shoulder_alignment_norm"] = _vertical_diff_norm(
        p["l_elbow"], p["l_shoulder"], scale
    )
    out["right_elbow_shoulder_alignment_norm"] = _vertical_diff_norm(
        p["r_elbow"], p["r_shoulder"], scale
    )
    out["mean_elbow_shoulder_alignment_norm"] = nan_mean(
        [
            out["left_elbow_shoulder_alignment_norm"],
            out["right_elbow_shoulder_alignment_norm"],
        ]
    )
    out["left_arm_abduction_deg"] = _arm_abduction_deg(p["l_shoulder"], p["l_elbow"])
    out["right_arm_abduction_deg"] = _arm_abduction_deg(p["r_shoulder"], p["r_elbow"])

    gate_by_view(out, view)
    return out


# A heel-to-toe segment shorter than this fraction of the shoulder width is too
# foreshortened for its direction to mean anything. This happens in a full
# Muzhumandi squat, where the foot points down towards the camera: reporting a
# turnout angle from a 10-pixel vector would be a measurement artefact, so the
# value is returned as unmeasurable instead.
MIN_FOOT_LENGTH_NORM = 0.15


def _turnout_deg(heel: np.ndarray, toe: np.ndarray, scale: float) -> float:
    """Foot turnout: 0 deg = pointing at the camera, 90 deg = fully sideways."""
    if not valid(heel, toe):
        return NAN
    length = safe_ratio(distance(heel, toe), scale)
    if not np.isfinite(length) or length < MIN_FOOT_LENGTH_NORM:
        return NAN
    # angle of the heel->toe vector away from the image horizontal, in [0, 90];
    # a sideways foot lies along the horizontal, so turnout = 90 - that angle
    from_horizontal = angle_from_horizontal_deg(heel, toe)
    if not np.isfinite(from_horizontal):
        return NAN
    return float(90.0 - from_horizontal)


def _knee_foot_alignment_deg(
    knee: np.ndarray, ankle: np.ndarray, heel: np.ndarray, toe: np.ndarray
) -> float:
    """Angle between the shank direction and the foot direction.

    In Aramandi the knee should track over the turned-out foot, so the shank
    (knee -> ankle) and the foot (heel -> toe) should share a plane; a large
    angle means the knee has collapsed inwards or drifted past the foot.
    """
    if not valid(knee, ankle, heel, toe):
        return NAN
    shank = np.asarray(ankle, float) - np.asarray(knee, float)
    foot = np.asarray(toe, float) - np.asarray(heel, float)
    angle = angle_between_vectors_deg(shank, foot)
    if not np.isfinite(angle):
        return NAN
    # knee tracking over the foot gives a right angle between shank and foot;
    # report the deviation from that ideal
    return float(abs(angle - 90.0))


def _lateral_offset_norm(
    upper: np.ndarray, base: np.ndarray, scale: float
) -> float:
    if not valid(upper, base):
        return NAN
    return safe_ratio(float(upper[0] - base[0]), scale)


def _vertical_diff_norm(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    if not valid(a, b):
        return NAN
    return safe_ratio(abs(float(a[1] - b[1])), scale)


def _arm_abduction_deg(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """0 deg = elbow hanging straight down, 90 deg = elbow level with shoulder."""
    if not valid(shoulder, elbow):
        return NAN
    v = np.asarray(elbow, float) - np.asarray(shoulder, float)
    if np.linalg.norm(v) < 1e-6:
        return NAN
    return float(np.degrees(np.arctan2(abs(v[0]), v[1])))


def _shin_crossing_parameters(
    p: dict[str, np.ndarray], scale: float
) -> dict[str, float]:
    """Swastikapadam geometry: crossing angle, crossing distance, placement."""
    out: dict[str, float] = {}

    l_shin = (
        np.asarray(p["l_ankle"], float) - np.asarray(p["l_knee"], float)
        if valid(p["l_ankle"], p["l_knee"])
        else np.array([np.nan, np.nan])
    )
    r_shin = (
        np.asarray(p["r_ankle"], float) - np.asarray(p["r_knee"], float)
        if valid(p["r_ankle"], p["r_knee"])
        else np.array([np.nan, np.nan])
    )
    out["shin_crossing_angle_deg"] = angle_between_vectors_deg(l_shin, r_shin)

    out["shin_crossing_distance_norm"] = safe_ratio(
        segment_min_distance(p["l_knee"], p["l_ankle"], p["r_knee"], p["r_ankle"]),
        scale,
    )

    # With the dancer facing the camera the person's left side appears on the
    # image right, so a NEGATIVE signed gap means the legs have crossed over.
    out["ankle_crossing_signed_norm"] = safe_ratio(
        float(p["l_ankle"][0] - p["r_ankle"][0])
        if valid(p["l_ankle"], p["r_ankle"])
        else NAN,
        scale,
    )
    out["knee_crossing_signed_norm"] = safe_ratio(
        float(p["l_knee"][0] - p["r_knee"][0])
        if valid(p["l_knee"], p["r_knee"])
        else NAN,
        scale,
    )

    # which leg travelled furthest across its own hip -> the crossing leg
    left_dev = (
        abs(float(p["l_ankle"][0] - p["l_hip"][0]))
        if valid(p["l_ankle"], p["l_hip"])
        else NAN
    )
    right_dev = (
        abs(float(p["r_ankle"][0] - p["r_hip"][0]))
        if valid(p["r_ankle"], p["r_hip"])
        else NAN
    )
    out["left_ankle_hip_deviation_norm"] = safe_ratio(left_dev, scale)
    out["right_ankle_hip_deviation_norm"] = safe_ratio(right_dev, scale)
    return out


def crossing_side(params: dict[str, float]) -> str:
    """'RIGHT'/'LEFT' for the leg that crosses in front, '' when undetermined."""
    left = params.get("left_ankle_hip_deviation_norm", NAN)
    right = params.get("right_ankle_hip_deviation_norm", NAN)
    if not np.isfinite(left) or not np.isfinite(right):
        return ""
    if abs(left - right) < 0.05:
        return ""
    return "LEFT" if left > right else "RIGHT"
