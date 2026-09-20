"""Mudra (hand) parameters from the 21-point hand blocks of the common layout.

Every hand distance is normalised by the palm size (wrist -> middle MCP) so the
values do not depend on how far the dancer stands from the camera.

Convention note: `*_straightness_deg` is a RAW joint angle (180 deg = perfectly
straight finger); `*_flexion_deg` is 180 - straightness. The fixed
finger-straightness range (160-180 deg correct) applies to the straightness
fields.
"""

from __future__ import annotations

import numpy as np

from ..landmarks import common as C
from ..landmarks.common import Landmarks
from .geometry import (
    angle_between_vectors_deg,
    distance,
    flexion_from_joint_angle,
    joint_angle_deg,
    nan_mean,
    safe_ratio,
    valid,
)

NAN = float("nan")
ADJACENT_FINGERS = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))


def _hand_points(
    lm: Landmarks, side: str, mask: np.ndarray
) -> dict[int, np.ndarray] | None:
    base = C.LEFT_HAND_START if side == "left" else C.RIGHT_HAND_START
    points: dict[int, np.ndarray] = {}
    for offset in range(21):
        idx = base + offset
        points[offset] = (
            lm.xy[idx].astype(float) if mask[idx] else np.array([np.nan, np.nan])
        )
    if not valid(points[C.H_WRIST]):
        return None
    return points


def compute_hand_parameters(
    lm: Landmarks, side: str, score_threshold: float
) -> dict[str, float]:
    """Parameters for one hand. Returns all-NaN when the hand is unavailable."""
    prefix = f"{side}_hand"
    mask = lm.available(score_threshold)
    keys = _parameter_keys(prefix)
    out: dict[str, float] = {k: NAN for k in keys}

    pts = _hand_points(lm, side, mask)
    if pts is None:
        return out

    wrist = pts[C.H_WRIST]
    middle_mcp = pts[C.H_MIDDLE_MCP]
    palm_size = distance(wrist, middle_mcp)
    out[f"{prefix}_palm_size_px"] = palm_size
    if not np.isfinite(palm_size) or palm_size < 1e-6:
        return out

    # palm centre: average of wrist and the four long-finger MCPs
    mcp_ids = [C.H_INDEX_MCP, C.H_MIDDLE_MCP, C.H_RING_MCP, C.H_PINKY_MCP]
    mcps = [pts[i] for i in mcp_ids if valid(pts[i])]
    palm_center = (
        np.mean(np.vstack([wrist] + mcps), axis=0)
        if mcps
        else np.array([np.nan, np.nan])
    )

    # --- per-finger straightness / flexion --------------------------------
    straightness: dict[str, float] = {}
    for finger, chain in C.FINGER_CHAINS.items():
        a, b, c, d = (pts[i] for i in chain)
        angles = [joint_angle_deg(a, b, c), joint_angle_deg(b, c, d)]
        value = nan_mean(angles)
        straightness[finger] = value
        out[f"{prefix}_{finger}_straightness_deg"] = value
        out[f"{prefix}_{finger}_flexion_deg"] = flexion_from_joint_angle(value)

    long_values = [straightness[f] for f in C.LONG_FINGERS]
    out[f"{prefix}_mean_finger_straightness_deg"] = nan_mean(long_values)
    out[f"{prefix}_min_finger_straightness_deg"] = (
        float(np.nanmin(long_values))
        if np.any(np.isfinite(long_values))
        else NAN
    )
    out[f"{prefix}_mean_finger_flexion_deg"] = flexion_from_joint_angle(
        out[f"{prefix}_mean_finger_straightness_deg"]
    )

    # --- inter-finger spread ----------------------------------------------
    directions: dict[str, np.ndarray] = {}
    for finger, chain in C.FINGER_CHAINS.items():
        mcp, tip = pts[chain[0]], pts[chain[3]]
        directions[finger] = (
            np.asarray(tip, float) - np.asarray(mcp, float)
            if valid(mcp, tip)
            else np.array([np.nan, np.nan])
        )

    spreads = [
        angle_between_vectors_deg(directions[a], directions[b])
        for a, b in ADJACENT_FINGERS
    ]
    out[f"{prefix}_mean_interfinger_spread_deg"] = nan_mean(spreads)
    out[f"{prefix}_max_interfinger_spread_deg"] = (
        float(np.nanmax(spreads)) if np.any(np.isfinite(spreads)) else NAN
    )
    out[f"{prefix}_thumb_index_spread_deg"] = angle_between_vectors_deg(
        directions["thumb"], directions["index"]
    )

    # --- fingertip geometry -------------------------------------------------
    tips = {f: pts[C.FINGER_CHAINS[f][3]] for f in C.FINGER_CHAINS}
    adjacent_tip_gaps = [
        safe_ratio(distance(tips[a], tips[b]), palm_size) for a, b in ADJACENT_FINGERS
    ]
    out[f"{prefix}_mean_adjacent_fingertip_gap_norm"] = nan_mean(adjacent_tip_gaps)
    out[f"{prefix}_max_adjacent_fingertip_gap_norm"] = (
        float(np.nanmax(adjacent_tip_gaps))
        if np.any(np.isfinite(adjacent_tip_gaps))
        else NAN
    )
    out[f"{prefix}_index_pinky_tip_gap_norm"] = safe_ratio(
        distance(tips["index"], tips["pinky"]), palm_size
    )

    # --- palm opening -------------------------------------------------------
    tip_to_palm = {
        f: safe_ratio(distance(tips[f], palm_center), palm_size)
        for f in C.FINGER_CHAINS
    }
    for finger, value in tip_to_palm.items():
        out[f"{prefix}_{finger}_tip_to_palm_norm"] = value
    long_tip_to_palm = [tip_to_palm[f] for f in C.LONG_FINGERS]
    out[f"{prefix}_palm_opening_norm"] = nan_mean(long_tip_to_palm)
    out[f"{prefix}_mean_fingertip_to_palm_norm"] = out[f"{prefix}_palm_opening_norm"]

    # --- Mushti: pinky drop / thumb position -------------------------------
    others = [tip_to_palm[f] for f in ("index", "middle", "ring")]
    mean_others = nan_mean(others)
    out[f"{prefix}_pinky_relative_extension"] = safe_ratio(
        tip_to_palm["pinky"], mean_others
    )
    out[f"{prefix}_thumb_tip_to_palm_norm"] = tip_to_palm["thumb"]
    out[f"{prefix}_thumb_tip_to_index_mcp_norm"] = safe_ratio(
        distance(tips["thumb"], pts[C.H_INDEX_MCP]), palm_size
    )

    # --- Katakamukha: thumb-finger contact and clustering ------------------
    # Katakamukha keeps the ring and little fingers extended while the thumb,
    # index and middle join; Mushti closes everything. The contrast between the
    # two finger groups separates the two mudras far better than overall
    # straightness does.
    outer = nan_mean([tip_to_palm["ring"], tip_to_palm["pinky"]])
    inner = nan_mean([tip_to_palm["index"], tip_to_palm["middle"]])
    out[f"{prefix}_extended_finger_contrast"] = (
        float(outer - inner) if np.isfinite(outer) and np.isfinite(inner) else NAN
    )

    out[f"{prefix}_thumb_index_tip_distance_norm"] = safe_ratio(
        distance(tips["thumb"], tips["index"]), palm_size
    )
    out[f"{prefix}_thumb_middle_tip_distance_norm"] = safe_ratio(
        distance(tips["thumb"], tips["middle"]), palm_size
    )
    cluster = [tips["thumb"], tips["index"], tips["middle"]]
    if all(valid(t) for t in cluster):
        centroid = np.mean(np.vstack(cluster), axis=0)
        out[f"{prefix}_fingertip_cluster_spread_norm"] = safe_ratio(
            float(np.mean([distance(t, centroid) for t in cluster])), palm_size
        )
    out[f"{prefix}_ring_pinky_straightness_deg"] = nan_mean(
        [straightness["ring"], straightness["pinky"]]
    )

    # --- wrist orientation --------------------------------------------------
    elbow_idx = C.LEFT_ELBOW if side == "left" else C.RIGHT_ELBOW
    elbow = lm.xy[elbow_idx].astype(float) if mask[elbow_idx] else np.array([np.nan] * 2)
    out[f"{prefix}_wrist_joint_angle_deg"] = joint_angle_deg(elbow, wrist, middle_mcp)
    out[f"{prefix}_wrist_deviation_deg"] = (
        abs(180.0 - out[f"{prefix}_wrist_joint_angle_deg"])
        if np.isfinite(out[f"{prefix}_wrist_joint_angle_deg"])
        else NAN
    )
    # orientation of the palm axis relative to the image vertical
    if valid(wrist, middle_mcp):
        v = np.asarray(middle_mcp, float) - np.asarray(wrist, float)
        out[f"{prefix}_palm_axis_from_vertical_deg"] = float(
            np.degrees(np.arctan2(abs(v[0]), -v[1]))
        )

    return out


def _parameter_keys(prefix: str) -> list[str]:
    keys = [
        f"{prefix}_palm_size_px",
        f"{prefix}_mean_finger_straightness_deg",
        f"{prefix}_min_finger_straightness_deg",
        f"{prefix}_mean_finger_flexion_deg",
        f"{prefix}_mean_interfinger_spread_deg",
        f"{prefix}_max_interfinger_spread_deg",
        f"{prefix}_thumb_index_spread_deg",
        f"{prefix}_mean_adjacent_fingertip_gap_norm",
        f"{prefix}_max_adjacent_fingertip_gap_norm",
        f"{prefix}_index_pinky_tip_gap_norm",
        f"{prefix}_palm_opening_norm",
        f"{prefix}_mean_fingertip_to_palm_norm",
        f"{prefix}_pinky_relative_extension",
        f"{prefix}_thumb_tip_to_palm_norm",
        f"{prefix}_thumb_tip_to_index_mcp_norm",
        f"{prefix}_thumb_index_tip_distance_norm",
        f"{prefix}_thumb_middle_tip_distance_norm",
        f"{prefix}_fingertip_cluster_spread_norm",
        f"{prefix}_extended_finger_contrast",
        f"{prefix}_ring_pinky_straightness_deg",
        f"{prefix}_wrist_joint_angle_deg",
        f"{prefix}_wrist_deviation_deg",
        f"{prefix}_palm_axis_from_vertical_deg",
    ]
    for finger in C.FINGER_CHAINS:
        keys += [
            f"{prefix}_{finger}_straightness_deg",
            f"{prefix}_{finger}_flexion_deg",
            f"{prefix}_{finger}_tip_to_palm_norm",
        ]
    return keys


def compute_mudra_parameters(
    lm: Landmarks, score_threshold: float
) -> dict[str, float | str]:
    """Both hands plus the combined 'best hand' summary used by the rules."""
    out: dict[str, float | str] = {}
    for side in ("left", "right"):
        out.update(compute_hand_parameters(lm, side, score_threshold))

    out["left_hand_available"] = float(lm.hand_available("left", score_threshold))
    out["right_hand_available"] = float(lm.hand_available("right", score_threshold))
    out.update(_wrist_lift(lm, score_threshold))

    # The mudra rules act on whichever hand is measured more completely; the
    # dataset shows the same mudra on both hands.
    out["mudra_hand_used"] = _pick_hand(out)
    return out


def _wrist_lift(lm: Landmarks, score_threshold: float) -> dict[str, float]:
    """Wrist height above the hip line, as a fraction of TORSO height.

    This is the cue that separates the dataset's two halves: body postures are
    held with the hands on the waist (lift ~= 0) while the mudra variations are
    presented with the hands raised in front of the chest (lift ~= 1 torso).

    Torso height is the normaliser rather than shoulder width because shoulder
    width collapses in a profile view, which would inflate the ratio and make
    every side-on frame look like a raised-hand mudra.
    """
    mask = lm.available(score_threshold)
    out: dict[str, float] = {
        "left_hand_wrist_lift_norm": NAN,
        "right_hand_wrist_lift_norm": NAN,
    }

    def point(idx: int) -> np.ndarray:
        return lm.xy[idx].astype(float) if mask[idx] else np.array([np.nan, np.nan])

    l_sh, r_sh = point(C.LEFT_SHOULDER), point(C.RIGHT_SHOULDER)
    l_hip, r_hip = point(C.LEFT_HIP), point(C.RIGHT_HIP)
    if not (valid(l_hip, r_hip) and valid(l_sh, r_sh)):
        return out

    mid_shoulder = 0.5 * (l_sh + r_sh)
    mid_hip = 0.5 * (l_hip + r_hip)
    torso = distance(mid_shoulder, mid_hip)
    if not np.isfinite(torso) or torso < 1e-6:
        return out

    hip_y = float(mid_hip[1])
    for side in ("left", "right"):
        wrist = point(C.hand_index(side, C.H_WRIST))
        if valid(wrist):
            # y grows downwards, so a raised wrist sits above the hip line
            out[f"{side}_hand_wrist_lift_norm"] = safe_ratio(
                hip_y - float(wrist[1]), torso
            )
    return out


def _pick_hand(params: dict[str, float | str]) -> str:
    left_ok = params.get("left_hand_available", 0.0) >= 1.0
    right_ok = params.get("right_hand_available", 0.0) >= 1.0
    if left_ok and right_ok:
        left_palm = params.get("left_hand_palm_size_px", NAN)
        right_palm = params.get("right_hand_palm_size_px", NAN)
        if np.isfinite(left_palm) and np.isfinite(right_palm):
            return "left" if left_palm >= right_palm else "right"
        return "left"
    if left_ok:
        return "left"
    if right_ok:
        return "right"
    return ""


def hand_param(params: dict[str, float | str], side: str, name: str) -> float:
    if not side:
        return NAN
    value = params.get(f"{side}_hand_{name}", NAN)
    return float(value) if isinstance(value, (int, float)) else NAN
