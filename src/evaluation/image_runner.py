"""Runs the full pipeline over the 208 labelled images for one ONNX model."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import IMAGE_RESULTS_DIR, KEYPOINT_SCORE_THRESHOLD
from ..correctness.rules import evaluate
from ..dataset import DatasetItem, build_manifest
from ..features import compute_parameters
from ..landmarks import common as C
from ..pipeline import AnalysisResult, PostureAnalyzer
from ..pose.registry import ModelSpec
from ..recognition.posture import recognise
from .landmark_cache import META_COLUMNS, landmarks_from_row, read_cache

# Parameters written to the per-image CSV, in report order.
BODY_PARAM_COLUMNS = [
    "left_knee_flexion_deg",
    "right_knee_flexion_deg",
    "mean_knee_flexion_deg",
    "left_knee_joint_angle_deg",
    "right_knee_joint_angle_deg",
    "knee_asymmetry_deg",
    "foot_separation_norm",
    "heel_separation_norm",
    "toe_separation_norm",
    "toe_heel_separation_ratio",
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
    "trunk_inclination_deg",
    "hip_tilt_deg",
    "shoulder_tilt_deg",
    "pelvic_symmetry_deg",
    "com_lateral_offset_norm",
    "pelvis_lateral_offset_norm",
    "leg_length_ratio_proxy",
    "weight_shift_proxy",
    "left_heel_lift_norm",
    "right_heel_lift_norm",
    "mean_heel_lift_norm",
    "hip_height_norm",
    "knee_height_norm",
    "hip_height_torso_norm",
    "knee_height_torso_norm",
    "trunk_foreshortening_ratio",
    "left_foot_length_norm",
    "right_foot_length_norm",
    "shin_crossing_angle_deg",
    "shin_crossing_distance_norm",
    "ankle_crossing_signed_norm",
    "knee_crossing_signed_norm",
    "left_ankle_hip_deviation_norm",
    "right_ankle_hip_deviation_norm",
    "left_elbow_shoulder_alignment_norm",
    "right_elbow_shoulder_alignment_norm",
    "mean_elbow_shoulder_alignment_norm",
    "left_arm_abduction_deg",
    "right_arm_abduction_deg",
    "shoulder_width_px",
    "scale_px",
]

HAND_PARAM_SUFFIXES = [
    "palm_size_px",
    "mean_finger_straightness_deg",
    "min_finger_straightness_deg",
    "mean_finger_flexion_deg",
    "mean_interfinger_spread_deg",
    "max_interfinger_spread_deg",
    "mean_adjacent_fingertip_gap_norm",
    "max_adjacent_fingertip_gap_norm",
    "index_pinky_tip_gap_norm",
    "palm_opening_norm",
    "pinky_relative_extension",
    "thumb_index_tip_distance_norm",
    "thumb_middle_tip_distance_norm",
    "fingertip_cluster_spread_norm",
    "extended_finger_contrast",
    "ring_pinky_straightness_deg",
    "wrist_joint_angle_deg",
    "wrist_deviation_deg",
    "palm_axis_from_vertical_deg",
    "wrist_lift_norm",
]

AVAILABILITY_COLUMNS = [
    "landmarks_available_count",
    "body_landmarks_available",
    "feet_landmarks_available",
    "arm_landmarks_available",
    "left_hand_available",
    "right_hand_available",
]


def _row_for(
    meta: dict[str, object],
    result: AnalysisResult,
    elapsed_ms: float,
) -> dict[str, object]:
    ground_truth_posture = str(meta.get("ground_truth_posture", ""))
    ground_truth_status = str(meta.get("ground_truth_status", ""))
    ground_truth_error = str(meta.get("ground_truth_error", "") or "")

    params = result.parameters
    row: dict[str, object] = {
        **meta,
        "landmark_detection_status": result.detection_status,
        "processing_ms": round(elapsed_ms, 2),
        "detect_ms": round(result.detect_ms, 2),
        "pose_ms": round(result.pose_ms, 2),
    }

    mask = result.landmarks.available(KEYPOINT_SCORE_THRESHOLD)
    supported = result.landmarks.supported
    row["landmarks_supported_by_model"] = int(supported.sum())
    row["required_landmarks_available"] = int(
        result.landmarks.has(C.REQUIRED_BODY, KEYPOINT_SCORE_THRESHOLD)
    )
    row["parameter_calculation_ok"] = int(bool(params))

    for column in AVAILABILITY_COLUMNS:
        row[column] = params.get(column, np.nan)

    for column in BODY_PARAM_COLUMNS:
        row[column] = params.get(column, np.nan)

    for column in ("view_class", "facing", "mudra_measurable", "shoulder_torso_ratio",
                   "torso_height_px"):
        row[column] = params.get(column, "" if column in {"view_class", "facing"} else np.nan)

    row["mudra_hand_used"] = params.get("mudra_hand_used", "")
    for side in ("left", "right"):
        for suffix in HAND_PARAM_SUFFIXES:
            key = f"{side}_hand_{suffix}"
            row[key] = params.get(key, np.nan)

    if result.recognition is not None:
        row["recognition_family"] = result.recognition.family
        row["recognition_confidence"] = round(result.recognition.confidence, 4)
        for posture, score in result.recognition.scores.items():
            row[f"score_{posture}"] = round(float(score), 4)
    else:
        row["recognition_family"] = ""
        row["recognition_confidence"] = np.nan

    if result.verdict is not None:
        row.update(result.verdict.as_row())
    else:
        row.update(
            {
                "detected_posture": "",
                "predicted_status": "UNKNOWN",
                "predicted_error": "",
                "predicted_error_text": "",
                "predicted_error_kind": "",
                "primary_parameter": "",
                "primary_value": np.nan,
                "primary_expected": "",
                "predicted_side": "",
                "borderline": False,
                "rule_message": result.message,
            }
        )

    row["posture_correct"] = int(row.get("detected_posture") == ground_truth_posture)
    row["status_correct"] = int(row.get("predicted_status") == ground_truth_status)
    row["error_correct"] = int(
        bool(ground_truth_error)
        and row.get("predicted_error") == ground_truth_error
    )
    row["unused_landmark_slots"] = int(C.NUM_COMMON - int(mask.sum()))
    return row


def _meta_from_item(item: DatasetItem, spec: ModelSpec) -> dict[str, object]:
    label = item.label
    return {
        "image_name": item.image_name,
        "camera": item.camera,
        "variation_id": item.variation_id,
        "ground_truth_posture": label.posture,
        "ground_truth_status": label.status,
        "ground_truth_error": label.error_code,
        "ground_truth_side": label.side,
        "model_name": spec.key,
        "model_layout": spec.layout.name,
    }


def run_model(
    spec: ModelSpec,
    items: list[DatasetItem] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Full run: ONNX inference plus the analysis stages, image by image."""
    items = items or build_manifest()
    analyzer = PostureAnalyzer(spec)
    rows: list[dict[str, object]] = []

    for i, item in enumerate(items, start=1):
        start = time.perf_counter()
        meta = _meta_from_item(item, spec)
        image = cv2.imread(str(item.path))
        if image is None:
            rows.append({**meta, "landmark_detection_status": "READ_ERROR"})
            continue

        try:
            result = analyzer.analyse(image)
        except Exception as exc:
            result = AnalysisResult(
                landmarks=C.Landmarks.empty(spec.key),
                detection_status="ERROR",
                message=str(exc),
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        rows.append(_row_for(meta, result, elapsed))

        if progress and i % 20 == 0:
            print(f"    {spec.key}: {i}/{len(items)} images")

    return pd.DataFrame(rows)


def run_model_from_cache(spec: ModelSpec, progress: bool = True) -> pd.DataFrame:
    """Analysis stages only, replayed from the cached landmark CSV.

    Lets the parameter, recognition and rule layers be re-run and recalibrated
    without repeating ONNX inference.
    """
    cache = read_cache(spec.key)
    rows: list[dict[str, object]] = []

    for i, (_, cached) in enumerate(cache.iterrows(), start=1):
        meta = {column: cached.get(column) for column in META_COLUMNS if column in cache}
        meta.pop("landmark_detection_status", None)
        meta.pop("detect_ms", None)
        meta.pop("pose_ms", None)

        status = str(cached.get("landmark_detection_status", ""))
        detect_ms = float(cached.get("detect_ms", 0.0) or 0.0)
        pose_ms = float(cached.get("pose_ms", 0.0) or 0.0)

        if status != "OK":
            rows.append({**meta, "landmark_detection_status": status})
            continue

        landmarks = landmarks_from_row(cached, spec.key)
        landmarks.supported = spec.layout.supported_mask
        try:
            params = compute_parameters(landmarks, KEYPOINT_SCORE_THRESHOLD)
            recognition = recognise(params)
            verdict = evaluate(recognition.posture, params, side=recognition.side)
            result = AnalysisResult(
                landmarks=landmarks,
                detection_status="OK",
                parameters=params,
                recognition=recognition,
                verdict=verdict,
                detect_ms=detect_ms,
                pose_ms=pose_ms,
            )
        except Exception as exc:
            result = AnalysisResult(
                landmarks=landmarks,
                detection_status="PARAMETER_ERROR",
                message=str(exc),
            )
        rows.append(_row_for(meta, result, detect_ms + pose_ms))

        if progress and i % 100 == 0:
            print(f"    {spec.key}: {i}/{len(cache)} cached rows")

    return pd.DataFrame(rows)


def write_results(df: pd.DataFrame, spec: ModelSpec) -> Path:
    IMAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = IMAGE_RESULTS_DIR / f"image_results_{spec.key}.csv"
    df.to_csv(target, index=False)
    return target
