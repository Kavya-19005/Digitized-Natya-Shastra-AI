"""Report-ready summary tables built from the per-image result CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import IMAGE_RESULTS_DIR
from ..labels import VARIATIONS


def model_processing_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in frames.items():
        total = len(df)
        detected = int((df["landmark_detection_status"] == "OK").sum())
        rows.append(
            {
                "model_name": model,
                "model_layout": _first(df, "model_layout"),
                "total_images_processed": total,
                "successful_detections": detected,
                "detection_rate_pct": _pct(detected, total),
                "required_landmarks_available": int(
                    df.get("required_landmarks_available", pd.Series(dtype=float))
                    .fillna(0)
                    .sum()
                ),
                "parameter_calculation_success": int(
                    df.get("parameter_calculation_ok", pd.Series(dtype=float))
                    .fillna(0)
                    .sum()
                ),
                "posture_recognised": int((df["detected_posture"].fillna("") != "").sum()),
                "posture_accuracy_pct": _pct(int(df["posture_correct"].sum()), total),
                "status_accuracy_pct": _pct(int(df["status_correct"].sum()), total),
                "mean_processing_ms": round(float(df["processing_ms"].mean()), 1),
                "mean_detect_ms": round(float(df["detect_ms"].mean()), 1),
                "mean_pose_ms": round(float(df["pose_ms"].mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def landmark_availability_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in frames.items():
        total = len(df)
        rows.append(
            {
                "model_name": model,
                "model_layout": _first(df, "model_layout"),
                "landmark_slots_supported_by_layout": int(
                    df["landmarks_supported_by_model"].max()
                ),
                "mean_landmarks_available": round(
                    float(df["landmarks_available_count"].mean()), 1
                ),
                "body_landmarks_available_pct": _pct(
                    int(df["body_landmarks_available"].fillna(0).sum()), total
                ),
                "feet_landmarks_available_pct": _pct(
                    int(df["feet_landmarks_available"].fillna(0).sum()), total
                ),
                "arm_landmarks_available_pct": _pct(
                    int(df["arm_landmarks_available"].fillna(0).sum()), total
                ),
                "left_hand_available_pct": _pct(
                    int(df["left_hand_available"].fillna(0).sum()), total
                ),
                "right_hand_available_pct": _pct(
                    int(df["right_hand_available"].fillna(0).sum()), total
                ),
                "either_hand_available_pct": _pct(
                    int(
                        (
                            df["left_hand_available"].fillna(0)
                            + df["right_hand_available"].fillna(0)
                            > 0
                        ).sum()
                    ),
                    total,
                ),
            }
        )
    return pd.DataFrame(rows)


def parameter_calculation_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-parameter completeness and descriptive statistics."""
    rows = []
    skip = {
        "image_name",
        "camera",
        "model_name",
        "model_layout",
        "landmark_detection_status",
        "mudra_hand_used",
        "detected_posture",
        "predicted_status",
        "predicted_error",
        "predicted_error_text",
        "predicted_error_kind",
        "primary_parameter",
        "primary_expected",
        "predicted_side",
        "rule_message",
        "recognition_family",
        "ground_truth_posture",
        "ground_truth_status",
        "ground_truth_error",
        "ground_truth_side",
    }
    for model, df in frames.items():
        numeric = df.select_dtypes(include=[np.number])
        for column in numeric.columns:
            if column in skip:
                continue
            series = numeric[column]
            finite = series[np.isfinite(series)]
            rows.append(
                {
                    "model_name": model,
                    "parameter": column,
                    "computed_count": int(finite.size),
                    "total_images": int(series.size),
                    "completeness_pct": _pct(int(finite.size), int(series.size)),
                    "mean": _round(finite.mean()),
                    "std": _round(finite.std()),
                    "min": _round(finite.min()),
                    "median": _round(finite.median()),
                    "max": _round(finite.max()),
                }
            )
    return pd.DataFrame(rows)


def posture_correctness_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-variation ground truth vs prediction."""
    rows = []
    for model, df in frames.items():
        for variation in VARIATIONS:
            subset = df[df["variation_id"] == variation.variation_id]
            if subset.empty:
                continue
            total = len(subset)
            predicted = subset["detected_posture"].fillna("")
            rows.append(
                {
                    "model_name": model,
                    "variation_id": variation.variation_id,
                    "ground_truth_posture": variation.posture,
                    "ground_truth_status": variation.status,
                    "ground_truth_error": variation.error_code,
                    "images": total,
                    "posture_correct": int(subset["posture_correct"].sum()),
                    "posture_accuracy_pct": _pct(
                        int(subset["posture_correct"].sum()), total
                    ),
                    "status_correct": int(subset["status_correct"].sum()),
                    "status_accuracy_pct": _pct(
                        int(subset["status_correct"].sum()), total
                    ),
                    "error_correct": int(subset["error_correct"].sum()),
                    "most_common_prediction": (
                        predicted.mode().iat[0] if not predicted.mode().empty else ""
                    ),
                    "most_common_error": _mode(subset["predicted_error"]),
                }
            )
    return pd.DataFrame(rows)


def camera_view_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-camera results, with the viewpoint each camera was classified as.

    The four cameras see the dancer from different angles, and a single RGB view
    cannot measure every parameter, so results are only interpretable per camera.
    """
    rows = []
    for model, df in frames.items():
        for camera, subset in df.groupby("camera"):
            total = len(subset)
            views = subset["view_class"].fillna("UNKNOWN")
            facings = subset["facing"].fillna("")
            measurable = subset["detected_posture"].fillna("") != ""
            rows.append(
                {
                    "model_name": model,
                    "camera": camera,
                    "images": total,
                    "dominant_view": views.mode().iat[0] if not views.mode().empty else "",
                    "dominant_facing": (
                        facings[facings != ""].mode().iat[0]
                        if not facings[facings != ""].mode().empty
                        else ""
                    ),
                    "coronal_pct": _pct(int((views == "CORONAL").sum()), total),
                    "sagittal_pct": _pct(int((views == "SAGITTAL").sum()), total),
                    "posture_recognised_pct": _pct(int(measurable.sum()), total),
                    "posture_accuracy_pct": _pct(int(subset["posture_correct"].sum()), total),
                    "posture_accuracy_of_recognised_pct": _pct(
                        int(subset.loc[measurable, "posture_correct"].sum()),
                        int(measurable.sum()),
                    ),
                    "status_accuracy_pct": _pct(int(subset["status_correct"].sum()), total),
                    "mudra_measurable_pct": _pct(
                        int(subset.get("mudra_measurable", pd.Series(dtype=float)).fillna(0).sum()),
                        total,
                    ),
                }
            )
    return pd.DataFrame(rows)


def error_identification_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How often the specific error code matches the labelled error."""
    rows = []
    for model, df in frames.items():
        labelled = df[df["ground_truth_error"].fillna("") != ""]
        for error, subset in labelled.groupby("ground_truth_error"):
            rows.append(
                {
                    "model_name": model,
                    "ground_truth_error": error,
                    "images": len(subset),
                    "posture_correct": int(subset["posture_correct"].sum()),
                    "flagged_incorrect": int(
                        (subset["predicted_status"] == "INCORRECT").sum()
                    ),
                    "flagged_incorrect_pct": _pct(
                        int((subset["predicted_status"] == "INCORRECT").sum()),
                        len(subset),
                    ),
                    "exact_error_match": int(subset["error_correct"].sum()),
                    "exact_error_match_pct": _pct(
                        int(subset["error_correct"].sum()), len(subset)
                    ),
                    "most_common_predicted_error": _mode(subset["predicted_error"]),
                }
            )
    return pd.DataFrame(rows)


def confusion_table(frames: dict[str, pd.DataFrame], model: str) -> pd.DataFrame:
    df = frames[model]
    return pd.crosstab(
        df["ground_truth_posture"], df["detected_posture"].fillna("NONE")
    )


def write_all(frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    IMAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    tables = {
        "model_processing_summary": model_processing_summary(frames),
        "landmark_availability_summary": landmark_availability_summary(frames),
        "parameter_calculation_summary": parameter_calculation_summary(frames),
        "posture_correctness_summary": posture_correctness_summary(frames),
        "camera_view_summary": camera_view_summary(frames),
        "error_identification_summary": error_identification_summary(frames),
    }
    for name, table in tables.items():
        target = IMAGE_RESULTS_DIR / f"{name}.csv"
        table.to_csv(target, index=False)
        written[name] = target

    for model in frames:
        target = IMAGE_RESULTS_DIR / f"confusion_posture_{model}.csv"
        confusion_table(frames, model).to_csv(target)
        written[f"confusion_{model}"] = target
    return written


def _pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


def _round(value) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return float("nan")


def _first(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df.empty:
        return ""
    values = df[column].dropna()
    return str(values.iat[0]) if not values.empty else ""


def _mode(series: pd.Series) -> str:
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return ""
    mode = cleaned.mode()
    return str(mode.iat[0]) if not mode.empty else ""
