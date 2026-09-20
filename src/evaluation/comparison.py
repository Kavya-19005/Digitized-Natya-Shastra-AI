"""Five-approach pose-estimator comparison on the labelled 208-image set.

All estimators are adapted into the common 67-landmark representation, then the
same parameter, recognition and correctness engines are applied. This file is
an evaluation mode; it is not on the RTMW video production path.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import EVALUATION_DIR, ensure_output_dirs
from ..dataset import DatasetItem
from ..evaluation.image_runner import _row_for
from ..labels import BODY_POSTURES, MUDRA_POSTURES
from ..landmarks import common as C
from ..pipeline import AnalysisResult, PostureAnalyzer
from ..pose.approaches import Approach


def run_approach(
    approach: Approach,
    backend,
    items: list[DatasetItem],
    progress: bool = True,
) -> pd.DataFrame:
    """Run one estimator adapter through the shared analysis stages."""
    analyzer = PostureAnalyzer(backend=backend)
    rows: list[dict[str, object]] = []
    for i, item in enumerate(items, start=1):
        start = time.perf_counter()
        label = item.label
        meta = {
            "image_name": item.image_name,
            "camera": item.camera,
            "variation_id": item.variation_id,
            "ground_truth_posture": label.posture,
            "ground_truth_status": label.status,
            "ground_truth_error": label.error_code,
            "ground_truth_side": label.side,
            "model_name": approach.key,
            "model_layout": backend.layout.name,
        }
        image = cv2.imread(str(item.path))
        if image is None:
            rows.append({**meta, "landmark_detection_status": "READ_ERROR"})
            continue
        try:
            result = analyzer.analyse(image)
        except Exception as exc:
            result = AnalysisResult(
                landmarks=C.Landmarks.empty(approach.key),
                detection_status="ERROR",
                message=str(exc),
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        rows.append(_row_for(meta, result, elapsed))
        if progress and i % 20 == 0:
            print(f"    {approach.display_name}: {i}/{len(items)} images")
    return pd.DataFrame(rows)


def summarise_approach(
    approach: Approach,
    df: pd.DataFrame | None,
    runnable: bool,
    probe_note: str,
    init_error: str = "",
) -> dict[str, object]:
    """Build one comparison row. Unavailable metrics are NaN, not zeros."""
    total = 0 if df is None else len(df)
    row: dict[str, object] = {
        "model": approach.display_name,
        "model_key": approach.key,
        "runnable": int(runnable and df is not None),
        "total_images": total,
        "successful_images": 0,
        "failed_images": total,
        "mean_landmarks_available": np.nan,
        "landmark_availability_percent": np.nan,
        "landmarks_supported": np.nan,
        "posture_accuracy": np.nan,
        "correctness_accuracy": np.nan,
        "body_posture_accuracy": np.nan,
        "mudra_posture_accuracy": np.nan,
        "processing_time_per_image_ms": np.nan,
        "body_parameter_availability": np.nan,
        "hand_parameter_availability": np.nan,
        "feet_parameter_availability": np.nan,
        "notes": init_error or probe_note,
    }
    if df is None or df.empty:
        row["failed_images"] = total
        return row

    ok = df["landmark_detection_status"] == "OK"
    successful = int(ok.sum())
    row["successful_images"] = successful
    row["failed_images"] = int((~ok).sum())
    row["processing_time_per_image_ms"] = round(float(df["processing_ms"].mean()), 1)

    if successful == 0:
        row["notes"] = (row["notes"] + "; no successful detections").strip("; ")
        return row

    work = df.loc[ok]
    mean_lm = float(pd.to_numeric(work["landmarks_available_count"], errors="coerce").mean())
    row["mean_landmarks_available"] = round(mean_lm, 2)
    row["landmark_availability_percent"] = round(100.0 * mean_lm / C.NUM_COMMON, 1)
    if "landmarks_supported_by_model" in work:
        row["landmarks_supported"] = int(
            pd.to_numeric(work["landmarks_supported_by_model"], errors="coerce").max()
        )

    row["body_parameter_availability"] = _pct_true(work, "body_landmarks_available")
    row["feet_parameter_availability"] = _pct_true(work, "feet_landmarks_available")
    row["hand_parameter_availability"] = _hand_availability(work)

    has_hands = int(row["landmarks_supported"] or 0) > 40

    row["posture_accuracy"] = _accuracy(df, "posture_correct")
    row["correctness_accuracy"] = _accuracy(df, "status_correct")

    body_mask = df["ground_truth_posture"].isin(BODY_POSTURES)
    body = df.loc[body_mask]
    body_measurable = body
    if "body_landmarks_available" in body:
        body_measurable = body.loc[pd.to_numeric(body["body_landmarks_available"], errors="coerce").fillna(0) >= 1]
    if len(body_measurable):
        row["body_posture_accuracy"] = _accuracy(body_measurable, "posture_correct")
    else:
        row["body_posture_accuracy"] = np.nan

    if has_hands:
        mudra_mask = df["ground_truth_posture"].isin(MUDRA_POSTURES)
        mudra = df.loc[mudra_mask]
        if len(mudra):
            row["mudra_posture_accuracy"] = _accuracy(mudra, "posture_correct")
    else:
        row["mudra_posture_accuracy"] = np.nan
        extra = "hand landmarks not produced by this estimator; mudra accuracy not reported"
        row["notes"] = (str(row["notes"]) + "; " + extra).strip("; ")

    return row


def _pct_true(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return float("nan")
    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return round(100.0 * float(series.mean()), 1)


def _hand_availability(df: pd.DataFrame) -> float:
    if "left_hand_available" not in df or "right_hand_available" not in df:
        return float("nan")
    left = pd.to_numeric(df["left_hand_available"], errors="coerce").fillna(0.0)
    right = pd.to_numeric(df["right_hand_available"], errors="coerce").fillna(0.0)
    return round(100.0 * float(((left >= 1) | (right >= 1)).mean()), 1)


def _accuracy(df: pd.DataFrame, column: str) -> float:
    if column not in df or df.empty:
        return float("nan")
    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return round(100.0 * float(series.mean()), 1)


def confusion_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        table = pd.crosstab(
            df["ground_truth_posture"].fillna("NONE"),
            df["detected_posture"].fillna("NONE"),
        )
        for gt in table.index:
            for pred in table.columns:
                count = int(table.loc[gt, pred])
                if count:
                    rows.append(
                        {
                            "model": name,
                            "ground_truth_posture": gt,
                            "detected_posture": pred,
                            "count": count,
                        }
                    )
    return pd.DataFrame(rows)


def write_comparison(
    summary: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    ensure_output_dirs()
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    summary_path = EVALUATION_DIR / "model_comparison.csv"
    summary.to_csv(summary_path, index=False)
    written["summary"] = summary_path

    long_path = EVALUATION_DIR / "model_comparison_confusion_matrix.csv"
    confusion_long(frames).to_csv(long_path, index=False)
    written["confusion_long"] = long_path

    for name, df in frames.items():
        per = pd.crosstab(
            df["ground_truth_posture"].fillna("NONE"),
            df["detected_posture"].fillna("NONE"),
        )
        path = EVALUATION_DIR / f"confusion_{name}.csv"
        per.to_csv(path)
        written[f"confusion_{name}"] = path
        detail = EVALUATION_DIR / f"image_results_{name}.csv"
        df.to_csv(detail, index=False)
        written[f"detail_{name}"] = detail
    return written
