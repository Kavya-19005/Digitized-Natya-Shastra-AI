"""Persist the extracted 67-landmark arrays so the downstream layers can be
re-run (and re-calibrated) without paying for ONNX inference again.

The CSV is also a report artefact in its own right: one row per image with
x/y/score for each of the 67 common landmarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import IMAGE_RESULTS_DIR
from ..landmarks import common as C
from ..landmarks.common import Landmarks

META_COLUMNS = [
    "image_name",
    "camera",
    "variation_id",
    "ground_truth_posture",
    "ground_truth_status",
    "ground_truth_error",
    "ground_truth_side",
    "model_name",
    "model_layout",
    "landmark_detection_status",
    "detect_ms",
    "pose_ms",
    "hand_refined_left",
    "hand_refined_right",
]


def landmark_columns() -> list[str]:
    columns: list[str] = []
    for index in range(C.NUM_COMMON):
        name = C.COMMON_NAMES[index] or f"lm{index}"
        columns += [f"{index:02d}_{name}_x", f"{index:02d}_{name}_y", f"{index:02d}_{name}_score"]
    return columns


def row_from_landmarks(lm: Landmarks) -> dict[str, float]:
    row: dict[str, float] = {}
    for index in range(C.NUM_COMMON):
        name = C.COMMON_NAMES[index] or f"lm{index}"
        row[f"{index:02d}_{name}_x"] = float(lm.xy[index, 0])
        row[f"{index:02d}_{name}_y"] = float(lm.xy[index, 1])
        row[f"{index:02d}_{name}_score"] = float(lm.score[index])
    return row


def landmarks_from_row(row: pd.Series, model_name: str) -> Landmarks:
    lm = Landmarks.empty(model_name=model_name)
    for index in range(C.NUM_COMMON):
        name = C.COMMON_NAMES[index] or f"lm{index}"
        lm.xy[index, 0] = row[f"{index:02d}_{name}_x"]
        lm.xy[index, 1] = row[f"{index:02d}_{name}_y"]
        lm.score[index] = row[f"{index:02d}_{name}_score"]
    lm.supported = np.isfinite(lm.xy).all(axis=1)
    return lm


def cache_path(model_key: str) -> Path:
    return IMAGE_RESULTS_DIR / f"landmarks_67_{model_key}.csv"


def write_cache(df: pd.DataFrame, model_key: str) -> Path:
    IMAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = cache_path(model_key)
    df.to_csv(target, index=False)
    return target


def read_cache(model_key: str) -> pd.DataFrame:
    target = cache_path(model_key)
    if not target.exists():
        raise FileNotFoundError(
            f"landmark cache missing: {target}\n"
            "run scripts/extract_landmarks.py first"
        )
    return pd.read_csv(target)
