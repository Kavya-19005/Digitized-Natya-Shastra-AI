"""Report what the labelled images actually measure, per camera and variation.

This produces DATA_CALIBRATED reference values for the write-up. It never edits
the PROJECT_SPEC ranges in `src/correctness/thresholds.py`; the numbers are
reported so the report can compare specification against measurement.

    python scripts/calibrate_thresholds.py
    python scripts/calibrate_thresholds.py --views CORONAL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import IMAGE_RESULTS_DIR
from src.labels import VARIATION_BY_ID

CUE_COLUMNS = [
    "mean_knee_flexion_deg",
    "knee_asymmetry_deg",
    "foot_separation_norm",
    "mean_foot_turnout_deg",
    "foot_line_deviation_deg",
    "trunk_inclination_deg",
    "mean_heel_lift_norm",
    "weight_shift_proxy",
    "hip_height_norm",
    "knee_spread_norm",
    "ankle_crossing_signed_norm",
    "shin_crossing_distance_norm",
    "shin_crossing_angle_deg",
    "mean_elbow_shoulder_alignment_norm",
]

HAND_SUFFIXES = [
    "mean_finger_straightness_deg",
    "mean_finger_flexion_deg",
    "mean_interfinger_spread_deg",
    "max_adjacent_fingertip_gap_norm",
    "palm_opening_norm",
    "thumb_index_tip_distance_norm",
    "fingertip_cluster_spread_norm",
    "ring_pinky_straightness_deg",
    "pinky_relative_extension",
    "wrist_lift_norm",
]


def chosen_hand_columns(df: pd.DataFrame) -> pd.DataFrame:
    side = df["mudra_hand_used"].fillna("")
    for suffix in HAND_SUFFIXES:
        values = []
        for hand, (_, row) in zip(side, df.iterrows()):
            values.append(row.get(f"{hand}_hand_{suffix}", np.nan) if hand else np.nan)
        df[f"hand_{suffix}"] = values
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rtmw_l_wholebody")
    parser.add_argument("--views", nargs="*", default=None)
    args = parser.parse_args()

    df = pd.read_csv(IMAGE_RESULTS_DIR / f"image_results_{args.model}.csv")
    df = chosen_hand_columns(df)
    if args.views:
        df = df[df["view_class"].isin(args.views)]

    pd.set_option("display.width", 260, "display.max_columns", 80)

    print("=== VIEW CLASS PER CAMERA ===")
    print(pd.crosstab(df["camera"], df["view_class"].fillna("NONE")).to_string())
    print()
    print(pd.crosstab(df["camera"], df["facing"].fillna("NONE")).to_string())

    print("\n=== ACCURACY PER CAMERA ===")
    print(
        df.groupby("camera")
        .agg(
            images=("image_name", "count"),
            detected=("landmark_detection_status", lambda s: (s == "OK").sum()),
            posture_acc=("posture_correct", "mean"),
            status_acc=("status_correct", "mean"),
        )
        .assign(
            posture_acc=lambda d: (100 * d.posture_acc).round(1),
            status_acc=lambda d: (100 * d.status_acc).round(1),
        )
        .to_string()
    )

    coronal = df[df["view_class"] == "CORONAL"]
    print("\n=== CORONAL-VIEW BODY CUES BY VARIATION (mean / std) ===")
    grouped = coronal.groupby(["variation_id", "ground_truth_posture"])
    print(grouped[CUE_COLUMNS].mean().round(2).to_string())

    print("\n=== FRONT-FACING HAND CUES BY VARIATION ===")
    front = df[(df["facing"] == "FRONT")]
    hand_cols = [f"hand_{s}" for s in HAND_SUFFIXES]
    print(
        front.groupby(["variation_id", "ground_truth_posture"])[hand_cols]
        .mean()
        .round(2)
        .to_string()
    )

    print("\n=== DATA_CALIBRATED REFERENCE (correct variations only, coronal) ===")
    correct = coronal[coronal["ground_truth_status"] == "CORRECT"]
    reference = (
        correct.groupby("ground_truth_posture")[CUE_COLUMNS]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
    )
    target = IMAGE_RESULTS_DIR / "data_calibrated_reference.csv"
    reference.to_csv(target)
    print(reference.to_string())
    print(f"\nwritten: {target}")

    print("\n=== PREDICTION BREAKDOWN ===")
    print(
        df.groupby(["variation_id", "ground_truth_posture"])
        .agg(
            n=("image_name", "count"),
            posture_hits=("posture_correct", "sum"),
            status_hits=("status_correct", "sum"),
            predicted=("detected_posture", lambda s: "/".join(sorted(set(s.fillna(""))))),
        )
        .to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
