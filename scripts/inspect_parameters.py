"""Print measured parameters grouped by variation, to calibrate recognition."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import IMAGE_RESULTS_DIR

DEFAULT_COLUMNS = [
    "mean_knee_flexion_deg",
    "foot_separation_norm",
    "hip_height_norm",
    "ankle_crossing_signed_norm",
    "shin_crossing_distance_norm",
    "mean_foot_turnout_deg",
    "trunk_inclination_deg",
    "mean_heel_lift_norm",
    "weight_shift_proxy",
    "foot_line_deviation_deg",
    "knee_spread_norm",
    "knee_over_ankle_spread_ratio",
    "mean_elbow_shoulder_alignment_norm",
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "rtmw_l_wholebody"
    columns = sys.argv[2:] or DEFAULT_COLUMNS

    df = pd.read_csv(IMAGE_RESULTS_DIR / f"image_results_{model}.csv")
    hand_side = df["mudra_hand_used"].fillna("")
    if "knee_over_ankle_spread_ratio" not in df.columns:
        df["knee_over_ankle_spread_ratio"] = (
            df["knee_spread_norm"] / df["foot_separation_norm"]
        )

    # lift the chosen hand's parameters into unified columns for readability
    for suffix in (
        "mean_finger_straightness_deg",
        "mean_interfinger_spread_deg",
        "palm_opening_norm",
        "thumb_index_tip_distance_norm",
        "pinky_relative_extension",
        "fingertip_cluster_spread_norm",
        "wrist_lift_norm",
    ):
        df[f"hand_{suffix}"] = [
            row.get(f"{side}_hand_{suffix}") if side else float("nan")
            for side, (_, row) in zip(hand_side, df.iterrows())
        ]

    grouped = df.groupby(["variation_id", "ground_truth_posture"])
    pd.set_option("display.width", 250, "display.max_columns", 60)

    print("=== BODY CUES ===")
    print(grouped[columns].mean().round(2).to_string())

    print("\n=== HAND CUES (chosen hand) ===")
    hand_cols = [c for c in df.columns if c.startswith("hand_")]
    print(grouped[hand_cols].mean().round(2).to_string())

    print("\n=== PREDICTIONS ===")
    print(
        df.groupby(["variation_id", "ground_truth_posture"])
        .agg(
            predicted=("detected_posture", lambda s: "/".join(sorted(set(s.fillna(""))))),
            status=("predicted_status", lambda s: "/".join(sorted(set(s.fillna(""))))),
            error=("predicted_error", lambda s: "/".join(sorted(set(s.fillna("").astype(str))))),
        )
        .to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
