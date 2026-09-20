"""Verify the project's acceptance criteria against the generated artefacts.

    python scripts/acceptance_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import IMAGE_RESULTS_DIR, VIDEO_RESULTS_DIR
from src.pose.registry import MODEL_SPECS, available_models

CHECKS: list[tuple[str, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, "PASS" if passed else "FAIL"))
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {name}" + (f"\n        {detail}" if detail else ""))


def main() -> int:
    print("Acceptance criteria\n" + "=" * 60)

    specs = available_models()
    record(
        "B) selected ONNX models present and loadable",
        len(specs) == len(MODEL_SPECS),
        f"{len(specs)}/{len(MODEL_SPECS)} registered models available",
    )

    frames: dict[str, pd.DataFrame] = {}
    for key in MODEL_SPECS:
        path = IMAGE_RESULTS_DIR / f"image_results_{key}.csv"
        if path.exists():
            frames[key] = pd.read_csv(path)

    record(
        "A) 208 labelled images processed",
        all(len(df) == 208 for df in frames.values()) and len(frames) == len(MODEL_SPECS),
        ", ".join(f"{k}={len(v)} rows" for k, v in frames.items()),
    )

    primary = frames.get("rtmw_l_wholebody")
    if primary is None:
        record("B) landmarks produced", False, "primary model results missing")
        return 1

    detected = int((primary["landmark_detection_status"] == "OK").sum())
    record(
        "B) landmarks produced by the ONNX models",
        detected == len(primary),
        f"{detected}/{len(primary)} images produced landmarks",
    )

    computed = int(primary["mean_knee_flexion_deg"].notna().sum())
    mudra = int(primary["left_hand_mean_finger_straightness_deg"].notna().sum())
    record(
        "C) Bharatanatyam parameters calculated from the landmarks",
        computed > 0 and mudra > 0,
        f"knee flexion on {computed} images, finger straightness on {mudra} images",
    )

    recognised = int((primary["detected_posture"].fillna("") != "").sum())
    postures = sorted(set(primary["detected_posture"].dropna()) - {""})
    record(
        "D) system identifies a posture",
        recognised > 0 and len(postures) >= 6,
        f"{recognised}/{len(primary)} images, postures: {', '.join(postures)}",
    )

    statuses = primary["predicted_status"].value_counts().to_dict()
    record(
        "E) Correct/Incorrect decided by the fixed rules",
        {"CORRECT", "INCORRECT"} <= set(statuses),
        f"{statuses}",
    )

    errors = sorted(set(primary["predicted_error"].dropna()) - {""})
    record(
        "F) specific error reported",
        len(errors) >= 6,
        f"{len(errors)} distinct error codes: {', '.join(errors[:8])}...",
    )

    frames_csv = sorted(VIDEO_RESULTS_DIR.glob("*_frames.csv"))
    record(
        "G/H) a video was uploaded/processed frame by frame",
        bool(frames_csv),
        f"{len(frames_csv)} per-frame CSV(s): {[p.name for p in frames_csv]}",
    )

    if frames_csv:
        # Each video contains as many transitions as the dancer performed, so
        # the criterion is judged on the richest one rather than on whichever
        # file happens to sort first.
        best = ("", 0, 0, 0)
        for path in frames_csv:
            video = pd.read_csv(path)
            label = (
                video["smoothed_posture"].fillna("")
                + "|"
                + video["smoothed_status"].fillna("")
            )
            transitions = int(label.ne(label.shift()).sum() - 1)
            parameter_moves = int(video["mean_knee_flexion_deg"].notna().sum())
            if transitions > best[2]:
                best = (path.name, len(video), transitions, parameter_moves)

        name, frames, transitions, parameter_moves = best
        record(
            "I) posture/status/parameters change as the dancer changes",
            transitions >= 3 and parameter_moves > 0,
            f"{name}: {frames} frames, {transitions} posture/status transitions, "
            f"knee flexion tracked on {parameter_moves} frames",
        )
        annotated = sorted(VIDEO_RESULTS_DIR.glob("*_annotated.mp4"))
        record(
            "   annotated video written",
            bool(annotated),
            f"{[p.name for p in annotated]}",
        )

    summaries = [
        "model_processing_summary.csv",
        "landmark_availability_summary.csv",
        "parameter_calculation_summary.csv",
        "posture_correctness_summary.csv",
        "camera_view_summary.csv",
        "error_identification_summary.csv",
        "parameter_thresholds_used.csv",
    ]
    missing = [name for name in summaries if not (IMAGE_RESULTS_DIR / name).exists()]
    record("report summary tables generated", not missing, f"missing: {missing or 'none'}")

    failed = [name for name, status in CHECKS if status == "FAIL"]
    print("=" * 60)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
