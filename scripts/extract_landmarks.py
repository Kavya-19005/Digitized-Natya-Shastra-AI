"""Stage 1 of the image pipeline: run the ONNX models and cache the landmarks.

    python scripts/extract_landmarks.py
    python scripts/extract_landmarks.py --models rtmw_l_wholebody

Writes outputs/image_results/landmarks_67_<model>.csv (one row per image).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_output_dirs
from src.dataset import build_manifest, write_manifest
from src.evaluation.landmark_cache import row_from_landmarks, write_cache
from src.pose.estimator import PoseEstimator
from src.pose.hand_refiner import was_refined
from src.pose.registry import MODEL_SPECS, get_spec, missing_models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    args = parser.parse_args()

    missing = missing_models()
    if missing:
        print("MISSING MODEL FILES (run scripts/download_models.py):")
        for item in missing:
            print("  -", item)
        return 1

    ensure_output_dirs()
    print("manifest:", write_manifest())
    items = build_manifest()
    print(f"images: {len(items)}")

    for key in args.models:
        spec = get_spec(key)
        estimator = PoseEstimator(spec)
        print(f"\n== {spec.display_name}")
        rows = []
        started = time.perf_counter()

        for i, item in enumerate(items, start=1):
            image = cv2.imread(str(item.path))
            label = item.label
            meta = {
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
            if image is None:
                rows.append({**meta, "landmark_detection_status": "READ_ERROR"})
                continue

            result = estimator(image)
            meta.update(
                {
                    "landmark_detection_status": result.detection_status,
                    "detect_ms": round(result.detect_ms, 2),
                    "pose_ms": round(result.pose_ms, 2),
                    "hand_refined_left": int(was_refined(result.landmarks, "left")),
                    "hand_refined_right": int(was_refined(result.landmarks, "right")),
                }
            )
            rows.append({**meta, **row_from_landmarks(result.landmarks)})

            if i % 40 == 0:
                print(f"    {i}/{len(items)}  ({time.perf_counter() - started:.0f}s)")

        path = write_cache(pd.DataFrame(rows), spec.key)
        print(f"   -> {path}  ({time.perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
