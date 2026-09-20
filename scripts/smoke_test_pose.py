"""Quick sanity check: run every registered ONNX model on one dataset image."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CORE_CAMERAS, DATASET_ROOT, KEYPOINT_SCORE_THRESHOLD
from src.landmarks import common as C
from src.pose.estimator import PoseEstimator
from src.pose.registry import MODEL_SPECS, missing_models


def main() -> int:
    missing = missing_models()
    if missing:
        print("MISSING MODEL FILES:")
        for item in missing:
            print("  -", item)
        return 1

    cam_dir = DATASET_ROOT / CORE_CAMERAS[0]
    image_path = sorted(cam_dir.glob("*.JPG"))[0]
    image = cv2.imread(str(image_path))
    print(f"image: {image_path.name}  shape={image.shape}")

    for key, spec in MODEL_SPECS.items():
        estimator = PoseEstimator(spec)
        result = estimator(image)
        lm = result.landmarks
        mask = lm.available(KEYPOINT_SCORE_THRESHOLD)
        print(f"\n== {key} ({spec.layout.name})")
        print(f"   status={result.detection_status} bbox={lm.bbox}")
        print(f"   detect={result.detect_ms:.0f}ms pose={result.pose_ms:.0f}ms")
        print(f"   available landmarks: {int(mask.sum())}/{C.NUM_COMMON}")
        print(f"   left hand usable : {lm.hand_available('left', KEYPOINT_SCORE_THRESHOLD)}")
        print(f"   right hand usable: {lm.hand_available('right', KEYPOINT_SCORE_THRESHOLD)}")
        for idx in (C.LEFT_SHOULDER, C.LEFT_HIP, C.LEFT_KNEE, C.LEFT_ANKLE, C.LEFT_HEEL):
            xy = lm.xy[idx]
            print(f"   {C.COMMON_NAMES[idx]:>16}: ({xy[0]:8.1f}, {xy[1]:8.1f}) s={lm.score[idx]:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
