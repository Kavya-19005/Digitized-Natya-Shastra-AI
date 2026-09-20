"""Import and initialise the five pose-estimation approach adapters.

    python scripts/smoke_test_approaches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CORE_CAMERAS, DATASET_ROOT, KEYPOINT_SCORE_THRESHOLD
from src.landmarks import common as C
from src.pose.approaches import APPROACHES


def main() -> int:
    cam_dir = DATASET_ROOT / CORE_CAMERAS[0]
    image_path = sorted(cam_dir.glob("*.JPG"))[0]
    image = cv2.imread(str(image_path))
    print(f"image: {image_path.name}  shape={image.shape}\n")

    failures = 0
    for approach in APPROACHES:
        print(f"== {approach.display_name} ({approach.key})")
        ok, note = approach.probe()
        print(f"   probe: {note}")
        if not ok:
            print("   SKIP (not runnable)")
            failures += 1
            continue
        try:
            backend = approach.factory()
        except Exception as exc:
            print(f"   INIT FAIL: {exc}")
            failures += 1
            continue
        try:
            result = backend(image)
        except Exception as exc:
            print(f"   INFER FAIL: {exc}")
            failures += 1
            continue
        lm = result.landmarks
        mask = lm.available(KEYPOINT_SCORE_THRESHOLD)
        print(f"   status={result.detection_status} pose_ms={result.pose_ms:.0f}")
        print(
            f"   available {int(mask.sum())}/{C.NUM_COMMON} "
            f"(supported {int(lm.supported.sum())})"
        )
        print(
            f"   hands L={lm.hand_available('left', KEYPOINT_SCORE_THRESHOLD)} "
            f"R={lm.hand_available('right', KEYPOINT_SCORE_THRESHOLD)}"
        )
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    print(f"\n{len(APPROACHES) - failures}/{len(APPROACHES)} approaches initialised")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
