"""Crop the hand region (located via the whole-body model) at full resolution.

Used to judge whether the hands carry enough pixels for mudra analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_ROOT, KEYPOINT_SCORE_THRESHOLD, LOGS_DIR
from src.dataset import camera_images
from src.labels import VARIATION_BY_ID
from src.landmarks import common as C
from src.pose.estimator import PoseEstimator
from src.pose.registry import get_spec

TILE = 260


def hand_box(lm, side: str) -> tuple[int, int, int, int] | None:
    rng = C.LEFT_HAND_RANGE if side == "left" else C.RIGHT_HAND_RANGE
    pts = np.array([lm.xy[i] for i in rng])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 5:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * 1.1
    half = max(half, 40.0)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def main() -> int:
    camera = "Cam 5"
    ids = [int(v) for v in sys.argv[1:]] or [19, 20, 21, 22, 23, 24, 25, 26]
    estimator = PoseEstimator(get_spec("rtmw_l_wholebody"))
    files = camera_images(camera)

    tiles = []
    for vid in ids:
        image = cv2.imread(str(files[(vid - 1) * 2]))
        result = estimator(image)
        lm = result.landmarks
        for side in ("left", "right"):
            box = hand_box(lm, side)
            tile = np.zeros((TILE, TILE, 3), dtype=np.uint8)
            if box is not None:
                x0, y0, x1, y1 = box
                x0, y0 = max(x0, 0), max(y0, 0)
                x1 = min(x1, image.shape[1])
                y1 = min(y1, image.shape[0])
                crop = image[y0:y1, x0:x1]
                if crop.size:
                    scale = TILE / max(crop.shape[:2])
                    resized = cv2.resize(
                        crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale))
                    )
                    tile[: resized.shape[0], : resized.shape[1]] = resized
                    # overlay the model's hand keypoints
                    rng = C.LEFT_HAND_RANGE if side == "left" else C.RIGHT_HAND_RANGE
                    for i in rng:
                        p = lm.xy[i]
                        if np.all(np.isfinite(p)):
                            px = int((p[0] - x0) * scale)
                            py = int((p[1] - y0) * scale)
                            cv2.circle(tile, (px, py), 2, (0, 255, 255), -1)
                    px_size = x1 - x0
                    cv2.putText(
                        tile, f"{px_size}px", (4, TILE - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                    )
            label = VARIATION_BY_ID[vid]
            cv2.putText(
                tile, f"{vid} {label.posture[:11]} {side[0].upper()}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1
            )
            tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * TILE, cols * TILE, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / "preview_hands.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
