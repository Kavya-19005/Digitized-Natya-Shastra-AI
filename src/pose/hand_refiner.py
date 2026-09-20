"""Stage 3: re-measure the finger landmarks from a full-resolution hand crop.

Why this stage exists: the whole-body models carry 21 hand keypoints per hand,
but they infer them from a 192x256 (or 288x384) crop of the WHOLE body. At the
college camera distance the dancer's hand covers only a handful of pixels in
that input, so the finger geometry collapses and the mudra parameters become
meaningless.

The whole-body model is still used to LOCATE each hand; a dedicated 21-keypoint
hand ONNX model then re-estimates the fingers from the original full-resolution
pixels. The refined points replace the hand block of the common representation
and `hand_refined_*` flags record that this happened.
"""

from __future__ import annotations

import numpy as np

from ..config import MODELS_DIR
from ..landmarks import common as C
from ..landmarks.common import Landmarks
from .simcc_pose import SimCCPoseModel

HAND_MODEL_FILE = "rtmpose-m_hand5_256x256.onnx"
HAND_INPUT_SIZE = (256, 256)

# A hand spans roughly this fraction of the forearm length; used as a fallback
# box size when the coarse hand keypoints are too clustered to trust.
HAND_TO_FOREARM_RATIO = 1.05
MIN_BOX_PX = 48.0


class HandRefiner:
    def __init__(self, score_threshold: float = 0.30) -> None:
        self.model = SimCCPoseModel(MODELS_DIR / HAND_MODEL_FILE, HAND_INPUT_SIZE)
        self.score_threshold = score_threshold

    @property
    def available(self) -> bool:
        return True

    def refine(self, image: np.ndarray, lm: Landmarks) -> Landmarks:
        """Overwrite both hand blocks with full-resolution estimates."""
        for side in ("left", "right"):
            box = self._hand_box(image, lm, side)
            if box is None:
                lm.extras[f"hand_refined_{side}"] = np.array([0.0], dtype=np.float32)
                continue
            try:
                keypoints, scores = self.model(image, np.asarray(box, dtype=np.float32))
            except Exception:
                lm.extras[f"hand_refined_{side}"] = np.array([0.0], dtype=np.float32)
                continue

            # Every refined point is written with its own score. Closed mudras
            # such as Mushti legitimately hide the fingertips, so the scores run
            # low; the hand-specific confidence floor in
            # `landmarks.common.available` decides what is usable rather than
            # discarding the geometry here.
            base = C.LEFT_HAND_START if side == "left" else C.RIGHT_HAND_START
            for offset in range(min(21, keypoints.shape[0])):
                idx = base + offset
                lm.xy[idx] = keypoints[offset]
                lm.score[idx] = float(scores[offset])
            lm.extras[f"hand_refined_{side}"] = np.array([1.0], dtype=np.float32)
            lm.extras[f"hand_box_{side}"] = np.asarray(box, dtype=np.float32)
        return lm

    def _hand_box(
        self, image: np.ndarray, lm: Landmarks, side: str
    ) -> tuple[float, float, float, float] | None:
        rng = C.LEFT_HAND_RANGE if side == "left" else C.RIGHT_HAND_RANGE
        points = np.array([lm.xy[i] for i in rng], dtype=float)
        points = points[np.isfinite(points).all(axis=1)]

        wrist_idx = C.hand_index(side, C.H_WRIST)
        elbow_idx = C.LEFT_ELBOW if side == "left" else C.RIGHT_ELBOW
        wrist = lm.xy[wrist_idx].astype(float)
        elbow = lm.xy[elbow_idx].astype(float)

        if points.shape[0] >= 5:
            center = points.mean(axis=0)
            extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
        elif np.all(np.isfinite(wrist)):
            center = wrist
            extent = 0.0
        else:
            return None

        forearm = (
            float(np.linalg.norm(elbow - wrist))
            if np.all(np.isfinite(elbow)) and np.all(np.isfinite(wrist))
            else 0.0
        )
        side_px = max(extent * 1.35, forearm * HAND_TO_FOREARM_RATIO, MIN_BOX_PX)
        half = side_px / 2.0

        h, w = image.shape[:2]
        x0 = max(center[0] - half, 0.0)
        y0 = max(center[1] - half, 0.0)
        x1 = min(center[0] + half, float(w - 1))
        y1 = min(center[1] + half, float(h - 1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1, y1)


def was_refined(lm: Landmarks, side: str) -> bool:
    flag = lm.extras.get(f"hand_refined_{side}")
    return bool(flag is not None and float(flag[0]) >= 1.0)
