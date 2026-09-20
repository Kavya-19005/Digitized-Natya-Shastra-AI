"""MediaPipe Holistic adapter -> common 67 landmarks.

Uses the already-installed `mediapipe.solutions.holistic` pipeline so body,
feet and the 21-point hands are real MediaPipe outputs. Pose-list wrist/finger
stubs (indices 15-22) are not copied; the dedicated hand blocks cover them.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import KEYPOINT_SCORE_THRESHOLD
from ..landmarks.adapters import MEDIAPIPE_HOLISTIC, to_common
from .estimator import PoseResult

APPROACH_KEY = "mediapipe"
DISPLAY_NAME = "MediaPipe"


def probe() -> tuple[bool, str]:
    try:
        import mediapipe as mp
    except ImportError:
        return False, "package mediapipe is not installed"
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "holistic"):
        return False, "mediapipe.solutions.holistic is unavailable"
    return True, "mediapipe.solutions.holistic is importable"


class MediaPipeAdapter:
    key = APPROACH_KEY
    display_name = DISPLAY_NAME
    layout = MEDIAPIPE_HOLISTIC

    def __init__(self, score_threshold: float = KEYPOINT_SCORE_THRESHOLD) -> None:
        ok, reason = probe()
        if not ok:
            raise RuntimeError(f"MediaPipe is not runnable: {reason}")
        import mediapipe as mp

        self.score_threshold = score_threshold
        self._mp = mp
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset(self) -> None:
        return

    def close(self) -> None:
        try:
            self._holistic.close()
        except Exception:
            pass

    def __call__(self, image: np.ndarray) -> PoseResult:
        if image is None or image.size == 0:
            return PoseResult(
                _empty(self.key), "ERROR", "empty image buffer"
            )
        rgb = image[:, :, ::-1]
        start = time.perf_counter()
        try:
            result = self._holistic.process(rgb)
        except Exception as exc:
            return PoseResult(
                _empty(self.key),
                "ERROR",
                f"MediaPipe inference failed: {exc}",
            )
        pose_ms = (time.perf_counter() - start) * 1000.0

        if result.pose_landmarks is None:
            return PoseResult(
                _empty(self.key),
                "NO_PERSON",
                "MediaPipe did not detect a pose",
                detect_ms=0.0,
                pose_ms=pose_ms,
            )

        h, w = image.shape[:2]
        packed = np.full((75, 2), np.nan, dtype=np.float32)
        scores = np.zeros(75, dtype=np.float32)
        _copy_landmarks(result.pose_landmarks.landmark, packed, scores, 0, w, h)
        if result.left_hand_landmarks is not None:
            _copy_landmarks(
                result.left_hand_landmarks.landmark, packed, scores, 33, w, h
            )
        if result.right_hand_landmarks is not None:
            _copy_landmarks(
                result.right_hand_landmarks.landmark, packed, scores, 54, w, h
            )

        landmarks = to_common(
            packed,
            scores,
            self.layout,
            model_name=self.key,
            score_threshold=self.score_threshold,
        )
        return PoseResult(landmarks, "OK", "", detect_ms=0.0, pose_ms=pose_ms)


def _copy_landmarks(lms, packed, scores, offset: int, width: int, height: int) -> None:
    for i, lm in enumerate(lms):
        packed[offset + i, 0] = lm.x * width
        packed[offset + i, 1] = lm.y * height
        visibility = float(getattr(lm, "visibility", 0.0) or 0.0)
        presence = float(getattr(lm, "presence", 0.0) or 0.0)
        score = max(visibility, presence)
        # Holistic omits visibility on many hand points; a returned landmark is
        # a real detection, so do not drop it at the body score threshold.
        if score <= 0:
            score = 0.90
        scores[offset + i] = score


def _empty(model_name: str):
    from ..landmarks.common import Landmarks

    empty = Landmarks.empty(model_name)
    empty.supported = MEDIAPIPE_HOLISTIC.supported_mask
    return empty
