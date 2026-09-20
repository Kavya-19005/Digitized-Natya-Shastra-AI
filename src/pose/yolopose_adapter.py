"""YOLO Pose adapter (Ultralytics YOLOv8-pose) -> common 67 landmarks.

This is a pose model (COCO-17 body keypoints), not object-detection YOLO.
Feet and hands are not in the layout and stay NaN.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import KEYPOINT_SCORE_THRESHOLD, MODELS_DIR
from ..landmarks.adapters import COCO17, to_common
from .estimator import PoseResult

APPROACH_KEY = "yolopose"
DISPLAY_NAME = "YOLO Pose"
_WEIGHTS = MODELS_DIR / "yolov8n-pose.pt"
_CONF = 0.25


def probe() -> tuple[bool, str]:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False, "package ultralytics is not installed"
    return True, "ultralytics is importable; YOLOv8n-pose weights load on first use"


class YOLOPoseAdapter:
    key = APPROACH_KEY
    display_name = DISPLAY_NAME
    layout = COCO17

    def __init__(self, score_threshold: float = KEYPOINT_SCORE_THRESHOLD) -> None:
        ok, reason = probe()
        if not ok:
            raise RuntimeError(f"YOLO Pose is not runnable: {reason}")
        from ultralytics import YOLO

        self.score_threshold = score_threshold
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weights = str(_WEIGHTS) if _WEIGHTS.exists() else "yolov8n-pose.pt"
        self._model = YOLO(weights)

    def reset(self) -> None:
        return

    def __call__(self, image: np.ndarray) -> PoseResult:
        if image is None or image.size == 0:
            return PoseResult(_empty(self.key), "ERROR", "empty image buffer")

        start = time.perf_counter()
        try:
            results = self._model.predict(
                image, verbose=False, conf=_CONF, device="cpu"
            )
        except Exception as exc:
            return PoseResult(
                _empty(self.key),
                "ERROR",
                f"YOLO Pose inference failed: {exc}",
            )
        pose_ms = (time.perf_counter() - start) * 1000.0

        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return PoseResult(
                _empty(self.key),
                "NO_PERSON",
                "YOLO Pose did not detect a person",
                detect_ms=0.0,
                pose_ms=pose_ms,
            )

        result = results[0]
        person = _largest_person_index(result)
        xy = np.asarray(result.keypoints.xy[person].cpu(), dtype=np.float32)
        conf = np.asarray(result.keypoints.conf[person].cpu(), dtype=np.float32)
        if xy.shape[0] < 17:
            return PoseResult(
                _empty(self.key),
                "ERROR",
                f"YOLO Pose returned {xy.shape[0]} keypoints, expected 17",
                detect_ms=0.0,
                pose_ms=pose_ms,
            )

        landmarks = to_common(
            xy[:17],
            conf[:17],
            self.layout,
            model_name=self.key,
            score_threshold=self.score_threshold,
        )
        return PoseResult(landmarks, "OK", "", detect_ms=0.0, pose_ms=pose_ms)


def _largest_person_index(result) -> int:
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    areas = []
    xyxy = result.boxes.xyxy.cpu().numpy()
    for box in xyxy:
        areas.append(float((box[2] - box[0]) * (box[3] - box[1])))
    return int(np.argmax(areas)) if areas else 0


def _empty(model_name: str):
    from ..landmarks.common import Landmarks

    empty = Landmarks.empty(model_name)
    empty.supported = COCO17.supported_mask
    return empty
