"""End-to-end pose estimation: image -> common 67-landmark representation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import DETECTION_SCORE_THRESHOLD, KEYPOINT_SCORE_THRESHOLD
from ..landmarks.adapters import to_common
from ..landmarks.common import Landmarks
from .detector import PersonDetector
from .hand_refiner import HandRefiner
from .registry import ModelSpec
from .simcc_pose import SimCCPoseModel


@dataclass
class PoseResult:
    landmarks: Landmarks
    detection_status: str  # "OK" | "NO_PERSON" | "ERROR"
    message: str = ""
    detect_ms: float = 0.0
    pose_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.detection_status == "OK"


class PoseEstimator:
    """Top-down estimator: YOLOX person box -> SimCC keypoints -> 67 landmarks.

    `reuse_detection_for` lets video processing skip the (expensive) detector on
    most frames by reusing the previous, slightly expanded person box.
    """

    def __init__(
        self,
        spec: ModelSpec,
        score_threshold: float = KEYPOINT_SCORE_THRESHOLD,
        reuse_detection_for: int = 0,
        refine_hands: bool = True,
    ) -> None:
        self.spec = spec
        self.score_threshold = score_threshold
        self.reuse_detection_for = reuse_detection_for

        self.detector = PersonDetector(
            spec.detector_path,
            input_size=spec.detector_input_size,
            score_threshold=DETECTION_SCORE_THRESHOLD,
        )
        self.pose_model = SimCCPoseModel(spec.pose_path, spec.pose_input_size)

        # Hand refinement only makes sense for layouts that locate the hands.
        self.hand_refiner: HandRefiner | None = None
        if refine_hands and spec.layout.has_hands:
            try:
                self.hand_refiner = HandRefiner(score_threshold)
            except FileNotFoundError:
                self.hand_refiner = None

        self._cached_bbox: np.ndarray | None = None
        self._cache_age = 0

    def reset(self) -> None:
        self._cached_bbox = None
        self._cache_age = 0

    def _person_bbox(self, image: np.ndarray) -> tuple[np.ndarray | None, float]:
        if (
            self._cached_bbox is not None
            and self._cache_age < self.reuse_detection_for
        ):
            self._cache_age += 1
            return self._cached_bbox, 0.0

        start = time.perf_counter()
        try:
            boxes = self.detector(image)
        except Exception:
            return None, (time.perf_counter() - start) * 1000.0
        detect_ms = (time.perf_counter() - start) * 1000.0

        if boxes.shape[0] == 0:
            self._cached_bbox = None
            self._cache_age = 0
            return None, detect_ms

        # the dancer is the largest detected person
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        bbox = boxes[int(np.argmax(areas)), :4]
        self._cached_bbox = self._expand(bbox, image.shape[:2])
        self._cache_age = 0
        return self._cached_bbox, detect_ms

    @staticmethod
    def _expand(bbox: np.ndarray, hw: tuple[int, int], margin: float = 0.12):
        h, w = hw
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return np.array(
            [
                max(bbox[0] - bw * margin, 0),
                max(bbox[1] - bh * margin, 0),
                min(bbox[2] + bw * margin, w - 1),
                min(bbox[3] + bh * margin, h - 1),
            ],
            dtype=np.float32,
        )

    def __call__(self, image: np.ndarray) -> PoseResult:
        if image is None or image.size == 0:
            return PoseResult(
                Landmarks.empty(self.spec.key), "ERROR", "empty image buffer"
            )

        bbox, detect_ms = self._person_bbox(image)
        if bbox is None:
            empty = Landmarks.empty(self.spec.key)
            empty.supported = self.spec.layout.supported_mask
            return PoseResult(empty, "NO_PERSON", "no person detected", detect_ms, 0.0)

        start = time.perf_counter()
        try:
            keypoints, scores = self.pose_model(image, bbox)
        except Exception as exc:
            empty = Landmarks.empty(self.spec.key)
            empty.supported = self.spec.layout.supported_mask
            return PoseResult(
                empty, "ERROR", f"pose inference failed: {exc}", detect_ms, 0.0
            )
        pose_ms = (time.perf_counter() - start) * 1000.0

        landmarks = to_common(
            keypoints,
            scores,
            self.spec.layout,
            model_name=self.spec.key,
            score_threshold=self.score_threshold,
            bbox=tuple(float(v) for v in bbox),
        )

        if self.hand_refiner is not None:
            start = time.perf_counter()
            try:
                landmarks = self.hand_refiner.refine(image, landmarks)
            except Exception:
                pass  # coarse hand points remain; the frame is still usable
            pose_ms += (time.perf_counter() - start) * 1000.0

        return PoseResult(landmarks, "OK", "", detect_ms, pose_ms)
