"""OpenPose BODY_25 adapter via OpenCV DNN (Caffe).

The full CMU OpenPose C++ runtime is not installed. This is the practical
locally runnable BODY_25 model: the official prototxt + caffemodel, executed
with OpenCV DNN which is already a project dependency. Hands are not produced
by BODY_25 and stay NaN.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..config import KEYPOINT_SCORE_THRESHOLD, PROJECT_ROOT
from ..landmarks.adapters import OPENPOSE_BODY25, to_common
from .estimator import PoseResult
from .fetch import ensure_file

APPROACH_KEY = "openpose"
DISPLAY_NAME = "OpenPose"

_WEIGHTS_DIR = PROJECT_ROOT / "models" / "openpose"
_PROTO = _WEIGHTS_DIR / "pose_deploy.prototxt"
_CAFFE = _WEIGHTS_DIR / "pose_iter_584000.caffemodel"
_INPUT_SIZE = 368
_PEAK_THRESHOLD = 0.10

_PROTO_URLS = [
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt",
    "https://huggingface.co/camenduru/openpose/raw/main/models/pose/body_25/pose_deploy.prototxt",
]
_CAFFE_URLS = [
    "https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/pose_iter_584000.caffemodel?download=true",
    "https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/pose_iter_584000.caffemodel",
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
]


def probe() -> tuple[bool, str]:
    if _PROTO.exists() and _CAFFE.exists() and _CAFFE.stat().st_size > 1_000_000:
        return True, f"BODY_25 Caffe weights present ({_CAFFE.name})"
    return True, (
        "OpenCV DNN BODY_25 adapter is implemented; weights will be fetched "
        "on first construction if the network is reachable"
    )


def _ensure_weights() -> tuple[Path, Path]:
    proto = ensure_file(_PROTO_URLS, _PROTO, min_bytes=1_000)
    caffe = ensure_file(_CAFFE_URLS, _CAFFE, min_bytes=1_000_000)
    return proto, caffe


class OpenPoseAdapter:
    key = APPROACH_KEY
    display_name = DISPLAY_NAME
    layout = OPENPOSE_BODY25

    def __init__(self, score_threshold: float = KEYPOINT_SCORE_THRESHOLD) -> None:
        self.score_threshold = score_threshold
        try:
            proto, caffe = _ensure_weights()
        except Exception as exc:
            raise RuntimeError(f"OpenPose weights could not be obtained: {exc}") from exc
        try:
            self._net = cv2.dnn.readNetFromCaffe(str(proto), str(caffe))
        except Exception as exc:
            raise RuntimeError(f"OpenCV DNN could not load BODY_25: {exc}") from exc
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def reset(self) -> None:
        return

    def __call__(self, image: np.ndarray) -> PoseResult:
        if image is None or image.size == 0:
            return PoseResult(_empty(self.key), "ERROR", "empty image buffer")

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / 255.0,
            (_INPUT_SIZE, _INPUT_SIZE),
            (0, 0, 0),
            swapRB=False,
            crop=False,
        )
        start = time.perf_counter()
        try:
            self._net.setInput(blob)
            output = self._net.forward()
        except Exception as exc:
            return PoseResult(
                _empty(self.key),
                "ERROR",
                f"OpenPose inference failed: {exc}",
            )
        pose_ms = (time.perf_counter() - start) * 1000.0

        # BODY_25: channels 0-24 are parts, 25 is background, the rest are PAFs.
        keypoints = np.full((25, 2), np.nan, dtype=np.float32)
        scores = np.zeros(25, dtype=np.float32)
        heat_h, heat_w = output.shape[2], output.shape[3]
        for part in range(25):
            heat = output[0, part, :, :]
            _, conf, _, peak = cv2.minMaxLoc(heat)
            if conf < _PEAK_THRESHOLD:
                continue
            keypoints[part, 0] = peak[0] * (w / float(heat_w))
            keypoints[part, 1] = peak[1] * (h / float(heat_h))
            scores[part] = float(conf)

        if not np.isfinite(keypoints).any():
            return PoseResult(
                _empty(self.key),
                "NO_PERSON",
                "OpenPose found no BODY_25 peaks",
                detect_ms=0.0,
                pose_ms=pose_ms,
            )

        landmarks = to_common(
            keypoints,
            scores,
            self.layout,
            model_name=self.key,
            score_threshold=self.score_threshold,
        )
        return PoseResult(landmarks, "OK", "", detect_ms=0.0, pose_ms=pose_ms)


def _empty(model_name: str):
    from ..landmarks.common import Landmarks

    empty = Landmarks.empty(model_name)
    empty.supported = OPENPOSE_BODY25.supported_mask
    return empty
