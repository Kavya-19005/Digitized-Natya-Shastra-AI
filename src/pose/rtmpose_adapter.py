"""RTMPose approach adapter. Uses the existing Halpe26 RTMPose-m ONNX model."""

from __future__ import annotations

from ..config import KEYPOINT_SCORE_THRESHOLD
from .estimator import PoseEstimator, PoseResult
from .registry import get_spec

APPROACH_KEY = "rtmpose"
DISPLAY_NAME = "RTMPose"
_SPEC_KEY = "rtmpose_m_halpe26"


def probe() -> tuple[bool, str]:
    spec = get_spec(_SPEC_KEY)
    if spec.available:
        return True, (
            f"ONNX weights present: {spec.pose_file} "
            "(body+feet; no hand landmarks by design)"
        )
    missing = []
    if not spec.pose_path.exists():
        missing.append(spec.pose_file)
    if not spec.detector_path.exists():
        missing.append(spec.detector_file)
    return False, "missing " + ", ".join(missing)


class RTMPoseAdapter:
    """Existing RTMPose-m Halpe26 estimator, labelled as the RTMPose approach."""

    key = APPROACH_KEY
    display_name = DISPLAY_NAME

    def __init__(self, score_threshold: float = KEYPOINT_SCORE_THRESHOLD) -> None:
        ok, reason = probe()
        if not ok:
            raise RuntimeError(f"RTMPose is not runnable: {reason}")
        self.spec = get_spec(_SPEC_KEY)
        self.layout = self.spec.layout
        self._inner = PoseEstimator(self.spec, score_threshold=score_threshold)

    def reset(self) -> None:
        self._inner.reset()

    def __call__(self, image) -> PoseResult:
        result = self._inner(image)
        result.landmarks.model_name = self.key
        return result
