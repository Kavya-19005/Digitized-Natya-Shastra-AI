"""The single analysis entry point used by the image, video and UI layers.

    frame -> ONNX pose -> common 67 landmarks -> parameters
          -> posture recognition -> fixed correctness rules -> feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import KEYPOINT_SCORE_THRESHOLD
from .correctness.rules import Verdict, evaluate
from .correctness.thresholds import UNKNOWN
from .features import compute_parameters
from .landmarks.common import Landmarks
from .pose.estimator import PoseEstimator, PoseResult
from .pose.registry import ModelSpec
from .recognition.posture import RecognitionResult, recognise


@dataclass
class AnalysisResult:
    landmarks: Landmarks
    detection_status: str
    parameters: dict[str, float | str] = field(default_factory=dict)
    recognition: RecognitionResult | None = None
    verdict: Verdict | None = None
    detect_ms: float = 0.0
    pose_ms: float = 0.0
    message: str = ""

    @property
    def posture(self) -> str:
        return self.recognition.posture if self.recognition else ""

    @property
    def status(self) -> str:
        return self.verdict.status if self.verdict else UNKNOWN

    @property
    def error_text(self) -> str:
        return self.verdict.error_text if self.verdict else ""

    @property
    def complete(self) -> bool:
        """True when the frame has a recognition outcome the video layer can show.

        TRANSITION is a valid identity ("not yet a stable posture") and must be
        published, even though correctness rules do not apply to it. Dropping
        those frames made the smoother hold the previous label, which is how a
        deepening Aramandi could appear to jump into a false Muzhumandi.
        """
        if self.recognition is not None and self.recognition.is_transition:
            return True
        return bool(self.posture) and self.status != UNKNOWN


class PostureAnalyzer:
    """Wraps a pose estimator with the parameter, recognition and rule layers."""

    def __init__(
        self,
        spec: ModelSpec | None = None,
        score_threshold: float = KEYPOINT_SCORE_THRESHOLD,
        reuse_detection_for: int = 0,
        backend=None,
    ) -> None:
        self.score_threshold = score_threshold
        if backend is not None:
            # Comparison backends share this wrapper; the video path still
            # constructs PoseEstimator from a ModelSpec and is unchanged.
            self.spec = spec
            self.estimator = backend
            return
        if spec is None:
            raise TypeError("PostureAnalyzer requires a ModelSpec or a pose backend")
        self.spec = spec
        self.estimator = PoseEstimator(
            spec,
            score_threshold=score_threshold,
            reuse_detection_for=reuse_detection_for,
        )

    def reset(self) -> None:
        self.estimator.reset()

    def analyse(self, image: np.ndarray) -> AnalysisResult:
        pose: PoseResult = self.estimator(image)
        if not pose.ok:
            return AnalysisResult(
                landmarks=pose.landmarks,
                detection_status=pose.detection_status,
                detect_ms=pose.detect_ms,
                pose_ms=pose.pose_ms,
                message=pose.message,
            )

        try:
            params = compute_parameters(pose.landmarks, self.score_threshold)
        except Exception as exc:  # a bad frame must not kill a whole video
            return AnalysisResult(
                landmarks=pose.landmarks,
                detection_status="PARAMETER_ERROR",
                detect_ms=pose.detect_ms,
                pose_ms=pose.pose_ms,
                message=f"parameter computation failed: {exc}",
            )

        recognition = recognise(params)
        verdict = evaluate(recognition.posture, params, side=recognition.side)

        return AnalysisResult(
            landmarks=pose.landmarks,
            detection_status=pose.detection_status,
            parameters=params,
            recognition=recognition,
            verdict=verdict,
            detect_ms=pose.detect_ms,
            pose_ms=pose.pose_ms,
            message=recognition.message,
        )
