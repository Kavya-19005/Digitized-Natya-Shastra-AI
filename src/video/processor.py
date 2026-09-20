"""Frame-by-frame video analysis with dynamic feedback.

For every frame: read -> ONNX pose -> landmarks -> validate -> parameters ->
posture -> rules -> smoothed feedback. An invalid frame keeps the last stable
state, is recorded as incomplete, and processing continues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import pandas as pd

from ..config import KEYPOINT_SCORE_THRESHOLD, SMOOTHING_WINDOW, VIDEO_RESULTS_DIR
from ..correctness.thresholds import UNKNOWN
from ..pipeline import AnalysisResult, PostureAnalyzer
from ..pose.registry import ModelSpec
from .overlay import draw_feedback_panel, draw_skeleton
from .smoothing import SmoothedState, TemporalSmoother

# Parameters recorded per frame and shown live in the UI.
TRACKED_PARAMETERS = (
    "mean_knee_flexion_deg",
    "left_knee_flexion_deg",
    "right_knee_flexion_deg",
    "knee_asymmetry_deg",
    "foot_separation_norm",
    "mean_foot_turnout_deg",
    "foot_line_deviation_deg",
    "trunk_inclination_deg",
    "mean_heel_lift_norm",
    "weight_shift_proxy",
    "ankle_crossing_signed_norm",
    "shin_crossing_distance_norm",
    "shin_crossing_angle_deg",
    "mean_elbow_shoulder_alignment_norm",
    "hip_height_norm",
)


@dataclass
class FrameOutcome:
    frame_index: int
    timestamp_s: float
    state: SmoothedState
    result: AnalysisResult
    frame: np.ndarray | None = None
    row: dict[str, object] = field(default_factory=dict)


@dataclass
class VideoSummary:
    frames_total: int = 0
    frames_analysed: int = 0
    frames_incomplete: int = 0
    posture_changes: int = 0
    elapsed_s: float = 0.0
    output_video: Path | None = None
    output_csv: Path | None = None
    segments: list[dict[str, object]] = field(default_factory=list)

    @property
    def fps_processed(self) -> float:
        return self.frames_total / self.elapsed_s if self.elapsed_s > 0 else 0.0


class VideoProcessor:
    def __init__(
        self,
        spec: ModelSpec,
        frame_stride: int = 1,
        smoothing_window: int = SMOOTHING_WINDOW,
        detector_interval: int = 5,
        max_width: int | None = 1280,
        draw: bool = True,
    ) -> None:
        self.spec = spec
        self.frame_stride = max(1, frame_stride)
        self.max_width = max_width
        self.draw = draw
        self.analyzer = PostureAnalyzer(
            spec, reuse_detection_for=max(0, detector_interval - 1)
        )
        self.smoother = TemporalSmoother(window=smoothing_window)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            return cv2.resize(
                frame,
                (self.max_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def iter_frames(self, video_path: str | Path) -> Iterator[FrameOutcome]:
        """Analyse a video lazily, yielding one outcome per processed frame."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.analyzer.reset()
        self.smoother.reset()
        index = -1

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if index % self.frame_stride:
                    continue

                frame = self._prepare(frame)
                try:
                    result = self.analyzer.analyse(frame)
                except Exception as exc:  # one bad frame must not stop the video
                    result = AnalysisResult(
                        landmarks=self.analyzer.estimator.pose_model
                        and _empty_landmarks(self.spec.key),
                        detection_status="ERROR",
                        message=str(exc),
                    )

                state = self._smooth(result)
                canvas = frame
                if self.draw:
                    canvas = frame.copy()
                    if result.detection_status == "OK":
                        draw_skeleton(canvas, result.landmarks, KEYPOINT_SCORE_THRESHOLD)
                    draw_feedback_panel(
                        canvas,
                        state,
                        model_name=self.spec.key,
                        frame_index=index,
                        extra_lines=self._extra_lines(result),
                    )

                yield FrameOutcome(
                    frame_index=index,
                    timestamp_s=index / fps,
                    state=state,
                    result=result,
                    frame=canvas,
                    row=self._row(index, index / fps, result, state),
                )
        finally:
            capture.release()

    def _smooth(self, result: AnalysisResult) -> SmoothedState:
        valid = result.detection_status == "OK" and result.complete
        verdict = result.verdict
        return self.smoother.update(
            posture=result.posture,
            status=verdict.status if verdict else UNKNOWN,
            error_code=verdict.error_code if verdict else "",
            error_text=verdict.error_text if verdict else "",
            primary_parameter=verdict.primary_parameter if verdict else "",
            primary_value=verdict.primary_value if verdict else float("nan"),
            primary_expected=verdict.primary_expected if verdict else "",
            parameters={
                name: float(result.parameters[name])
                for name in TRACKED_PARAMETERS
                if isinstance(result.parameters.get(name), (int, float))
            },
            valid=valid,
        )

    def _extra_lines(self, result: AnalysisResult) -> list[str]:
        lines = []
        view = result.parameters.get("view_class")
        facing = result.parameters.get("facing")
        if view:
            lines.append(f"view: {view} {facing or ''}".strip())
        if result.detection_status != "OK":
            lines.append(f"frame incomplete: {result.detection_status}")
        return lines

    def _row(
        self,
        index: int,
        timestamp: float,
        result: AnalysisResult,
        state: SmoothedState,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "frame_index": index,
            "timestamp_s": round(timestamp, 3),
            "model_name": self.spec.key,
            "landmark_detection_status": result.detection_status,
            "frame_complete": int(result.complete),
            "raw_posture": result.posture,
            "raw_status": result.status,
            "raw_error": result.verdict.error_code if result.verdict else "",
            "smoothed_posture": state.posture,
            "smoothed_status": state.status,
            "smoothed_error": state.error_code,
            "smoothed_error_text": state.error_text,
            "primary_parameter": state.primary_parameter,
            "primary_value": round(state.primary_value, 3)
            if np.isfinite(state.primary_value)
            else np.nan,
            "primary_expected": state.primary_expected,
            "view_class": result.parameters.get("view_class", ""),
            "facing": result.parameters.get("facing", ""),
            "detect_ms": round(result.detect_ms, 2),
            "pose_ms": round(result.pose_ms, 2),
        }
        for name in TRACKED_PARAMETERS:
            value = result.parameters.get(name, np.nan)
            row[name] = round(float(value), 4) if isinstance(value, (int, float)) else np.nan
            smoothed = state.parameters.get(name, np.nan)
            row[f"{name}_smoothed"] = (
                round(float(smoothed), 4) if np.isfinite(smoothed) else np.nan
            )
        return row

    def process(
        self,
        video_path: str | Path,
        output_name: str | None = None,
        write_video: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> VideoSummary:
        video_path = Path(video_path)
        VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = output_name or f"{video_path.stem}_{self.spec.key}"

        capture = cv2.VideoCapture(str(video_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()

        writer: cv2.VideoWriter | None = None
        rows: list[dict[str, object]] = []
        summary = VideoSummary()
        started = time.perf_counter()
        previous_label = ("", "")

        for outcome in self.iter_frames(video_path):
            summary.frames_total += 1
            if outcome.result.detection_status == "OK":
                summary.frames_analysed += 1
            else:
                summary.frames_incomplete += 1

            label = (outcome.state.posture, outcome.state.status)
            if label != previous_label and outcome.state.posture:
                if previous_label != ("", ""):
                    summary.posture_changes += 1
                summary.segments.append(
                    {
                        "frame_index": outcome.frame_index,
                        "timestamp_s": round(outcome.timestamp_s, 2),
                        "posture": outcome.state.posture,
                        "status": outcome.state.status,
                        "error": outcome.state.error_text,
                        "primary_parameter": outcome.state.primary_parameter,
                        "primary_value": round(outcome.state.primary_value, 2)
                        if np.isfinite(outcome.state.primary_value)
                        else None,
                    }
                )
                previous_label = label

            rows.append(outcome.row)

            if write_video and outcome.frame is not None:
                if writer is None:
                    h, w = outcome.frame.shape[:2]
                    target = VIDEO_RESULTS_DIR / f"{stem}_annotated.mp4"
                    writer = cv2.VideoWriter(
                        str(target),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(1.0, source_fps / self.frame_stride),
                        (w, h),
                    )
                    summary.output_video = target
                writer.write(outcome.frame)

            if progress and total:
                progress(summary.frames_total, max(1, total // self.frame_stride))

        if writer is not None:
            writer.release()

        summary.elapsed_s = time.perf_counter() - started
        summary.output_csv = VIDEO_RESULTS_DIR / f"{stem}_frames.csv"
        pd.DataFrame(rows).to_csv(summary.output_csv, index=False)

        if summary.segments:
            pd.DataFrame(summary.segments).to_csv(
                VIDEO_RESULTS_DIR / f"{stem}_segments.csv", index=False
            )
        return summary


def _empty_landmarks(model_key: str):
    from ..landmarks.common import Landmarks

    return Landmarks.empty(model_key)
