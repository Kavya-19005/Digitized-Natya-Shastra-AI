"""The five pose-estimation approaches the project is required to compare.

Each approach is an estimator adapter that emits the common 67-landmark
representation. Downstream parameter extraction, recognition and correctness
are shared and are not reimplemented per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .mediapipe_adapter import DISPLAY_NAME as MP_NAME
from .mediapipe_adapter import MediaPipeAdapter
from .mediapipe_adapter import probe as probe_mediapipe
from .openpose_adapter import DISPLAY_NAME as OP_NAME
from .openpose_adapter import OpenPoseAdapter
from .openpose_adapter import probe as probe_openpose
from .rtmpose_adapter import DISPLAY_NAME as RTMPOSE_NAME
from .rtmpose_adapter import RTMPoseAdapter
from .rtmpose_adapter import probe as probe_rtmpose
from .rtmw_adapter import DISPLAY_NAME as RTMW_NAME
from .rtmw_adapter import RTMWAdapter
from .rtmw_adapter import probe as probe_rtmw
from .yolopose_adapter import DISPLAY_NAME as YOLO_NAME
from .yolopose_adapter import YOLOPoseAdapter
from .yolopose_adapter import probe as probe_yolo


@dataclass(frozen=True)
class Approach:
    key: str
    display_name: str
    factory: Callable
    probe: Callable[[], tuple[bool, str]]


# Order matches the project requirement: MediaPipe, OpenPose, YOLO Pose, RTMPose, RTMW.
APPROACHES: tuple[Approach, ...] = (
    Approach("mediapipe", MP_NAME, MediaPipeAdapter, probe_mediapipe),
    Approach("openpose", OP_NAME, OpenPoseAdapter, probe_openpose),
    Approach("yolopose", YOLO_NAME, YOLOPoseAdapter, probe_yolo),
    Approach("rtmpose", RTMPOSE_NAME, RTMPoseAdapter, probe_rtmpose),
    Approach("rtmw", RTMW_NAME, RTMWAdapter, probe_rtmw),
)


def get_approach(key: str) -> Approach:
    for approach in APPROACHES:
        if approach.key == key:
            return approach
    raise KeyError(f"unknown approach {key!r}")
