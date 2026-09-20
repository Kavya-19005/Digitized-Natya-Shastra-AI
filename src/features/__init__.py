"""Bharatanatyam parameter extraction from the common landmark representation."""

from __future__ import annotations

from ..config import KEYPOINT_SCORE_THRESHOLD
from ..landmarks import common as C
from ..landmarks.common import Landmarks
from .body_params import compute_body_parameters, crossing_side
from .hand_params import compute_mudra_parameters, hand_param
from .view import FACING_BACK

__all__ = [
    "compute_parameters",
    "compute_body_parameters",
    "compute_mudra_parameters",
    "crossing_side",
    "hand_param",
]


def compute_parameters(
    lm: Landmarks, score_threshold: float = KEYPOINT_SCORE_THRESHOLD
) -> dict[str, float | str]:
    """All Bharatanatyam parameters plus landmark-availability bookkeeping."""
    params: dict[str, float | str] = {}
    params.update(compute_body_parameters(lm, score_threshold))
    params.update(compute_mudra_parameters(lm, score_threshold))

    # Seen from behind, the palms face away from the camera, so the finger
    # geometry cannot be measured at all. Blank it rather than reporting the
    # model's guess as if it were a measurement.
    if params.get("facing") == FACING_BACK:
        for key in list(params):
            if key.endswith("_hand_available"):
                params[key] = 0.0
            elif key.endswith("wrist_lift_norm"):
                # wrist HEIGHT is still measurable from behind and is what tells
                # the recogniser that a mudra is being presented at all
                continue
            elif "_hand_" in key and isinstance(params[key], (int, float)):
                params[key] = float("nan")
        params["mudra_hand_used"] = ""
        params["mudra_measurable"] = 0.0
    else:
        params["mudra_measurable"] = float(bool(params.get("mudra_hand_used")))

    mask = lm.available(score_threshold)
    params["landmarks_available_count"] = float(int(mask.sum()))
    params["body_landmarks_available"] = float(
        lm.has(C.REQUIRED_BODY, score_threshold)
    )
    params["feet_landmarks_available"] = float(
        lm.has(C.REQUIRED_FEET, score_threshold)
    )
    params["arm_landmarks_available"] = float(lm.has(C.REQUIRED_ARMS, score_threshold))
    return params
