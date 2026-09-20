"""Adapters from each model's raw keypoint layout to the common 67 landmarks.

Nothing is fabricated. A landmark a given layout does not contain stays NaN and
is reported as unsupported, so `docs/landmark_availability.md` and the result
CSVs state honestly what each ONNX model can actually measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import common as C

# ---------------------------------------------------------------------------
# COCO-WholeBody 133 (RTMW-l cocktail14, DWPose-l ucoco)
#   0-16   body (COCO)
#   17-22  feet (l big toe, l small toe, l heel, r big toe, r small toe, r heel)
#   23-90  face (68-point layout)
#   91-111 left hand (21)
#   112-132 right hand (21)
# ---------------------------------------------------------------------------
_FACE = 23

WHOLEBODY133_MAP: dict[int, int] = {
    C.NOSE: 0,
    C.LEFT_EYE: 1,
    C.RIGHT_EYE: 2,
    C.LEFT_EAR: 3,
    C.RIGHT_EAR: 4,
    C.LEFT_SHOULDER: 5,
    C.RIGHT_SHOULDER: 6,
    C.LEFT_ELBOW: 7,
    C.RIGHT_ELBOW: 8,
    C.LEFT_HIP: 11,
    C.RIGHT_HIP: 12,
    C.LEFT_KNEE: 13,
    C.RIGHT_KNEE: 14,
    C.LEFT_ANKLE: 15,
    C.RIGHT_ANKLE: 16,
    # feet: the common representation's "foot index" is the big toe
    C.LEFT_FOOT_INDEX: 17,
    C.LEFT_HEEL: 19,
    C.RIGHT_FOOT_INDEX: 20,
    C.RIGHT_HEEL: 22,
    # eye/mouth detail comes from the 68-point face block
    C.RIGHT_EYE_OUTER: _FACE + 36,
    C.RIGHT_EYE_INNER: _FACE + 39,
    C.LEFT_EYE_INNER: _FACE + 42,
    C.LEFT_EYE_OUTER: _FACE + 45,
    C.MOUTH_RIGHT: _FACE + 48,
    C.MOUTH_LEFT: _FACE + 54,
}
for _off in range(21):
    WHOLEBODY133_MAP[C.LEFT_HAND_START + _off] = 91 + _off
    WHOLEBODY133_MAP[C.RIGHT_HAND_START + _off] = 112 + _off

# Small toes have no slot in the common representation but are useful for the
# foot-line parameters, so adapters expose them as named extras.
WHOLEBODY133_EXTRAS = {"left_small_toe": 18, "right_small_toe": 21}

# ---------------------------------------------------------------------------
# Halpe26 (RTMPose-m body7-halpe26) -- body + feet, NO hands
#   0-16 body (COCO), 17 head, 18 neck, 19 pelvis,
#   20 l big toe, 21 r big toe, 22 l small toe, 23 r small toe,
#   24 l heel, 25 r heel
# ---------------------------------------------------------------------------
HALPE26_MAP: dict[int, int] = {
    C.NOSE: 0,
    C.LEFT_EYE: 1,
    C.RIGHT_EYE: 2,
    C.LEFT_EAR: 3,
    C.RIGHT_EAR: 4,
    C.LEFT_SHOULDER: 5,
    C.RIGHT_SHOULDER: 6,
    C.LEFT_ELBOW: 7,
    C.RIGHT_ELBOW: 8,
    C.LEFT_HIP: 11,
    C.RIGHT_HIP: 12,
    C.LEFT_KNEE: 13,
    C.RIGHT_KNEE: 14,
    C.LEFT_ANKLE: 15,
    C.RIGHT_ANKLE: 16,
    C.LEFT_FOOT_INDEX: 20,
    C.RIGHT_FOOT_INDEX: 21,
    C.LEFT_HEEL: 24,
    C.RIGHT_HEEL: 25,
}
HALPE26_EXTRAS = {"left_small_toe": 22, "right_small_toe": 23}

# ---------------------------------------------------------------------------
# COCO 17 (plain body models) -- no feet, no hands
# ---------------------------------------------------------------------------
COCO17_MAP = {k: v for k, v in HALPE26_MAP.items() if v <= 16}


@dataclass(frozen=True)
class LandmarkLayout:
    """Describes a raw keypoint layout and how it maps to the common 67."""

    name: str
    num_keypoints: int
    mapping: dict[int, int]
    extras: dict[str, int]
    has_hands: bool
    has_feet: bool

    @property
    def supported_mask(self) -> np.ndarray:
        mask = np.zeros(C.NUM_COMMON, dtype=bool)
        mask[list(self.mapping.keys())] = True
        return mask


WHOLEBODY133 = LandmarkLayout(
    name="coco_wholebody_133",
    num_keypoints=133,
    mapping=WHOLEBODY133_MAP,
    extras=WHOLEBODY133_EXTRAS,
    has_hands=True,
    has_feet=True,
)
HALPE26 = LandmarkLayout(
    name="halpe_26",
    num_keypoints=26,
    mapping=HALPE26_MAP,
    extras=HALPE26_EXTRAS,
    has_hands=False,
    has_feet=True,
)
COCO17 = LandmarkLayout(
    name="coco_17",
    num_keypoints=17,
    mapping=COCO17_MAP,
    extras={},
    has_hands=False,
    has_feet=False,
)

LAYOUTS = {layout.name: layout for layout in (WHOLEBODY133, HALPE26, COCO17)}


def to_common(
    keypoints: np.ndarray,
    scores: np.ndarray,
    layout: LandmarkLayout,
    model_name: str,
    score_threshold: float,
    bbox: tuple[float, float, float, float] | None = None,
) -> C.Landmarks:
    """Scatter a raw (K,2)/(K,) prediction into the common 67-landmark array."""
    landmarks = C.Landmarks.empty(model_name=model_name)
    landmarks.supported = layout.supported_mask
    landmarks.bbox = bbox

    if keypoints is None or keypoints.size == 0:
        return landmarks

    k_available = keypoints.shape[0]
    for common_idx, raw_idx in layout.mapping.items():
        if raw_idx >= k_available:
            continue
        xy = keypoints[raw_idx]
        if not np.all(np.isfinite(xy)) or float(scores[raw_idx]) < score_threshold:
            continue
        landmarks.xy[common_idx] = xy
        landmarks.score[common_idx] = float(scores[raw_idx])

    extras_xy: dict[str, np.ndarray] = {}
    for name, raw_idx in layout.extras.items():
        if raw_idx < k_available and float(scores[raw_idx]) >= score_threshold:
            extras_xy[name] = keypoints[raw_idx].astype(np.float32)
    landmarks.extras = extras_xy
    return landmarks
