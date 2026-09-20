"""The common 67-landmark representation shared by every ONNX model adapter.

Layout (exactly the ordering fixed by the project specification):

    0-14   pose 0-14   (head, shoulders, elbows)
    15-24  pose 23-32  (hips, knees, ankles, heels, foot index)
    25-45  left hand 0-20
    46-66  right hand 0-20

The 8 wrist/finger points of the MediaPipe pose list (15-22) are deliberately
dropped because the dedicated hand blocks cover them.

Any landmark a model cannot supply is stored as NaN with score 0.0. Nothing is
ever fabricated: the parameter layer checks availability before computing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_COMMON = 67

# --- body block -------------------------------------------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_HIP = 15
RIGHT_HIP = 16
LEFT_KNEE = 17
RIGHT_KNEE = 18
LEFT_ANKLE = 19
RIGHT_ANKLE = 20
LEFT_HEEL = 21
RIGHT_HEEL = 22
LEFT_FOOT_INDEX = 23
RIGHT_FOOT_INDEX = 24

# --- hand blocks ------------------------------------------------------------
LEFT_HAND_START = 25
RIGHT_HAND_START = 46

# offsets inside a 21-point hand block
H_WRIST = 0
H_THUMB_CMC, H_THUMB_MCP, H_THUMB_IP, H_THUMB_TIP = 1, 2, 3, 4
H_INDEX_MCP, H_INDEX_PIP, H_INDEX_DIP, H_INDEX_TIP = 5, 6, 7, 8
H_MIDDLE_MCP, H_MIDDLE_PIP, H_MIDDLE_DIP, H_MIDDLE_TIP = 9, 10, 11, 12
H_RING_MCP, H_RING_PIP, H_RING_DIP, H_RING_TIP = 13, 14, 15, 16
H_PINKY_MCP, H_PINKY_PIP, H_PINKY_DIP, H_PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (H_THUMB_CMC, H_THUMB_MCP, H_THUMB_IP, H_THUMB_TIP),
    "index": (H_INDEX_MCP, H_INDEX_PIP, H_INDEX_DIP, H_INDEX_TIP),
    "middle": (H_MIDDLE_MCP, H_MIDDLE_PIP, H_MIDDLE_DIP, H_MIDDLE_TIP),
    "ring": (H_RING_MCP, H_RING_PIP, H_RING_DIP, H_RING_TIP),
    "pinky": (H_PINKY_MCP, H_PINKY_PIP, H_PINKY_DIP, H_PINKY_TIP),
}
LONG_FINGERS = ("index", "middle", "ring", "pinky")


def hand_index(side: str, offset: int) -> int:
    """Absolute common index of a hand keypoint. `side` is 'left' or 'right'."""
    base = LEFT_HAND_START if side == "left" else RIGHT_HAND_START
    return base + offset


COMMON_NAMES: list[str] = [""] * NUM_COMMON
for _name, _idx in {
    "nose": NOSE,
    "left_eye_inner": LEFT_EYE_INNER,
    "left_eye": LEFT_EYE,
    "left_eye_outer": LEFT_EYE_OUTER,
    "right_eye_inner": RIGHT_EYE_INNER,
    "right_eye": RIGHT_EYE,
    "right_eye_outer": RIGHT_EYE_OUTER,
    "left_ear": LEFT_EAR,
    "right_ear": RIGHT_EAR,
    "mouth_left": MOUTH_LEFT,
    "mouth_right": MOUTH_RIGHT,
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
    "left_heel": LEFT_HEEL,
    "right_heel": RIGHT_HEEL,
    "left_foot_index": LEFT_FOOT_INDEX,
    "right_foot_index": RIGHT_FOOT_INDEX,
}.items():
    COMMON_NAMES[_idx] = _name

_HAND_PART_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]
for _side, _base in (("left", LEFT_HAND_START), ("right", RIGHT_HAND_START)):
    for _off, _part in enumerate(_HAND_PART_NAMES):
        COMMON_NAMES[_base + _off] = f"{_side}_hand_{_part}"

# Landmarks the body-parameter layer cannot work without.
REQUIRED_BODY = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)
# Landmarks needed for foot orientation / heel rules.
REQUIRED_FEET = (LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX)
# Landmarks needed for the elbow-shoulder alignment rule of the mudra classes.
REQUIRED_ARMS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW)

LEFT_HAND_RANGE = range(LEFT_HAND_START, LEFT_HAND_START + 21)
RIGHT_HAND_RANGE = range(RIGHT_HAND_START, RIGHT_HAND_START + 21)
HAND_INDICES = list(LEFT_HAND_RANGE) + list(RIGHT_HAND_RANGE)

# Hand keypoints are held to a lower confidence bar than body keypoints. Closed
# mudras (Mushti, Katakamukha) hide the fingertips behind the palm, so the pose
# model reports genuinely low scores for points whose position is still useful.
# Judging fingers at the body threshold would simply erase those mudras.
HAND_SCORE_FLOOR = 0.10


@dataclass
class Landmarks:
    """A single person's landmarks in the common representation.

    `xy` holds pixel coordinates in the original image frame; missing points are
    NaN. `score` is the model confidence (0.0 for missing points).
    """

    xy: np.ndarray  # (67, 2) float32
    score: np.ndarray  # (67,) float32
    model_name: str = ""
    supported: np.ndarray = field(default=None)  # (67,) bool: model *can* supply it
    bbox: tuple[float, float, float, float] | None = None
    # Model-specific points with no slot in the common list (e.g. small toes).
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.supported is None:
            self.supported = np.isfinite(self.xy).all(axis=1)

    @classmethod
    def empty(cls, model_name: str = "") -> Landmarks:
        return cls(
            xy=np.full((NUM_COMMON, 2), np.nan, dtype=np.float32),
            score=np.zeros(NUM_COMMON, dtype=np.float32),
            model_name=model_name,
            supported=np.zeros(NUM_COMMON, dtype=bool),
        )

    def available(self, threshold: float) -> np.ndarray:
        """Boolean mask of landmarks usable at the given score threshold.

        Hand keypoints use `HAND_SCORE_FLOOR` when that is the looser bar.
        """
        finite = np.isfinite(self.xy).all(axis=1)
        mask = finite & (self.score >= threshold)
        hand_bar = min(threshold, HAND_SCORE_FLOOR)
        mask[HAND_INDICES] = (finite & (self.score >= hand_bar))[HAND_INDICES]
        return mask

    def has(self, indices, threshold: float) -> bool:
        mask = self.available(threshold)
        return bool(np.all([mask[i] for i in indices]))

    def point(self, index: int) -> np.ndarray:
        return self.xy[index]

    def hand_available(self, side: str, threshold: float) -> bool:
        rng = LEFT_HAND_RANGE if side == "left" else RIGHT_HAND_RANGE
        mask = self.available(threshold)
        # a hand is usable when the wrist plus most of the finger points landed
        return bool(mask[rng.start] and mask[list(rng)].sum() >= 15)
