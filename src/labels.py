"""The fixed 26-variation label taxonomy for the college dataset.

These labels are FINAL. Variation ids are 1-based and map onto the sorted JPG
files of each core camera folder as `variation_id = index // 2 + 1`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Posture identifiers used everywhere downstream (recognition, rules, UI).
ARAMANDI = "ARAMANDI"
SAMAPADHAM = "SAMAPADHAM"
MUZHUMANDI = "MUZHUMANDI"
SWASTIKAPADAM = "SWASTIKAPADAM"
PATAKA = "PATAKA"
ALAPADMA = "ALAPADMA"
MUSHTI = "MUSHTI"
KATAKAMUKHA = "KATAKAMUKHA"

BODY_POSTURES = (ARAMANDI, SAMAPADHAM, MUZHUMANDI, SWASTIKAPADAM)
MUDRA_POSTURES = (PATAKA, ALAPADMA, MUSHTI, KATAKAMUKHA)
ALL_POSTURES = BODY_POSTURES + MUDRA_POSTURES

STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"


@dataclass(frozen=True)
class VariationLabel:
    variation_id: int
    posture: str
    status: str
    error_code: str  # "" when the variation is correct
    error_text: str
    side: str  # "", "RIGHT" or "LEFT"

    @property
    def label_name(self) -> str:
        parts = [self.posture, self.status]
        if self.error_text:
            parts.append(self.error_text)
        if self.side:
            parts.append(f"{self.side.lower()} leg")
        return " - ".join(parts)


def _c(vid: int, posture: str, side: str = "") -> VariationLabel:
    return VariationLabel(vid, posture, STATUS_CORRECT, "", "", side)


def _i(vid: int, posture: str, code: str, text: str, side: str = "") -> VariationLabel:
    return VariationLabel(vid, posture, STATUS_INCORRECT, code, text, side)


VARIATIONS: tuple[VariationLabel, ...] = (
    # --- body postures, correct ---
    _c(1, ARAMANDI),
    _c(2, SAMAPADHAM),
    _c(3, MUZHUMANDI),
    _c(4, SWASTIKAPADAM, side="RIGHT"),
    _c(5, SWASTIKAPADAM, side="LEFT"),
    # --- Aramandi errors ---
    _i(6, ARAMANDI, "INSUFFICIENT_KNEE_BEND", "Insufficient knee bend"),
    _i(7, ARAMANDI, "FEET_TOO_FAR_APART", "Feet too far apart"),
    _i(8, ARAMANDI, "FEET_NOT_IN_LINE", "Feet not spread in a straight line / V-shaped"),
    # --- Samapadham errors ---
    _i(9, SAMAPADHAM, "FEET_TOO_FAR_APART", "Feet too far apart"),
    _i(10, SAMAPADHAM, "FEET_NOT_ALIGNED", "Feet not aligned / V-shaped"),
    _i(11, SAMAPADHAM, "WEIGHT_ON_ONE_LEG", "More weight on one leg"),
    # --- Muzhumandi errors ---
    _i(12, MUZHUMANDI, "KNEES_FACING_FORWARD", "Feet and knees facing forward"),
    _i(13, MUZHUMANDI, "BACK_BENT_FORWARD", "Back bent forward"),
    _i(
        14,
        MUZHUMANDI,
        "FEET_APART_HEEL_DOWN",
        "Feet far apart and heel pressing on floor",
    ),
    # --- Swastikapadam errors ---
    _i(15, SWASTIKAPADAM, "OVERCROSSING_SHINS", "Overcrossing shins", side="RIGHT"),
    _i(16, SWASTIKAPADAM, "OVERCROSSING_SHINS", "Overcrossing shins", side="LEFT"),
    _i(
        17,
        SWASTIKAPADAM,
        "INSUFFICIENT_ARAMANDI",
        "Insufficient Aramandi",
        side="RIGHT",
    ),
    _i(18, SWASTIKAPADAM, "INSUFFICIENT_ARAMANDI", "Insufficient Aramandi", side="LEFT"),
    # --- full body + mudra ---
    _i(19, PATAKA, "ELBOWS_NOT_ALIGNED", "Elbows down / not aligned with shoulders"),
    _i(20, ALAPADMA, "ELBOWS_NOT_ALIGNED", "Elbows not aligned with shoulders"),
    _c(21, ALAPADMA),
    _c(22, MUSHTI),
    _c(23, KATAKAMUKHA),
    _i(24, PATAKA, "FINGERS_TOO_FAR_APART", "Fingers too far apart"),
    _i(25, KATAKAMUKHA, "FINGERS_BENT", "Fingers bent"),
    _i(26, MUSHTI, "PINKY_DROPPED", "Pinky dropped / fingers loose"),
)

VARIATION_BY_ID: dict[int, VariationLabel] = {v.variation_id: v for v in VARIATIONS}

assert len(VARIATIONS) == 26, "the taxonomy must contain exactly 26 variations"


def variation_for_index(index: int) -> VariationLabel:
    """Map a 0-based index within a camera's sorted JPG list to its label."""
    return VARIATION_BY_ID[index // 2 + 1]
