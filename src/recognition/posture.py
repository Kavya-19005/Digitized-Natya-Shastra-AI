"""Parameter-based posture recognition (no deep-learning classifier).

Each posture is described by a small template of parameter targets with
tolerances. A posture's score is the mean Gaussian similarity over the
parameters that could actually be measured, so a missing landmark degrades the
score gracefully instead of crashing.

Two families are recognised:

* BODY postures   - Aramandi, Samapadham, Muzhumandi, Swastikapadam. In the
                    dataset these are performed with the hands on the waist.
* MUDRA postures  - Pataka, Alapadma, Mushti, Katakamukha. These are performed
                    standing, with the hands raised in front of the chest, and
                    are recognised from the hand geometry.

The family is chosen from how high the wrists are carried relative to the hips,
which is the visual cue that separates the two halves of the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.hand_params import hand_param
from ..labels import (
    ALAPADMA,
    ARAMANDI,
    KATAKAMUKHA,
    MUSHTI,
    MUZHUMANDI,
    PATAKA,
    SAMAPADHAM,
    SWASTIKAPADAM,
)

NAN = float("nan")

# Wrists carried this far above the hips (as a fraction of torso height) mean the
# dancer is presenting a mudra rather than working the legs with the hands on the
# waist. DATA_CALIBRATED on the labelled set: across the four cameras the body
# variations stay below ~0.68 at the 95th percentile while the mudra variations
# start at ~0.68, so this is the crossover point.
MUDRA_WRIST_LIFT_THRESHOLD = 0.70


@dataclass(frozen=True)
class Cue:
    parameter: str
    target: float
    tolerance: float
    weight: float = 1.0


@dataclass(frozen=True)
class Template:
    posture: str
    cues: tuple[Cue, ...]


# Cue targets are the values measured on the labelled images (see
# `scripts/calibrate_thresholds.py` and outputs/image_results/
# data_calibrated_reference.csv). They only select WHICH posture is being
# performed; whether it is performed correctly is decided afterwards by the
# fixed ranges in `src/correctness/thresholds.py`.
BODY_TEMPLATES = (
    Template(
        SAMAPADHAM,
        (
            Cue("mean_knee_flexion_deg", 6.0, 16.0, 1.6),
            Cue("hip_height_torso_norm", 1.90, 0.55, 1.2),
            Cue("foot_separation_norm", 0.20, 0.45),
        ),
    ),
    Template(
        ARAMANDI,
        (
            Cue("mean_knee_flexion_deg", 40.0, 22.0, 1.6),
            Cue("hip_height_torso_norm", 1.60, 0.50, 1.2),
            Cue("foot_separation_norm", 0.70, 0.55),
        ),
    ),
    Template(
        MUZHUMANDI,
        (
            Cue("mean_knee_flexion_deg", 135.0, 40.0, 2.0),
            Cue("hip_height_torso_norm", 0.70, 0.45, 1.6),
        ),
    ),
    Template(
        SWASTIKAPADAM,
        (
            Cue("mean_knee_flexion_deg", 18.0, 22.0),
            Cue("ankle_crossing_signed_norm", -0.05, 0.30, 1.8),
            Cue("shin_crossing_distance_norm", 0.08, 0.28, 1.8),
            Cue("foot_line_deviation_deg", 30.0, 22.0, 1.4),
        ),
    ),
)

MUDRA_TEMPLATES = (
    Template(
        PATAKA,
        (
            Cue("mean_finger_straightness_deg", 172.0, 15.0, 1.6),
            Cue("mean_interfinger_spread_deg", 8.0, 18.0, 1.4),
            Cue("palm_opening_norm", 1.15, 0.40, 1.2),
            Cue("extended_finger_contrast", 0.0, 0.35),
        ),
    ),
    Template(
        ALAPADMA,
        (
            Cue("mean_finger_straightness_deg", 160.0, 20.0),
            Cue("mean_interfinger_spread_deg", 50.0, 26.0, 1.8),
            Cue("max_adjacent_fingertip_gap_norm", 1.05, 0.45, 1.4),
            Cue("palm_opening_norm", 1.05, 0.40),
        ),
    ),
    Template(
        MUSHTI,
        (
            Cue("mean_finger_flexion_deg", 54.0, 22.0, 1.6),
            Cue("palm_opening_norm", 0.54, 0.26, 1.8),
            Cue("extended_finger_contrast", 0.0, 0.30, 1.6),
            Cue("thumb_index_tip_distance_norm", 0.37, 0.35),
        ),
    ),
    Template(
        KATAKAMUKHA,
        (
            Cue("thumb_index_tip_distance_norm", 0.40, 0.30, 1.4),
            Cue("fingertip_cluster_spread_norm", 0.30, 0.22, 1.2),
            Cue("extended_finger_contrast", 0.55, 0.30, 2.0),
            Cue("palm_opening_norm", 0.80, 0.35),
        ),
    ),
)


@dataclass
class RecognitionResult:
    posture: str
    confidence: float
    family: str  # "BODY" | "MUDRA" | ""
    scores: dict[str, float]
    side: str = ""
    message: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.posture)


def _similarity(value: float, cue: Cue) -> float:
    if not np.isfinite(value) or cue.tolerance <= 0:
        return NAN
    z = (value - cue.target) / cue.tolerance
    return float(np.exp(-0.5 * z * z))


def _score(template: Template, values: dict[str, float]) -> tuple[float, int]:
    total = 0.0
    weight = 0.0
    used = 0
    for cue in template.cues:
        similarity = _similarity(values.get(cue.parameter, NAN), cue)
        if not np.isfinite(similarity):
            continue
        total += similarity * cue.weight
        weight += cue.weight
        used += 1
    if weight <= 0:
        return NAN, 0
    return total / weight, used


def wrist_lift_norm(params: dict[str, float | str]) -> float:
    """How far above the hips the wrists are carried, in shoulder widths."""
    values = []
    for side in ("left", "right"):
        # the hand block's wrist is the reliable wrist point in our layout
        lift = params.get(f"{side}_hand_wrist_lift_norm", NAN)
        if isinstance(lift, (int, float)) and np.isfinite(lift):
            values.append(float(lift))
    if not values:
        return NAN
    return float(np.mean(values))


def _mudra_values(params: dict[str, float | str]) -> tuple[dict[str, float], str]:
    side = params.get("mudra_hand_used", "")
    side = side if isinstance(side, str) else ""
    if not side:
        return {}, ""
    names = (
        "mean_finger_straightness_deg",
        "mean_finger_flexion_deg",
        "mean_interfinger_spread_deg",
        "max_adjacent_fingertip_gap_norm",
        "mean_adjacent_fingertip_gap_norm",
        "palm_opening_norm",
        "thumb_index_tip_distance_norm",
        "fingertip_cluster_spread_norm",
        "extended_finger_contrast",
        "ring_pinky_straightness_deg",
        "pinky_relative_extension",
    )
    return {name: hand_param(params, side, name) for name in names}, side


def _numeric(params: dict[str, float | str]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in params.items()
        if isinstance(v, (int, float)) and np.isfinite(float(v))
    }


def recognise(params: dict[str, float | str]) -> RecognitionResult:
    """Pick the posture whose template best explains the measured parameters."""
    body_values = _numeric(params)
    mudra_values, mudra_side = _mudra_values(params)

    body_scores: dict[str, float] = {}
    for template in BODY_TEMPLATES:
        score, used = _score(template, body_values)
        if used:
            body_scores[template.posture] = score

    mudra_scores: dict[str, float] = {}
    if mudra_values:
        for template in MUDRA_TEMPLATES:
            score, used = _score(template, mudra_values)
            if used:
                mudra_scores[template.posture] = score

    scores = {**body_scores, **mudra_scores}
    lift = wrist_lift_norm(params)
    hands_raised = np.isfinite(lift) and lift >= MUDRA_WRIST_LIFT_THRESHOLD

    if hands_raised:
        if mudra_scores:
            posture = max(mudra_scores, key=mudra_scores.get)
            return RecognitionResult(
                posture=posture,
                confidence=mudra_scores[posture],
                family="MUDRA",
                scores=scores,
                message=f"hands raised {lift:.2f} torso heights above the hips",
            )
        # The dancer is clearly presenting a mudra, but the fingers cannot be
        # measured from this viewpoint (typically a rear view, where the palms
        # face away). Reporting a leg posture instead would be misleading.
        return RecognitionResult(
            posture="",
            confidence=0.0,
            family="MUDRA",
            scores=scores,
            message=(
                "mudra presented but finger landmarks are not measurable from "
                "this viewpoint"
            ),
        )

    if not body_scores:
        return RecognitionResult(
            posture="",
            confidence=0.0,
            family="",
            scores=scores,
            message="not enough landmarks to recognise a posture",
        )

    posture = max(body_scores, key=body_scores.get)
    side = ""
    if posture == SWASTIKAPADAM:
        from ..features.body_params import crossing_side

        side = crossing_side(body_values)
    return RecognitionResult(
        posture=posture,
        confidence=body_scores[posture],
        family="BODY",
        scores=scores,
        side=side,
        message=(
            f"hands at waist level (wrist lift {lift:.2f})"
            if np.isfinite(lift)
            else "wrist height unavailable"
        ),
    )
