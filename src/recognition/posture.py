"""Parameter-based posture recognition (no deep-learning classifier).

This module answers ONE question: which posture is the dancer attempting? It
never decides whether the posture is performed correctly -- that is
`src/correctness/rules.py`, working from the fixed specification ranges. A
posture recognised here with out-of-range parameters stays that posture and is
reported as INCORRECT by the rules; it is never relabelled as a different
posture because one of its parameters is wrong.

Recognition works in three steps:

1. GATES - necessary conditions. Each posture declares the parameter intervals
   it must satisfy to be a candidate at all, taken from the ranges the 208
   labelled images actually occupy. A gate on an unavailable parameter is
   skipped, so a side view (where the frontal-plane parameters are blanked) is
   judged on whatever remains rather than being rejected outright.
2. SCORING - among the surviving candidates, each posture's score is the
   weighted mean Gaussian similarity over its cues.
3. DECISION - the winner must clear a minimum confidence AND beat the
   runner-up by a minimum margin. If it does not, or if no candidate survives
   gating, the frame is reported as TRANSITION rather than being forced into
   the nearest label.

Step 1 is what stops a deepening knee bend from being read as Muzhumandi: a
deep squat is not merely a large knee angle, it also drops the hips to the
ankles. On the labelled set the two never overlap -- Muzhumandi hips sit at
0.35-0.91 torso heights above the ankles while Aramandi hips sit at 1.17-1.70,
and knee flexion is 95.8-157.5 deg for Muzhumandi against 2.7-68.0 deg for
Aramandi. A frame at 89 deg flexion with the hips still high is therefore
still an Aramandi (an over-deep one, which the rules will mark INCORRECT). It
is not a Muzhumandi, and it is not relabelled as one just because the knee
angle has left the *correct* Aramandi range.

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

from dataclasses import dataclass, field

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
INF = float("inf")

# Reported when the frame does not confidently correspond to one stable posture.
# This is a recognition outcome, not a posture: no correctness rules apply to it.
TRANSITION = "TRANSITION"

# Wrists carried this far above the hips (as a fraction of torso height) mean the
# dancer is presenting a mudra rather than working the legs with the hands on the
# waist. DATA_CALIBRATED on the labelled set: across the four cameras the body
# variations stay below ~0.68 at the 95th percentile while the mudra variations
# start at ~0.68, so this is the crossover point.
MUDRA_WRIST_LIFT_THRESHOLD = 0.70
# Dataset mudras are performed standing (labelled flexion 0.7-10.8 deg). A squat
# drops the hips toward waist-level hands and inflates wrist-lift, which must
# not be read as a raised-hand mudra. Body work is underway once the knees bend
# past a standing Samapadham.
MUDRA_MAX_KNEE_FLEXION_DEG = 28.0

# The winning template must explain the frame at least this well.
MIN_CONFIDENCE = 0.35
# A winner this good is accepted even when a sibling posture scores close to it.
# Held postures frequently produce near-ties (Samapadham and Swastikapadam both
# stand with the feet together, for instance) and on the labelled images the
# score does not separate the right pick from the wrong one in those ties, so
# demanding a margin there would only discard correct answers.
DECISIVE_CONFIDENCE = 0.55
# Below DECISIVE_CONFIDENCE, a winner must also beat the runner-up by this much.
# A mediocre fit plus a near-tie is what a frame between two postures looks like.
MIN_MARGIN = 0.06


@dataclass(frozen=True)
class Cue:
    parameter: str
    target: float
    tolerance: float
    weight: float = 1.0


@dataclass(frozen=True)
class Gate:
    """A necessary condition: the posture is impossible outside this interval.

    Bounds are taken from the range the labelled images occupy, widened by a
    margin. A gate whose parameter is unavailable for this frame is skipped
    rather than failed, so viewpoints that cannot measure it still work.
    """

    parameter: str
    low: float = -INF
    high: float = INF
    observed: str = ""  # what the labelled set measures, for the record

    def check(self, values: dict[str, float]) -> tuple[bool, str]:
        value = values.get(self.parameter, NAN)
        if not np.isfinite(value):
            return True, ""
        if value < self.low or value > self.high:
            bound = (
                f">= {self.low:g}"
                if self.high == INF
                else f"<= {self.high:g}"
                if self.low == -INF
                else f"in [{self.low:g}, {self.high:g}]"
            )
            return False, f"{self.parameter}={value:.2f} needs {bound}"
        return True, ""


@dataclass(frozen=True)
class Template:
    posture: str
    cues: tuple[Cue, ...]
    gates: tuple[Gate, ...] = ()


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
        gates=(
            # standing tall on straight legs
            Gate("mean_knee_flexion_deg", high=25.0, observed="0.9-10.4"),
            Gate("hip_height_torso_norm", low=1.15, observed="1.43-1.78"),
        ),
    ),
    Template(
        ARAMANDI,
        (
            Cue("mean_knee_flexion_deg", 40.0, 22.0, 1.6),
            Cue("hip_height_torso_norm", 1.60, 0.50, 1.2),
            Cue("foot_separation_norm", 0.70, 0.55),
            Cue("mean_foot_turnout_deg", 68.0, 35.0, 0.7),
            Cue("mean_knee_foot_alignment_deg", 8.0, 22.0, 0.8),
            Cue("trunk_inclination_deg", 3.0, 12.0, 0.6),
        ),
        gates=(
            # Identity, not correctness: an over-deep half-sit is still
            # Aramandi. There is no upper flexion bound here because that is
            # what the fixed 30-45 deg rule judges afterwards. The hip floor
            # sits just above the labelled Muzhumandi ceiling (0.91) so a
            # true squat cannot sneak through as Aramandi, while a dancer
            # who has only gone a little deeper than the labelled Aramandi
            # still stays Aramandi.
            Gate("hip_height_torso_norm", low=0.92, observed="1.17-1.70"),
        ),
    ),
    Template(
        MUZHUMANDI,
        (
            Cue("mean_knee_flexion_deg", 135.0, 40.0, 2.0),
            Cue("hip_height_torso_norm", 0.70, 0.45, 1.6),
            Cue("knee_height_torso_norm", 0.20, 0.40, 1.0),
        ),
        gates=(
            # A full squat has BOTH a deeply folded knee AND the hips dropped
            # towards the ankles. Either cue alone is not enough: a deepening
            # Aramandi raises flexion while the hips are still high, and that
            # must remain Aramandi. Bounds sit in the empty band between the
            # labelled ranges (flexion 68.0-95.8, hip height 0.91-1.17).
            Gate("mean_knee_flexion_deg", low=92.0, observed="95.8-157.5"),
            Gate("hip_height_torso_norm", high=1.05, observed="0.35-0.91"),
        ),
    ),
    Template(
        SWASTIKAPADAM,
        (
            Cue("mean_knee_flexion_deg", 18.0, 22.0),
            Cue("crossing_signed_norm_view", -0.05, 0.30, 1.8),
            Cue("shin_crossing_distance_norm", 0.08, 0.28, 1.8),
            Cue("foot_line_deviation_deg", 30.0, 22.0, 1.4),
        ),
        gates=(
            # held at Aramandi depth, standing, with the legs actually together
            Gate("mean_knee_flexion_deg", high=70.0, observed="1.8-50.9"),
            Gate("hip_height_torso_norm", low=0.95, observed="1.06-1.62"),
            Gate("shin_crossing_distance_norm", high=0.45, observed="0.00-0.44"),
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
    posture: str  # "" when nothing could be recognised, TRANSITION when unstable
    confidence: float
    family: str  # "BODY" | "MUDRA" | "TRANSITION" | ""
    scores: dict[str, float]
    side: str = ""
    message: str = ""
    rejected: dict[str, str] = field(default_factory=dict)
    runner_up: str = ""
    margin: float = NAN

    @property
    def ok(self) -> bool:
        return bool(self.posture) and self.posture != TRANSITION

    @property
    def is_transition(self) -> bool:
        return self.posture == TRANSITION


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


def _legs_standing(values: dict[str, float]) -> bool:
    """True when the legs are not already in a bent-knee body posture."""
    flexion = values.get("mean_knee_flexion_deg", NAN)
    if not np.isfinite(flexion):
        return True
    return flexion <= MUDRA_MAX_KNEE_FLEXION_DEG


def _body_values(params: dict[str, float | str]) -> dict[str, float]:
    """Numeric parameters plus the cues that recognition derives from them."""
    values = _numeric(params)

    # `ankle_crossing_signed_norm` is left_ankle_x - right_ankle_x, so its sign
    # depends on which way the dancer faces: uncrossed legs read positive from
    # the front and negative from the rear. Flip it to a view-independent
    # convention (positive = uncrossed) before using it as a crossing cue.
    crossing = values.get("ankle_crossing_signed_norm", NAN)
    facing = params.get("facing", "")
    if np.isfinite(crossing) and isinstance(facing, str) and facing:
        sign = 1.0 if facing == "FRONT" else -1.0
        values["crossing_signed_norm_view"] = sign * crossing

    return values


def _evaluate_templates(
    templates: tuple[Template, ...], values: dict[str, float]
) -> tuple[dict[str, float], dict[str, str]]:
    """Score every template that passes its gates; record why others failed."""
    scores: dict[str, float] = {}
    rejected: dict[str, str] = {}

    for template in templates:
        blockers = [
            reason
            for gate in template.gates
            for passed, reason in [gate.check(values)]
            if not passed
        ]
        if blockers:
            rejected[template.posture] = "; ".join(blockers)
            continue
        score, used = _score(template, values)
        if used:
            scores[template.posture] = score
        else:
            rejected[template.posture] = "no cue parameter available"

    return scores, rejected


def _decide(
    scores: dict[str, float], family: str, rejected: dict[str, str], note: str
) -> RecognitionResult:
    """Apply the confidence and margin tests, falling back to TRANSITION."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up, runner_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = best_score - runner_score

    if best_score < MIN_CONFIDENCE:
        return RecognitionResult(
            posture=TRANSITION,
            confidence=best_score,
            family="TRANSITION",
            scores=scores,
            rejected=rejected,
            runner_up=runner_up,
            margin=margin,
            message=(
                f"closest posture {best} scores {best_score:.2f}, below the "
                f"{MIN_CONFIDENCE:.2f} confidence needed for a stable label"
            ),
        )

    if runner_up and margin < MIN_MARGIN and best_score < DECISIVE_CONFIDENCE:
        return RecognitionResult(
            posture=TRANSITION,
            confidence=best_score,
            family="TRANSITION",
            scores=scores,
            rejected=rejected,
            runner_up=runner_up,
            margin=margin,
            message=(
                f"{best} ({best_score:.2f}) and {runner_up} ({runner_score:.2f}) "
                f"are within {margin:.2f}, so the frame is between postures"
            ),
        )

    return RecognitionResult(
        posture=best,
        confidence=best_score,
        family=family,
        scores=scores,
        rejected=rejected,
        runner_up=runner_up,
        margin=margin,
        message=note,
    )


def recognise(params: dict[str, float | str]) -> RecognitionResult:
    """Identify the posture being attempted, or report TRANSITION.

    Correctness is deliberately not considered here: a posture whose parameters
    fall outside the specification still wins its own template, so the rules can
    report it as INCORRECT under the right posture name.
    """
    body_values = _body_values(params)
    mudra_values, _ = _mudra_values(params)

    body_scores, body_rejected = _evaluate_templates(BODY_TEMPLATES, body_values)
    mudra_scores, mudra_rejected = (
        _evaluate_templates(MUDRA_TEMPLATES, mudra_values)
        if mudra_values
        else ({}, {})
    )

    all_scores = {**body_scores, **mudra_scores}
    all_rejected = {**body_rejected, **mudra_rejected}
    lift = wrist_lift_norm(params)
    hands_raised = np.isfinite(lift) and lift >= MUDRA_WRIST_LIFT_THRESHOLD
    # A squat drops the hips toward waist-level hands, which inflates wrist
    # lift. Dataset mudras are standing, so bent knees mean body work.
    presenting_mudra = hands_raised and _legs_standing(body_values)

    if presenting_mudra:
        if mudra_scores:
            result = _decide(
                mudra_scores,
                "MUDRA",
                all_rejected,
                f"hands raised {lift:.2f} torso heights above the hips",
            )
            result.scores = all_scores
            return result
        # The dancer is clearly presenting a mudra, but the fingers cannot be
        # measured from this viewpoint (typically a rear view, where the palms
        # face away). Reporting a leg posture instead would be misleading.
        return RecognitionResult(
            posture="",
            confidence=0.0,
            family="MUDRA",
            scores=all_scores,
            rejected=all_rejected,
            message=(
                "mudra presented but finger landmarks are not measurable from "
                "this viewpoint"
            ),
        )

    if not body_scores:
        # Every body posture was ruled out by its own necessary conditions, so
        # the dancer is somewhere between postures rather than in any of them.
        if body_rejected:
            return RecognitionResult(
                posture=TRANSITION,
                confidence=0.0,
                family="TRANSITION",
                scores=all_scores,
                rejected=all_rejected,
                message="no posture satisfies its required configuration: "
                + "; ".join(f"{k} ({v})" for k, v in body_rejected.items()),
            )
        return RecognitionResult(
            posture="",
            confidence=0.0,
            family="",
            scores=all_scores,
            rejected=all_rejected,
            message="not enough landmarks to recognise a posture",
        )

    note = (
        f"hands at waist level (wrist lift {lift:.2f})"
        if np.isfinite(lift)
        else "wrist height unavailable"
    )
    result = _decide(body_scores, "BODY", all_rejected, note)
    result.scores = all_scores
    if result.posture == SWASTIKAPADAM:
        from ..features.body_params import crossing_side

        result.side = crossing_side(body_values)
    return result
