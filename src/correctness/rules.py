"""Correct/Incorrect decision plus a specific error, per detected posture.

The canonical error codes are exactly those of the fixed 26-variation taxonomy.
When a posture fails on a parameter that the taxonomy has no label for (for
example trunk inclination during Aramandi) the verdict is still INCORRECT, but
the error code is marked as a SECONDARY finding so the evaluation can tell the
two cases apart.
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
    STATUS_CORRECT,
    STATUS_INCORRECT,
    SWASTIKAPADAM,
)
from .thresholds import (
    ARAMANDI_FOOT_SEPARATION_TARGET_NORM,
    FEET_TOGETHER_BASELINE_NORM,
    BORDERLINE,
    CORRECT,
    INCORRECT,
    POSTURE_THRESHOLDS,
    TARGET_TURNOUT_DEG,
    UNKNOWN,
    Threshold,
)

NAN = float("nan")

CANONICAL = "CANONICAL"
SECONDARY = "SECONDARY"


@dataclass
class Check:
    """One evaluated parameter."""

    rule_key: str
    parameter: str
    value: float
    band: str
    expected: str
    error_code: str
    error_text: str
    kind: str = CANONICAL
    severity: float = 0.0

    @property
    def failed(self) -> bool:
        return self.band == INCORRECT


@dataclass
class Verdict:
    posture: str
    status: str  # CORRECT | INCORRECT | UNKNOWN
    error_code: str = ""
    error_text: str = ""
    error_kind: str = ""
    primary_parameter: str = ""
    primary_value: float = NAN
    primary_expected: str = ""
    checks: list[Check] = field(default_factory=list)
    borderline: bool = False
    side: str = ""
    message: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "detected_posture": self.posture,
            "predicted_status": self.status,
            "predicted_error": self.error_code,
            "predicted_error_text": self.error_text,
            "predicted_error_kind": self.error_kind,
            "primary_parameter": self.primary_parameter,
            "primary_value": self.primary_value,
            "primary_expected": self.primary_expected,
            "predicted_side": self.side,
            "borderline": self.borderline,
            "rule_message": self.message,
        }


# ---------------------------------------------------------------------------
# posture-dependent derived parameters
# ---------------------------------------------------------------------------
def derived_rule_inputs(
    posture: str, params: dict[str, float | str]
) -> dict[str, float]:
    """Parameters whose definition depends on which posture is being judged."""
    out: dict[str, float] = {}

    separation = _num(params.get("foot_separation_norm"))
    if np.isfinite(separation):
        # gap between the feet, i.e. ankle-to-ankle distance minus the distance
        # that remains when the feet are actually touching
        out["foot_gap_norm"] = max(0.0, separation - FEET_TOGETHER_BASELINE_NORM)

    if posture == ARAMANDI and np.isfinite(separation):
        target = ARAMANDI_FOOT_SEPARATION_TARGET_NORM
        out["foot_separation_deviation_pct"] = abs(separation - target) / target * 100.0
        out["foot_separation_signed_deviation_pct"] = (
            (separation - target) / target * 100.0
        )

    turnout = _num(params.get("mean_foot_turnout_deg"))
    target_turnout = TARGET_TURNOUT_DEG.get(posture)
    if np.isfinite(turnout) and target_turnout is not None:
        out["turnout_deviation_deg"] = abs(turnout - target_turnout)
        out["turnout_signed_deviation_deg"] = turnout - target_turnout

    return out


def _num(value: object) -> float:
    if isinstance(value, (int, float)):
        v = float(value)
        return v if np.isfinite(v) else NAN
    return NAN


def _evaluate(
    threshold: Threshold,
    rule_key: str,
    value: float,
    error_code: str,
    error_text: str,
    kind: str = CANONICAL,
) -> Check:
    severity = threshold.severity(value)
    return Check(
        rule_key=rule_key,
        parameter=threshold.parameter,
        value=value,
        band=threshold.classify(value),
        expected=threshold.expected_text(),
        error_code=error_code,
        error_text=error_text,
        kind=kind,
        severity=0.0 if not np.isfinite(severity) else float(severity),
    )


# ---------------------------------------------------------------------------
# per-posture rule sets, in priority order
# ---------------------------------------------------------------------------
def _aramandi_checks(values: dict[str, float]) -> list[Check]:
    table = POSTURE_THRESHOLDS[ARAMANDI]
    checks: list[Check] = []

    flexion = values.get("mean_knee_flexion_deg", NAN)
    knee = _evaluate(
        table["mean_knee_flexion_deg"],
        "mean_knee_flexion_deg",
        flexion,
        "INSUFFICIENT_KNEE_BEND",
        "Insufficient knee bend",
    )
    if knee.failed and np.isfinite(flexion) and flexion > 45.0:
        knee.error_code = "EXCESSIVE_KNEE_BEND"
        knee.error_text = "Knee bend deeper than Aramandi"
        knee.kind = SECONDARY
    checks.append(knee)

    if "foot_separation_deviation_pct" in values:
        signed = values.get("foot_separation_signed_deviation_pct", NAN)
        too_far = np.isfinite(signed) and signed > 0
        checks.append(
            _evaluate(
                table["foot_separation_deviation_pct"],
                "foot_separation_deviation_pct",
                values["foot_separation_deviation_pct"],
                "FEET_TOO_FAR_APART" if too_far else "FEET_TOO_CLOSE",
                "Feet too far apart" if too_far else "Feet too close together",
                CANONICAL if too_far else SECONDARY,
            )
        )

    checks.append(
        _evaluate(
            table["foot_line_deviation_deg"],
            "foot_line_deviation_deg",
            values.get("foot_line_deviation_deg", NAN),
            "FEET_NOT_IN_LINE",
            "Feet not spread in a straight line / V-shaped",
        )
    )
    checks.append(
        _evaluate(
            table["turnout_deviation_deg"],
            "turnout_deviation_deg",
            values.get("turnout_deviation_deg", NAN),
            "FEET_NOT_IN_LINE",
            "Feet not turned out symmetrically / V-shaped",
        )
    )
    checks.append(
        _evaluate(
            table["mean_knee_foot_alignment_deg"],
            "mean_knee_foot_alignment_deg",
            values.get("mean_knee_foot_alignment_deg", NAN),
            "KNEE_FOOT_MISALIGNED",
            "Knees not tracking over the turned-out feet",
            SECONDARY,
        )
    )
    checks.append(
        _evaluate(
            table["knee_asymmetry_deg"],
            "knee_asymmetry_deg",
            values.get("knee_asymmetry_deg", NAN),
            "KNEE_ASYMMETRY",
            "Knees bent unequally",
            SECONDARY,
        )
    )
    checks.append(
        _evaluate(
            table["trunk_inclination_deg"],
            "trunk_inclination_deg",
            values.get("trunk_inclination_deg", NAN),
            "TRUNK_NOT_UPRIGHT",
            "Trunk not held upright",
            SECONDARY,
        )
    )
    checks.append(
        _evaluate(
            table["pelvic_symmetry_deg"],
            "pelvic_symmetry_deg",
            values.get("pelvic_symmetry_deg", NAN),
            "PELVIS_NOT_LEVEL",
            "Pelvis not level",
            SECONDARY,
        )
    )
    return checks


def _samapadham_checks(values: dict[str, float]) -> list[Check]:
    table = POSTURE_THRESHOLDS[SAMAPADHAM]
    return [
        _evaluate(
            table["foot_gap_norm"],
            "foot_gap_norm",
            values.get("foot_gap_norm", NAN),
            "FEET_TOO_FAR_APART",
            "Feet too far apart",
        ),
        _evaluate(
            table["turnout_deviation_deg"],
            "turnout_deviation_deg",
            values.get("turnout_deviation_deg", NAN),
            "FEET_NOT_ALIGNED",
            "Feet not aligned / V-shaped",
        ),
        _evaluate(
            table["weight_shift_proxy"],
            "weight_shift_proxy",
            values.get("weight_shift_proxy", NAN),
            "WEIGHT_ON_ONE_LEG",
            "More weight on one leg (landmark-based loading proxy)",
        ),
        _evaluate(
            table["hip_tilt_deg"],
            "hip_tilt_deg",
            values.get("hip_tilt_deg", NAN),
            "WEIGHT_ON_ONE_LEG",
            "Pelvis dropped on one side (landmark-based loading proxy)",
        ),
        _evaluate(
            table["mean_knee_flexion_deg"],
            "mean_knee_flexion_deg",
            values.get("mean_knee_flexion_deg", NAN),
            "KNEES_NOT_STRAIGHT",
            "Knees not fully extended",
            SECONDARY,
        ),
        _evaluate(
            table["knee_asymmetry_deg"],
            "knee_asymmetry_deg",
            values.get("knee_asymmetry_deg", NAN),
            "KNEE_ASYMMETRY",
            "Knees extended unequally",
            SECONDARY,
        ),
        _evaluate(
            table["trunk_inclination_deg"],
            "trunk_inclination_deg",
            values.get("trunk_inclination_deg", NAN),
            "TRUNK_NOT_UPRIGHT",
            "Trunk not held upright",
            SECONDARY,
        ),
    ]


def _muzhumandi_checks(values: dict[str, float]) -> list[Check]:
    table = POSTURE_THRESHOLDS[MUZHUMANDI]
    checks = [
        _evaluate(
            table["turnout_deviation_deg"],
            "turnout_deviation_deg",
            values.get("turnout_deviation_deg", NAN),
            "KNEES_FACING_FORWARD",
            "Feet and knees facing forward",
        ),
        _evaluate(
            table["trunk_inclination_deg"],
            "trunk_inclination_deg",
            values.get("trunk_inclination_deg", NAN),
            "BACK_BENT_FORWARD",
            "Back bent forward",
        ),
        _evaluate(
            table["mean_heel_lift_norm"],
            "mean_heel_lift_norm",
            values.get("mean_heel_lift_norm", NAN),
            "FEET_APART_HEEL_DOWN",
            "Heel pressing on the floor",
        ),
        _evaluate(
            table["foot_separation_norm"],
            "foot_separation_norm",
            values.get("foot_separation_norm", NAN),
            "FEET_APART_HEEL_DOWN",
            "Feet far apart",
        ),
        _evaluate(
            table["mean_knee_flexion_deg"],
            "mean_knee_flexion_deg",
            values.get("mean_knee_flexion_deg", NAN),
            "INSUFFICIENT_SQUAT_DEPTH",
            "Squat not deep enough for Muzhumandi",
            SECONDARY,
        ),
        _evaluate(
            table["knee_asymmetry_deg"],
            "knee_asymmetry_deg",
            values.get("knee_asymmetry_deg", NAN),
            "KNEE_ASYMMETRY",
            "Knees bent unequally",
            SECONDARY,
        ),
    ]
    return checks


def _swastikapadam_checks(values: dict[str, float]) -> list[Check]:
    table = POSTURE_THRESHOLDS[SWASTIKAPADAM]
    crossing = values.get("ankle_crossing_signed_norm", NAN)

    overcross = _evaluate(
        table["overcrossing_limit_norm"],
        "overcrossing_limit_norm",
        crossing,
        "OVERCROSSING_SHINS",
        "Overcrossing shins",
    )
    not_crossed = _evaluate(
        table["ankle_crossing_signed_norm"],
        "ankle_crossing_signed_norm",
        crossing,
        "LEGS_NOT_CROSSED",
        "Legs not crossed into Swastikapadam",
        SECONDARY,
    )
    depth = _evaluate(
        table["mean_knee_flexion_deg"],
        "mean_knee_flexion_deg",
        values.get("mean_knee_flexion_deg", NAN),
        "INSUFFICIENT_ARAMANDI",
        "Insufficient Aramandi",
    )
    return [
        overcross,
        depth,
        not_crossed,
        _evaluate(
            table["shin_crossing_distance_norm"],
            "shin_crossing_distance_norm",
            values.get("shin_crossing_distance_norm", NAN),
            "SHINS_NOT_TOGETHER",
            "Shins not brought together",
            SECONDARY,
        ),
        _evaluate(
            table["trunk_inclination_deg"],
            "trunk_inclination_deg",
            values.get("trunk_inclination_deg", NAN),
            "TRUNK_NOT_UPRIGHT",
            "Trunk not held upright",
            SECONDARY,
        ),
    ]


def _mudra_checks(posture: str, values: dict[str, float]) -> list[Check]:
    table = POSTURE_THRESHOLDS[posture]
    checks: list[Check] = []

    if posture == PATAKA:
        checks += [
            _evaluate(
                table["mean_elbow_shoulder_alignment_norm"],
                "mean_elbow_shoulder_alignment_norm",
                values.get("mean_elbow_shoulder_alignment_norm", NAN),
                "ELBOWS_NOT_ALIGNED",
                "Elbows down / not aligned with shoulders",
            ),
            _evaluate(
                table["mean_interfinger_spread_deg"],
                "mean_interfinger_spread_deg",
                values.get("mean_interfinger_spread_deg", NAN),
                "FINGERS_TOO_FAR_APART",
                "Fingers too far apart",
            ),
            _evaluate(
                table["mean_adjacent_fingertip_gap_norm"],
                "mean_adjacent_fingertip_gap_norm",
                values.get("mean_adjacent_fingertip_gap_norm", NAN),
                "FINGERS_TOO_FAR_APART",
                "Fingertips too far apart",
            ),
            _evaluate(
                table["mean_finger_straightness_deg"],
                "mean_finger_straightness_deg",
                values.get("mean_finger_straightness_deg", NAN),
                "FINGERS_BENT",
                "Fingers not held straight",
                SECONDARY,
            ),
        ]
    elif posture == ALAPADMA:
        checks += [
            _evaluate(
                table["mean_elbow_shoulder_alignment_norm"],
                "mean_elbow_shoulder_alignment_norm",
                values.get("mean_elbow_shoulder_alignment_norm", NAN),
                "ELBOWS_NOT_ALIGNED",
                "Elbows not aligned with shoulders",
            ),
            _evaluate(
                table["mean_interfinger_spread_deg"],
                "mean_interfinger_spread_deg",
                values.get("mean_interfinger_spread_deg", NAN),
                "FINGERS_NOT_SPREAD",
                "Fingers not spread into the lotus shape",
                SECONDARY,
            ),
            _evaluate(
                table["palm_opening_norm"],
                "palm_opening_norm",
                values.get("palm_opening_norm", NAN),
                "PALM_NOT_OPEN",
                "Palm not fully opened",
                SECONDARY,
            ),
        ]
    elif posture == MUSHTI:
        checks += [
            _evaluate(
                table["pinky_relative_extension"],
                "pinky_relative_extension",
                values.get("pinky_relative_extension", NAN),
                "PINKY_DROPPED",
                "Pinky dropped / fingers loose",
            ),
            _evaluate(
                table["mean_finger_flexion_deg"],
                "mean_finger_flexion_deg",
                values.get("mean_finger_flexion_deg", NAN),
                "PINKY_DROPPED",
                "Fist not fully closed / fingers loose",
            ),
            _evaluate(
                table["palm_opening_norm"],
                "palm_opening_norm",
                values.get("palm_opening_norm", NAN),
                "FIST_NOT_CLOSED",
                "Palm still open",
                SECONDARY,
            ),
        ]
    elif posture == KATAKAMUKHA:
        checks += [
            _evaluate(
                table["ring_pinky_straightness_deg"],
                "ring_pinky_straightness_deg",
                values.get("ring_pinky_straightness_deg", NAN),
                "FINGERS_BENT",
                "Fingers bent",
            ),
            _evaluate(
                table["thumb_index_tip_distance_norm"],
                "thumb_index_tip_distance_norm",
                values.get("thumb_index_tip_distance_norm", NAN),
                "THUMB_NOT_JOINED",
                "Thumb not joined to the index and middle fingertips",
                SECONDARY,
            ),
            _evaluate(
                table["fingertip_cluster_spread_norm"],
                "fingertip_cluster_spread_norm",
                values.get("fingertip_cluster_spread_norm", NAN),
                "FINGERTIPS_NOT_CLUSTERED",
                "Fingertips not clustered",
                SECONDARY,
            ),
            _evaluate(
                table["mean_elbow_shoulder_alignment_norm"],
                "mean_elbow_shoulder_alignment_norm",
                values.get("mean_elbow_shoulder_alignment_norm", NAN),
                "ELBOWS_NOT_ALIGNED",
                "Elbows not aligned with shoulders",
                SECONDARY,
            ),
        ]
    return checks


_BODY_CHECKS = {
    ARAMANDI: _aramandi_checks,
    SAMAPADHAM: _samapadham_checks,
    MUZHUMANDI: _muzhumandi_checks,
    SWASTIKAPADAM: _swastikapadam_checks,
}
MUDRA_POSTURE_SET = {PATAKA, ALAPADMA, MUSHTI, KATAKAMUKHA}


def rule_values(posture: str, params: dict[str, float | str]) -> dict[str, float]:
    """Assemble every value the posture's rules need, mudra params included."""
    values = {k: _num(v) for k, v in params.items()}
    values.update(derived_rule_inputs(posture, params))

    if posture in MUDRA_POSTURE_SET:
        side = params.get("mudra_hand_used", "")
        side = side if isinstance(side, str) else ""
        for name in (
            "mean_finger_straightness_deg",
            "mean_finger_flexion_deg",
            "mean_interfinger_spread_deg",
            "mean_adjacent_fingertip_gap_norm",
            "max_adjacent_fingertip_gap_norm",
            "palm_opening_norm",
            "pinky_relative_extension",
            "thumb_index_tip_distance_norm",
            "fingertip_cluster_spread_norm",
            "ring_pinky_straightness_deg",
        ):
            values[name] = hand_param(params, side, name)
    return values


def evaluate(
    posture: str, params: dict[str, float | str], side: str = ""
) -> Verdict:
    """Judge a detected posture against the fixed parameter ranges."""
    if not posture:
        return Verdict(
            posture="",
            status=UNKNOWN,
            message="no posture recognised, cannot apply correctness rules",
        )

    values = rule_values(posture, params)
    if posture in _BODY_CHECKS:
        checks = _BODY_CHECKS[posture](values)
    elif posture in MUDRA_POSTURE_SET:
        checks = _mudra_checks(posture, values)
    else:
        return Verdict(posture=posture, status=UNKNOWN, message="no rules for posture")

    measured = [c for c in checks if c.band != UNKNOWN]
    if not measured:
        return Verdict(
            posture=posture,
            status=UNKNOWN,
            checks=checks,
            side=side,
            message="required parameters unavailable for this frame",
        )

    failures = [c for c in measured if c.failed]
    borderline = any(c.band == BORDERLINE for c in measured)

    if not failures:
        return Verdict(
            posture=posture,
            status=STATUS_CORRECT,
            checks=checks,
            borderline=borderline,
            side=side,
            primary_parameter=measured[0].parameter,
            primary_value=measured[0].value,
            primary_expected=measured[0].expected,
            message=f"{len(measured)} parameter checks within range",
        )

    # Canonical taxonomy errors win over secondary findings. Among equals the
    # most severe violation is reported (how far past its correct limit the
    # value sits, in borderline-band widths), with the rule order as tie-break.
    canonical = [c for c in failures if c.kind == CANONICAL]
    primary = max(canonical or failures, key=lambda c: c.severity)

    error_text = primary.error_text
    if side and primary.error_code in {"OVERCROSSING_SHINS", "INSUFFICIENT_ARAMANDI"}:
        error_text = f"{error_text} ({side.lower()} leg)"

    return Verdict(
        posture=posture,
        status=STATUS_INCORRECT,
        error_code=primary.error_code,
        error_text=error_text,
        error_kind=primary.kind,
        primary_parameter=primary.parameter,
        primary_value=primary.value,
        primary_expected=primary.expected,
        checks=checks,
        borderline=borderline,
        side=side,
        message=f"{len(failures)} of {len(measured)} parameter checks out of range",
    )
