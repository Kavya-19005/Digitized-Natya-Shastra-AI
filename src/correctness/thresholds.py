"""Fixed parameter ranges for the correctness rules.

Provenance is recorded for every threshold and must stay honest:

* PROJECT_SPEC        - fixed by the project specification; never edited to make
                        sample data look better.
* ENGINEERING_DEFAULT - the specification did not fix this value, so the
                        implementation chose a defensible starting value.
* DATA_CALIBRATED     - derived from the 208 labelled images by
                        `scripts/calibrate_thresholds.py`. These are REPORTED
                        for the write-up and are never auto-applied on top of
                        the specification values.

Every angular threshold below refers to FLEXION where the parameter name says
flexion (flexion = 180 - raw joint angle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PROJECT_SPEC = "PROJECT_SPEC"
ENGINEERING_DEFAULT = "ENGINEERING_DEFAULT"
DATA_CALIBRATED = "DATA_CALIBRATED"

CORRECT = "CORRECT"
BORDERLINE = "BORDERLINE"
INCORRECT = "INCORRECT"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Threshold:
    """A three-band classifier for one parameter.

    kind:
      'max'   - smaller is better: correct <= correct_limit <= borderline_limit
      'min'   - larger is better:  correct >= correct_limit >= borderline_limit
      'range' - correct inside `correct_range`, borderline inside
                `borderline_range`, incorrect outside
    """

    parameter: str
    kind: str
    provenance: str
    correct_limit: float = float("nan")
    borderline_limit: float = float("nan")
    correct_range: tuple[float, float] = (float("nan"), float("nan"))
    borderline_range: tuple[float, float] = (float("nan"), float("nan"))
    unit: str = "deg"
    note: str = ""

    def classify(self, value: float) -> str:
        if value is None or not np.isfinite(value):
            return UNKNOWN
        if self.kind == "max":
            if value <= self.correct_limit:
                return CORRECT
            if value <= self.borderline_limit:
                return BORDERLINE
            return INCORRECT
        if self.kind == "min":
            if value >= self.correct_limit:
                return CORRECT
            if value >= self.borderline_limit:
                return BORDERLINE
            return INCORRECT
        if self.kind == "range":
            lo, hi = self.correct_range
            if lo <= value <= hi:
                return CORRECT
            blo, bhi = self.borderline_range
            if blo <= value <= bhi:
                return BORDERLINE
            return INCORRECT
        raise ValueError(f"unknown threshold kind: {self.kind}")

    def severity(self, value: float) -> float:
        """How far past the correct limit a value sits, in borderline-band widths.

        Used to pick the most serious violation when several canonical checks
        fail at once, instead of relying on a hand-written priority order.
        """
        if value is None or not np.isfinite(value):
            return float("nan")
        eps = 1e-9
        if self.kind == "max":
            band = max(self.borderline_limit - self.correct_limit, eps)
            return max(0.0, (value - self.correct_limit) / band)
        if self.kind == "min":
            band = max(self.correct_limit - self.borderline_limit, eps)
            return max(0.0, (self.correct_limit - value) / band)
        lo, hi = self.correct_range
        blo, bhi = self.borderline_range
        if value < lo:
            return max(0.0, (lo - value) / max(lo - blo, eps))
        if value > hi:
            return max(0.0, (value - hi) / max(bhi - hi, eps))
        return 0.0

    def expected_text(self) -> str:
        if self.kind == "max":
            return f"<= {self.correct_limit:g}{_unit_suffix(self.unit)}"
        if self.kind == "min":
            return f">= {self.correct_limit:g}{_unit_suffix(self.unit)}"
        lo, hi = self.correct_range
        return f"{lo:g}-{hi:g}{_unit_suffix(self.unit)}"


def _unit_suffix(unit: str) -> str:
    return "\u00b0" if unit == "deg" else ""


# ---------------------------------------------------------------------------
# Targets that the specification left open. Declared explicitly so the report
# can state that these are implementation choices, not tradition.
# ---------------------------------------------------------------------------
# DATA_CALIBRATED from the labelled correct Aramandi samples of the frontal
# camera (Cam 5): the reference stance measures ~0.60 shoulder widths between
# the ankle joints. The +/-5% / +/-15% BAND WIDTHS around it remain PROJECT_SPEC.
ARAMANDI_FOOT_SEPARATION_TARGET_NORM = 0.60  # shoulder widths

# DATA_CALIBRATED: with the feet actually touching (correct Samapadham) the two
# ANKLE JOINTS still sit ~0.30 shoulder widths apart, because an ankle joint is
# inset from the inner edge of the foot. The specification's "<= 0.05
# shoulder-width" separation therefore describes the GAP BETWEEN THE FEET, not
# the distance between ankle joints, so the rule is applied to
# `foot_gap_norm = foot_separation_norm - FEET_TOGETHER_BASELINE_NORM`.
FEET_TOGETHER_BASELINE_NORM = 0.30

MUZHUMANDI_FOOT_SEPARATION_MAX_NORM = 1.60  # shoulder widths
ARAMANDI_TARGET_TURNOUT_DEG = 45.0
SAMAPADHAM_TARGET_TURNOUT_DEG = 0.0
MUZHUMANDI_TARGET_TURNOUT_DEG = 45.0


@dataclass(frozen=True)
class PostureThresholds:
    posture: str
    thresholds: dict[str, Threshold] = field(default_factory=dict)

    def get(self, name: str) -> Threshold | None:
        return self.thresholds.get(name)


# ---------------------------------------------------------------------------
# ARAMANDI
# ---------------------------------------------------------------------------
ARAMANDI_THRESHOLDS = {
    "mean_knee_flexion_deg": Threshold(
        parameter="mean_knee_flexion_deg",
        kind="range",
        provenance=PROJECT_SPEC,
        correct_range=(30.0, 45.0),
        borderline_range=(25.0, 55.0),
        note=(
            "30-45 correct / 25-30 borderline / <25 incorrect are PROJECT_SPEC. "
            "The 45-55 upper borderline band is an ENGINEERING_DEFAULT: the "
            "specification lists no 'too deep' Aramandi error, deeper bends are "
            "recognised as Muzhumandi instead."
        ),
    ),
    "knee_asymmetry_deg": Threshold(
        parameter="knee_asymmetry_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=10.0,
        borderline_limit=15.0,
    ),
    "foot_separation_deviation_pct": Threshold(
        parameter="foot_separation_deviation_pct",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=5.0,
        borderline_limit=15.0,
        unit="pct",
        note=(
            "Band widths are PROJECT_SPEC. The target separation itself "
            f"({ARAMANDI_FOOT_SEPARATION_TARGET_NORM:g} shoulder widths) is an "
            "ENGINEERING_DEFAULT."
        ),
    ),
    "foot_line_deviation_deg": Threshold(
        parameter="foot_line_deviation_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=8.0,
        borderline_limit=15.0,
    ),
    "mean_knee_foot_alignment_deg": Threshold(
        parameter="mean_knee_foot_alignment_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=8.0,
        borderline_limit=15.0,
    ),
    "trunk_inclination_deg": Threshold(
        parameter="trunk_inclination_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=5.0,
        borderline_limit=10.0,
    ),
    "pelvic_symmetry_deg": Threshold(
        parameter="pelvic_symmetry_deg",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=8.0,
        borderline_limit=15.0,
        note="Pelvic symmetry band mirrors the foot-line band; not fixed by spec.",
    ),
    "turnout_deviation_deg": Threshold(
        parameter="turnout_deviation_deg",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=20.0,
        borderline_limit=32.0,
        note=(
            "Deviation from a "
            f"{ARAMANDI_TARGET_TURNOUT_DEG:g}deg turnout target. Wider than the "
            "8/15 orientation band because monocular turnout estimation from a "
            "heel-to-toe vector is noisy."
        ),
    ),
}

# ---------------------------------------------------------------------------
# SAMAPADHAM
# ---------------------------------------------------------------------------
SAMAPADHAM_THRESHOLDS = {
    "mean_knee_flexion_deg": Threshold(
        parameter="mean_knee_flexion_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=5.0,
        borderline_limit=10.0,
    ),
    "foot_gap_norm": Threshold(
        parameter="foot_gap_norm",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=0.05,
        borderline_limit=0.15,
        unit="shoulder_width",
        note=(
            "PROJECT_SPEC bands, applied to the gap between the feet. The "
            f"feet-together baseline ({FEET_TOGETHER_BASELINE_NORM:g} shoulder "
            "widths of ankle-to-ankle distance) is DATA_CALIBRATED."
        ),
    ),
    "turnout_deviation_deg": Threshold(
        parameter="turnout_deviation_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=8.0,
        borderline_limit=15.0,
        note=(
            "Foot orientation deviation from the "
            f"{SAMAPADHAM_TARGET_TURNOUT_DEG:g}deg (parallel, forward) target."
        ),
    ),
    "knee_asymmetry_deg": Threshold(
        parameter="knee_asymmetry_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=5.0,
        borderline_limit=10.0,
    ),
    "trunk_inclination_deg": Threshold(
        parameter="trunk_inclination_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=5.0,
        borderline_limit=10.0,
    ),
    # Landmark-based loading PROXY -- NOT a force measurement.
    "weight_shift_proxy": Threshold(
        parameter="weight_shift_proxy",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.10,
        borderline_limit=0.18,
        unit="shoulder_width",
        note=(
            "The specification deliberately fixes no numeric band here. This is "
            "a landmark-based lateral weight-shift PROXY (offset of the "
            "shoulder/pelvis midline from the ankle midline); it does not "
            "measure ground reaction force."
        ),
    ),
    "hip_tilt_deg": Threshold(
        parameter="hip_tilt_deg",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=6.0,
        borderline_limit=12.0,
        note="Secondary loading proxy used together with weight_shift_proxy.",
    ),
}

# ---------------------------------------------------------------------------
# MUZHUMANDI
# ---------------------------------------------------------------------------
MUZHUMANDI_THRESHOLDS = {
    "mean_knee_flexion_deg": Threshold(
        parameter="mean_knee_flexion_deg",
        kind="min",
        provenance=PROJECT_SPEC,
        correct_limit=120.0,
        borderline_limit=105.0,
    ),
    "knee_asymmetry_deg": Threshold(
        parameter="knee_asymmetry_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=10.0,
        borderline_limit=15.0,
    ),
    "turnout_deviation_deg": Threshold(
        parameter="turnout_deviation_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=8.0,
        borderline_limit=15.0,
        note=(
            "Knee/foot orientation deviation from the "
            f"{MUZHUMANDI_TARGET_TURNOUT_DEG:g}deg turnout target; detects the "
            "'feet and knees facing forward' error."
        ),
    ),
    "trunk_inclination_deg": Threshold(
        parameter="trunk_inclination_deg",
        kind="max",
        provenance=PROJECT_SPEC,
        correct_limit=8.0,
        borderline_limit=15.0,
    ),
    "mean_heel_lift_norm": Threshold(
        parameter="mean_heel_lift_norm",
        kind="min",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.04,
        borderline_limit=0.015,
        unit="shoulder_width",
        note=(
            "The specification says to 'check whether the required heel "
            "configuration is maintained' without a number. Heel lift is "
            "measured as the toe-to-heel vertical offset normalised by shoulder "
            "width; a heel pressed on the floor gives ~0."
        ),
    ),
    "foot_separation_norm": Threshold(
        parameter="foot_separation_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=MUZHUMANDI_FOOT_SEPARATION_MAX_NORM,
        borderline_limit=MUZHUMANDI_FOOT_SEPARATION_MAX_NORM + 0.35,
        unit="shoulder_width",
        note="Supports the 'feet far apart and heel pressing on floor' error.",
    ),
}

# ---------------------------------------------------------------------------
# SWASTIKAPADAM
# There is no universally published crossing-angle threshold. The bands below
# are ENGINEERING_DEFAULTs; `scripts/calibrate_thresholds.py` reports what the
# labelled correct/incorrect samples actually measure.
# ---------------------------------------------------------------------------
SWASTIKAPADAM_THRESHOLDS = {
    "shin_crossing_distance_norm": Threshold(
        parameter="shin_crossing_distance_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.35,
        borderline_limit=0.55,
        unit="shoulder_width",
        note="Legs must actually cross; a large gap means they never met.",
    ),
    "ankle_crossing_signed_norm": Threshold(
        parameter="ankle_crossing_signed_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.25,
        borderline_limit=0.45,
        unit="shoulder_width",
        note=(
            "Signed ankle gap; strongly negative values indicate OVERCROSSED "
            "shins. Interpreted from the labelled samples, not from literature."
        ),
    ),
    "overcrossing_limit_norm": Threshold(
        parameter="ankle_crossing_signed_norm",
        kind="min",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=-0.55,
        borderline_limit=-0.80,
        unit="shoulder_width",
        note="Lower bound on the signed ankle gap: below it the shins overcross.",
    ),
    "mean_knee_flexion_deg": Threshold(
        parameter="mean_knee_flexion_deg",
        kind="range",
        provenance=PROJECT_SPEC,
        correct_range=(30.0, 45.0),
        borderline_range=(25.0, 60.0),
        note=(
            "Swastikapadam is held in Aramandi depth, so the Aramandi "
            "PROJECT_SPEC flexion range is reused to detect 'insufficient "
            "Aramandi'."
        ),
    ),
    "trunk_inclination_deg": Threshold(
        parameter="trunk_inclination_deg",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=8.0,
        borderline_limit=15.0,
    ),
    "shin_crossing_angle_deg": Threshold(
        parameter="shin_crossing_angle_deg",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=35.0,
        borderline_limit=50.0,
        note=(
            "Reported for the record. No universal published crossing-angle "
            "threshold exists, so this band is NOT used to fail a posture on "
            "its own."
        ),
    ),
}

# ---------------------------------------------------------------------------
# MUDRA thresholds
# ---------------------------------------------------------------------------
FINGER_STRAIGHTNESS = Threshold(
    parameter="mean_finger_straightness_deg",
    kind="min",
    provenance=PROJECT_SPEC,
    correct_limit=160.0,
    borderline_limit=145.0,
)
ELBOW_SHOULDER_ALIGNMENT = Threshold(
    parameter="mean_elbow_shoulder_alignment_norm",
    kind="max",
    provenance=PROJECT_SPEC,
    correct_limit=0.08,
    borderline_limit=0.15,
    unit="shoulder_width",
)

PATAKA_THRESHOLDS = {
    "mean_finger_straightness_deg": FINGER_STRAIGHTNESS,
    "mean_elbow_shoulder_alignment_norm": ELBOW_SHOULDER_ALIGNMENT,
    "mean_interfinger_spread_deg": Threshold(
        parameter="mean_interfinger_spread_deg",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=10.0,
        borderline_limit=16.0,
        note="Pataka keeps the fingers together; spread detects 'fingers too far apart'.",
    ),
    "mean_adjacent_fingertip_gap_norm": Threshold(
        parameter="mean_adjacent_fingertip_gap_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.30,
        borderline_limit=0.45,
        unit="palm_size",
    ),
}

ALAPADMA_THRESHOLDS = {
    "mean_elbow_shoulder_alignment_norm": ELBOW_SHOULDER_ALIGNMENT,
    "mean_interfinger_spread_deg": Threshold(
        parameter="mean_interfinger_spread_deg",
        kind="min",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=12.0,
        borderline_limit=8.0,
        note="Alapadma is a fully splayed lotus hand: fingers must be spread.",
    ),
    "palm_opening_norm": Threshold(
        parameter="palm_opening_norm",
        kind="min",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=1.30,
        borderline_limit=1.05,
        unit="palm_size",
    ),
}

MUSHTI_THRESHOLDS = {
    "mean_finger_flexion_deg": Threshold(
        parameter="mean_finger_flexion_deg",
        kind="min",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=55.0,
        borderline_limit=38.0,
        note="Mushti is a closed fist, so the long fingers must be strongly flexed.",
    ),
    "palm_opening_norm": Threshold(
        parameter="palm_opening_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.95,
        borderline_limit=1.15,
        unit="palm_size",
    ),
    "pinky_relative_extension": Threshold(
        parameter="pinky_relative_extension",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=1.25,
        borderline_limit=1.45,
        unit="ratio",
        note=(
            "Pinky tip-to-palm distance divided by the mean of the other three "
            "fingers; detects 'pinky dropped / fingers loose'."
        ),
    ),
}

KATAKAMUKHA_THRESHOLDS = {
    "thumb_index_tip_distance_norm": Threshold(
        parameter="thumb_index_tip_distance_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.45,
        borderline_limit=0.70,
        unit="palm_size",
        note="Katakamukha joins the thumb to the index and middle fingertips.",
    ),
    "fingertip_cluster_spread_norm": Threshold(
        parameter="fingertip_cluster_spread_norm",
        kind="max",
        provenance=ENGINEERING_DEFAULT,
        correct_limit=0.35,
        borderline_limit=0.55,
        unit="palm_size",
    ),
    "ring_pinky_straightness_deg": Threshold(
        parameter="ring_pinky_straightness_deg",
        kind="min",
        provenance=PROJECT_SPEC,
        correct_limit=160.0,
        borderline_limit=145.0,
        note=(
            "The ring and little fingers stay extended in Katakamukha, so the "
            "PROJECT_SPEC finger-straightness range applies to them; a drop "
            "below it is the 'fingers bent' error."
        ),
    ),
    "mean_elbow_shoulder_alignment_norm": ELBOW_SHOULDER_ALIGNMENT,
}

POSTURE_THRESHOLDS: dict[str, dict[str, Threshold]] = {
    "ARAMANDI": ARAMANDI_THRESHOLDS,
    "SAMAPADHAM": SAMAPADHAM_THRESHOLDS,
    "MUZHUMANDI": MUZHUMANDI_THRESHOLDS,
    "SWASTIKAPADAM": SWASTIKAPADAM_THRESHOLDS,
    "PATAKA": PATAKA_THRESHOLDS,
    "ALAPADMA": ALAPADMA_THRESHOLDS,
    "MUSHTI": MUSHTI_THRESHOLDS,
    "KATAKAMUKHA": KATAKAMUKHA_THRESHOLDS,
}

TARGET_TURNOUT_DEG = {
    "ARAMANDI": ARAMANDI_TARGET_TURNOUT_DEG,
    "SAMAPADHAM": SAMAPADHAM_TARGET_TURNOUT_DEG,
    "MUZHUMANDI": MUZHUMANDI_TARGET_TURNOUT_DEG,
    "SWASTIKAPADAM": ARAMANDI_TARGET_TURNOUT_DEG,
}


def export_rows() -> list[dict[str, object]]:
    """Flatten every threshold for `outputs/image_results/parameter_thresholds_*.csv`."""
    rows: list[dict[str, object]] = []
    for posture, table in POSTURE_THRESHOLDS.items():
        for key, th in table.items():
            rows.append(
                {
                    "posture": posture,
                    "rule_key": key,
                    "parameter": th.parameter,
                    "kind": th.kind,
                    "unit": th.unit,
                    "correct_limit": th.correct_limit,
                    "borderline_limit": th.borderline_limit,
                    "correct_range_low": th.correct_range[0],
                    "correct_range_high": th.correct_range[1],
                    "borderline_range_low": th.borderline_range[0],
                    "borderline_range_high": th.borderline_range[1],
                    "provenance": th.provenance,
                    "note": th.note,
                }
            )
    return rows
