"""Temporal smoothing for the video pipeline.

Deliberately simple: a rolling majority vote for the posture/status/error labels
and an exponential moving average for the displayed parameter values. That is
enough to stop the readout flickering frame to frame while still reacting within
a fraction of a second when the dancer genuinely changes posture. No tracker, no
state machine.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from ..correctness.thresholds import UNKNOWN


@dataclass
class SmoothedState:
    posture: str = ""
    status: str = UNKNOWN
    error_code: str = ""
    error_text: str = ""
    primary_parameter: str = ""
    primary_value: float = float("nan")
    primary_expected: str = ""
    parameters: dict[str, float] = field(default_factory=dict)
    stable: bool = False
    frames_incomplete: int = 0
    changed: bool = False


class TemporalSmoother:
    """Majority vote over a rolling window, plus EMA on numeric parameters.

    `min_agreement` is the fraction of the window that must agree before a new
    label is published; until then the previous stable label is retained, which
    is what keeps the UI steady during the transition between two postures.
    """

    def __init__(
        self,
        window: int = 9,
        min_agreement: float = 0.5,
        ema_alpha: float = 0.35,
        max_hold_frames: int = 45,
    ) -> None:
        self.window = max(1, window)
        self.min_agreement = min_agreement
        self.ema_alpha = ema_alpha
        self.max_hold_frames = max_hold_frames

        self._postures: deque[str] = deque(maxlen=self.window)
        self._verdicts: deque[tuple[str, str, str]] = deque(maxlen=self.window)
        self._values: dict[str, float] = {}
        self._state = SmoothedState()
        self._held = 0

    def reset(self) -> None:
        self._postures.clear()
        self._verdicts.clear()
        self._values.clear()
        self._state = SmoothedState()
        self._held = 0

    def update(
        self,
        posture: str,
        status: str,
        error_code: str,
        error_text: str,
        primary_parameter: str = "",
        primary_value: float = float("nan"),
        primary_expected: str = "",
        parameters: dict[str, float] | None = None,
        valid: bool = True,
    ) -> SmoothedState:
        previous_posture = self._state.posture
        previous_status = self._state.status

        if not valid:
            # A dropped frame must not blank the readout: keep the last stable
            # state, but stop holding it forever if detection never recovers.
            self._held += 1
            self._state.frames_incomplete += 1
            self._state.changed = False
            if self._held > self.max_hold_frames:
                self.reset()
            return self._state

        self._held = 0
        self._postures.append(posture)
        self._verdicts.append((posture, status, error_code))
        self._blend(parameters or {})

        posture_vote, posture_share = _vote(self._postures)
        if posture_share >= self.min_agreement and posture_vote:
            self._state.posture = posture_vote

        matching = [v for v in self._verdicts if v[0] == self._state.posture]
        if matching:
            status_vote, status_share = _vote([v[1] for v in matching])
            if status_share >= self.min_agreement and status_vote:
                self._state.status = status_vote
                errors = [v[2] for v in matching if v[1] == status_vote]
                error_vote, _ = _vote(errors)
                self._state.error_code = error_vote
                # keep the human-readable text aligned with the voted code
                if error_vote == error_code:
                    self._state.error_text = error_text
                elif not error_vote:
                    self._state.error_text = ""

        if posture == self._state.posture:
            self._state.primary_parameter = primary_parameter
            self._state.primary_expected = primary_expected
            self._state.primary_value = self._values.get(
                primary_parameter, primary_value
            )

        self._state.parameters = dict(self._values)
        self._state.stable = len(self._postures) >= self.window
        self._state.changed = (
            self._state.posture != previous_posture
            or self._state.status != previous_status
        )
        return self._state

    def _blend(self, parameters: dict[str, float]) -> None:
        for key, raw in parameters.items():
            if not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if not np.isfinite(value):
                continue
            if key in self._values:
                self._values[key] = (
                    self.ema_alpha * value + (1.0 - self.ema_alpha) * self._values[key]
                )
            else:
                self._values[key] = value


def _vote(values) -> tuple[str, float]:
    items = [v for v in values if v]
    if not items:
        return "", 0.0
    counter = Counter(items)
    label, count = counter.most_common(1)[0]
    return label, count / len(items)
