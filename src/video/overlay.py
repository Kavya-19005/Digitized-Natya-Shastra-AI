"""Skeleton and feedback overlay drawing for the processed video."""

from __future__ import annotations

import cv2
import numpy as np

from ..correctness.thresholds import UNKNOWN
from ..labels import STATUS_CORRECT, STATUS_INCORRECT
from ..landmarks import common as C
from ..landmarks.common import Landmarks
from .smoothing import SmoothedState

GREEN = (80, 220, 100)
RED = (70, 70, 235)
AMBER = (60, 190, 240)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
PANEL = (32, 30, 28)

BODY_EDGES = (
    (C.LEFT_SHOULDER, C.RIGHT_SHOULDER),
    (C.LEFT_SHOULDER, C.LEFT_ELBOW),
    (C.RIGHT_SHOULDER, C.RIGHT_ELBOW),
    (C.LEFT_SHOULDER, C.LEFT_HIP),
    (C.RIGHT_SHOULDER, C.RIGHT_HIP),
    (C.LEFT_HIP, C.RIGHT_HIP),
    (C.LEFT_HIP, C.LEFT_KNEE),
    (C.RIGHT_HIP, C.RIGHT_KNEE),
    (C.LEFT_KNEE, C.LEFT_ANKLE),
    (C.RIGHT_KNEE, C.RIGHT_ANKLE),
    (C.LEFT_ANKLE, C.LEFT_HEEL),
    (C.RIGHT_ANKLE, C.RIGHT_HEEL),
    (C.LEFT_HEEL, C.LEFT_FOOT_INDEX),
    (C.RIGHT_HEEL, C.RIGHT_FOOT_INDEX),
    (C.LEFT_ANKLE, C.LEFT_FOOT_INDEX),
    (C.RIGHT_ANKLE, C.RIGHT_FOOT_INDEX),
)

HAND_EDGES = tuple(
    (chain[i], chain[i + 1])
    for chain in C.FINGER_CHAINS.values()
    for i in range(len(chain) - 1)
) + tuple((C.H_WRIST, chain[0]) for chain in C.FINGER_CHAINS.values())


def status_colour(status: str) -> tuple[int, int, int]:
    if status == STATUS_CORRECT:
        return GREEN
    if status == STATUS_INCORRECT:
        return RED
    return GREY


def draw_skeleton(
    frame: np.ndarray, lm: Landmarks, threshold: float, colour=(230, 190, 70)
) -> np.ndarray:
    mask = lm.available(threshold)
    thickness = max(1, int(round(min(frame.shape[:2]) / 400)))
    radius = max(2, thickness + 1)

    for a, b in BODY_EDGES:
        if mask[a] and mask[b]:
            cv2.line(
                frame,
                tuple(np.int32(lm.xy[a])),
                tuple(np.int32(lm.xy[b])),
                colour,
                thickness,
                cv2.LINE_AA,
            )
    for index in range(C.LEFT_HAND_START):
        if mask[index]:
            cv2.circle(frame, tuple(np.int32(lm.xy[index])), radius, WHITE, -1, cv2.LINE_AA)

    for base in (C.LEFT_HAND_START, C.RIGHT_HAND_START):
        for a, b in HAND_EDGES:
            ia, ib = base + a, base + b
            if mask[ia] and mask[ib]:
                cv2.line(
                    frame,
                    tuple(np.int32(lm.xy[ia])),
                    tuple(np.int32(lm.xy[ib])),
                    AMBER,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )
        for offset in range(21):
            index = base + offset
            if mask[index]:
                cv2.circle(
                    frame, tuple(np.int32(lm.xy[index])), max(1, radius - 1), AMBER, -1
                )
    return frame


def draw_feedback_panel(
    frame: np.ndarray,
    state: SmoothedState,
    model_name: str = "",
    frame_index: int = 0,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    """Draw the live posture / status / parameter readout."""
    h, w = frame.shape[:2]
    scale = w / 1280.0
    panel_w = int(430 * scale)
    panel_h = int(275 * scale)
    x0, y0 = int(18 * scale), int(18 * scale)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    colour = status_colour(state.status)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), colour, max(1, int(2 * scale)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = int(16 * scale)
    y = y0 + int(34 * scale)

    def put(text: str, size: float, col, bold: int = 1, step: float = 30.0) -> None:
        nonlocal y
        # the Hershey fonts have no degree glyph, so spell it out when drawing
        drawable = text.replace("\u00b0", " deg")
        cv2.putText(
            frame, drawable, (x0 + pad, y), font, size * scale, col, bold, cv2.LINE_AA
        )
        y += int(step * scale)

    put("DETECTED POSTURE", 0.46, GREY, 1, 26)
    put(state.posture or "-- searching --", 0.85, WHITE, 2, 40)
    put(f"STATUS: {state.status if state.posture else UNKNOWN}", 0.68, colour, 2, 34)

    if state.error_text:
        put("ISSUE", 0.44, GREY, 1, 22)
        for line in _wrap(state.error_text, 34):
            put(line, 0.55, AMBER, 1, 24)

    if state.primary_parameter and np.isfinite(state.primary_value):
        put(_pretty(state.primary_parameter), 0.44, GREY, 1, 22)
        put(
            f"{state.primary_value:.1f}   (expected {state.primary_expected})",
            0.56,
            WHITE,
            1,
            26,
        )

    for line in extra_lines or []:
        put(line, 0.44, GREY, 1, 20)

    footer = f"{model_name}   frame {frame_index}"
    if not state.stable:
        footer += "   [warming up]"
    cv2.putText(
        frame,
        footer,
        (x0 + pad, y0 + panel_h - int(10 * scale)),
        font,
        0.40 * scale,
        GREY,
        1,
        cv2.LINE_AA,
    )
    return frame


def _pretty(name: str) -> str:
    return name.replace("_deg", " (deg)").replace("_norm", " (norm)").replace("_", " ").upper()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]
