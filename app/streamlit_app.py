"""Digitized Natya Shastra - Bharatanatyam posture analysis UI.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SMOOTHING_WINDOW, VIDEO_RESULTS_DIR, ensure_output_dirs
from src.correctness.thresholds import POSTURE_THRESHOLDS
from src.labels import STATUS_CORRECT, STATUS_INCORRECT
from src.pose.registry import MODEL_SPECS, available_models, missing_models
from src.recognition.posture import TRANSITION
from src.video.processor import TRACKED_PARAMETERS, VideoProcessor

st.set_page_config(
    page_title="Digitized Natya Shastra", page_icon="\U0001f483", layout="wide"
)

PARAMETER_LABELS = {
    "mean_knee_flexion_deg": "Mean knee flexion",
    "left_knee_flexion_deg": "Left knee flexion",
    "right_knee_flexion_deg": "Right knee flexion",
    "knee_asymmetry_deg": "Knee asymmetry",
    "foot_separation_norm": "Foot separation (shoulder widths)",
    "mean_foot_turnout_deg": "Foot turnout",
    "foot_line_deviation_deg": "Foot-line deviation",
    "trunk_inclination_deg": "Trunk inclination",
    "mean_heel_lift_norm": "Heel lift (shoulder widths)",
    "weight_shift_proxy": "Weight-shift proxy",
    "ankle_crossing_signed_norm": "Ankle crossing (signed)",
    "shin_crossing_distance_norm": "Shin crossing distance",
    "shin_crossing_angle_deg": "Shin crossing angle",
    "mean_elbow_shoulder_alignment_norm": "Elbow-shoulder alignment",
    "hip_height_norm": "Hip height (shoulder widths)",
    "hip_height_torso_norm": "Squat depth (hip height / torso)",
}


def status_banner(posture: str, status: str, error_text: str) -> None:
    if not posture:
        st.info("Searching for the dancer...")
        return
    if posture == TRANSITION:
        st.warning(
            "### Detected posture: **TRANSITION / UNCERTAIN**\n\n"
            "The dancer is between postures, so no correctness verdict is given "
            "for this frame."
        )
        return
    headline = f"### Detected posture: **{posture}**"
    if status == STATUS_CORRECT:
        st.success(f"{headline}\n\n**Status: CORRECT**")
    elif status == STATUS_INCORRECT:
        st.error(
            f"{headline}\n\n**Status: INCORRECT**"
            + (f"\n\n**Issue:** {error_text}" if error_text else "")
        )
    else:
        st.warning(f"{headline}\n\n**Status: UNKNOWN** (parameters unavailable)")


def expected_for(posture: str, parameter: str) -> str:
    table = POSTURE_THRESHOLDS.get(posture, {})
    for threshold in table.values():
        if threshold.parameter == parameter:
            return threshold.expected_text()
    return "-"


def main() -> None:
    ensure_output_dirs()
    st.title("Digitized Natya Shastra")
    st.caption(
        "AI-assisted Bharatanatyam posture analysis - ONNX pose estimation, "
        "geometric parameters and fixed correctness rules, frame by frame."
    )

    specs = available_models()
    if not specs:
        st.error(
            "No ONNX models found. Run `python scripts/download_models.py`.\n\n"
            + "\n".join(f"- missing: {item}" for item in missing_models())
        )
        return

    with st.sidebar:
        st.header("Analysis settings")
        labels = {spec.display_name: spec.key for spec in specs}
        chosen = st.selectbox("ONNX pose model", list(labels))
        model_key = labels[chosen]
        st.caption(MODEL_SPECS[model_key].notes)

        frame_stride = st.slider(
            "Analyse every Nth frame", 1, 10, 3,
            help="Higher values process faster on CPU.",
        )
        smoothing_window = st.slider(
            "Temporal smoothing window (frames)", 1, 21, SMOOTHING_WINDOW
        )
        detector_interval = st.slider(
            "Re-run person detector every N frames", 1, 30, 8,
            help="The person box is reused in between, which saves most of the time.",
        )
        preview_every = st.slider("Refresh preview every N analysed frames", 1, 10, 2)
        write_video = st.checkbox("Save annotated video", value=True)

    uploaded = st.file_uploader(
        "Upload a dance video", type=["mp4", "mov", "avi", "mkv", "m4v"]
    )
    start = st.button("Start analysis", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.info(
            "Upload a video to begin. The readout updates frame by frame and "
            "changes as the dancer moves between postures."
        )
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getbuffer())
        video_path = Path(handle.name)

    if not start:
        st.video(str(video_path))
        return

    processor = VideoProcessor(
        MODEL_SPECS[model_key],
        frame_stride=frame_stride,
        smoothing_window=smoothing_window,
        detector_interval=detector_interval,
    )

    left, right = st.columns([3, 2])
    frame_slot = left.empty()
    banner_slot = right.empty()
    metric_slot = right.empty()
    table_slot = right.empty()
    progress_slot = st.empty()
    timeline_slot = st.empty()

    rows: list[dict[str, object]] = []
    segments: list[dict[str, object]] = []
    previous_label = ("", "")
    analysed = 0

    for outcome in processor.iter_frames(video_path):
        analysed += 1
        rows.append(outcome.row)
        state = outcome.state

        label = (state.posture, state.status)
        if state.posture and label != previous_label:
            segments.append(
                {
                    "time (s)": round(outcome.timestamp_s, 2),
                    "posture": state.posture,
                    "status": state.status,
                    "issue": state.error_text or "-",
                }
            )
            previous_label = label

        if analysed % preview_every == 0 or analysed == 1:
            if outcome.frame is not None:
                frame_slot.image(
                    outcome.frame[:, :, ::-1], caption=f"frame {outcome.frame_index}",
                    use_container_width=True,
                )
            with banner_slot.container():
                status_banner(state.posture, state.status, state.error_text)

            if state.primary_parameter and np.isfinite(state.primary_value):
                metric_slot.metric(
                    PARAMETER_LABELS.get(
                        state.primary_parameter, state.primary_parameter
                    ),
                    f"{state.primary_value:.1f}",
                    help=f"Expected {state.primary_expected}",
                )

            live = []
            for name in TRACKED_PARAMETERS:
                value = state.parameters.get(name, np.nan)
                if not np.isfinite(value):
                    continue
                live.append(
                    {
                        "Parameter": PARAMETER_LABELS.get(name, name),
                        "Value": round(float(value), 2),
                        "Expected": expected_for(state.posture, name),
                    }
                )
            if live:
                table_slot.dataframe(
                    pd.DataFrame(live), hide_index=True, use_container_width=True
                )
            progress_slot.caption(
                f"analysed {analysed} frames | "
                f"{outcome.result.detect_ms + outcome.result.pose_ms:.0f} ms/frame | "
                f"view {outcome.row.get('view_class', '-')}"
            )
            if segments:
                timeline_slot.dataframe(
                    pd.DataFrame(segments[::-1]), hide_index=True,
                    use_container_width=True,
                )

    frames = pd.DataFrame(rows)
    stem = f"{Path(uploaded.name).stem}_{model_key}"
    csv_path = VIDEO_RESULTS_DIR / f"{stem}_frames.csv"
    frames.to_csv(csv_path, index=False)

    st.success(
        f"Analysed {len(frames)} frames. "
        f"Per-frame results written to `{csv_path}`."
    )
    if write_video:
        st.caption(
            "Tip: `python scripts/run_video_analysis.py <video>` writes an "
            "annotated MP4 of the same analysis."
        )

    st.subheader("Posture timeline")
    st.dataframe(pd.DataFrame(segments), hide_index=True, use_container_width=True)

    plot_columns = [
        f"{name}_smoothed"
        for name in ("mean_knee_flexion_deg", "trunk_inclination_deg")
        if f"{name}_smoothed" in frames
    ]
    if plot_columns:
        st.subheader("Parameter traces")
        st.line_chart(frames.set_index("timestamp_s")[plot_columns])

    st.download_button(
        "Download per-frame CSV",
        frames.to_csv(index=False).encode(),
        file_name=csv_path.name,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
