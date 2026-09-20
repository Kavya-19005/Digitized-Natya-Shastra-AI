"""Process a video end-to-end and write an annotated MP4 plus per-frame CSVs.

    python scripts/run_video_analysis.py "path/to/video.mp4"
    python scripts/run_video_analysis.py video.mp4 --model rtmw_l_wholebody --stride 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SMOOTHING_WINDOW, ensure_output_dirs
from src.pose.registry import DEFAULT_MODEL_KEY, MODEL_SPECS, get_spec, missing_models
from src.video.processor import VideoProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list(MODEL_SPECS))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--smoothing", type=int, default=SMOOTHING_WINDOW)
    parser.add_argument("--detector-interval", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    missing = missing_models()
    if missing:
        print("MISSING MODEL FILES (run scripts/download_models.py):")
        for item in missing:
            print("  -", item)
        return 1

    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}")
        return 1

    ensure_output_dirs()
    spec = get_spec(args.model)
    processor = VideoProcessor(
        spec,
        frame_stride=args.stride,
        smoothing_window=args.smoothing,
        detector_interval=args.detector_interval,
    )

    print(f"video : {video.name}")
    print(f"model : {spec.display_name}")
    print(f"stride: every {args.stride} frame(s)\n")

    last = [0]

    def progress(done: int, total: int) -> None:
        if done - last[0] >= 25:
            last[0] = done
            print(f"  {done}/{total} frames")

    if args.max_frames:
        # bounded run for a quick demonstration clip
        summary = _process_limited(processor, video, args.max_frames, args.output_name)
    else:
        summary = processor.process(
            video, output_name=args.output_name, progress=progress
        )

    print(f"\nframes processed : {summary.frames_total}")
    print(f"frames analysed  : {summary.frames_analysed}")
    print(f"frames incomplete: {summary.frames_incomplete}")
    print(f"posture changes  : {summary.posture_changes}")
    print(f"speed            : {summary.fps_processed:.1f} analysed fps")
    print(f"annotated video  : {summary.output_video}")
    print(f"per-frame CSV    : {summary.output_csv}")

    if summary.segments:
        print("\nposture timeline:")
        for segment in summary.segments:
            issue = f" | {segment['error']}" if segment["error"] else ""
            print(
                f"  t={segment['timestamp_s']:>6.2f}s  {segment['posture']:<14}"
                f"{segment['status']:<10}{issue}"
            )
    return 0


def _process_limited(processor, video: Path, max_frames: int, output_name: str | None):
    """Same as `process` but stops after `max_frames` analysed frames."""
    import cv2
    import pandas as pd

    from src.config import VIDEO_RESULTS_DIR
    from src.video.processor import VideoSummary

    stem = output_name or f"{video.stem}_{processor.spec.key}"
    capture = cv2.VideoCapture(str(video))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()

    writer = None
    rows, summary = [], VideoSummary()
    previous_label = ("", "")
    import time

    started = time.perf_counter()

    for outcome in processor.iter_frames(video):
        summary.frames_total += 1
        if outcome.result.detection_status == "OK":
            summary.frames_analysed += 1
        else:
            summary.frames_incomplete += 1
        rows.append(outcome.row)

        label = (outcome.state.posture, outcome.state.status)
        if outcome.state.posture and label != previous_label:
            if previous_label != ("", ""):
                summary.posture_changes += 1
            summary.segments.append(
                {
                    "frame_index": outcome.frame_index,
                    "timestamp_s": round(outcome.timestamp_s, 2),
                    "posture": outcome.state.posture,
                    "status": outcome.state.status,
                    "error": outcome.state.error_text,
                    "primary_parameter": outcome.state.primary_parameter,
                    "primary_value": outcome.state.primary_value,
                }
            )
            previous_label = label

        if outcome.frame is not None:
            if writer is None:
                h, w = outcome.frame.shape[:2]
                target = VIDEO_RESULTS_DIR / f"{stem}_annotated.mp4"
                writer = cv2.VideoWriter(
                    str(target),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, source_fps / processor.frame_stride),
                    (w, h),
                )
                summary.output_video = target
            writer.write(outcome.frame)

        if summary.frames_total % 25 == 0:
            print(f"  {summary.frames_total}/{max_frames} frames")
        if summary.frames_total >= max_frames:
            break

    if writer is not None:
        writer.release()
    summary.elapsed_s = time.perf_counter() - started
    summary.output_csv = VIDEO_RESULTS_DIR / f"{stem}_frames.csv"
    pd.DataFrame(rows).to_csv(summary.output_csv, index=False)
    if summary.segments:
        pd.DataFrame(summary.segments).to_csv(
            VIDEO_RESULTS_DIR / f"{stem}_segments.csv", index=False
        )
    return summary


if __name__ == "__main__":
    sys.exit(main())
