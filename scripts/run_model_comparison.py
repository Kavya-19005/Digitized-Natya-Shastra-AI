"""Run the five pose-estimation approaches on the same 208 labelled images.

    python scripts/run_model_comparison.py
    python scripts/run_model_comparison.py --approaches rtmw rtmpose

This is an evaluation mode. It does not replace the RTMW Streamlit video path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVALUATION_DIR, ensure_output_dirs
from src.dataset import build_manifest
from src.evaluation.comparison import run_approach, summarise_approach, write_comparison
from src.pose.approaches import APPROACHES, get_approach


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approaches",
        nargs="*",
        default=[a.key for a in APPROACHES],
        help="Subset of: mediapipe openpose yolopose rtmpose rtmw",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    items = build_manifest()
    print(f"labelled images: {len(items)}")
    print(f"output directory: {EVALUATION_DIR}\n")

    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for key in args.approaches:
        approach = get_approach(key)
        ok, note = approach.probe()
        print(f"== {approach.display_name}")
        print(f"   probe: {'runnable' if ok else 'NOT RUNNABLE'} ({note})")
        if not ok:
            rows.append(summarise_approach(approach, None, False, note))
            continue
        try:
            backend = approach.factory()
        except Exception as exc:
            print(f"   init FAILED: {exc}")
            rows.append(
                summarise_approach(approach, None, False, note, init_error=str(exc))
            )
            continue
        df = run_approach(approach, backend, items)
        frames[approach.key] = df
        summary = summarise_approach(approach, df, True, note)
        rows.append(summary)
        print(
            f"   success {summary['successful_images']}/{summary['total_images']} | "
            f"landmarks {summary['mean_landmarks_available']} | "
            f"posture {summary['posture_accuracy']}% | "
            f"{summary['processing_time_per_image_ms']} ms/image"
        )
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    summary_df = pd.DataFrame(rows)
    written = write_comparison(summary_df, frames)
    print("\nwritten:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print("\ncomparison table:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
