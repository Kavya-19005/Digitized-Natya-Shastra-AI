"""Run the image pipeline over the labelled college images.

    python scripts/run_image_analysis.py                  # replay cached landmarks
    python scripts/run_image_analysis.py --full           # re-run ONNX inference too
    python scripts/run_image_analysis.py --models rtmw_l_wholebody --cameras "Cam 5"

By default the analysis stages replay the landmark caches written by
`scripts/extract_landmarks.py`, so parameters and rules can be re-run in seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CORE_CAMERAS, IMAGE_RESULTS_DIR, ensure_output_dirs
from src.correctness.thresholds import export_rows
from src.dataset import build_manifest, write_manifest
from src.evaluation.image_runner import run_model, run_model_from_cache, write_results
from src.evaluation.summaries import write_all
from src.pose.registry import MODEL_SPECS, get_spec, missing_models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    parser.add_argument("--cameras", nargs="*", default=list(CORE_CAMERAS))
    parser.add_argument("--no-summaries", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="re-run ONNX inference instead of replaying the landmark cache",
    )
    args = parser.parse_args()

    if args.full:
        missing = missing_models()
        if missing:
            print("MISSING MODEL FILES (run scripts/download_models.py):")
            for item in missing:
                print("  -", item)
            return 1

    ensure_output_dirs()
    manifest_path = write_manifest()
    print(f"manifest: {manifest_path}")

    items = [item for item in build_manifest() if item.camera in args.cameras]
    print(f"images: {len(items)} from {', '.join(args.cameras)}")

    pd.DataFrame(export_rows()).to_csv(
        IMAGE_RESULTS_DIR / "parameter_thresholds_used.csv", index=False
    )

    frames: dict[str, pd.DataFrame] = {}
    for key in args.models:
        spec = get_spec(key)
        print(f"\n== {spec.display_name}")
        if args.full:
            df = run_model(spec, items)
        else:
            df = run_model_from_cache(spec)
            df = df[df["camera"].isin(args.cameras)].reset_index(drop=True)
        path = write_results(df, spec)
        frames[key] = df
        print(f"   -> {path}")
        print(
            f"   posture accuracy {100 * df['posture_correct'].mean():.1f}% | "
            f"status accuracy {100 * df['status_correct'].mean():.1f}%"
        )

    if not args.no_summaries:
        written = write_all(frames)
        print("\nsummaries:")
        for name, path in written.items():
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
