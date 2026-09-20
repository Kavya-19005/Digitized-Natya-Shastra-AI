"""Project-wide paths and runtime configuration.

DATASET_ROOT points at the college dataset, which lives OUTSIDE this repository
and is never copied into git. Override it with the NATYA_DATASET_ROOT
environment variable when running on another machine.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- dataset (external, read-only) -----------------------------------------
DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\kavya\Desktop\VIT\SEM 9 YEAR 5\dataset\DATA FROM COLLEGE"
)
DATASET_ROOT = Path(os.environ.get("NATYA_DATASET_ROOT", DEFAULT_DATASET_ROOT))

# Only these four camera folders form the labelled image dataset.
# "Cam 6 extra", "Cam 7 extra" and "param_calc" are historical and excluded.
CORE_CAMERAS = ("Cam 5", "Cam 6", "Cam 7", "Cam 8")
IMAGES_PER_CAMERA = 52
IMAGES_PER_VARIATION = 2
NUM_VARIATIONS = 26

# --- repository paths -------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models" / "onnx"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGE_RESULTS_DIR = OUTPUTS_DIR / "image_results"
VIDEO_RESULTS_DIR = OUTPUTS_DIR / "video_results"
LOGS_DIR = OUTPUTS_DIR / "logs"

# --- inference defaults -----------------------------------------------------
# Landmarks below this score are treated as unavailable by the parameter layer.
KEYPOINT_SCORE_THRESHOLD = 0.30
# Person detector confidence.
DETECTION_SCORE_THRESHOLD = 0.35
# Temporal smoothing window for video (frames).
SMOOTHING_WINDOW = 9


def ensure_output_dirs() -> None:
    for path in (IMAGE_RESULTS_DIR, VIDEO_RESULTS_DIR, LOGS_DIR, ANNOTATIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)
