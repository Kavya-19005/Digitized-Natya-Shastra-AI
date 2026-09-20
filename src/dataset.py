"""Builds the labelled manifest of the 208 college JPG images.

52 JPGs per camera x 4 core cameras = 208 images.
Within a camera the sorted JPG list maps two consecutive files to one variation:
indices 0,1 -> variation 1 ... indices 50,51 -> variation 26.

The historical folders ("Cam 6 extra", "Cam 7 extra", "param_calc") and the old
CSV outputs are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    ANNOTATIONS_DIR,
    CORE_CAMERAS,
    DATASET_ROOT,
    IMAGES_PER_CAMERA,
    IMAGES_PER_VARIATION,
)
from .labels import VariationLabel, variation_for_index


@dataclass(frozen=True)
class DatasetItem:
    path: Path
    camera: str
    index_in_camera: int
    variation_id: int
    label: VariationLabel

    @property
    def image_name(self) -> str:
        return self.path.name


def camera_images(camera: str, root: Path | None = None) -> list[Path]:
    """Sorted JPGs of one camera folder (filename sort, as the labels require)."""
    base = (root or DATASET_ROOT) / camera
    if not base.is_dir():
        raise FileNotFoundError(f"camera folder not found: {base}")
    files = sorted(
        (p for p in base.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
        key=lambda p: p.name,
    )
    return files


def build_manifest(root: Path | None = None, strict: bool = True) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for camera in CORE_CAMERAS:
        files = camera_images(camera, root)
        if strict and len(files) != IMAGES_PER_CAMERA:
            raise ValueError(
                f"{camera}: expected {IMAGES_PER_CAMERA} JPGs, found {len(files)}"
            )
        for index, path in enumerate(files[:IMAGES_PER_CAMERA]):
            label = variation_for_index(index)
            items.append(
                DatasetItem(
                    path=path,
                    camera=camera,
                    index_in_camera=index,
                    variation_id=label.variation_id,
                    label=label,
                )
            )
    return items


def manifest_dataframe(items: list[DatasetItem] | None = None) -> pd.DataFrame:
    items = items or build_manifest()
    rows = []
    for item in items:
        label = item.label
        rows.append(
            {
                "image_name": item.image_name,
                "camera": item.camera,
                "index_in_camera": item.index_in_camera,
                "pair_index": item.index_in_camera % IMAGES_PER_VARIATION,
                "variation_id": item.variation_id,
                "ground_truth_posture": label.posture,
                "ground_truth_status": label.status,
                "ground_truth_error": label.error_code,
                "ground_truth_error_text": label.error_text,
                "ground_truth_side": label.side,
                "label_name": label.label_name,
                "image_path": str(item.path),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(path: Path | None = None) -> Path:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = path or (ANNOTATIONS_DIR / "dataset_manifest_208.csv")
    manifest_dataframe().to_csv(target, index=False)
    return target
