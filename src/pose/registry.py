"""Registry of the selected ONNX pose models.

Every entry goes through the SAME downstream pipeline
(ONNX -> raw keypoints -> common 67 landmarks -> parameters -> rules).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MODELS_DIR
from ..landmarks.adapters import HALPE26, WHOLEBODY133, LandmarkLayout


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    pose_file: str
    pose_input_size: tuple[int, int]  # (width, height)
    layout: LandmarkLayout
    detector_file: str
    detector_input_size: tuple[int, int] = (640, 640)
    notes: str = ""

    @property
    def pose_path(self):
        return MODELS_DIR / self.pose_file

    @property
    def detector_path(self):
        return MODELS_DIR / self.detector_file

    @property
    def available(self) -> bool:
        return self.pose_path.exists() and self.detector_path.exists()


MODEL_SPECS: dict[str, ModelSpec] = {
    "rtmw_l_wholebody": ModelSpec(
        key="rtmw_l_wholebody",
        display_name="RTMW-l whole-body (133 kpts, body+feet+hands)",
        pose_file="rtmw-dw-x-l_cocktail14_256x192.onnx",
        pose_input_size=(192, 256),
        layout=WHOLEBODY133,
        detector_file="yolox_m_humanart_640.onnx",
        notes="Primary model: only layout giving both body and finger landmarks.",
    ),
    "dwpose_l_wholebody": ModelSpec(
        key="dwpose_l_wholebody",
        display_name="DWPose-l whole-body (133 kpts, body+feet+hands)",
        pose_file="dwpose-l_ucoco_384x288.onnx",
        pose_input_size=(288, 384),
        layout=WHOLEBODY133,
        detector_file="yolox_m_humanart_640.onnx",
        notes="Second whole-body model at higher input resolution.",
    ),
    "rtmpose_m_halpe26": ModelSpec(
        key="rtmpose_m_halpe26",
        display_name="RTMPose-m Halpe26 (26 kpts, body+feet, no hands)",
        pose_file="rtmpose-m_halpe26_256x192.onnx",
        pose_input_size=(192, 256),
        layout=HALPE26,
        detector_file="yolox_s_humanart_640.onnx",
        notes="Body-only baseline: mudra parameters are unavailable by design.",
    ),
}

DEFAULT_MODEL_KEY = "rtmw_l_wholebody"


def available_models() -> list[ModelSpec]:
    return [spec for spec in MODEL_SPECS.values() if spec.available]


def missing_models() -> list[str]:
    missing: list[str] = []
    for spec in MODEL_SPECS.values():
        if not spec.pose_path.exists():
            missing.append(f"{spec.key}: {spec.pose_file}")
        if not spec.detector_path.exists():
            missing.append(f"{spec.key} detector: {spec.detector_file}")
    return missing


def get_spec(key: str) -> ModelSpec:
    if key not in MODEL_SPECS:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[key]
