"""SimCC top-down pose model runner (RTMPose / RTMW / DWPose ONNX exports).

The class is deliberately layout-agnostic: it returns the model's *raw* keypoint
array. Mapping raw keypoints onto the common 67-landmark representation is the
job of `src.landmarks.adapters`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .transforms import bbox_xyxy2cs, decode_simcc, top_down_affine

# ImageNet statistics used by every RTMPose-family export (applied to BGR).
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class SimCCPoseModel:
    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int],  # (width, height)
        simcc_split_ratio: float = 2.0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"pose model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio

    def __call__(
        self, image: np.ndarray, bbox: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one person through the model.

        Returns (keypoints (K,2) in image pixels, scores (K,)).
        """
        if bbox is None:
            bbox = np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float32)

        center, scale = bbox_xyxy2cs(np.asarray(bbox, dtype=np.float32))
        crop, scale = top_down_affine(self.input_size, scale, center, image)

        blob = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.session.run(None, {self.input_name: blob})[:2]

        locs, scores = decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)
        keypoints = locs[0] / np.array(self.input_size, dtype=np.float32) * scale
        keypoints = keypoints + center - scale / 2
        return keypoints.astype(np.float32), scores[0].astype(np.float32)

    @property
    def num_keypoints(self) -> int:
        shape = self.session.get_outputs()[0].shape
        return int(shape[1]) if isinstance(shape[1], int) else -1
