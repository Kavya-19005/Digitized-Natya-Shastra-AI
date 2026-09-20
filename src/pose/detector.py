"""YOLOX person detector (stage 1 of the top-down pose pipeline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .transforms import letterbox, nms_xyxy

PERSON_CLASS_ID = 0


class PersonDetector:
    """Detects people with a YOLOX ONNX model exported by MMDeploy.

    Two export flavours exist in the wild; both are handled:
      * with NMS baked in  -> outputs `dets` (N,5 = xyxy+score) and `labels`
      * without NMS        -> a single raw (N, 85) prediction tensor
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"detector model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return person boxes as an (N, 5) array of xyxy + score, best first."""
        padded, ratio = letterbox(image, self.input_size)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: blob})

        boxes, scores = self._parse(outputs)
        if boxes.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        boxes = boxes / ratio
        keep = scores >= self.score_threshold
        boxes, scores = boxes[keep], scores[keep]
        if boxes.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        kept = nms_xyxy(boxes, scores, self.nms_threshold)
        boxes, scores = boxes[kept], scores[kept]

        h, w = image.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h - 1)
        return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)

    def _parse(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) >= 2 and outputs[0].shape[-1] == 5:
            dets = outputs[0][0]
            labels = outputs[1][0]
            person = labels == PERSON_CLASS_ID
            return dets[person, :4].copy(), dets[person, 4].copy()

        raw = outputs[0]
        preds = raw[0] if raw.ndim == 3 else raw
        return self._decode_raw(preds)

    def _decode_raw(self, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode grid-relative YOLOX predictions (export without NMS)."""
        strides = (8, 16, 32)
        grids, expanded = [], []
        for stride in strides:
            hs, ws = self.input_size[0] // stride, self.input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            expanded.append(np.full((hs * ws, 1), stride, dtype=np.float32))
        grid = np.concatenate(grids, 0).astype(np.float32)
        stride_col = np.concatenate(expanded, 0)

        cxcy = (preds[:, :2] + grid) * stride_col
        wh = np.exp(preds[:, 2:4]) * stride_col
        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        scores = (obj * cls)[:, PERSON_CLASS_ID]

        boxes = np.empty((preds.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = cxcy[:, 0] - wh[:, 0] / 2
        boxes[:, 1] = cxcy[:, 1] - wh[:, 1] / 2
        boxes[:, 2] = cxcy[:, 0] + wh[:, 0] / 2
        boxes[:, 3] = cxcy[:, 1] + wh[:, 1] / 2
        return boxes, scores.astype(np.float32)
