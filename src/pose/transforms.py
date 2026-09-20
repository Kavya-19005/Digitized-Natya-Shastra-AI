"""Geometric pre/post-processing shared by the top-down ONNX pose models.

These are the standard MMPose top-down transforms (bbox -> centre/scale ->
affine warp -> SimCC decode). They are reimplemented here so the project has no
runtime dependency on mmpose/mmdeploy.
"""

from __future__ import annotations

import cv2
import numpy as np


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.25
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an xyxy box to (centre, scale) with the usual 1.25 padding."""
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray, scale: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    dst_w, dst_h = output_size
    src_dir = np.array([0.0, scale[0] * -0.5], dtype=np.float32)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _third_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _third_point(dst[0], dst[1])

    return cv2.getAffineTransform(src, dst)


def top_down_affine(
    input_size: tuple[int, int],
    scale: np.ndarray,
    center: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop `image` around (centre, scale) into the model's input resolution."""
    w, h = input_size
    aspect_ratio = w / h
    b_w, b_h = float(scale[0]), float(scale[1])
    if b_w > b_h * aspect_ratio:
        scale = np.array([b_w, b_w / aspect_ratio], dtype=np.float32)
    else:
        scale = np.array([b_h * aspect_ratio, b_h], dtype=np.float32)

    warp_mat = get_warp_matrix(center, scale, (int(w), int(h)))
    warped = cv2.warpAffine(image, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    return warped, scale


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax-decode SimCC outputs into input-space coordinates and scores."""
    n, k, _ = simcc_x.shape
    flat_x = simcc_x.reshape(n * k, -1)
    flat_y = simcc_y.reshape(n * k, -1)

    x_locs = np.argmax(flat_x, axis=1)
    y_locs = np.argmax(flat_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = 0.5 * (np.amax(flat_x, axis=1) + np.amax(flat_y, axis=1))
    locs[vals <= 0.0] = -1.0

    return locs.reshape(n, k, 2) / split_ratio, vals.reshape(n, k)


def letterbox(
    image: np.ndarray, size: tuple[int, int], pad_value: int = 114
) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio and pad bottom/right (YOLOX convention)."""
    h, w = image.shape[:2]
    out_h, out_w = size
    if (h, w) == (out_h, out_w):
        return image.copy(), 1.0

    ratio = min(out_h / h, out_w / w)
    resized = cv2.resize(
        image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, ratio


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain greedy NMS on xyxy boxes."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= iou_thr]
    return keep
