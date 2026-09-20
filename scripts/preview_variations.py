"""Write a downscaled contact sheet of the first image of chosen variations.

Inspection aid only; output lands in outputs/logs/ which is git-ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_ROOT, LOGS_DIR
from src.labels import VARIATION_BY_ID

TILE_W, TILE_H = 320, 240


def main() -> int:
    camera = sys.argv[1] if len(sys.argv) > 1 else "Cam 5"
    ids = [int(v) for v in sys.argv[2:]] or list(range(1, 27))

    files = sorted((DATASET_ROOT / camera).glob("*.JPG"))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    tiles = []
    for vid in ids:
        idx = (vid - 1) * 2
        if idx >= len(files):
            continue
        img = cv2.imread(str(files[idx]))
        tile = cv2.resize(img, (TILE_W, TILE_H))
        label = VARIATION_BY_ID[vid]
        cv2.putText(
            tile, f"{vid}. {label.posture}", (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )
        text = label.error_text or "CORRECT"
        cv2.putText(
            tile, text[:34], (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1
        )
        tiles.append(tile)

    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * TILE_H, cols * TILE_W, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * TILE_H : (r + 1) * TILE_H, c * TILE_W : (c + 1) * TILE_W] = tile

    out = LOGS_DIR / f"preview_{camera.replace(' ', '')}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
