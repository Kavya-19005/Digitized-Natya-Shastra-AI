"""Download the selected ONNX pose-estimation models into models/onnx/.

Model binaries are intentionally kept OUT of git (see .gitignore). Run this once
after cloning:

    python scripts/download_models.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = PROJECT_ROOT / "models" / "onnx"

BASE = "https://download.openmmlab.com/mmpose/v1/projects"

# name -> (url, destination .onnx filename)
DOWNLOADS: dict[str, tuple[str, str]] = {
    # --- person detectors (top-down pipeline stage 1) ---
    "yolox_m_humanart": (
        f"{BASE}/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "yolox_m_humanart_640.onnx",
    ),
    "yolox_s_humanart": (
        f"{BASE}/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip",
        "yolox_s_humanart_640.onnx",
    ),
    # --- pose models (top-down pipeline stage 2) ---
    "rtmw_l_wholebody133": (
        f"{BASE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        "rtmw-dw-x-l_cocktail14_256x192.onnx",
    ),
    "dwpose_l_wholebody133": (
        f"{BASE}/rtmposev1/onnx_sdk/"
        "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip",
        "dwpose-l_ucoco_384x288.onnx",
    ),
    "rtmpose_m_halpe26": (
        f"{BASE}/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "rtmpose-m_halpe26_256x192.onnx",
    ),
    # --- hand refinement (stage 3): the whole-body models locate the hand but
    # resolve the fingers at only a few pixels, so the mudra parameters are
    # re-measured from a full-resolution hand crop.
    "rtmpose_m_hand21": (
        f"{BASE}/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
        "rtmpose-m_hand5_256x256.onnx",
    ),
}


def fetch(name: str, url: str, dest_name: str) -> bool:
    dest = ONNX_DIR / dest_name
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[skip] {name}: already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"[get ] {name} <- {url}")
    try:
        with urllib.request.urlopen(url, timeout=600) as resp:
            payload = resp.read()
    except Exception as exc:  # network/host problems are reported, not fatal
        print(f"[FAIL] {name}: {exc}")
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".onnx")]
            if not members:
                print(f"[FAIL] {name}: archive contains no .onnx member")
                return False
            # SDK archives ship exactly one end2end.onnx
            with zf.open(members[0]) as src:
                dest.write_bytes(src.read())
    except zipfile.BadZipFile:
        # some links serve the raw .onnx directly
        dest.write_bytes(payload)

    print(f"[ok  ] {name} -> {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    failures = [
        name for name, (url, dest) in DOWNLOADS.items() if not fetch(name, url, dest)
    ]
    if failures:
        print("\nMissing models:", ", ".join(failures))
        return 1
    print("\nAll models present in", ONNX_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
