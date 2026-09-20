"""Download helper for comparison-only weight files (not used by the RTMW video path)."""

from __future__ import annotations

import urllib.request
from pathlib import Path


def ensure_file(urls: list[str], dest: Path, min_bytes: int = 1024) -> Path:
    """Return `dest` if it already exists, otherwise try each URL in order."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; natya-shastra-eval/1.0)"}
    for url in urls:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as resp:
                payload = resp.read()
            if len(payload) < min_bytes:
                last_error = f"{url}: payload too small ({len(payload)} bytes)"
                continue
            dest.write_bytes(payload)
            return dest
        except Exception as exc:
            last_error = f"{url}: {exc}"
    raise FileNotFoundError(
        f"could not fetch {dest.name}. Last error: {last_error}"
    )
