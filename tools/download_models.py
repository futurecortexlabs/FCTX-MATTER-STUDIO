"""Fetch the MediaPipe hand landmarker model pack.

Run once after cloning:

    uv run python tools/download_models.py

The model is not vendored into the repository: it is a 7.5 MB binary owned by
Google, and pinning its checksum here is enough to make the download
reproducible.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fctx.config import HAND_MODEL_PATH, HAND_MODEL_SHA256, HAND_MODEL_URL  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expect: str | None = None,
             force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        digest = sha256(dest)
        if expect is None or digest == expect:
            print(f"[ok]   {dest.name} already present ({dest.stat().st_size:,} B)")
            return dest
        print(f"[warn] {dest.name} checksum mismatch, re-downloading")

    print(f"[get]  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        with tmp.open("wb") as fh:
            while chunk := response.read(1 << 16):
                fh.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100.0 * read / total
                    print(f"\r       {read:,} / {total:,} B ({pct:5.1f}%)",
                          end="", flush=True)
    print()
    digest = sha256(tmp)
    if expect is not None and digest != expect:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"checksum mismatch for {dest.name}\n"
            f"  expected {expect}\n  got      {digest}")
    tmp.replace(dest)
    print(f"[ok]   {dest} ({dest.stat().st_size:,} B)\n       sha256 {digest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    # "--print-hash" downloads without verifying and reports the digest, which
    # is how HAND_MODEL_SHA256 gets refreshed when Google publishes a new pack.
    expect = None if "--print-hash" in argv else HAND_MODEL_SHA256
    download(HAND_MODEL_URL, HAND_MODEL_PATH, expect=expect, force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
