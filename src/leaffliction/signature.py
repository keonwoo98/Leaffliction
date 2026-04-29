"""SHA1 signing & verification for dataset/model zip artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 64 * 1024


def compute_sha1(path: Path) -> str:
    """Stream the file through sha1; matches shasum/sha1sum."""
    h = hashlib.sha1()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_signature(paths: list[Path], out: Path) -> None:
    """Write `<sha1>  <basename>\\n` per file (compatible with `shasum -c`)."""
    lines: list[str] = []
    for p in paths:
        p = Path(p)
        digest = compute_sha1(p)
        lines.append(f"{digest}  {p.name}")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_signature(sig_path: Path) -> bool:
    """Return True iff every recorded hash matches a sibling file of `sig_path`."""
    sig_path = Path(sig_path)
    if not sig_path.exists():
        return False
    base = sig_path.parent
    for raw in sig_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        target = base / name.strip()
        if not target.exists():
            return False
        if compute_sha1(target) != digest.strip():
            return False
    return True
