"""Tests for SHA1 signing and signature.txt round-trip."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from leaffliction.signature import compute_sha1, verify_signature, write_signature


def test_compute_sha1_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hello leaffliction"
    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert compute_sha1(f) == hashlib.sha1(payload).hexdigest()


def test_compute_sha1_matches_shasum(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"sha-cmd-cross-check")
    has_shasum = subprocess.run(["which", "shasum"], capture_output=True).returncode == 0
    cmd = "shasum" if has_shasum else "sha1sum"
    out = subprocess.check_output([cmd, str(f)], text=True).split()[0]
    assert compute_sha1(f) == out


def test_write_and_verify_roundtrip(tmp_path: Path) -> None:
    a = tmp_path / "a.zip"
    a.write_bytes(b"AAAA")
    b = tmp_path / "b.zip"
    b.write_bytes(b"BBBB")
    sig_path = tmp_path / "signature.txt"
    write_signature([a, b], sig_path)
    assert sig_path.exists()
    assert verify_signature(sig_path) is True


def test_verify_detects_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.zip"
    a.write_bytes(b"original")
    sig_path = tmp_path / "signature.txt"
    write_signature([a], sig_path)
    a.write_bytes(b"tampered")
    assert verify_signature(sig_path) is False


def test_verify_missing_file_returns_false(tmp_path: Path) -> None:
    sig_path = tmp_path / "signature.txt"
    sig_path.write_text("0000000000000000000000000000000000000000  ghost.zip\n")
    assert verify_signature(sig_path) is False
