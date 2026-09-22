"""Fetch, hash-verify and load the UCI and Kaggle heart disease data."""
from __future__ import annotations

import hashlib
import http.client
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
# No Kaggle API dependency: two public mirrors, pinned by commit, both carrying the same md5.
_KAGGLE_MIRRORS = (
    "https://raw.githubusercontent.com/clinicalml/TabLLM/"
    "71f6647c523aa6fd35df94ad76207c4377248c49/datasets/heart/heart.csv",
    "https://raw.githubusercontent.com/PacktPublishing/Building-Data-Science-Solutions-with-Anaconda/"
    "46bea78d4db5398f00862f298ad91b197c7d521f/Chapter10/heart.csv",
)

# filename -> (urls tried in order, hash algorithm, pinned digest)
RAW_FILES = {
    "processed.cleveland.data": (
        (f"{_UCI}/processed.cleveland.data",),
        "sha256",
        "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    ),
    "processed.hungarian.data": (
        (f"{_UCI}/processed.hungarian.data",),
        "sha256",
        "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    ),
    "processed.switzerland.data": (
        (f"{_UCI}/processed.switzerland.data",),
        "sha256",
        "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    ),
    "processed.va.data": (
        (f"{_UCI}/processed.va.data",),
        "sha256",
        "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
    ),
    "heart.csv": (_KAGGLE_MIRRORS, "md5", "ab21f2524241ed14b321bcaf40c8b86e"),
}


class HashMismatch(RuntimeError):
    pass


def file_digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def verify(path: Path, algorithm: str, expected: str) -> None:
    actual = file_digest(path, algorithm)
    if actual != expected:
        raise HashMismatch(f"{Path(path).name}: {algorithm} {actual} != pinned {expected}")


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "heart-disease-audit"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def _download_first_valid(urls: tuple[str, ...], algorithm: str, digest: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    last_error: Exception = FileNotFoundError(f"no source URL for {dest.name}")
    try:
        for url in urls:
            try:
                _download(url, part)
                verify(part, algorithm, digest)
                part.replace(dest)
                return
            except (OSError, http.client.HTTPException, HashMismatch) as exc:
                last_error = exc
    finally:
        part.unlink(missing_ok=True)
    raise last_error


def fetch_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Download any missing raw file, then verify every file against its pin.

    A file already on disk is never overwritten: if it fails its pin, that is a hard failure.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (urls, algorithm, digest) in RAW_FILES.items():
        dest = raw_dir / name
        if not dest.exists():
            _download_first_valid(urls, algorithm, digest, dest)
        verify(dest, algorithm, digest)
    return raw_dir
