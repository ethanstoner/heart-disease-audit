import shutil

import pytest

import heart_audit.data as data
from heart_audit.data import RAW_FILES, HashMismatch, fetch_raw, file_digest, verify


def _copy_all_but(raw_dir, dest, missing):
    for name in RAW_FILES:
        if name != missing:
            shutil.copy(raw_dir / name, dest / name)


def test_all_raw_files_are_pinned():
    assert sorted(RAW_FILES) == [
        "heart.csv",
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data",
        "statlog.heart.dat",
    ]


def test_fetched_files_match_pins(raw_dir):
    for name, (_, algorithm, digest) in RAW_FILES.items():
        assert file_digest(raw_dir / name, algorithm) == digest


def test_tampered_file_is_a_hard_failure(tmp_path):
    f = tmp_path / "x.data"
    f.write_bytes(b"not the data")
    with pytest.raises(HashMismatch):
        verify(f, "sha256", "0" * 64)


def test_existing_corrupt_file_is_not_silently_replaced(tmp_path, raw_dir):
    _copy_all_but(raw_dir, tmp_path, "processed.va.data")
    (tmp_path / "processed.va.data").write_bytes(b"corrupt")
    with pytest.raises(HashMismatch):
        fetch_raw(tmp_path)
    assert (tmp_path / "processed.va.data").read_bytes() == b"corrupt"


def test_falls_back_to_second_mirror(tmp_path, raw_dir, monkeypatch):
    _copy_all_but(raw_dir, tmp_path, "heart.csv")
    first, second = RAW_FILES["heart.csv"][0]
    tried = []

    def fake_download(url, dest):
        tried.append(url)
        if url == first:
            raise OSError("mirror down")
        shutil.copy(raw_dir / "heart.csv", dest)

    monkeypatch.setattr(data, "_download", fake_download)
    fetch_raw(tmp_path)
    assert tried == [first, second]
    assert file_digest(tmp_path / "heart.csv", "md5") == RAW_FILES["heart.csv"][2]


def test_all_mirrors_bad_leaves_nothing_behind(tmp_path, raw_dir, monkeypatch):
    _copy_all_but(raw_dir, tmp_path, "heart.csv")

    def fake_download(url, dest):
        dest.write_bytes(b"wrong content")

    monkeypatch.setattr(data, "_download", fake_download)
    with pytest.raises(HashMismatch):
        fetch_raw(tmp_path)
    assert not (tmp_path / "heart.csv").exists()
    assert not (tmp_path / "heart.csv.part").exists()
