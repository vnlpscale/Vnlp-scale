from pathlib import Path

import numpy as np

from vnlp_scale.codec import encode_tensor, preset
from vnlp_scale.store import StoreReader, StoreWriter


def _write_store(path: Path, source: np.ndarray):
    with StoreWriter(path, meta={"test": True}) as writer:
        writer.ensure_tensor("weight", source.shape, str(source.dtype))
        midpoint = max(1, source.shape[0] // 2)
        for start, stop in [(0, midpoint), (midpoint, source.shape[0])]:
            encoded = encode_tensor("weight", source[start:stop], preset("lossless"))
            writer.add_chunk(
                "weight",
                start=start,
                stop=stop,
                shape=source[start:stop].shape,
                encoded=encoded,
            )
            writer.flush()


def test_store_roundtrip_and_checksums(tmp_path):
    source = np.arange(24, dtype=np.float32).reshape(6, 4) / 7
    path = tmp_path / "store"
    _write_store(path, source)
    with StoreReader(path, verify_on_open=True) as reader:
        restored = reader.decode("weight")
        report = reader.verify(checksums=True)
        summary = reader.summary()
    np.testing.assert_array_equal(restored, source.astype(np.float16).astype(np.float32))
    assert report["ok"]
    assert summary["chunks"] == 2
    assert summary["finalized"] is True


def test_store_detects_blob_corruption(tmp_path):
    source = np.arange(24, dtype=np.float32).reshape(6, 4)
    path = tmp_path / "store"
    _write_store(path, source)
    blob = path / "blobs.bin"
    with blob.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
    with StoreReader(path) as reader:
        report = reader.verify(checksums=True)
    assert not report["ok"]
    assert any("checksum mismatch" in error for error in report["errors"])


def test_writer_truncates_uncommitted_tail_on_resume(tmp_path):
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    path = tmp_path / "store"
    _write_store(path, source)
    blob = path / "blobs.bin"
    committed_size = blob.stat().st_size
    with blob.open("ab") as handle:
        handle.write(b"orphaned-tail")
    writer = StoreWriter(path)
    try:
        assert blob.stat().st_size == committed_size
    finally:
        writer.close(finalize=True)


def test_incomplete_store_is_inspectable_but_warned(tmp_path):
    source = np.arange(24, dtype=np.float32).reshape(6, 4)
    path = tmp_path / "store"
    writer = StoreWriter(path)
    writer.ensure_tensor("weight", source.shape, str(source.dtype))
    encoded = encode_tensor("weight", source[:3], preset("lossless"))
    writer.add_chunk("weight", start=0, stop=3, shape=source[:3].shape, encoded=encoded)
    writer.close(finalize=False, allow_incomplete=True)
    with StoreReader(path) as reader:
        report = reader.verify(checksums=True)
        summary = reader.summary()
    assert report["ok"]
    assert report["warnings"]
    assert summary["finalized"] is False
