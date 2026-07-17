"""Crash-resilient, checksummed, chunk-addressable Vnlp-scale model store."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import numpy as np

from . import codec
from .errors import StoreError, StoreLockedError

STORE_FORMAT = "vnlp-scale-store"
STORE_VERSION = 2
_ALIGNMENT = 64
_MANIFEST = "manifest.json"
_BLOBS = "blobs.bin"
_LOCK = ".writer.lock"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_manifest(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise StoreError(f"store manifest not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"cannot read store manifest: {path}: {exc}") from exc
    if manifest.get("format") != STORE_FORMAT:
        raise StoreError(f"unsupported store format: {manifest.get('format')!r}")
    if manifest.get("version") != STORE_VERSION:
        raise StoreError(
            f"unsupported store version {manifest.get('version')!r}; expected {STORE_VERSION}"
        )
    if not isinstance(manifest.get("tensors"), dict):
        raise StoreError("manifest.tensors must be an object")
    if not isinstance(manifest.get("meta"), dict):
        raise StoreError("manifest.meta must be an object")
    return manifest


class _WriterLock:
    def __init__(self, path: Path, *, force: bool = False):
        self.path = path
        if force:
            with suppress(FileNotFoundError):
                path.unlink()
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": time.time(),
        }
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            detail = ""
            with suppress(OSError):
                detail = path.read_text(encoding="utf-8").strip()
            suffix = f" ({detail})" if detail else ""
            raise StoreLockedError(f"store is locked by another writer{suffix}") from exc
        try:
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        with suppress(FileNotFoundError):
            self.path.unlink()
        self._released = True


class StoreWriter:
    """Append-only writer with chunk-level checkpoints and crash recovery.

    A checkpoint becomes committed only after the blob file has been fsynced and the
    manifest has been atomically replaced. Any uncommitted tail is truncated when a
    writer resumes the store.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        meta: dict | None = None,
        overwrite: bool = False,
        force_lock: bool = False,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / _MANIFEST
        self.blob_path = self.path / _BLOBS
        self._lock = _WriterLock(self.path / _LOCK, force=force_lock)
        self._closed = False
        try:
            if overwrite:
                for target in (self.manifest_path, self.blob_path):
                    with suppress(FileNotFoundError):
                        target.unlink()
            if self.manifest_path.exists():
                self.manifest = _load_manifest(self.manifest_path)
            else:
                if self.blob_path.exists() and self.blob_path.stat().st_size:
                    raise StoreError(
                        "blobs.bin exists without a manifest; use overwrite=True to reset"
                    )
                now = time.time()
                self.manifest = {
                    "format": STORE_FORMAT,
                    "version": STORE_VERSION,
                    "created_at": now,
                    "updated_at": now,
                    "finalized": False,
                    "meta": {},
                    "tensors": {},
                }
            if meta:
                self.manifest["meta"].update(meta)
            self.blob_path.touch(exist_ok=True)
            self._recover_uncommitted_tail()
            self._blob = self.blob_path.open("ab", buffering=0)
        except Exception:
            self._lock.release()
            raise

    def _recover_uncommitted_tail(self) -> None:
        committed_end = 0
        for tensor in self.manifest["tensors"].values():
            for chunk in tensor.get("chunks", []):
                for stage in chunk.get("stages", []):
                    for blob in stage.get("blobs", []):
                        committed_end = max(
                            committed_end, int(blob["offset"]) + int(blob["length"])
                        )
        actual = self.blob_path.stat().st_size if self.blob_path.exists() else 0
        if actual < committed_end:
            raise StoreError(
                f"blob file is truncated: {actual} bytes, manifest references {committed_end}"
            )
        if actual > committed_end:
            with self.blob_path.open("r+b") as handle:
                handle.truncate(committed_end)
                handle.flush()
                os.fsync(handle.fileno())

    def ensure_tensor(
        self,
        name: str,
        shape: tuple[int, ...] | list[int],
        source_dtype: str,
        *,
        chunk_axis: int = 0,
    ) -> dict:
        if not name or "\x00" in name:
            raise StoreError("tensor name is empty or contains NUL")
        normalized_shape = [int(value) for value in shape]
        if not normalized_shape or any(value <= 0 for value in normalized_shape):
            raise StoreError(f"invalid shape for {name!r}: {normalized_shape}")
        if chunk_axis != 0:
            raise StoreError("store version 2 only supports chunk_axis=0")
        existing = self.manifest["tensors"].get(name)
        if existing is not None:
            expected = (normalized_shape, str(source_dtype), chunk_axis)
            actual = (existing["shape"], existing["source_dtype"], existing["chunk_axis"])
            if actual != expected:
                raise StoreError(f"tensor metadata changed while resuming {name!r}")
            return existing
        record = {
            "shape": normalized_shape,
            "source_dtype": str(source_dtype),
            "decoded_dtype": "float32",
            "chunk_axis": chunk_axis,
            "complete": False,
            "rel_error": None,
            "chunks": [],
        }
        self.manifest["tensors"][name] = record
        return record

    def has_chunk(self, name: str, start: int, stop: int) -> bool:
        tensor = self.manifest["tensors"].get(name)
        if tensor is None:
            return False
        return any(
            int(chunk["start"]) == start and int(chunk["stop"]) == stop
            for chunk in tensor["chunks"]
        )

    def add_chunk(
        self,
        name: str,
        *,
        start: int,
        stop: int,
        shape: tuple[int, ...] | list[int],
        encoded: codec.EncodedTensor,
    ) -> None:
        if name not in self.manifest["tensors"]:
            raise StoreError(f"tensor must be declared before adding chunks: {name!r}")
        tensor = self.manifest["tensors"][name]
        if not 0 <= start < stop <= int(tensor["shape"][0]):
            raise StoreError(f"invalid chunk interval [{start}, {stop}) for {name!r}")
        chunk_shape = [int(value) for value in shape]
        expected_shape = list(tensor["shape"])
        expected_shape[0] = stop - start
        if chunk_shape != expected_shape:
            raise StoreError(
                f"chunk shape mismatch for {name!r}: got {chunk_shape}, expected {expected_shape}"
            )
        for current in tensor["chunks"]:
            current_start, current_stop = int(current["start"]), int(current["stop"])
            if start == current_start and stop == current_stop:
                raise StoreError(f"chunk [{start}, {stop}) already exists for {name!r}")
            if max(start, current_start) < min(stop, current_stop):
                raise StoreError(f"overlapping chunks for {name!r}")

        stage_records: list[dict] = []
        for stage in encoded.stages:
            blob_records = []
            for payload in stage["blobs"]:
                data = bytes(payload)
                pad = (-self._blob.tell()) % _ALIGNMENT
                if pad:
                    self._blob.write(b"\0" * pad)
                offset = self._blob.tell()
                self._blob.write(data)
                blob_records.append(
                    {"offset": int(offset), "length": len(data), "sha256": _sha256(data)}
                )
            stage_records.append(
                {"kind": stage["kind"], "meta": stage["meta"], "blobs": blob_records}
            )
        tensor["chunks"].append(
            {
                "start": int(start),
                "stop": int(stop),
                "shape": chunk_shape,
                "rel_error": float(encoded.rel_error),
                "error_l2_sq": float(encoded.error_l2_sq),
                "source_l2_sq": float(encoded.source_l2_sq),
                "stages": stage_records,
            }
        )
        tensor["chunks"].sort(key=lambda chunk: int(chunk["start"]))
        self._update_tensor_metrics(tensor)

    @staticmethod
    def _update_tensor_metrics(tensor: dict) -> None:
        expected = 0
        complete = True
        error = 0.0
        source = 0.0
        for chunk in tensor["chunks"]:
            if int(chunk["start"]) != expected:
                complete = False
            expected = int(chunk["stop"])
            error += float(chunk["error_l2_sq"])
            source += float(chunk["source_l2_sq"])
        if expected != int(tensor["shape"][0]):
            complete = False
        tensor["complete"] = complete
        tensor["rel_error"] = (
            float(np.sqrt(error / max(source, 1e-30))) if tensor["chunks"] else None
        )

    def flush(self) -> None:
        if self._closed:
            raise StoreError("writer is closed")
        self._blob.flush()
        os.fsync(self._blob.fileno())
        self.manifest["updated_at"] = time.time()
        _atomic_write_json(self.manifest_path, self.manifest)

    def close(self, *, finalize: bool = True, allow_incomplete: bool = False) -> None:
        if self._closed:
            return
        try:
            incomplete = [
                name for name, tensor in self.manifest["tensors"].items() if not tensor["complete"]
            ]
            if finalize and incomplete and not allow_incomplete:
                raise StoreError(
                    "cannot finalize an incomplete store; incomplete tensors: "
                    + ", ".join(incomplete[:8])
                )
            self.manifest["finalized"] = bool(finalize and not incomplete)
            if self.manifest["finalized"]:
                self.manifest["finalized_at"] = time.time()
            self.flush()
        finally:
            self._blob.close()
            self._closed = True
            self._lock.release()

    def __enter__(self) -> StoreWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close(finalize=True)
        else:
            with suppress(Exception):
                self.close(finalize=False, allow_incomplete=True)


class StoreReader:
    """Read-only random-access view over a Vnlp-scale store."""

    def __init__(self, path: str | os.PathLike[str], *, verify_on_open: bool = False):
        self.path = Path(path)
        self.manifest_path = self.path / _MANIFEST
        self.blob_path = self.path / _BLOBS
        self.manifest = _load_manifest(self.manifest_path)
        try:
            self.blob_bytes = self.blob_path.stat().st_size
        except FileNotFoundError as exc:
            raise StoreError(f"blob file not found: {self.blob_path}") from exc
        self._mm = (
            np.memmap(self.blob_path, dtype=np.uint8, mode="r")
            if self.blob_bytes
            else np.empty(0, dtype=np.uint8)
        )
        structural = self.verify(checksums=verify_on_open)
        if not structural["ok"]:
            raise StoreError("invalid store: " + "; ".join(structural["errors"][:5]))

    def names(self) -> list[str]:
        return list(self.manifest["tensors"])

    def info(self, name: str) -> dict:
        try:
            return self.manifest["tensors"][name]
        except KeyError as exc:
            raise StoreError(f"tensor not found in store: {name}") from exc

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in self.info(name)["shape"])

    def chunk_count(self, name: str) -> int:
        return len(self.info(name)["chunks"])

    def _views_for_stage(self, stage: dict) -> list[np.ndarray]:
        views = []
        for blob in stage["blobs"]:
            offset, length = int(blob["offset"]), int(blob["length"])
            views.append(self._mm[offset : offset + length])
        return views

    def decode_chunk(self, name: str, index: int, *, max_stages: int | None = None) -> np.ndarray:
        tensor = self.info(name)
        try:
            chunk = tensor["chunks"][index]
        except IndexError as exc:
            raise StoreError(f"chunk index out of range for {name!r}: {index}") from exc
        stages = [
            {"kind": stage["kind"], "meta": stage["meta"], "views": self._views_for_stage(stage)}
            for stage in chunk["stages"]
        ]
        return codec.decode_tensor(stages, chunk["shape"], max_stages=max_stages)

    def iter_decoded_chunks(
        self, name: str, *, max_stages: int | None = None
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        tensor = self.info(name)
        for index, chunk in enumerate(tensor["chunks"]):
            yield (
                int(chunk["start"]),
                int(chunk["stop"]),
                self.decode_chunk(name, index, max_stages=max_stages),
            )

    def decode(self, name: str, *, max_stages: int | None = None) -> np.ndarray:
        pieces = [chunk for _, _, chunk in self.iter_decoded_chunks(name, max_stages=max_stages)]
        if not pieces:
            raise StoreError(f"tensor has no chunks: {name!r}")
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    def read_rows(
        self, name: str, rows: np.ndarray, *, max_stages: int | None = None
    ) -> np.ndarray:
        tensor = self.info(name)
        shape = tuple(int(value) for value in tensor["shape"])
        if len(shape) != 2:
            raise StoreError("read_rows requires a matrix")
        requested = np.asarray(rows, dtype=np.int64)
        if requested.size and (requested.min() < 0 or requested.max() >= shape[0]):
            raise StoreError(f"row index out of range for {name!r}")
        result = np.empty((*requested.shape, shape[1]), dtype=np.float32)
        flat = requested.reshape(-1)
        output = result.reshape(-1, shape[1])
        for index, chunk in enumerate(tensor["chunks"]):
            start, stop = int(chunk["start"]), int(chunk["stop"])
            mask = (flat >= start) & (flat < stop)
            if not np.any(mask):
                continue
            values = self.decode_chunk(name, index, max_stages=max_stages)
            output[mask] = values[flat[mask] - start]
        return result

    def verify(self, *, checksums: bool = True) -> dict:
        errors: list[str] = []
        warnings: list[str] = []
        checked_blobs = 0
        tensor_count = 0
        with self.blob_path.open("rb") as handle:
            for name, tensor in self.manifest["tensors"].items():
                tensor_count += 1
                shape = tensor.get("shape")
                if (
                    not isinstance(shape, list)
                    or not shape
                    or any(not isinstance(value, int) or value <= 0 for value in shape)
                ):
                    errors.append(f"{name}: invalid shape")
                    continue
                expected = 0
                for chunk in tensor.get("chunks", []):
                    start, stop = chunk.get("start"), chunk.get("stop")
                    if (
                        not isinstance(start, int)
                        or not isinstance(stop, int)
                        or start != expected
                        or stop <= start
                    ):
                        errors.append(f"{name}: non-contiguous or invalid chunk coverage")
                        break
                    expected = stop
                    expected_shape = list(shape)
                    expected_shape[0] = stop - start
                    if chunk.get("shape") != expected_shape:
                        errors.append(f"{name}: chunk shape mismatch at {start}")
                    for stage in chunk.get("stages", []):
                        if stage.get("kind") not in {"raw", "svd", "quant"}:
                            errors.append(f"{name}: unknown stage kind")
                        for blob in stage.get("blobs", []):
                            offset, length = blob.get("offset"), blob.get("length")
                            if (
                                not isinstance(offset, int)
                                or not isinstance(length, int)
                                or offset < 0
                                or length < 0
                                or offset + length > self.blob_bytes
                            ):
                                errors.append(f"{name}: blob range outside file")
                                continue
                            if checksums:
                                handle.seek(offset)
                                digest = hashlib.sha256()
                                remaining = length
                                while remaining:
                                    block = handle.read(min(1024 * 1024, remaining))
                                    if not block:
                                        errors.append(f"{name}: truncated blob")
                                        break
                                    digest.update(block)
                                    remaining -= len(block)
                                if remaining == 0 and digest.hexdigest() != blob.get("sha256"):
                                    errors.append(f"{name}: checksum mismatch at offset {offset}")
                            checked_blobs += 1
                coverage_complete = expected == shape[0]
                if not coverage_complete:
                    if self.manifest.get("finalized", False) or tensor.get("complete", False):
                        errors.append(f"{name}: incomplete chunk coverage")
                    else:
                        warnings.append(f"{name}: recording is incomplete")
                if bool(tensor.get("complete")) != coverage_complete:
                    errors.append(f"{name}: complete flag is inconsistent")
        if not self.manifest.get("finalized", False):
            warnings.append("store is not finalized")
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "tensors": tensor_count,
            "checked_blobs": checked_blobs,
            "checksums": checksums,
        }

    def summary(self) -> dict:
        parameters = 0
        stored = 0
        by_kind: dict[str, int] = {}
        errors = []
        chunk_count = 0
        for name, tensor in self.manifest["tensors"].items():
            parameters += int(np.prod(tensor["shape"]))
            chunk_count += len(tensor["chunks"])
            if tensor.get("rel_error") is not None:
                errors.append((float(tensor["rel_error"]), name))
            for chunk in tensor["chunks"]:
                for stage in chunk["stages"]:
                    for blob in stage["blobs"]:
                        length = int(blob["length"])
                        stored += length
                        by_kind[stage["kind"]] = by_kind.get(stage["kind"], 0) + length
        errors.sort(reverse=True)
        return {
            "format": self.manifest["format"],
            "version": self.manifest["version"],
            "finalized": bool(self.manifest.get("finalized")),
            "tensors": len(self.manifest["tensors"]),
            "chunks": chunk_count,
            "parameters": parameters,
            "original_fp16_bytes": parameters * 2,
            "stored_payload_bytes": stored,
            "blob_file_bytes": self.blob_bytes,
            "bits_per_parameter": stored * 8 / max(parameters, 1),
            "compression_vs_fp16": parameters * 2 / max(stored, 1),
            "bytes_by_stage_kind": by_kind,
            "mean_tensor_rel_error": float(np.mean([value for value, _ in errors]))
            if errors
            else 0.0,
            "worst_tensor_rel_error": errors[:5],
            "meta": self.manifest["meta"],
        }

    def close(self) -> None:
        if isinstance(self._mm, np.memmap):
            mmap_object = getattr(self._mm, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()

    def __enter__(self) -> StoreReader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
