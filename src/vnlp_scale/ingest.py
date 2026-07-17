"""Streaming safetensors ingestion with chunk-level resume semantics."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors import safe_open

from . import codec
from .errors import StoreError
from .store import StoreReader, StoreWriter

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"
_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "spiece.model",
    "merges.txt",
    "vocab.json",
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _safe_local_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise StoreError(f"checkpoint index contains path traversal: {relative!r}") from exc
    return candidate


def _read_weight_map(index_path: Path) -> dict[str, str]:
    try:
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"cannot read safetensors index {index_path}: {exc}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise StoreError(f"invalid or empty weight_map in {index_path}")
    result: dict[str, str] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise StoreError("weight_map keys and values must be strings")
        result[name] = shard
    return result


def _local_plan(source: Path) -> tuple[list[Path], dict[str, str] | None]:
    index = source / _INDEX_FILE
    if index.is_file():
        weight_map = _read_weight_map(index)
        shard_names = sorted(set(weight_map.values()))
        shards = [_safe_local_child(source, shard) for shard in shard_names]
        missing = [str(path) for path in shards if not path.is_file()]
        if missing:
            raise StoreError("checkpoint index references missing shards: " + ", ".join(missing))
        return shards, weight_map
    single = source / _SINGLE_FILE
    if single.is_file():
        return [single], None
    raise StoreError(f"no {_INDEX_FILE} or {_SINGLE_FILE} found under {source}")


def _remote_plan(
    repo_id: str, revision: str | None
) -> tuple[list[str], dict[str, str] | None, Callable[[str], str], str | None]:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError

    resolved_revision = revision
    try:
        info = HfApi().model_info(repo_id, revision=revision)
        resolved_revision = info.sha or revision
    except Exception:
        # Download calls below produce the authoritative user-facing error. Resolution
        # metadata is useful but not required for correctness.
        pass

    def fetch(filename: str) -> str:
        return hf_hub_download(repo_id, filename, revision=resolved_revision)

    try:
        index_path = Path(fetch(_INDEX_FILE))
    except (EntryNotFoundError, RemoteEntryNotFoundError):
        return [_SINGLE_FILE], None, fetch, resolved_revision
    weight_map = _read_weight_map(index_path)
    return sorted(set(weight_map.values())), weight_map, fetch, resolved_revision


def _chunk_ranges(
    shape: tuple[int, ...], dtype_name: str, max_chunk_bytes: int
) -> Iterable[tuple[int, int]]:
    if not shape:
        raise StoreError("scalar safetensors are not supported as model parameters")
    element_bytes = _DTYPE_BYTES.get(dtype_name)
    if element_bytes is None:
        raise StoreError(f"unsupported safetensors dtype: {dtype_name}")
    row_elements = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    row_bytes = max(1, row_elements * element_bytes)
    rows_per_chunk = max(1, max_chunk_bytes // row_bytes)
    for start in range(0, shape[0], rows_per_chunk):
        yield start, min(shape[0], start + rows_per_chunk)


def _copy_local_metadata(source: Path, output: Path) -> list[str]:
    copied = []
    for filename in _METADATA_FILES:
        src = source / filename
        if src.is_file():
            shutil.copy2(src, output / filename)
            copied.append(filename)
    return copied


def _copy_remote_metadata(fetch: Callable[[str], str], output: Path) -> list[str]:
    from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError

    copied = []
    for filename in _METADATA_FILES:
        try:
            source_path = fetch(filename)
        except (EntryNotFoundError, RemoteEntryNotFoundError):
            continue
        shutil.copy2(source_path, output / filename)
        copied.append(filename)
    return copied


def _validate_resume_identity(output: Path, identity: dict, *, overwrite: bool) -> None:
    manifest_path = output / "manifest.json"
    if overwrite or not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"cannot inspect existing store identity: {exc}") from exc
    existing = manifest.get("meta", {}).get("recording_identity")
    if existing is None and manifest.get("tensors"):
        raise StoreError(
            "existing store has no recording identity; use --overwrite instead of resuming"
        )
    if existing is not None and existing != identity:
        changed = sorted(
            key for key in set(existing) | set(identity) if existing.get(key) != identity.get(key)
        )
        raise StoreError("recording settings differ from the existing store: " + ", ".join(changed))


def record(
    source: str,
    output: str,
    *,
    quality: str = "med",
    revision: str | None = None,
    max_chunk_bytes: int = 64 * 1024 * 1024,
    checkpoint_every: int = 8,
    overwrite: bool = False,
    force_lock: bool = False,
    progress: Callable[[str], None] | None = print,
) -> dict:
    """Encode a local or Hugging Face safetensors checkpoint into a Vnlp-scale store.

    The source checkpoint is never deserialized through pickle. Each tensor is sliced
    along axis 0 so peak encoding memory is bounded by ``max_chunk_bytes`` plus codec
    workspace. Existing committed chunks are skipped when the operation resumes.
    """

    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    config = codec.preset(quality)
    source_path = Path(source).expanduser()
    is_local = source_path.is_dir()
    fetch: Callable[[str], str] | None = None
    weight_map: dict[str, str] | None
    if is_local:
        shards, weight_map = _local_plan(source_path)
        resolved_revision = None
    else:
        shard_names, weight_map, fetch, resolved_revision = _remote_plan(source, revision)
        shards = shard_names

    output_path = Path(output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for filename in _METADATA_FILES:
            metadata_path = output_path / filename
            if metadata_path.is_file():
                metadata_path.unlink()
    canonical_source = str(source_path.resolve()) if is_local else source
    started = time.time()
    encoded_chunks = 0
    skipped_chunks = 0
    encoded_tensors: set[str] = set()
    checkpoint_counter = 0

    codec_metadata = asdict(config)
    # Convert dataclass quant stage dictionaries into JSON-compatible values.
    codec_metadata["quant_stages"] = [asdict(stage) for stage in config.quant_stages]
    recording_identity = {
        "source": canonical_source,
        "source_kind": "local" if is_local else "huggingface",
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "quality": quality,
        "max_chunk_bytes": max_chunk_bytes,
        "codec": codec_metadata,
    }
    recording_identity = json.loads(json.dumps(recording_identity, sort_keys=True))
    _validate_resume_identity(output_path, recording_identity, overwrite=overwrite)
    meta = {
        **recording_identity,
        "recording_identity": recording_identity,
        "started_at": started,
    }

    writer = StoreWriter(
        output_path,
        meta=meta,
        overwrite=overwrite,
        force_lock=force_lock,
    )
    try:
        for shard_index, shard in enumerate(shards):
            shard_path = Path(shard) if is_local else Path(fetch(str(shard)))  # type: ignore[arg-type]
            if progress:
                progress(f"[record] shard {shard_index + 1}/{len(shards)}: {shard_path.name}")
            with safe_open(str(shard_path), framework="numpy") as handle:
                names = list(handle.keys())
                if weight_map is not None:
                    expected_names = {
                        name
                        for name, mapped_shard in weight_map.items()
                        if mapped_shard == str(shard)
                    }
                    # Local plans contain absolute Path objects while weight maps contain
                    # relative names. Resolve by basename when needed.
                    if is_local:
                        expected_names = {
                            name
                            for name, mapped_shard in weight_map.items()
                            if _safe_local_child(source_path, mapped_shard) == shard_path.resolve()
                        }
                    unexpected = set(names) - expected_names
                    missing = expected_names - set(names)
                    if unexpected or missing:
                        raise StoreError(
                            f"index/shard mismatch for {shard_path.name}: "
                            f"unexpected={sorted(unexpected)[:5]}, missing={sorted(missing)[:5]}"
                        )
                for name in names:
                    tensor_slice = handle.get_slice(name)
                    shape = tuple(int(value) for value in tensor_slice.get_shape())
                    dtype_name = str(tensor_slice.get_dtype())
                    writer.ensure_tensor(name, shape, dtype_name)
                    for start, stop in _chunk_ranges(shape, dtype_name, max_chunk_bytes):
                        if writer.has_chunk(name, start, stop):
                            skipped_chunks += 1
                            continue
                        values = np.asarray(tensor_slice[start:stop])
                        encoded = codec.encode_tensor(name, values, config)
                        writer.add_chunk(
                            name,
                            start=start,
                            stop=stop,
                            shape=values.shape,
                            encoded=encoded,
                        )
                        encoded_chunks += 1
                        encoded_tensors.add(name)
                        checkpoint_counter += 1
                        if checkpoint_counter >= checkpoint_every:
                            writer.flush()
                            checkpoint_counter = 0
            writer.flush()

        copied = (
            _copy_local_metadata(source_path, output_path)
            if is_local
            else _copy_remote_metadata(fetch, output_path)  # type: ignore[arg-type]
        )
        writer.manifest["meta"].update(
            {
                "metadata_files": copied,
                "completed_at": time.time(),
            }
        )
        writer.close(finalize=True)
    except Exception:
        with suppress(Exception):
            writer.close(finalize=False, allow_incomplete=True)
        raise

    elapsed = time.time() - started
    with StoreReader(output_path) as reader:
        summary = reader.summary()
    result = {
        "encoded_tensors": len(encoded_tensors),
        "encoded_chunks": encoded_chunks,
        "skipped_chunks": skipped_chunks,
        "seconds": elapsed,
        "summary": summary,
    }
    if progress:
        progress(
            f"[record] complete: {encoded_chunks} chunks encoded, "
            f"{skipped_chunks} resumed, {summary['bits_per_parameter']:.2f} bits/parameter"
        )
    return result
