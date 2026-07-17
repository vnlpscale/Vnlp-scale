"""Deterministic recursive residual codec used by Vnlp-scale stores.

The codec is intentionally independent from the storage layer. Each call operates on
one bounded-size tensor chunk and returns byte blobs plus JSON-serializable metadata.
Large model tensors are split by :mod:`vnlp_scale.ingest` before reaching this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .errors import CodecError

_SUPPORTED_BITS = frozenset({2, 4, 8})


@dataclass(frozen=True)
class QuantStageConfig:
    bits: int
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.bits not in _SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {sorted(_SUPPORTED_BITS)}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")


@dataclass(frozen=True)
class CodecConfig:
    """Configuration for one tensor-chunk encoding operation."""

    name: str
    rank_fraction: float = 0.0
    quant_stages: tuple[QuantStageConfig, ...] = ()
    min_dimension: int = 64
    min_rank: int = 8
    raw_name_fragments: tuple[str, ...] = ("norm",)
    force_raw: bool = False
    seed: int = 0
    svd_oversample: int = 8
    svd_power_iterations: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.rank_fraction <= 1.0:
            raise ValueError("rank_fraction must be in [0, 1]")
        if self.min_dimension <= 0 or self.min_rank <= 0:
            raise ValueError("min_dimension and min_rank must be positive")
        if self.svd_oversample < 0 or self.svd_power_iterations < 0:
            raise ValueError("SVD tuning values must be non-negative")


@dataclass
class EncodedTensor:
    """Encoded representation plus metrics needed for aggregate error accounting."""

    stages: list[dict]
    rel_error: float
    error_l2_sq: float
    source_l2_sq: float


PRESETS: dict[str, dict] = {
    "lossless": {
        "force_raw": True,
        "rank_fraction": 0.0,
        "quant_stages": (),
    },
    "low": {
        "rank_fraction": 0.05,
        "quant_stages": (QuantStageConfig(2, 64),),
    },
    "med": {
        "rank_fraction": 0.08,
        "quant_stages": (QuantStageConfig(4, 64),),
    },
    "high": {
        "rank_fraction": 0.125,
        "quant_stages": (QuantStageConfig(4, 64), QuantStageConfig(2, 64)),
    },
}


def preset(name: str, **overrides) -> CodecConfig:
    """Return a validated immutable preset, optionally overriding any field."""

    if name not in PRESETS:
        raise ValueError(f"unknown quality preset {name!r}; choose {sorted(PRESETS)}")
    values = dict(PRESETS[name])
    values.update(overrides)
    values["name"] = name
    if "quant_stages" in values:
        values["quant_stages"] = tuple(
            stage if isinstance(stage, QuantStageConfig) else QuantStageConfig(*stage)
            for stage in values["quant_stages"]
        )
    return CodecConfig(**values)


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if bits not in _SUPPORTED_BITS:
        raise CodecError(f"unsupported bit width: {bits}")
    limit = 1 << bits
    if values.size and int(values.max()) >= limit:
        raise CodecError(f"value does not fit in {bits} bits")
    if bits == 8:
        return values.tobytes(order="C")
    per_byte = 8 // bits
    pad = (-values.size) % per_byte
    if pad:
        values = np.pad(values, (0, pad), mode="constant")
    values = values.reshape(-1, per_byte)
    shifts = np.arange(per_byte - 1, -1, -1, dtype=np.uint8) * bits
    packed = np.bitwise_or.reduce(values << shifts[None, :], axis=1)
    return packed.astype(np.uint8, copy=False).tobytes(order="C")


def _unpack_bits(buffer: bytes | memoryview | np.ndarray, bits: int, count: int) -> np.ndarray:
    if count < 0:
        raise CodecError("count must be non-negative")
    if bits not in _SUPPORTED_BITS:
        raise CodecError(f"unsupported bit width: {bits}")
    raw = np.frombuffer(buffer, dtype=np.uint8)
    if bits == 8:
        if raw.size < count:
            raise CodecError("packed buffer is truncated")
        return raw[:count].copy()
    per_byte = 8 // bits
    required = (count + per_byte - 1) // per_byte
    if raw.size < required:
        raise CodecError("packed buffer is truncated")
    shifts = np.arange(per_byte - 1, -1, -1, dtype=np.uint8) * bits
    unpacked = ((raw[:required, None] >> shifts[None, :]) & ((1 << bits) - 1)).reshape(-1)
    return unpacked[:count].astype(np.uint8, copy=False)


def _quant_encode(
    residual: np.ndarray, config: QuantStageConfig
) -> tuple[dict, list[bytes], np.ndarray]:
    x = np.asarray(residual, dtype=np.float32).reshape(-1)
    group = config.group_size
    pad = (-x.size) % group
    padded = np.pad(x, (0, pad), mode="constant") if pad else x
    groups = padded.reshape(-1, group)
    qmax = (1 << (config.bits - 1)) - 1
    scale = np.max(np.abs(groups), axis=1) / max(qmax, 1)
    scale = np.where(scale == 0.0, np.float32(1.0), scale).astype(np.float32)
    signed = np.clip(np.rint(groups / scale[:, None]), -qmax, qmax).astype(np.int8)
    reconstructed = (signed.astype(np.float32) * scale[:, None]).reshape(-1)[: x.size]
    codes = (signed.reshape(-1).astype(np.int16) + qmax).astype(np.uint8)
    meta = {
        "bits": config.bits,
        "group_size": group,
        "value_count": int(x.size),
        "group_count": int(groups.shape[0]),
    }
    return (
        meta,
        [_pack_bits(codes, config.bits), scale.astype(np.float16).tobytes()],
        reconstructed.reshape(residual.shape),
    )


def _quant_decode(
    meta: dict, views: Iterable[bytes | memoryview | np.ndarray], shape: tuple[int, ...]
) -> np.ndarray:
    views = list(views)
    if len(views) != 2:
        raise CodecError("quant stage requires code and scale blobs")
    bits = int(meta["bits"])
    group = int(meta["group_size"])
    count = int(meta["value_count"])
    group_count = int(meta["group_count"])
    if count != int(np.prod(shape)) or group_count <= 0 or group <= 0:
        raise CodecError("quant metadata does not match requested shape")
    qmax = (1 << (bits - 1)) - 1
    codes = _unpack_bits(views[0], bits, group_count * group).astype(np.int16)
    signed = codes - qmax
    scales = np.frombuffer(views[1], dtype=np.float16)
    if scales.size != group_count:
        raise CodecError("quant scale blob has the wrong length")
    dequantized = (
        signed.reshape(group_count, group).astype(np.float32) * scales.astype(np.float32)[:, None]
    )
    return dequantized.reshape(-1)[:count].reshape(shape)


def _randomized_svd(
    matrix: np.ndarray,
    rank: int,
    *,
    seed: int,
    oversample: int,
    power_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise CodecError("SVD stage only supports matrices")
    m, n = matrix.shape
    rank = min(rank, m, n)
    width = min(rank + oversample, n)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, width), dtype=np.float32)
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iterations):
        q2, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
        q, _ = np.linalg.qr(matrix @ q2, mode="reduced")
    small = q.T @ matrix
    u_small, singular, vt = np.linalg.svd(small, full_matrices=False)
    return q @ u_small[:, :rank], singular[:rank], vt[:rank]


def _svd_encode(
    residual: np.ndarray, rank: int, config: CodecConfig
) -> tuple[dict, list[bytes], np.ndarray]:
    u, singular, vt = _randomized_svd(
        residual,
        rank,
        seed=config.seed,
        oversample=config.svd_oversample,
        power_iterations=config.svd_power_iterations,
    )
    root = np.sqrt(np.maximum(singular, 0.0)).astype(np.float32)
    left = (u * root[None, :]).astype(np.float16)
    right = (root[:, None] * vt).astype(np.float16)
    reconstruction = left.astype(np.float32) @ right.astype(np.float32)
    meta = {"rows": int(residual.shape[0]), "cols": int(residual.shape[1]), "rank": int(rank)}
    return meta, [left.tobytes(order="C"), right.tobytes(order="C")], reconstruction


def _svd_decode(meta: dict, views: Iterable[bytes | memoryview | np.ndarray]) -> np.ndarray:
    views = list(views)
    if len(views) != 2:
        raise CodecError("SVD stage requires two factor blobs")
    rows, cols, rank = int(meta["rows"]), int(meta["cols"]), int(meta["rank"])
    left = np.frombuffer(views[0], dtype=np.float16)
    right = np.frombuffer(views[1], dtype=np.float16)
    if left.size != rows * rank or right.size != rank * cols:
        raise CodecError("SVD blob length does not match metadata")
    return left.reshape(rows, rank).astype(np.float32) @ right.reshape(rank, cols).astype(
        np.float32
    )


def _raw_encode(array: np.ndarray) -> tuple[dict, list[bytes], np.ndarray]:
    raw = np.asarray(array, dtype=np.float16)
    return {"dtype": "float16"}, [raw.tobytes(order="C")], raw.astype(np.float32)


def encode_tensor(name: str, array: np.ndarray, config: CodecConfig) -> EncodedTensor:
    """Encode one bounded-size tensor chunk.

    Non-finite inputs are rejected because silently quantizing NaN/Inf values makes a
    store unverifiable and usually indicates a damaged source checkpoint.
    """

    source = np.asarray(array)
    if source.ndim == 0:
        source = source.reshape(1)
    if not np.issubdtype(source.dtype, np.number):
        raise CodecError(f"tensor {name!r} has unsupported dtype {source.dtype}")
    work = source.astype(np.float32)
    if not np.all(np.isfinite(work)):
        raise CodecError(f"tensor {name!r} contains NaN or infinity")

    raw_required = (
        config.force_raw
        or work.ndim != 2
        or min(work.shape) < config.min_dimension
        or any(fragment in name for fragment in config.raw_name_fragments)
    )
    stages: list[dict] = []
    residual = work

    if raw_required:
        meta, blobs, reconstruction = _raw_encode(work)
        stages.append({"kind": "raw", "meta": meta, "blobs": blobs})
        residual = work - reconstruction
    else:
        if config.rank_fraction > 0.0:
            rank = min(
                min(work.shape),
                max(config.min_rank, round(min(work.shape) * config.rank_fraction)),
            )
            raw_bytes = work.size * np.dtype(np.float16).itemsize
            factor_bytes = (work.shape[0] * rank + rank * work.shape[1]) * np.dtype(
                np.float16
            ).itemsize
            if factor_bytes < raw_bytes:
                meta, blobs, reconstruction = _svd_encode(residual, rank, config)
                stages.append({"kind": "svd", "meta": meta, "blobs": blobs})
                residual = residual - reconstruction
        for stage_config in config.quant_stages:
            meta, blobs, reconstruction = _quant_encode(residual, stage_config)
            stages.append({"kind": "quant", "meta": meta, "blobs": blobs})
            residual = residual - reconstruction
        if not stages:
            meta, blobs, reconstruction = _raw_encode(work)
            stages.append({"kind": "raw", "meta": meta, "blobs": blobs})
            residual = work - reconstruction

    source_l2_sq = float(np.vdot(work, work).real)
    error_l2_sq = float(np.vdot(residual, residual).real)
    rel_error = float(np.sqrt(error_l2_sq / max(source_l2_sq, 1e-30)))
    return EncodedTensor(stages, rel_error, error_l2_sq, source_l2_sq)


def decode_tensor(
    stages: Iterable[dict], shape: Iterable[int], max_stages: int | None = None
) -> np.ndarray:
    """Decode a chunk as float32, optionally using only an initial stage prefix."""

    shape_tuple = tuple(int(value) for value in shape)
    if any(value <= 0 for value in shape_tuple):
        raise CodecError(f"invalid tensor shape: {shape_tuple}")
    stages_list = list(stages)
    if max_stages is not None:
        if max_stages < 0:
            raise CodecError("max_stages must be non-negative")
        stages_list = stages_list[:max_stages]
    output = np.zeros(shape_tuple, dtype=np.float32)
    for stage in stages_list:
        try:
            kind = stage["kind"]
            meta = stage["meta"]
            views = stage["views"]
        except KeyError as exc:
            raise CodecError(f"malformed codec stage: missing {exc.args[0]}") from exc
        if kind == "raw":
            views = list(views)
            if len(views) != 1:
                raise CodecError("raw stage requires one blob")
            values = np.frombuffer(views[0], dtype=np.float16)
            if values.size != int(np.prod(shape_tuple)):
                raise CodecError("raw blob length does not match shape")
            part = values.reshape(shape_tuple).astype(np.float32)
        elif kind == "svd":
            part = _svd_decode(meta, views)
            if part.shape != shape_tuple:
                raise CodecError("SVD metadata does not match shape")
        elif kind == "quant":
            part = _quant_decode(meta, views, shape_tuple)
        else:
            raise CodecError(f"unknown codec stage kind: {kind!r}")
        output += part
    return output
