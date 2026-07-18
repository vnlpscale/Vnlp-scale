"""Vnlp-scale: bounded-memory storage and inference primitives for very large LLMs."""

import ml_dtypes
import numpy as np

from . import codec as _codec
from .codec import PRESETS, CodecConfig, decode_tensor, preset
from .estimate import HardwareProfile, ModelProfile, plan_inference
from .ingest import record
from .store import StoreReader, StoreWriter

_BFLOAT16_DTYPE = np.dtype(ml_dtypes.bfloat16)
_ORIGINAL_ENCODE_TENSOR = _codec.encode_tensor


def encode_tensor(name: str, array: np.ndarray, config: CodecConfig):
    """Encode a tensor, normalizing NumPy-compatible BF16 chunks to float32."""

    source = np.asarray(array)
    if source.dtype == _BFLOAT16_DTYPE:
        source = source.astype(np.float32)
    return _ORIGINAL_ENCODE_TENSOR(name, source, config)


# Ingest keeps a module reference to codec, so update that entry point as well.
_codec.encode_tensor = encode_tensor

__all__ = [
    "PRESETS",
    "CodecConfig",
    "HardwareProfile",
    "ModelProfile",
    "StoreReader",
    "StoreWriter",
    "decode_tensor",
    "encode_tensor",
    "plan_inference",
    "preset",
    "record",
]

__version__ = "0.1.0"
