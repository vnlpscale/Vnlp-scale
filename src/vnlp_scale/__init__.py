"""Vnlp-scale: bounded-memory storage and inference primitives for very large LLMs."""

import ml_dtypes
import numpy as np

# Register BF16 with NumPy before safetensors opens tensors through its NumPy backend.
_BFLOAT16_DTYPE = np.dtype(ml_dtypes.bfloat16)

from .codec import PRESETS, CodecConfig, decode_tensor, encode_tensor, preset
from .estimate import HardwareProfile, ModelProfile, plan_inference
from .ingest import record
from .store import StoreReader, StoreWriter

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
