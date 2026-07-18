"""Vnlp-scale: bounded-memory storage and inference primitives for very large LLMs."""

# Import for its NumPy BF16 dtype-registration side effect.
import ml_dtypes  # noqa: F401

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
