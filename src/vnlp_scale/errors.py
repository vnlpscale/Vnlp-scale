"""Project-specific exception hierarchy."""


class VnlpScaleError(Exception):
    """Base exception for recoverable Vnlp-scale errors."""


class CodecError(VnlpScaleError):
    """Raised when a tensor cannot be encoded or decoded safely."""


class StoreError(VnlpScaleError):
    """Raised for malformed, incompatible, or corrupted stores."""


class StoreLockedError(StoreError):
    """Raised when another writer owns the store lock."""


class UnsupportedModelError(VnlpScaleError):
    """Raised when a model architecture is not supported by a runtime."""
