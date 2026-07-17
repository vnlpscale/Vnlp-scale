import numpy as np
import pytest

from vnlp_scale.codec import _pack_bits, _unpack_bits, decode_tensor, encode_tensor, preset
from vnlp_scale.errors import CodecError


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_bit_pack_roundtrip(bits):
    rng = np.random.default_rng(1)
    values = rng.integers(0, 1 << bits, size=137, dtype=np.uint8)
    packed = _pack_bits(values, bits)
    restored = _unpack_bits(packed, bits, values.size)
    np.testing.assert_array_equal(restored, values)


def _views(encoded):
    return [
        {"kind": stage["kind"], "meta": stage["meta"], "views": stage["blobs"]}
        for stage in encoded.stages
    ]


def test_lossless_preset_roundtrip_is_fp16_exact():
    rng = np.random.default_rng(2)
    source = rng.standard_normal((17, 11), dtype=np.float32)
    encoded = encode_tensor("weight", source, preset("lossless"))
    restored = decode_tensor(_views(encoded), source.shape)
    np.testing.assert_array_equal(restored, source.astype(np.float16).astype(np.float32))


def test_recursive_codec_is_bounded_and_progressive():
    rng = np.random.default_rng(3)
    source = rng.standard_normal((96, 80), dtype=np.float32)
    encoded = encode_tensor("linear.weight", source, preset("high"))
    full = decode_tensor(_views(encoded), source.shape)
    empty = decode_tensor(_views(encoded), source.shape, max_stages=0)
    assert len(encoded.stages) >= 2
    assert np.isfinite(encoded.rel_error)
    assert encoded.rel_error < 0.6
    np.testing.assert_array_equal(empty, np.zeros_like(source))
    assert np.linalg.norm(source - full) < np.linalg.norm(source)


def test_non_finite_source_is_rejected():
    source = np.ones((8, 8), dtype=np.float32)
    source[0, 0] = np.nan
    with pytest.raises(CodecError, match="NaN"):
        encode_tensor("weight", source, preset("med"))
