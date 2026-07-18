import numpy as np
import pytest

from vnlp_scale import codec


class _EncodedStageStore:
    def __init__(self, weight, left, right, signed, scales, *, bits, group_size):
        rows, cols = weight.shape
        qmax = (1 << (bits - 1)) - 1
        codes = (signed.reshape(-1).astype(np.int16) + qmax).astype(np.uint8)
        self._views = [
            left.astype(np.float16).view(np.uint8).reshape(-1),
            right.astype(np.float16).view(np.uint8).reshape(-1),
            np.frombuffer(codec._pack_bits(codes, bits), dtype=np.uint8),
            scales.astype(np.float16).view(np.uint8).reshape(-1),
        ]
        self._weight = weight.astype(np.float32)
        self.decode_calls = 0
        self._tensor = {
            "shape": [rows, cols],
            "chunks": [
                {
                    "start": 0,
                    "stop": rows,
                    "shape": [rows, cols],
                    "stages": [
                        {
                            "kind": "svd",
                            "meta": {"rows": rows, "cols": cols, "rank": left.shape[1]},
                            "blobs": [{"view": 0}, {"view": 1}],
                        },
                        {
                            "kind": "quant",
                            "meta": {
                                "bits": bits,
                                "group_size": group_size,
                                "value_count": rows * cols,
                                "group_count": scales.size,
                            },
                            "blobs": [{"view": 2}, {"view": 3}],
                        },
                    ],
                }
            ],
        }

    def info(self, name):
        assert name == "weight"
        return self._tensor

    def shape(self, name):
        return tuple(self.info(name)["shape"])

    def chunk_count(self, name):
        return len(self.info(name)["chunks"])

    def _views_for_stage(self, stage):
        return [self._views[blob["view"]] for blob in stage["blobs"]]

    def decode_chunk(self, name, index, *, max_stages=None):
        assert name == "weight"
        assert index == 0
        self.decode_calls += 1
        if max_stages == 0:
            return np.zeros_like(self._weight)
        if max_stages == 1:
            stage = self._tensor["chunks"][0]["stages"][0]
            views = self._views_for_stage(stage)
            return codec._svd_decode(stage["meta"], views)
        return self._weight.copy()

    def close(self):
        return None


def _encoded_store():
    rng = np.random.default_rng(19)
    rows, cols, rank = 11, 12, 3
    bits, group_size = 4, 4
    left = rng.normal(size=(rows, rank)).astype(np.float32)
    right = rng.normal(size=(rank, cols)).astype(np.float32)
    signed = rng.integers(-7, 8, size=(rows, cols), dtype=np.int8)
    scales = rng.uniform(0.01, 0.2, size=rows * cols // group_size).astype(np.float32)
    stored_left = left.astype(np.float16).astype(np.float32)
    stored_right = right.astype(np.float16).astype(np.float32)
    stored_scales = scales.astype(np.float16).astype(np.float32)
    quant = signed.reshape(-1, group_size).astype(np.float32) * stored_scales[:, None]
    weight = stored_left @ stored_right + quant.reshape(rows, cols)
    return _EncodedStageStore(
        weight,
        left,
        right,
        signed,
        scales,
        bits=bits,
        group_size=group_size,
    ), weight


def test_stage_aware_linear_matches_materialized_weight():
    torch = pytest.importorskip("torch")
    from vnlp_scale.engine_torch import TorchStoreProvider

    store, weight = _encoded_store()
    provider = TorchStoreProvider(
        store,
        device="cpu",
        dtype="float32",
        cache_bytes=0,
        quant_block_rows=3,
        triton_quant=False,
    )
    inputs = torch.arange(60, dtype=torch.float32).reshape(5, 12) / 17.0
    try:
        actual = provider.linear("weight", inputs)
    finally:
        provider.close()
    expected = inputs @ torch.from_numpy(weight).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert store.decode_calls == 0
    assert provider.stats["svd_matmul_stages"] == 1
    assert provider.stats["quant_matmul_stages"] == 1
    assert provider.stats["materialized_weight_bytes_avoided"] > 0


def test_stage_aware_linear_respects_progressive_stage_limit():
    torch = pytest.importorskip("torch")
    from vnlp_scale.engine_torch import TorchStoreProvider

    store, _ = _encoded_store()
    provider = TorchStoreProvider(
        store,
        device="cpu",
        dtype="float32",
        max_stages=1,
        triton_quant=False,
    )
    inputs = torch.eye(12, dtype=torch.float32)[:4]
    try:
        actual = provider.linear("weight", inputs)
        expected_weight = store.decode_chunk("weight", 0, max_stages=1)
    finally:
        provider.close()
    expected = inputs @ torch.from_numpy(expected_weight).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert provider.stats["svd_matmul_stages"] == 1
    assert provider.stats["quant_matmul_stages"] == 0
