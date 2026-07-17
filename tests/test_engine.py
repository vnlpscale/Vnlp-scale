import json

import numpy as np
import pytest

from vnlp_scale.engine_np import DictProvider, NumpyLlamaEngine
from vnlp_scale.errors import StoreError
from vnlp_scale.ingest import record


def test_numpy_engine_handles_multitoken_causal_mask(tiny_checkpoint):
    _, config, tensors = tiny_checkpoint
    engine = NumpyLlamaEngine(DictProvider(tensors), config)
    try:
        result = engine.generate([1, 2, 3, 4, 5], 2, trace_logits=True)
    finally:
        engine.close()
    assert len(result["tokens"]) == 2
    assert result["logits_trace"][0].shape == (config["vocab_size"],)
    assert np.all(np.isfinite(result["logits_trace"][0]))


def test_chunked_lossless_store_matches_in_memory_engine(tmp_path, tiny_checkpoint):
    source, config, tensors = tiny_checkpoint
    store = tmp_path / "store"
    record(
        str(source),
        str(store),
        quality="lossless",
        max_chunk_bytes=64,
        progress=None,
    )
    reference = NumpyLlamaEngine(DictProvider(tensors), config)
    streamed = NumpyLlamaEngine.from_store(store, cache_bytes=0, verify=True)
    try:
        expected = reference.generate([1, 7, 3], 3, trace_logits=True)
        actual = streamed.generate([1, 7, 3], 3, trace_logits=True)
    finally:
        reference.close()
        streamed.close()
    assert actual["tokens"] == expected["tokens"]
    for left, right in zip(actual["logits_trace"], expected["logits_trace"], strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-5)
    assert actual["provider_stats"]["chunks_decoded"] > 0


def test_torch_cpu_backend_matches_numpy(tmp_path, tiny_checkpoint):
    torch = pytest.importorskip("torch")
    from vnlp_scale.engine_torch import TorchLlamaEngine

    source, config, tensors = tiny_checkpoint
    store = tmp_path / "store"
    record(str(source), str(store), quality="lossless", max_chunk_bytes=64, progress=None)
    numpy_engine = NumpyLlamaEngine(DictProvider(tensors), config)
    torch_engine = TorchLlamaEngine.from_store(
        store, device="cpu", dtype="float32", cache_bytes=0, verify=True
    )
    try:
        numpy_result = numpy_engine.generate([2, 4, 6], 2)
        torch_result = torch_engine.generate([2, 4, 6], 2)
    finally:
        numpy_engine.close()
        torch_engine.close()
    assert torch_result["tokens"] == numpy_result["tokens"]
    assert torch.__version__


def test_runtime_rejects_unfinalized_store(tmp_path, tiny_checkpoint):
    source, _, _ = tiny_checkpoint
    store = tmp_path / "store"
    record(str(source), str(store), quality="lossless", progress=None)
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finalized"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StoreError, match="unfinalized"):
        NumpyLlamaEngine.from_store(store)
