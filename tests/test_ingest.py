import json

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

from vnlp_scale.errors import StoreError
from vnlp_scale.ingest import record
from vnlp_scale.store import StoreReader


def test_record_is_chunked_verified_and_resumable(tmp_path, tiny_checkpoint):
    source, _, tensors = tiny_checkpoint
    output = tmp_path / "encoded"
    first = record(
        str(source),
        str(output),
        quality="lossless",
        max_chunk_bytes=64,
        progress=None,
    )
    second = record(
        str(source),
        str(output),
        quality="lossless",
        max_chunk_bytes=64,
        progress=None,
    )
    assert first["encoded_chunks"] > len(tensors)
    assert second["encoded_chunks"] == 0
    assert second["skipped_chunks"] == first["encoded_chunks"]
    with StoreReader(output, verify_on_open=True) as reader:
        assert reader.summary()["tensors"] == len(tensors)
        assert reader.manifest["finalized"] is True
    assert (output / "config.json").is_file()


def test_record_accepts_bfloat16_safetensors(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": np.asarray(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=ml_dtypes.bfloat16,
            )
        },
        source / "model.safetensors",
    )
    (source / "config.json").write_text("{}", encoding="utf-8")

    output = tmp_path / "encoded"
    result = record(
        str(source),
        str(output),
        quality="lossless",
        max_chunk_bytes=8,
        progress=None,
    )

    assert result["encoded_tensors"] == 1
    with StoreReader(output, verify_on_open=True) as reader:
        assert reader.manifest["tensors"]["model.embed_tokens.weight"]["dtype"] == "BF16"


def test_local_index_path_traversal_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(StoreError, match="path traversal"):
        record(str(source), str(tmp_path / "out"), progress=None)


def test_resume_rejects_changed_codec_settings(tmp_path, tiny_checkpoint):
    source, _, _ = tiny_checkpoint
    output = tmp_path / "encoded"
    record(
        str(source),
        str(output),
        quality="lossless",
        max_chunk_bytes=64,
        progress=None,
    )
    with pytest.raises(StoreError, match="settings differ"):
        record(
            str(source),
            str(output),
            quality="med",
            max_chunk_bytes=64,
            progress=None,
        )


def test_overwrite_removes_stale_managed_metadata(tmp_path, tiny_checkpoint):
    source, _, _ = tiny_checkpoint
    output = tmp_path / "encoded"
    record(str(source), str(output), quality="lossless", progress=None)
    stale = output / "tokenizer.json"
    stale.write_text("stale", encoding="utf-8")
    record(
        str(source),
        str(output),
        quality="lossless",
        overwrite=True,
        progress=None,
    )
    assert not stale.exists()
