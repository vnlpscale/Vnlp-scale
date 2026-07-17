from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def tiny_checkpoint(tmp_path: Path):
    source = tmp_path / "tiny-model"
    source.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
    }
    rng = np.random.default_rng(7)

    def weight(*shape, scale=0.03):
        return (rng.standard_normal(shape) * scale).astype(np.float16)

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    kv_width = config["num_key_value_heads"] * (hidden // config["num_attention_heads"])
    tensors = {
        "model.embed_tokens.weight": weight(config["vocab_size"], hidden, scale=0.1),
        "model.norm.weight": np.ones(hidden, dtype=np.float16),
        "lm_head.weight": weight(config["vocab_size"], hidden, scale=0.1),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = weight(hidden, hidden)
        tensors[prefix + "self_attn.k_proj.weight"] = weight(kv_width, hidden)
        tensors[prefix + "self_attn.v_proj.weight"] = weight(kv_width, hidden)
        tensors[prefix + "self_attn.o_proj.weight"] = weight(hidden, hidden)
        tensors[prefix + "mlp.gate_proj.weight"] = weight(intermediate, hidden)
        tensors[prefix + "mlp.up_proj.weight"] = weight(intermediate, hidden)
        tensors[prefix + "mlp.down_proj.weight"] = weight(hidden, intermediate)
        tensors[prefix + "input_layernorm.weight"] = np.ones(hidden, dtype=np.float16)
        tensors[prefix + "post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float16)

    names = sorted(tensors)
    midpoint = len(names) // 2
    shard_map = {
        "model-00001-of-00002.safetensors": names[:midpoint],
        "model-00002-of-00002.safetensors": names[midpoint:],
    }
    weight_map = {}
    total_size = 0
    for filename, shard_names in shard_map.items():
        payload = {name: tensors[name] for name in shard_names}
        save_file(payload, source / filename)
        for name in shard_names:
            weight_map[name] = filename
            total_size += tensors[name].nbytes
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return source, config, tensors
