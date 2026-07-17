"""Experimental chunk-streaming PyTorch runtime for Llama-compatible stores.

Unlike integrations that materialize an entire transformer layer, this runtime decodes
and transfers one row chunk for each matrix multiplication. It therefore preserves the
same bounded-memory property as the NumPy reference backend. CUDA performance is not
claimed until hardware-specific benchmarks are published.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .engine_np import layer_parameter_names
from .errors import StoreError, UnsupportedModelError
from .store import StoreReader


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch backend requires: pip install 'vnlp-scale[torch]'") from exc
    return torch


def _resolve_dtype(torch, dtype):
    if dtype is None:
        return torch.float16
    if not isinstance(dtype, str):
        return dtype
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {dtype}") from exc


class TorchStoreProvider:
    def __init__(
        self,
        store: StoreReader,
        *,
        device: str = "cuda",
        dtype=None,
        cache_bytes: int = 0,
        max_stages: int | None = None,
        pin_memory: bool = True,
        max_materialize_bytes: int = 64 * 1024 * 1024,
    ):
        torch = _import_torch()
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        self.torch = torch
        self.store = store
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        self.dtype = _resolve_dtype(torch, dtype)
        self.cache_budget = cache_bytes
        self.max_stages = max_stages
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        self.max_materialize_bytes = max_materialize_bytes
        self._cache: OrderedDict[tuple[str, int, int | None], object] = OrderedDict()
        self._cache_bytes = 0
        self.stats = {
            "chunks_decoded": 0,
            "tensors_decoded": 0,
            "decode_seconds": 0.0,
            "transfer_seconds": 0.0,
            "decoded_bytes_total": 0,
            "cache_hits": 0,
            "cache_bytes": 0,
            "peak_live_bytes": 0,
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _decode_chunk(self, name: str, index: int):
        key = (name, index, self.max_stages)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return cached

        started = time.perf_counter()
        array = self.store.decode_chunk(name, index, max_stages=self.max_stages)
        self.stats["decode_seconds"] += time.perf_counter() - started
        self.stats["chunks_decoded"] += 1
        self.stats["decoded_bytes_total"] += array.nbytes

        cpu = self.torch.from_numpy(array)
        if self.pin_memory:
            cpu = cpu.pin_memory()
        self._sync()
        transfer_started = time.perf_counter()
        tensor = cpu.to(self.device, dtype=self.dtype, non_blocking=self.pin_memory)
        self._sync()
        self.stats["transfer_seconds"] += time.perf_counter() - transfer_started
        tensor_bytes = tensor.numel() * tensor.element_size()
        self.stats["peak_live_bytes"] = max(
            self.stats["peak_live_bytes"], self._cache_bytes + tensor_bytes
        )
        if tensor_bytes <= self.cache_budget:
            self._cache[key] = tensor
            self._cache_bytes += tensor_bytes
            while self._cache_bytes > self.cache_budget and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.numel() * evicted.element_size()
            self.stats["cache_bytes"] = self._cache_bytes
        return tensor

    def get(self, name: str):
        torch = self.torch
        decoded_bytes = int(np.prod(self.store.shape(name))) * 4
        if decoded_bytes > self.max_materialize_bytes:
            raise StoreError(
                f"refusing to materialize {name!r} ({decoded_bytes} bytes); use linear or embedding"
            )
        pieces = [self._decode_chunk(name, index) for index in range(self.store.chunk_count(name))]
        self.stats["tensors_decoded"] += 1
        if not pieces:
            raise StoreError(f"tensor has no chunks: {name}")
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def embedding(self, name: str, ids):
        torch = self.torch
        tensor = self.store.info(name)
        shape = tuple(int(value) for value in tensor["shape"])
        if len(shape) != 2:
            raise StoreError(f"embedding tensor must be a matrix: {name}")
        ids = ids.to(self.device, dtype=torch.long)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= shape[0]):
            raise ValueError("token ID is outside the embedding vocabulary")
        result = torch.empty((*tuple(ids.shape), shape[1]), device=self.device, dtype=self.dtype)
        flat_ids = ids.reshape(-1)
        flat_result = result.reshape(-1, shape[1])
        for chunk_index, chunk in enumerate(tensor["chunks"]):
            start, stop = int(chunk["start"]), int(chunk["stop"])
            mask = (flat_ids >= start) & (flat_ids < stop)
            if not bool(mask.any()):
                continue
            values = self._decode_chunk(name, chunk_index)
            flat_result[mask] = values[flat_ids[mask] - start]
        self.stats["tensors_decoded"] += 1
        return result

    def linear(self, name: str, inputs):
        torch = self.torch
        tensor = self.store.info(name)
        shape = tuple(int(value) for value in tensor["shape"])
        if len(shape) != 2:
            raise StoreError(f"linear weight must be a matrix: {name}")
        if inputs.shape[-1] != shape[1]:
            raise ValueError(f"linear dimension mismatch for {name}")
        flat = inputs.reshape(-1, shape[1])
        output = torch.empty((flat.shape[0], shape[0]), device=self.device, dtype=self.dtype)
        for chunk_index, chunk in enumerate(tensor["chunks"]):
            start, stop = int(chunk["start"]), int(chunk["stop"])
            weight = self._decode_chunk(name, chunk_index)
            output[:, start:stop] = flat @ weight.transpose(0, 1)
        self.stats["tensors_decoded"] += 1
        return output.reshape((*tuple(inputs.shape[:-1]), shape[0]))

    def close(self) -> None:
        self._cache.clear()
        self._cache_bytes = 0
        self.stats["cache_bytes"] = 0
        self.store.close()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


class TorchLlamaEngine:
    def __init__(self, provider: TorchStoreProvider, config: dict):
        torch = provider.torch
        self.torch = torch
        self.provider = provider
        self.config = config
        self.hidden_size = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.attention_heads = int(config["num_attention_heads"])
        self.kv_heads = int(config.get("num_key_value_heads", self.attention_heads))
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.attention_heads))
        self.vocabulary_size = int(config["vocab_size"])
        self.rms_epsilon = float(config.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(config.get("rope_theta", 10_000.0))
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        self._validate_config()
        self.final_norm = provider.get("model.norm.weight")
        self.layer_norms = []
        for layer_index in range(self.layers):
            names = layer_parameter_names(layer_index)
            self.layer_norms.append((provider.get(names["ln1"]), provider.get(names["ln2"])))
        self.lm_head_name = (
            "model.embed_tokens.weight" if self.tie_word_embeddings else "lm_head.weight"
        )
        half = self.head_dim // 2
        self.inverse_frequency = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(half, dtype=torch.float32, device=provider.device)
                * 2.0
                / self.head_dim
            )
        )

    def _validate_config(self) -> None:
        model_type = self.config.get("model_type")
        if model_type not in {None, "llama", "mistral"}:
            raise UnsupportedModelError(f"unsupported model_type={model_type!r}")
        if model_type == "mistral" and self.config.get("sliding_window") not in (None, 0):
            raise UnsupportedModelError("Mistral sliding-window attention is not implemented")
        if self.hidden_size != self.attention_heads * self.head_dim:
            raise UnsupportedModelError("hidden_size must equal heads * head_dim")
        if self.attention_heads % self.kv_heads:
            raise UnsupportedModelError("attention heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise UnsupportedModelError("RoPE requires an even head_dim")
        if self.config.get("rope_scaling") not in (None, {}):
            raise UnsupportedModelError("rope_scaling is not implemented")
        if self.config.get("attention_bias", False) or self.config.get("mlp_bias", False):
            raise UnsupportedModelError("bias-enabled variants are not implemented")

    @classmethod
    def from_store(
        cls,
        store_directory: str | os.PathLike[str],
        *,
        device: str = "cuda",
        dtype=None,
        cache_bytes: int = 0,
        max_stages: int | None = None,
        verify: bool = False,
    ) -> TorchLlamaEngine:
        directory = Path(store_directory)
        try:
            with (directory / "config.json").open(encoding="utf-8") as handle:
                config = json.load(handle)
        except FileNotFoundError as exc:
            raise StoreError("store does not include config.json") from exc
        reader = StoreReader(directory, verify_on_open=verify)
        if not reader.manifest.get("finalized", False):
            reader.close()
            raise StoreError("refusing to run an unfinalized store")
        provider = TorchStoreProvider(
            reader,
            device=device,
            dtype=dtype,
            cache_bytes=cache_bytes,
            max_stages=max_stages,
        )
        return cls(provider, config)

    def _rms_norm(self, values, weight):
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * self.torch.rsqrt(variance + self.rms_epsilon).to(values.dtype)
        return normalized * weight

    def _rope(self, values, positions):
        torch = self.torch
        half = self.head_dim // 2
        angles = positions.float().unsqueeze(1) * self.inverse_frequency.unsqueeze(0)
        cosine = torch.cos(angles).unsqueeze(1).to(values.dtype)
        sine = torch.sin(angles).unsqueeze(1).to(values.dtype)
        first, second = values[..., :half], values[..., half:]
        return torch.cat([first * cosine - second * sine, second * cosine + first * sine], dim=-1)

    def _attention(self, values, names, key_cache, value_cache, position_start):
        torch = self.torch
        token_count = values.shape[0]
        query = self.provider.linear(names["q"], values).reshape(
            token_count, self.attention_heads, self.head_dim
        )
        key = self.provider.linear(names["k"], values).reshape(
            token_count, self.kv_heads, self.head_dim
        )
        current_value = self.provider.linear(names["v"], values).reshape(
            token_count, self.kv_heads, self.head_dim
        )
        positions = torch.arange(
            position_start,
            position_start + token_count,
            device=self.provider.device,
            dtype=torch.long,
        )
        query, key = self._rope(query, positions), self._rope(key, positions)
        key_cache = key if key_cache is None else torch.cat([key_cache, key], dim=0)
        value_cache = (
            current_value if value_cache is None else torch.cat([value_cache, current_value], dim=0)
        )
        repetitions = self.attention_heads // self.kv_heads
        expanded_key = key_cache.repeat_interleave(repetitions, dim=1)
        expanded_value = value_cache.repeat_interleave(repetitions, dim=1)
        scores = torch.einsum("thd,shd->ths", query.float(), expanded_key.float())
        scores /= math.sqrt(self.head_dim)
        key_positions = torch.arange(key_cache.shape[0], device=self.provider.device)
        causal = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
        scores = scores.masked_fill(~causal.unsqueeze(1), -1e30)
        probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
        context = torch.einsum("ths,shd->thd", probabilities, expanded_value)
        context = context.reshape(token_count, self.hidden_size)
        return self.provider.linear(names["o"], context), key_cache, value_cache

    def _mlp(self, values, names):
        torch = self.torch
        gate = torch.nn.functional.silu(self.provider.linear(names["gate"], values))
        up = self.provider.linear(names["up"], values)
        return self.provider.linear(names["down"], gate * up)

    def _forward_step(self, token_ids, key_cache, value_cache, position_start):
        torch = self.torch
        ids = torch.as_tensor(token_ids, device=self.provider.device, dtype=torch.long)
        hidden = self.provider.embedding("model.embed_tokens.weight", ids)
        for layer_index in range(self.layers):
            names = layer_parameter_names(layer_index)
            first_norm, second_norm = self.layer_norms[layer_index]
            attention, key_cache[layer_index], value_cache[layer_index] = self._attention(
                self._rms_norm(hidden, first_norm),
                names,
                key_cache[layer_index],
                value_cache[layer_index],
                position_start,
            )
            hidden = hidden + attention
            hidden = hidden + self._mlp(self._rms_norm(hidden, second_norm), names)
        hidden = self._rms_norm(hidden, self.final_norm)
        return self.provider.linear(self.lm_head_name, hidden[-1:])[0]

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> dict:
        torch = self.torch
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if any(token < 0 or token >= self.vocabulary_size for token in prompt_ids):
            raise ValueError("prompt contains a token outside the vocabulary")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        generator = torch.Generator(device=self.provider.device)
        generator.manual_seed(seed)
        key_cache = [None] * self.layers
        value_cache = [None] * self.layers
        generated = []
        timings = []
        feed = list(prompt_ids)
        position_start = 0

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if self.provider.device.type == "cuda":
                    torch.cuda.synchronize(self.provider.device)
                started = time.perf_counter()
                logits = self._forward_step(feed, key_cache, value_cache, position_start)
                if self.provider.device.type == "cuda":
                    torch.cuda.synchronize(self.provider.device)
                timings.append(time.perf_counter() - started)
                if greedy:
                    next_token = int(torch.argmax(logits).item())
                else:
                    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
                    next_token = int(
                        torch.multinomial(probabilities, 1, generator=generator).item()
                    )
                generated.append(next_token)
                position_start += len(feed)
                feed = [next_token]

        kv_bytes = sum(
            value.numel() * value.element_size() for value in key_cache if value is not None
        ) + sum(value.numel() * value.element_size() for value in value_cache if value is not None)
        return {
            "tokens": generated,
            "all_tokens": list(prompt_ids) + generated,
            "step_seconds": timings,
            "tokens_per_second": len(timings) / sum(timings) if timings else 0.0,
            "kv_bytes": kv_bytes,
            "provider_stats": dict(self.provider.stats),
        }

    def close(self) -> None:
        self.provider.close()

    def __enter__(self) -> TorchLlamaEngine:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
