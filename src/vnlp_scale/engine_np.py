"""Bounded-memory NumPy reference runtime for Llama-family checkpoints.

This backend prioritizes auditability and memory bounds over speed. Matrix weights are
read and decoded by row chunk, multiplied, then released. It is suitable for codec
validation and CPU experiments; production GPU throughput requires dedicated kernels.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .errors import StoreError, UnsupportedModelError
from .store import StoreReader

_LAYER_NAMES = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
    "ln1": "input_layernorm.weight",
    "ln2": "post_attention_layernorm.weight",
}


def layer_parameter_names(index: int) -> dict[str, str]:
    return {short: f"model.layers.{index}.{suffix}" for short, suffix in _LAYER_NAMES.items()}


class DictProvider:
    """In-memory baseline provider used for numerical comparisons."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = {
            name: np.asarray(value, dtype=np.float32) for name, value in weights.items()
        }
        resident = sum(value.nbytes for value in self.weights.values())
        self.stats = {
            "chunks_decoded": 0,
            "tensors_decoded": 0,
            "decode_seconds": 0.0,
            "decoded_bytes_total": 0,
            "cache_hits": 0,
            "cache_bytes": resident,
            "peak_live_bytes": resident,
        }

    def get(self, name: str) -> np.ndarray:
        return self.weights[name]

    def embedding(self, name: str, ids: np.ndarray) -> np.ndarray:
        return self.weights[name][ids]

    def linear(self, name: str, inputs: np.ndarray) -> np.ndarray:
        weight = self.weights[name]
        if inputs.shape[-1] != weight.shape[1]:
            raise ValueError(f"linear dimension mismatch for {name}")
        return inputs @ weight.T

    def close(self) -> None:
        return None


class StoreProvider:
    """Chunk-streaming store provider with a byte-bounded decoded-chunk LRU."""

    def __init__(
        self,
        store: StoreReader,
        *,
        cache_bytes: int = 256 * 1024 * 1024,
        max_stages: int | None = None,
        max_materialize_bytes: int = 64 * 1024 * 1024,
    ):
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        self.store = store
        self.cache_budget = cache_bytes
        self.max_stages = max_stages
        self.max_materialize_bytes = max_materialize_bytes
        self._cache: OrderedDict[tuple[str, int, int | None], np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self.stats = {
            "chunks_decoded": 0,
            "tensors_decoded": 0,
            "decode_seconds": 0.0,
            "decoded_bytes_total": 0,
            "cache_hits": 0,
            "cache_bytes": 0,
            "peak_live_bytes": 0,
        }

    def _decode_chunk(self, name: str, index: int) -> np.ndarray:
        key = (name, index, self.max_stages)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return cached
        started = time.perf_counter()
        values = self.store.decode_chunk(name, index, max_stages=self.max_stages)
        elapsed = time.perf_counter() - started
        self.stats["chunks_decoded"] += 1
        self.stats["decode_seconds"] += elapsed
        self.stats["decoded_bytes_total"] += values.nbytes
        self.stats["peak_live_bytes"] = max(
            self.stats["peak_live_bytes"], self._cache_bytes + values.nbytes
        )
        if values.nbytes <= self.cache_budget:
            self._cache[key] = values
            self._cache_bytes += values.nbytes
            while self._cache_bytes > self.cache_budget and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.nbytes
            self.stats["cache_bytes"] = self._cache_bytes
        return values

    def get(self, name: str) -> np.ndarray:
        decoded_bytes = int(np.prod(self.store.shape(name))) * np.dtype(np.float32).itemsize
        if decoded_bytes > self.max_materialize_bytes:
            raise StoreError(
                f"refusing to materialize {name!r} ({decoded_bytes} bytes); use linear or embedding"
            )
        pieces = [self._decode_chunk(name, index) for index in range(self.store.chunk_count(name))]
        self.stats["tensors_decoded"] += 1
        if not pieces:
            raise StoreError(f"tensor has no chunks: {name}")
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    def embedding(self, name: str, ids: np.ndarray) -> np.ndarray:
        tensor = self.store.info(name)
        shape = tuple(int(value) for value in tensor["shape"])
        if len(shape) != 2:
            raise StoreError(f"embedding tensor must be a matrix: {name}")
        indices = np.asarray(ids, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= shape[0]):
            raise ValueError("token ID is outside the embedding vocabulary")
        result = np.empty((*indices.shape, shape[1]), dtype=np.float32)
        flat_indices = indices.reshape(-1)
        flat_result = result.reshape(-1, shape[1])
        for chunk_index, chunk in enumerate(tensor["chunks"]):
            start, stop = int(chunk["start"]), int(chunk["stop"])
            mask = (flat_indices >= start) & (flat_indices < stop)
            if not np.any(mask):
                continue
            values = self._decode_chunk(name, chunk_index)
            flat_result[mask] = values[flat_indices[mask] - start]
        self.stats["tensors_decoded"] += 1
        return result

    def linear(self, name: str, inputs: np.ndarray) -> np.ndarray:
        tensor = self.store.info(name)
        shape = tuple(int(value) for value in tensor["shape"])
        if len(shape) != 2:
            raise StoreError(f"linear weight must be a matrix: {name}")
        values = np.asarray(inputs, dtype=np.float32)
        if values.shape[-1] != shape[1]:
            raise ValueError(
                f"linear dimension mismatch for {name}: input={values.shape[-1]}, weight={shape}"
            )
        flat = values.reshape(-1, shape[1])
        output = np.empty((flat.shape[0], shape[0]), dtype=np.float32)
        for chunk_index, chunk in enumerate(tensor["chunks"]):
            start, stop = int(chunk["start"]), int(chunk["stop"])
            weight = self._decode_chunk(name, chunk_index)
            output[:, start:stop] = flat @ weight.T
        self.stats["tensors_decoded"] += 1
        return output.reshape((*values.shape[:-1], shape[0]))

    def close(self) -> None:
        self._cache.clear()
        self._cache_bytes = 0
        self.stats["cache_bytes"] = 0
        self.store.close()


class NumpyLlamaEngine:
    """Auditable Llama forward pass using a pluggable weight provider."""

    def __init__(self, provider: DictProvider | StoreProvider, config: dict):
        self.provider = provider
        self.config = config
        self.hidden_size = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.attention_heads = int(config["num_attention_heads"])
        self.kv_heads = int(config.get("num_key_value_heads", self.attention_heads))
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.attention_heads))
        self.intermediate_size = int(config["intermediate_size"])
        self.vocabulary_size = int(config["vocab_size"])
        self.rms_epsilon = float(config.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(config.get("rope_theta", 10_000.0))
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        self._validate_config()

        self.final_norm = provider.get("model.norm.weight").astype(np.float32, copy=False)
        self.layer_norms = []
        for layer_index in range(self.layers):
            names = layer_parameter_names(layer_index)
            self.layer_norms.append(
                (
                    provider.get(names["ln1"]).astype(np.float32, copy=False),
                    provider.get(names["ln2"]).astype(np.float32, copy=False),
                )
            )
        if self.tie_word_embeddings:
            self.lm_head_name = "model.embed_tokens.weight"
        else:
            self.lm_head_name = "lm_head.weight"
        half = self.head_dim // 2
        self._inverse_frequency = 1.0 / (
            self.rope_theta ** (np.arange(half, dtype=np.float64) * 2.0 / self.head_dim)
        )

    def _validate_config(self) -> None:
        model_type = self.config.get("model_type")
        if model_type not in {None, "llama", "mistral"}:
            raise UnsupportedModelError(
                f"NumPy backend supports Llama-compatible layouts, got model_type={model_type!r}"
            )
        if model_type == "mistral" and self.config.get("sliding_window") not in (None, 0):
            raise UnsupportedModelError("Mistral sliding-window attention is not implemented")
        if self.hidden_size != self.attention_heads * self.head_dim:
            raise UnsupportedModelError("hidden_size must equal num_attention_heads * head_dim")
        if self.attention_heads % self.kv_heads:
            raise UnsupportedModelError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise UnsupportedModelError("RoPE requires an even head_dim")
        if self.config.get("attention_bias", False) or self.config.get("mlp_bias", False):
            raise UnsupportedModelError(
                "bias-enabled Llama variants are not supported by NumPy backend"
            )
        rope_scaling = self.config.get("rope_scaling")
        if rope_scaling not in (None, {}):
            raise UnsupportedModelError(
                "rope_scaling is not implemented in the NumPy reference backend"
            )

    @classmethod
    def from_store(
        cls,
        store_directory: str | os.PathLike[str],
        *,
        cache_bytes: int = 256 * 1024 * 1024,
        max_stages: int | None = None,
        verify: bool = False,
    ) -> NumpyLlamaEngine:
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
        return cls(StoreProvider(reader, cache_bytes=cache_bytes, max_stages=max_stages), config)

    def _rms_norm(self, values: np.ndarray, weight: np.ndarray) -> np.ndarray:
        variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        return values * (1.0 / np.sqrt(variance + self.rms_epsilon)) * weight

    def _rope(self, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        half = self.head_dim // 2
        angles = positions[:, None].astype(np.float64) * self._inverse_frequency[None, :]
        cosine = np.cos(angles)[:, None, :].astype(np.float32)
        sine = np.sin(angles)[:, None, :].astype(np.float32)
        first, second = values[..., :half], values[..., half:]
        return np.concatenate(
            [first * cosine - second * sine, second * cosine + first * sine], axis=-1
        )

    def _attention(
        self,
        values: np.ndarray,
        names: dict[str, str],
        key_cache: np.ndarray | None,
        value_cache: np.ndarray | None,
        position_start: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        positions = np.arange(position_start, position_start + token_count, dtype=np.int64)
        query = self._rope(query, positions)
        key = self._rope(key, positions)
        key_cache = key if key_cache is None else np.concatenate([key_cache, key], axis=0)
        value_cache = (
            current_value
            if value_cache is None
            else np.concatenate([value_cache, current_value], axis=0)
        )
        repetitions = self.attention_heads // self.kv_heads
        expanded_key = np.repeat(key_cache, repetitions, axis=1)
        expanded_value = np.repeat(value_cache, repetitions, axis=1)
        scores = np.einsum("thd,shd->ths", query, expanded_key, optimize=True)
        scores /= math.sqrt(self.head_dim)

        key_positions = np.arange(key_cache.shape[0], dtype=np.int64)
        causal = key_positions[None, :] <= positions[:, None]
        scores = np.where(causal[:, None, :], scores, np.float32(-1e30))
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores, dtype=np.float32)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        context = np.einsum("ths,shd->thd", probabilities, expanded_value, optimize=True)
        context = context.reshape(token_count, self.hidden_size)
        return self.provider.linear(names["o"], context), key_cache, value_cache

    @staticmethod
    def _silu(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -80.0, 80.0)
        return values / (1.0 + np.exp(-clipped))

    def _mlp(self, values: np.ndarray, names: dict[str, str]) -> np.ndarray:
        gate = self._silu(self.provider.linear(names["gate"], values))
        up = self.provider.linear(names["up"], values)
        return self.provider.linear(names["down"], gate * up)

    def _forward_step(
        self,
        token_ids: list[int],
        key_cache: list[np.ndarray | None],
        value_cache: list[np.ndarray | None],
        position_start: int,
    ) -> np.ndarray:
        hidden = self.provider.embedding(
            "model.embed_tokens.weight", np.asarray(token_ids, dtype=np.int64)
        )
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
        trace_logits: bool = False,
    ) -> dict:
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if any(token < 0 or token >= self.vocabulary_size for token in prompt_ids):
            raise ValueError("prompt contains a token ID outside the vocabulary")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        key_cache: list[np.ndarray | None] = [None] * self.layers
        value_cache: list[np.ndarray | None] = [None] * self.layers
        random = np.random.default_rng(seed)
        generated: list[int] = []
        step_seconds: list[float] = []
        logits_trace: list[np.ndarray] = []
        feed = list(prompt_ids)
        position_start = 0

        for _ in range(max_new_tokens):
            started = time.perf_counter()
            logits = self._forward_step(feed, key_cache, value_cache, position_start)
            step_seconds.append(time.perf_counter() - started)
            if trace_logits:
                logits_trace.append(logits.copy())
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                scaled = logits / temperature
                scaled -= scaled.max()
                probabilities = np.exp(scaled, dtype=np.float64)
                probabilities /= probabilities.sum()
                next_token = int(random.choice(self.vocabulary_size, p=probabilities))
            generated.append(next_token)
            position_start += len(feed)
            feed = [next_token]

        kv_bytes = sum(value.nbytes for value in key_cache if value is not None) + sum(
            value.nbytes for value in value_cache if value is not None
        )
        provider_stats = dict(self.provider.stats)
        return {
            "tokens": generated,
            "all_tokens": list(prompt_ids) + generated,
            "logits_trace": logits_trace,
            "step_seconds": step_seconds,
            "tokens_per_second": (
                len(step_seconds) / sum(step_seconds)
                if step_seconds and sum(step_seconds) > 0
                else 0.0
            ),
            "kv_bytes": kv_bytes,
            "provider_stats": provider_stats,
        }

    def close(self) -> None:
        self.provider.close()

    def __enter__(self) -> NumpyLlamaEngine:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
