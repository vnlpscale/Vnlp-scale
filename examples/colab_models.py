"""Utilities used by the multi-model Colab chat demo."""

from __future__ import annotations

MODEL_CATALOG = {
    "Qwen3.5 0.8B": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "backend": "transformers",
        "loader": "multimodal",
        "context_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
    },
    "Qwen3 0.6B": {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
    },
    "Qwen3 1.7B": {
        "model_id": "Qwen/Qwen3-1.7B",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
    },
    "Qwen2.5 0.5B Instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
    },
    "SmolLM2 360M Instruct": {
        "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
    },
    "SmolLM2 1.7B Instruct": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
    },
    "Gemma 3 1B IT (license required)": {
        "model_id": "google/gemma-3-1b-it",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
        "gated": True,
    },
    "Llama 3.2 1B Instruct (gated)": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
        "gated": True,
    },
    "Phi-3.5 Mini Instruct 3.8B (4-bit)": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "backend": "transformers",
        "loader": "causal",
        "context_tokens": 4096,
        "load_in_4bit": True,
        "trust_remote_code": True,
    },
    "TinyLlama 1.1B Chat (Vnlp-scale compressed)": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "backend": "vnlp",
        "loader": "causal",
        "context_tokens": 2048,
    },
}


def model_profile(
    choice: str,
    *,
    custom_model_id: str = "",
    custom_loader: str = "auto",
    force_backend: str = "auto",
    load_in_4bit: bool | None = None,
) -> dict:
    profile = dict(MODEL_CATALOG[choice])
    if custom_model_id.strip():
        profile = {
            "model_id": custom_model_id.strip(),
            "backend": "transformers",
            "loader": custom_loader,
            "context_tokens": 4096,
        }
    if force_backend != "auto":
        profile["backend"] = force_backend
    if custom_loader != "auto":
        profile["loader"] = custom_loader
    if load_in_4bit is not None:
        profile["load_in_4bit"] = bool(load_in_4bit)
    profile.setdefault("load_in_4bit", False)
    profile.setdefault("trust_remote_code", False)
    profile.setdefault("gated", False)
    profile.setdefault("temperature", 0.7)
    profile.setdefault("top_p", 0.9)
    profile.setdefault("top_k", 50)
    return profile


def normalize_token_ids(value) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value.get("input_ids")
    if value is None:
        raise TypeError("tokenizer output does not contain input_ids")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected one tokenized conversation")
        value = list(value[0])
    return [int(token) for token in value]


def _text_multimodal_messages(messages: list[dict]) -> list[dict]:
    result = []
    for item in messages:
        content = item["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        result.append({"role": item["role"], "content": content})
    return result


def _fallback_prompt(messages: list[dict]) -> str:
    return (
        "\n".join(
            f"{item['role'].capitalize()}: {item['content']}" for item in messages
        )
        + "\nAssistant:"
    )
