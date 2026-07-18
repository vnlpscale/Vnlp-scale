"""Runtime loaders for the multi-model Colab chat demo."""

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from pathlib import Path

from colab_models import (
    _fallback_prompt,
    _text_multimodal_messages,
    normalize_token_ids,
)


class MultiModelRuntime:
    def __init__(
        self,
        profile: dict,
        *,
        quality: str = "low",
        store_root: str | Path = "/content/vnlp-stores",
        chunk_mib: int = 16,
        cache_mib: int = 2048,
        max_stages: int | None = None,
        quant_block_rows: int = 256,
    ):
        self.profile = dict(profile)
        self.model_id = str(profile["model_id"])
        self.backend = str(profile["backend"])
        self.loader = str(profile.get("loader", "auto"))
        self.max_context_tokens = int(profile.get("context_tokens", 4096))
        self.quality = quality
        self.store_dir = Path(store_root) / self.model_id.replace("/", "--") / quality
        self.chunk_mib = int(chunk_mib)
        self.cache_mib = int(cache_mib)
        self.max_stages = max_stages
        self.quant_block_rows = int(quant_block_rows)
        self.engine = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.torch = None
        self.info: dict = {}

    def load(self) -> dict:
        if self.backend == "vnlp":
            self._load_vnlp()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"unsupported backend: {self.backend}")
        return dict(self.info)

    def _dtype(self):
        torch = self.torch
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _resolve_loader(self):
        if self.loader in {"causal", "multimodal"}:
            return
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.model_id,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=bool(self.profile.get("trust_remote_code")),
        )
        model_type = str(getattr(config, "model_type", ""))
        names = [str(value).lower() for value in getattr(config, "architectures", [])]
        is_multimodal = (
            "vision" in model_type
            or "vl" in model_type
            or any("conditionalgeneration" in value for value in names)
        )
        self.loader = "multimodal" if is_multimodal else "causal"

    def _load_transformers(self):
        import torch
        import transformers
        from transformers import AutoProcessor, AutoTokenizer

        self.torch = torch
        self._resolve_loader()
        token = os.environ.get("HF_TOKEN")
        trust = bool(self.profile.get("trust_remote_code"))
        dtype = self._dtype()
        if self.loader == "multimodal":
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, token=token, trust_remote_code=trust
            )
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_class is None:
                model_class = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_class is None:
                raise RuntimeError("Transformers has no multimodal auto-model class")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, token=token, trust_remote_code=trust
            )
            self.processor = self.tokenizer
            model_class = transformers.AutoModelForCausalLM
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"token": token, "trust_remote_code": trust, "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        if self.profile.get("load_in_4bit"):
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit loading requires CUDA")
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        try:
            self.model = model_class.from_pretrained(
                self.model_id, dtype=dtype, **kwargs
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            self.model = model_class.from_pretrained(
                self.model_id, torch_dtype=dtype, **kwargs
            )
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.info = {
            "backend": self.backend,
            "model": self.model_id,
            "loader": self.loader,
            "dtype": str(dtype).removeprefix("torch."),
            "four_bit": bool(self.profile.get("load_in_4bit")),
            "device": str(next(self.model.parameters()).device),
        }

    def _load_vnlp(self):
        import torch
        from transformers import AutoTokenizer
        from vnlp_scale.engine_torch import TorchLlamaEngine
        from vnlp_scale.ingest import record
        from vnlp_scale.store import StoreReader

        self.torch = torch
        self.store_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.store_dir / "manifest.json"
        finalized = False
        if manifest_path.is_file():
            try:
                finalized = bool(
                    json.loads(manifest_path.read_text(encoding="utf-8")).get(
                        "finalized"
                    )
                )
            except (OSError, json.JSONDecodeError):
                finalized = False
        if not finalized:
            record(
                self.model_id,
                str(self.store_dir),
                quality=self.quality,
                max_chunk_bytes=self.chunk_mib * 1024 * 1024,
                checkpoint_every=4,
                overwrite=False,
                progress=print,
            )
        with StoreReader(self.store_dir) as reader:
            report = reader.verify(checksums=False)
            print("Store verification:", report)
            print(
                "Stored bits/parameter:",
                round(reader.summary()["bits_per_parameter"], 3),
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.store_dir, local_files_only=True
        )
        self.processor = self.tokenizer
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        options = {
            "device": device,
            "dtype": "auto" if device == "cuda" else "float32",
            "cache_bytes": self.cache_mib * 1024 * 1024,
            "max_stages": self.max_stages,
            "verify": False,
        }
        optional = {
            "attention_backend": "auto",
            "compile_resident": True,
            "cuda_graph_decode": True,
            "fused_linear": True,
            "quant_block_rows": self.quant_block_rows,
            "triton_quant": True,
        }
        parameters = inspect.signature(TorchLlamaEngine.from_store).parameters
        options.update(
            {key: value for key, value in optional.items() if key in parameters}
        )
        self.engine = TorchLlamaEngine.from_store(self.store_dir, **options)
        self.info = {
            "backend": self.backend,
            "model": self.model_id,
            **self.engine.optimization_report(),
        }

    def _transformer_inputs(self, messages):
        torch = self.torch
        formatted = (
            _text_multimodal_messages(messages)
            if self.loader == "multimodal"
            else messages
        )
        try:
            values = self.processor.apply_chat_template(
                formatted,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            values = self.tokenizer(
                _fallback_prompt(messages), return_tensors="pt", add_special_tokens=True
            )
        if not isinstance(values, dict):
            values = {"input_ids": values}
        if "attention_mask" not in values:
            values["attention_mask"] = torch.ones_like(values["input_ids"])
        length = values["input_ids"].shape[-1]
        if length > self.max_context_tokens:
            for key, value in list(values.items()):
                if (
                    torch.is_tensor(value)
                    and value.ndim >= 2
                    and value.shape[-1] == length
                ):
                    values[key] = value[..., -self.max_context_tokens :]
        device = self.model.get_input_embeddings().weight.device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in values.items()
        }

    def generate(
        self,
        messages,
        *,
        max_new_tokens=64,
        temperature=0.7,
        sample=True,
        top_p=0.9,
        top_k=50,
    ):
        if self.backend == "vnlp":
            return self._generate_vnlp(messages, max_new_tokens, temperature, sample)
        inputs = self._transformer_inputs(messages)
        input_length = inputs["input_ids"].shape[-1]
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            kwargs.update(
                temperature=max(float(temperature), 1e-5),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output[0, input_length:]
        decoder = (
            self.processor if hasattr(self.processor, "decode") else self.tokenizer
        )
        text = decoder.decode(generated, skip_special_tokens=True).strip()
        count = int(generated.numel())
        return text, {
            "tokens_per_second": count / elapsed if elapsed else 0.0,
            "generated_tokens": count,
            "runtime": self.info,
        }

    def _generate_vnlp(self, messages, max_new_tokens, temperature, sample):
        try:
            ids = normalize_token_ids(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
        except Exception:
            ids = normalize_token_ids(
                self.tokenizer.encode(
                    _fallback_prompt(messages), add_special_tokens=True
                )
            )
        result = self.engine.generate(
            ids[-self.max_context_tokens :],
            int(max_new_tokens),
            greedy=not bool(sample),
            temperature=max(float(temperature), 1e-5),
        )
        tokens = list(result["tokens"])
        eos = self.tokenizer.eos_token_id
        if eos is not None and eos in tokens:
            tokens = tokens[: tokens.index(eos) + 1]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return text, {
            "tokens_per_second": result["tokens_per_second"],
            "generated_tokens": len(tokens),
            "runtime": result.get("optimizations", self.info),
        }

    def close(self):
        if self.engine is not None:
            self.engine.close()
        self.engine = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
