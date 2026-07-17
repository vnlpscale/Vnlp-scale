"""Run a reproducible token-ID generation benchmark and emit JSON."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from vnlp_scale import __version__


def parse_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--prompt-ids", required=True, type=parse_ids)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache-mib", type=int, default=0)
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    cache_bytes = arguments.cache_mib * 1024 * 1024

    if arguments.backend == "numpy":
        from vnlp_scale.engine_np import NumpyLlamaEngine

        engine = NumpyLlamaEngine.from_store(
            arguments.store,
            cache_bytes=cache_bytes,
            max_stages=arguments.max_stages,
            verify=arguments.verify,
        )
        backend_versions = {"numpy": np.__version__}
    else:
        import torch

        from vnlp_scale.engine_torch import TorchLlamaEngine

        engine = TorchLlamaEngine.from_store(
            arguments.store,
            device=arguments.device,
            dtype=arguments.dtype,
            cache_bytes=cache_bytes,
            max_stages=arguments.max_stages,
            verify=arguments.verify,
        )
        backend_versions = {"torch": torch.__version__}

    try:
        result = engine.generate(arguments.prompt_ids, arguments.max_new)
    finally:
        engine.close()
    payload = {
        "benchmark": "generation",
        "vnlp_scale_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "store": str(arguments.store.resolve()),
        "backend": arguments.backend,
        "backend_versions": backend_versions,
        "device": arguments.device if arguments.backend == "torch" else "cpu",
        "dtype": arguments.dtype if arguments.backend == "torch" else "float32",
        "prompt_ids": arguments.prompt_ids,
        "max_new": arguments.max_new,
        "cache_mib": arguments.cache_mib,
        "max_stages": arguments.max_stages,
        "tokens": result["tokens"],
        "tokens_per_second": result["tokens_per_second"],
        "step_seconds": result["step_seconds"],
        "kv_bytes": result["kv_bytes"],
        "provider_stats": result["provider_stats"],
        "note": "report first-run and repeated-process results separately; OS cache state is external",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
