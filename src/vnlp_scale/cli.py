"""Command-line interface for Vnlp-scale."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .codec import PRESETS
from .errors import VnlpScaleError


def _parse_count(value: str) -> float:
    text = value.strip().upper().replace("_", "")
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    if text and text[-1] in multipliers:
        return float(text[:-1]) * multipliers[text[-1]]
    return float(text)


def _parse_prompt_ids(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prompt IDs must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one prompt ID is required")
    return result


def _json_print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vnlp-scale",
        description="Bounded-memory storage and inference for very large safetensors LLMs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    record_parser = subcommands.add_parser("record", help="encode a local or Hugging Face model")
    record_parser.add_argument("--source", required=True)
    record_parser.add_argument("--output", required=True)
    record_parser.add_argument("--quality", choices=sorted(PRESETS), default="med")
    record_parser.add_argument("--revision")
    record_parser.add_argument("--chunk-mib", type=int, default=64)
    record_parser.add_argument("--checkpoint-every", type=int, default=8)
    record_parser.add_argument("--overwrite", action="store_true")
    record_parser.add_argument("--force-unlock", action="store_true")
    record_parser.add_argument("--json", action="store_true")

    inspect_parser = subcommands.add_parser("inspect", help="show compression and store metadata")
    inspect_parser.add_argument("--store", required=True)

    verify_parser = subcommands.add_parser("verify", help="validate structure and blob checksums")
    verify_parser.add_argument("--store", required=True)
    verify_parser.add_argument("--structure-only", action="store_true")

    run_parser = subcommands.add_parser("run", help="run Llama-compatible streamed inference")
    run_parser.add_argument("--store", required=True)
    prompt_group = run_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-ids", type=_parse_prompt_ids)
    prompt_group.add_argument("--prompt")
    run_parser.add_argument("--max-new", type=int, default=16)
    run_parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--dtype", default="float16")
    run_parser.add_argument("--cache-mib", type=int, default=256)
    run_parser.add_argument("--max-stages", type=int)
    run_parser.add_argument("--sample", action="store_true")
    run_parser.add_argument("--temperature", type=float, default=1.0)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--verify", action="store_true")
    run_parser.add_argument("--json", action="store_true")

    plan_parser = subcommands.add_parser("plan", help="estimate storage, memory mode, and roofline")
    plan_parser.add_argument("--total-params", required=True, type=_parse_count)
    plan_parser.add_argument("--active-params", type=_parse_count)
    plan_parser.add_argument("--bits", type=float, required=True)
    plan_parser.add_argument("--layers", type=int, required=True)
    plan_parser.add_argument("--bandwidth-gbps", type=float, required=True)
    plan_parser.add_argument("--tflops", type=float, required=True)
    plan_parser.add_argument("--vram-gb", type=float, required=True)
    plan_parser.add_argument("--ram-gb", type=float, required=True)
    plan_parser.add_argument("--storage-gb", type=float)
    plan_parser.add_argument("--batch", type=int, default=1)
    plan_parser.add_argument("--prefetch-depth", type=int, default=1)
    plan_parser.add_argument("--efficiency", type=float, default=0.70)

    subcommands.add_parser("scenarios", help="print reference 1T roofline scenarios")
    return parser


def _load_tokenizer(store: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise VnlpScaleError(
            "text prompts require transformers: pip install 'vnlp-scale[tokenizer]'"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(store, local_files_only=True)
    except Exception as exc:
        raise VnlpScaleError(
            "could not load a tokenizer from the store; use --prompt-ids or re-record tokenizer metadata"
        ) from exc


def _run_command(arguments: argparse.Namespace) -> int:
    tokenizer = None
    prompt_ids = arguments.prompt_ids
    if arguments.prompt is not None:
        tokenizer = _load_tokenizer(arguments.store)
        prompt_ids = tokenizer.encode(arguments.prompt, add_special_tokens=True)

    cache_bytes = arguments.cache_mib * 1024 * 1024
    if arguments.backend == "numpy":
        from .engine_np import NumpyLlamaEngine

        engine = NumpyLlamaEngine.from_store(
            arguments.store,
            cache_bytes=cache_bytes,
            max_stages=arguments.max_stages,
            verify=arguments.verify,
        )
    else:
        from .engine_torch import TorchLlamaEngine

        engine = TorchLlamaEngine.from_store(
            arguments.store,
            device=arguments.device,
            dtype=arguments.dtype,
            cache_bytes=cache_bytes,
            max_stages=arguments.max_stages,
            verify=arguments.verify,
        )
    try:
        result = engine.generate(
            prompt_ids,
            arguments.max_new,
            greedy=not arguments.sample,
            temperature=arguments.temperature,
            seed=arguments.seed,
        )
    finally:
        engine.close()
    if tokenizer is not None:
        result["text"] = tokenizer.decode(result["all_tokens"], skip_special_tokens=False)
    if arguments.json:
        _json_print(result)
    else:
        print("tokens:", result["tokens"])
        if "text" in result:
            print(result["text"])
        print(f"tokens/s: {result['tokens_per_second']:.4f}")
        print(f"KV cache: {result['kv_bytes'] / 1024**2:.2f} MiB")
        print("provider:", json.dumps(result["provider_stats"], sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "record":
            from .ingest import record

            result = record(
                arguments.source,
                arguments.output,
                quality=arguments.quality,
                revision=arguments.revision,
                max_chunk_bytes=arguments.chunk_mib * 1024 * 1024,
                checkpoint_every=arguments.checkpoint_every,
                overwrite=arguments.overwrite,
                force_lock=arguments.force_unlock,
                progress=None if arguments.json else print,
            )
            if arguments.json:
                _json_print(result)
            return 0

        if arguments.command == "inspect":
            from .store import StoreReader

            with StoreReader(arguments.store) as reader:
                _json_print(reader.summary())
            return 0

        if arguments.command == "verify":
            from .store import StoreReader

            with StoreReader(arguments.store) as reader:
                report = reader.verify(checksums=not arguments.structure_only)
            _json_print(report)
            return 0 if report["ok"] and not report["warnings"] else 1

        if arguments.command == "run":
            return _run_command(arguments)

        if arguments.command == "plan":
            from .estimate import HardwareProfile, ModelProfile, plan_inference

            model = ModelProfile(
                total_parameters=arguments.total_params,
                active_parameters=arguments.active_params or arguments.total_params,
                stored_bits_per_parameter=arguments.bits,
                layers=arguments.layers,
            )
            hardware = HardwareProfile(
                storage_bandwidth_gbps=arguments.bandwidth_gbps,
                compute_tflops=arguments.tflops,
                vram_gb=arguments.vram_gb,
                ram_gb=arguments.ram_gb,
                batch_size=arguments.batch,
                prefetch_depth=arguments.prefetch_depth,
                storage_capacity_gb=arguments.storage_gb,
                efficiency=arguments.efficiency,
            )
            _json_print(plan_inference(model, hardware))
            return 0

        if arguments.command == "scenarios":
            from .estimate import scenario_table

            print(scenario_table())
            return 0
    except (VnlpScaleError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
