# Roadmap

## Release gate for 0.2

- Validate perplexity deltas on at least three public 1B–8B Llama-compatible checkpoints.
- Publish reproducible CPU and CUDA benchmark scripts with raw results.
- Implement RoPE scaling variants used by current Llama-family checkpoints.
- Add store migration tooling before changing the format again.
- Add fault-injection tests for interruption during every write phase.

## T-class execution path

1. **Chunk-streamed dense kernels** — fuse decode, dequantization, and GEMM so a decoded chunk is not materialized twice.
2. **Asynchronous I/O pipeline** — double-buffer direct reads, pinned host memory, and CUDA streams.
3. **Expert-aware MoE runtime** — route first, fetch only selected experts, preserve an expert cache across tokens.
4. **KV-cache policies** — quantization, offload, sliding windows, and capacity planning.
5. **Distributed home cluster** — optional weight partitioning across multiple machines without requiring a datacenter scheduler.
6. **1T reproducibility report** — hardware bill of materials, model license, exact revision, throughput, energy, and quality metrics.

## Explicit non-goals for 0.1

- Claiming interactive dense 1T inference on a single consumer GPU.
- Supporting pickle-based checkpoint formats.
- Hiding unsupported model configurations behind approximate behavior.
