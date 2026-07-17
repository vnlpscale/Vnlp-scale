# Vnlp-scale

Vnlp-scale is an open-source storage and inference substrate for language models that do not fit in conventional RAM or VRAM. It converts safetensors checkpoints into a checksummed, chunk-addressable store and executes Llama-compatible matrix operations by decoding one row block at a time.

The project goal is to make **trillion-parameter-class model storage and experimentation addressable on home hardware**. That goal is deliberately separated into three claims:

1. **Storage addressability:** the compressed model can be recorded and verified without materializing the full checkpoint.
2. **Execution addressability:** inference can proceed with bounded weight memory by streaming chunks.
3. **Interactive throughput:** useful tokens per second require sparse activation, caching, and optimized GPU kernels. A dense 1T model is not interactive on a single consumer machine.

## Status

| Capability | Status |
|---|---|
| Local and Hugging Face safetensors ingestion | Implemented |
| Chunk-level crash resume | Implemented |
| SHA-256 verification of every stored blob | Implemented |
| Recursive residual codec: low-rank + quantized residual stages | Implemented |
| Progressive partial-stage decode | Implemented |
| Bounded-memory NumPy Llama runtime | Implemented, reference backend |
| Bounded-memory PyTorch runtime | Experimental |
| Text prompts through a copied tokenizer | Optional |
| Dedicated CUDA fused decode/matmul kernels | Not implemented |
| Expert-aware MoE loading and routing | Not implemented |
| Published 1T hardware benchmark | Not yet available |

The repository is alpha software. Store format version 2 is validated and checksummed, but backward compatibility is not promised before version 1.0.

## Why block streaming is required

A 1T dense model stored at 1.5 bits per parameter occupies:

```text
1e12 × 1.5 / 8 = 187.5 GB
```

Reading those active weights from a 7 GB/s NVMe device once per generated token has an optimistic lower bound of:

```text
187.5 / 7 = 26.8 seconds/token ≈ 0.037 token/s
```

This excludes decoder work, kernel launch overhead, KV-cache traffic, and filesystem losses. Dense 1T inference can be technically executable but is not an interactive home-computing target.

A 1T MoE model with 32B active parameters changes the active transfer volume. At 1.5 bits per active parameter it reads 6 GB per decode step. That is the architectural path toward practical throughput, provided the runtime can select experts without scanning inactive weights.

Use the planner before downloading or encoding a checkpoint:

```bash
vnlp-scale plan \
  --total-params 1T \
  --active-params 32B \
  --bits 1.5 \
  --layers 120 \
  --bandwidth-gbps 64 \
  --tflops 80 \
  --vram-gb 24 \
  --ram-gb 128 \
  --storage-gb 4000
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

PyTorch and tokenizer support are optional:

```bash
pip install -e '.[torch,tokenizer]'
```

## Record a model

Local checkpoint:

```bash
vnlp-scale record \
  --source /models/example \
  --output /models/example.vnls \
  --quality med \
  --chunk-mib 64
```

Hugging Face repository:

```bash
vnlp-scale record \
  --source organization/model-name \
  --revision COMMIT_OR_TAG \
  --output /models/model.vnls \
  --quality high
```

Recording is resumable at chunk boundaries. Re-running the same command skips committed chunks. A model index that points outside a local checkpoint directory is rejected.

Quality presets:

| Preset | Encoding |
|---|---|
| `lossless` | fp16 storage, intended for correctness baselines |
| `low` | low-rank stage plus 2-bit residual |
| `med` | low-rank stage plus 4-bit residual |
| `high` | low-rank stage plus 4-bit and 2-bit recursive residuals |

These presets are generic. They are not a substitute for model-specific perplexity evaluation.

## Verify and inspect

```bash
vnlp-scale verify --store /models/model.vnls
vnlp-scale inspect --store /models/model.vnls
```

`verify` checks chunk coverage, blob boundaries, format invariants, and SHA-256 digests. The runtime can also verify on open with `--verify`.

## Run inference

Token IDs, NumPy reference backend:

```bash
vnlp-scale run \
  --store /models/model.vnls \
  --prompt-ids 1,42,99 \
  --max-new 16 \
  --backend numpy \
  --cache-mib 512
```

PyTorch backend:

```bash
vnlp-scale run \
  --store /models/model.vnls \
  --prompt-ids 1,42,99 \
  --max-new 16 \
  --backend torch \
  --device cuda \
  --dtype float16 \
  --cache-mib 2048
```

Text prompts require tokenizer files to have been copied into the store and the `tokenizer` extra:

```bash
vnlp-scale run --store /models/model.vnls --prompt 'Hello' --max-new 16
```

The current runtime supports the standard Llama/Mistral tensor naming layout without attention/MLP biases or custom RoPE scaling. Unsupported configurations fail explicitly.

## Store design

Each tensor is split along output rows. Every chunk contains one or more recursive codec stages and each stage references checksummed byte ranges in an append-only blob file.

```text
model.safetensors shards
        │ safe row slices
        ▼
recursive chunk codec
        │
        ├── manifest.json   format, tensor/chunk index, errors, SHA-256
        ├── blobs.bin       aligned append-only payloads
        └── config/tokenizer metadata
```

A matrix multiplication computes output row ranges independently:

```text
for W_rows in stored_weight_chunks:
    decoded = decode(W_rows)
    output[:, row_start:row_stop] = input @ decoded.T
    release(decoded)
```

Peak decoded-weight memory is therefore bounded by the largest chunk plus the configured cache, not by a full layer.

## Documentation

- [Architecture](docs/architecture.md)
- [Feasibility model](docs/feasibility.md)
- [Store format](docs/store-format.md)
- [Benchmarking](docs/benchmarking.md)
- [Security model](SECURITY.md)
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
python -m build
```

## License

MIT License. See [LICENSE](LICENSE).
