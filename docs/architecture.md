# Architecture

## Design constraints

Vnlp-scale is built around four constraints:

1. A complete checkpoint may exceed local RAM and VRAM.
2. A single transformer layer may also exceed VRAM.
3. Recording can run for hours or days and must resume after interruption.
4. Corruption and unsupported model layouts must fail explicitly.

These constraints rule out full-model loading and, at trillion-parameter scale, also rule out treating a complete layer as the minimum execution unit.

## Data plane

```text
safetensors shard
    │ get_slice(name)[row_start:row_stop]
    ▼
float32 codec workspace bounded by chunk size
    │
    ├─ optional randomized low-rank stage
    ├─ groupwise quantized residual stage
    └─ recursive residual stages
    ▼
aligned append-only blobs.bin
    │
    └─ atomic manifest checkpoint with offsets and SHA-256
```

Only a bounded row slice is presented to the codec. The store records axis-0 coverage, so recording can resume without re-encoding committed chunks.

## Recursive residual codec

For a source chunk `W`:

```text
R0 = W
C1 = encode_low_rank(R0)       R1 = R0 - decode(C1)
C2 = encode_quantized(R1)      R2 = R1 - decode(C2)
C3 = encode_quantized(R2)      R3 = R2 - decode(C3)
W_hat = decode(C1) + decode(C2) + decode(C3)
```

The stored representation is the ordered list of stages. `max_stages` permits progressive decoding. This is useful for controlled quality experiments but should not be enabled in production without downstream evaluation.

The low-rank stage is skipped when its fp16 factors would be larger than raw fp16 storage. Non-matrix tensors, small tensors, and names containing configured normalization fragments are stored as fp16.

## Commit protocol

For each checkpoint:

1. Payload blobs are appended at aligned offsets.
2. `blobs.bin` is flushed and fsynced.
3. A complete temporary manifest is written and fsynced.
4. The manifest is atomically replaced.
5. The containing directory is fsynced where supported.

After a process crash, bytes after the largest committed blob range are unreferenced. The next writer truncates that tail before appending new data.

An exclusive lock file prevents two writers from mutating the same store. `--force-unlock` is an operator override, not an automatic recovery policy.

## Runtime provider abstraction

The execution engine does not request a complete layer. It requests operations:

- `embedding(name, ids)`
- `get(name)` for small vectors
- `linear(name, input)`

For a stored matrix `W[out, in]`, `linear` allocates the output activation and fills row ranges:

```text
for chunk in W.axis0_chunks:
    Wc = decode(chunk)
    Y[:, chunk.start:chunk.stop] = X @ Wc.T
    release_or_cache(Wc)
```

The largest transient decoded weight is one chunk. An LRU may retain decoded chunks up to a byte budget. Cache accounting is based on decoded memory, not compressed payload size.

## NumPy backend

The NumPy engine is the numerical reference implementation for:

- RMSNorm
- rotary position embeddings
- grouped-query attention
- causal masking with arbitrary prefill length
- SwiGLU MLP
- greedy or temperature sampling
- KV-cache growth

It is intentionally direct rather than fused. Its purpose is correctness, codec evaluation, and memory-behavior validation.

## PyTorch backend

The PyTorch engine mirrors the same operation-level architecture and transfers decoded chunks to the selected device. It does not instantiate a Transformers model or execute remote model code.

Current limitations:

- no fused decode/GEMM kernel;
- synchronous decode and transfer;
- no custom RoPE scaling;
- no bias-enabled Llama variants;
- no expert-aware MoE routing;
- no tensor parallelism.

The backend is therefore experimental even though its memory boundary is explicit.

## Why a layer cache is insufficient

Let `P_active` be active parameters and `L` the layer count. A rough decoded fp16 layer size is:

```text
layer_bytes ≈ P_active / L × 2
```

For a dense 1T model with 120 layers this is roughly 16.7 GB per layer. Double buffering requires roughly 33.3 GB before activations and KV cache, which exceeds a 24 GB GPU. Row-block execution is therefore not an optimization; it is a feasibility requirement.

## Extension points

The store format separates logical chunks from codec stages. Future stage kinds can add:

- outlier channels;
- entropy coding;
- learned shared decoders;
- sparse blocks;
- expert-specific encodings;
- device-native packed formats.

A new stage kind requires a store-version compatibility decision, decoder implementation, structural validation, corruption tests, and migration documentation.
