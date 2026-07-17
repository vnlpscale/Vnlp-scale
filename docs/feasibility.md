# Feasibility model

## Definitions

- `P_total`: total stored parameters.
- `P_active`: parameters used for one decode step.
- `b_store`: stored bits per parameter.
- `b_decode`: decoded bits per parameter, normally 16.
- `B`: batch size.
- `BW`: effective weight-source bandwidth in bytes per second.
- `F`: effective compute throughput in floating-point operations per second.
- `L`: number of transformer layers.

## Storage

```text
store_bytes = P_total × b_store / 8
```

Representative values for 1T parameters:

| Bits/parameter | Payload size |
|---:|---:|
| 16 | 2.00 TB |
| 8 | 1.00 TB |
| 4 | 500 GB |
| 2 | 250 GB |
| 1.5 | 187.5 GB |

Manifest, alignment, filesystem, tokenizer, and metadata overhead are additional.

## Per-step transfer

```text
transfer_bytes = P_active × b_store / 8
```

Dense models have `P_active ≈ P_total`. MoE models can have a much smaller active set, but only if routing occurs before inactive expert weights are loaded.

## I/O roofline

```text
T_io = transfer_bytes / BW
```

The planner applies an efficiency factor to nominal bandwidth. Real effective bandwidth depends on access pattern, queue depth, filesystem, page cache, decompression, PCIe topology, and thermal throttling.

## Compute roofline

A dense matrix-based decoder requires approximately two operations per active weight per batch item:

```text
T_compute ≈ 2 × P_active × B / F
```

This is a coarse lower bound. Attention, normalization, activation functions, routing, KV-cache operations, and codec decode add work.

## Optimistic throughput ceiling

Assuming perfect overlap:

```text
T_step = max(T_io, T_compute)
tokens_per_second = B / T_step
```

The planner intentionally reports this as a ceiling. A measured benchmark must be lower or equal unless its assumptions differ, for example because weights remain cached across steps.

## Working-set mode

Approximate decoded bytes per active layer:

```text
decoded_layer_bytes = P_active / L × b_decode / 8
```

With prefetch depth `d`:

```text
full_layer_working_set = decoded_layer_bytes × (1 + d)
```

The planner reserves 20% of VRAM for activations, KV cache, allocator fragmentation, and runtime state. If the estimated full-layer working set exceeds the remaining budget, it selects `block-streaming-required`.

## Cache interpretation

RAM or VRAM caching changes the source bandwidth for cache hits. A correct benchmark must report:

- cold start after dropping or bypassing page cache;
- warm OS page cache;
- warm decoded RAM cache;
- warm GPU cache;
- cache size and eviction policy.

Reporting a warm-cache number as storage-streaming throughput is misleading.

## Example: dense 1T on home NVMe

```text
P_active = 1e12
b_store = 1.5
BW = 7 GB/s
transfer = 187.5 GB
T_io = 26.8 s
```

The optimistic ceiling is about 0.037 token/s before codec and runtime overhead.

## Example: 1T MoE, 32B active

```text
P_total = 1e12
P_active = 32e9
b_store = 1.5
transfer = 6 GB
```

At 64 GB/s effective bandwidth, the I/O lower bound is 0.09375 seconds per batch step. This scenario still requires expert-aware storage access; scanning all experts would revert to the dense transfer volume.

## What qualifies as a 1T result

A reproducible claim must publish:

- exact model and immutable revision;
- total and active parameter definitions;
- storage format and measured bits per parameter;
- hardware, links, filesystem, and software versions;
- cold and warm throughput;
- peak RAM and VRAM;
- prompt length, generated length, batch size, and KV-cache policy;
- downstream quality relative to an uncompressed reference;
- energy or wall-power measurement when available.
