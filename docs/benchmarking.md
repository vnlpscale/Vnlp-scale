# Benchmarking

Performance claims must be reproducible and must separate estimates from measurements.

## Tensor decode

```bash
python benchmarks/benchmark_tensor.py \
  --store /models/model.vnls \
  --tensor model.layers.0.self_attn.q_proj.weight \
  --iterations 10 \
  --verify > tensor-decode.json
```

The first iteration is reported separately. It is not guaranteed to represent cold storage because the operating-system page cache is external to the process.

## Generation

```bash
python benchmarks/benchmark_generate.py \
  --store /models/model.vnls \
  --prompt-ids 1,42,99 \
  --max-new 32 \
  --backend torch \
  --device cuda \
  --dtype float16 \
  --cache-mib 2048 \
  --verify > generation.json
```

Run separate processes for each condition:

1. first process after recording or copying;
2. repeated process with warm OS page cache;
3. decoded-cache sizes of zero and the intended deployment value;
4. progressive-stage settings, if evaluated.

## Required report fields

- model identifier and immutable revision;
- Vnlp-scale commit/version and store format;
- quality preset and measured bits per parameter;
- CPU, GPU, RAM, storage, links, filesystem, driver, and OS;
- backend, dtype, cache size, prompt length, generation length, and batch;
- all per-token timings, not only the best aggregate;
- peak RAM/VRAM from an external monitor where possible;
- quality metric against an uncompressed reference.

Do not describe the planner output as a benchmark. It is an optimistic roofline used for capacity planning.
