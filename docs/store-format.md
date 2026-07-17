# Store format version 2

A store is a directory containing:

```text
manifest.json
blobs.bin
config.json                 optional but required by bundled runtimes
[tokenizer metadata]        optional
```

## Manifest root

```json
{
  "format": "vnlp-scale-store",
  "version": 2,
  "created_at": 0.0,
  "updated_at": 0.0,
  "finalized": true,
  "meta": {},
  "tensors": {}
}
```

`finalized=false` means recording did not complete. Readers may inspect such a store, but the bundled runtime rejects structural gaps during open verification.

## Tensor record

```json
{
  "shape": [4096, 4096],
  "source_dtype": "F16",
  "decoded_dtype": "float32",
  "chunk_axis": 0,
  "complete": true,
  "rel_error": 0.031,
  "chunks": []
}
```

Version 2 supports axis-0 chunking only. Chunks must form a contiguous, non-overlapping cover of `[0, shape[0])`.

## Chunk record

```json
{
  "start": 0,
  "stop": 1024,
  "shape": [1024, 4096],
  "rel_error": 0.030,
  "error_l2_sq": 12.3,
  "source_l2_sq": 13600.0,
  "stages": []
}
```

Tensor relative error is aggregated from squared norms, not by averaging chunk percentages:

```text
rel_error = sqrt(sum(error_l2_sq) / sum(source_l2_sq))
```

## Stage record

```json
{
  "kind": "quant",
  "meta": {
    "bits": 4,
    "group_size": 64,
    "value_count": 4194304,
    "group_count": 65536
  },
  "blobs": []
}
```

Defined stage kinds:

- `raw`: one fp16 payload.
- `svd`: fp16 left and right factors.
- `quant`: packed signed codes plus fp16 scales.

Stages are decoded in order and summed.

## Blob reference

```json
{
  "offset": 4096,
  "length": 1048576,
  "sha256": "..."
}
```

Offsets are aligned to 64-byte boundaries when written. Padding bytes are not referenced and are excluded from payload compression statistics.

## Validation invariants

A conforming reader must verify:

- format and version;
- positive tensor dimensions;
- contiguous chunk coverage;
- chunk shape consistency;
- known stage kinds;
- non-negative blob offsets and lengths;
- blob ranges contained in `blobs.bin`;
- SHA-256 when integrity verification is requested;
- consistency between coverage and the `complete` flag.

## Compatibility policy

Before version 1.0, incompatible changes may increment the store version without an in-place migration. Once a released store version is declared stable, readers will retain support or provide an explicit conversion command.
