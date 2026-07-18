# Stage-aware linear inference

Vnlp-scale stores a compressed matrix chunk as an additive sequence of codec stages. For
a low-rank stage followed by quantized residual stages,

\[
W \approx LR + \sum_{j=1}^{s} Q_j.
\]

Materializing the complete matrix before every linear operation performs unnecessary work.
The PyTorch backend instead applies distributivity and associativity:

\[
XW^\mathsf{T}
= X(LR)^\mathsf{T} + \sum_j XQ_j^\mathsf{T}
= (XR^\mathsf{T})L^\mathsf{T} + \sum_j XQ_j^\mathsf{T}.
\]

For `X` with batch size `b`, a matrix `W` of shape `m × n`, and rank `r`, the low-rank
path changes the arithmetic cost from constructing `LR` in `O(mnr)` and multiplying in
`O(bmn)` to two direct products in `O(bnr + bmr)`. Autoregressive decode commonly has
`b = 1`, where this reduction is most significant.

## Runtime paths

- **SVD stages:** transfer and optionally cache only the `L` and `R` factors, then execute
  the two matrix products directly.
- **Packed quant stages on CUDA:** when Triton is available, a decode-oriented kernel reads
  packed 2-bit, 4-bit, or 8-bit codes, applies group scales, and accumulates the matvec
  result without creating a dequantized weight tensor.
- **Portable quant fallback:** unpack codes once, retain them as `int8`, and dequantize only
  a bounded output-row block before each matrix product.
- **Raw stages:** multiply directly from the stored FP16 row chunk.

All stage results are accumulated into the output chunk. `max_stages` continues to select a
progressive prefix of the codec stages, and `fused_linear=False` restores the previous full
materialization path for comparison or debugging.

## Memory behavior

The SVD path does not allocate an `m × n` reconstructed matrix. The portable quant path
limits temporary dequantized storage to `quant_block_rows × n`. The Triton path avoids that
temporary entirely and reads packed codes in the kernel.

The existing cache budget is shared by decoded chunks, factors, packed codes, unpacked
integer codes, and scale tensors. This prevents the optimized path from silently exceeding
the configured cache bound.

## Numerical behavior

The transformation is algebraically exact for the stored codec stages. Floating-point
rounding can differ because the low-rank product is evaluated as two matrix multiplications
instead of first constructing `LR`. The fallback remains available for numerical comparison.
CUDA-specific performance and tolerance should be measured on each target GPU and model.
