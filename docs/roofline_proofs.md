# CachePilot Roofline Notes

This repo now includes a first-principles scorecard for decode efficiency.
The goal is not to predict exact production tok/s, but to establish hard upper
bounds from bandwidth, compute, and KV-cache capacity.

## 1. KV Bandwidth Law

For one cached context token, the KV footprint across all layers is:

`B_kv = 2 * n_layers * n_heads * head_dim * bytes_per_scalar`

The factor `2` is for `K` and `V`.

If the active decode context is `C` tokens, one output token requires reading:

`B_decode = C * B_kv`

If GPU memory bandwidth is `BW` bytes/s, then no implementation can sustain:

`tok/s > BW / B_decode`

This is a conservation law on bytes moved. It is independent of scheduler
details.

## 2. Attention FLOP Law

A simplified decode-attention cost for one output token is:

`F_decode ~= 4 * n_layers * n_heads * head_dim * C`

This covers the dominant `QK` and `AV` terms.

If GPU compute is `P` FLOP/s, then:

`tok/s > P / F_decode`

is impossible.

## 3. Roofline Bound

The realizable upper bound is the lower of the compute and bandwidth limits:

`tok/s_roofline = min(P / F_decode, BW / B_decode)`

The arithmetic intensity is:

`I = F_decode / B_decode`

Substituting the two formulas above:

`I ~= 2 / bytes_per_scalar`

So:
- FP16 KV gives `~1 FLOP/byte`
- FP8 / INT8 KV gives `~2 FLOP/byte`

Modern inference GPUs have ridge points far above this, which means decode is
typically bandwidth-bound, not math-bound.

## 4. Compression Law

Halving `bytes_per_scalar` from FP16 to FP8 or INT8:
- halves `B_kv`
- halves `B_decode`
- doubles cache-token capacity
- doubles the bandwidth-bound decode ceiling

That is why KV compression is a direct throughput and cache-headroom lever.

## 5. What This Proves

These formulas prove three useful things:
- KV cache compression gives a near-linear headroom improvement before
  implementation overheads.
- Large-memory, high-bandwidth GPUs dominate long-context decode workloads.
- Policy work such as PERC matters because every prevented eviction avoids
  recompute that would otherwise consume the same scarce bandwidth budget.
