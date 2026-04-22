"""
kv_block_copy.py — Triton kernel for async KV block eviction/restore.

Copies 16 MB KV pages between GPU HBM and CPU pinned memory.
On CPU-only machines, falls back to a NumPy memcpy simulation
that reports the theoretical PCIe latency for the block size.

On real hardware: the copy is dispatched on a background CUDA stream
so it overlaps with the ongoing decode step on the compute stream.
Latency: 16 MB / 64 GB/s ≈ 250 µs, hidden inside ~28 ms decode step.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

BLOCK_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB

# Theoretical PCIe Gen5 x16 bandwidth (bytes/s) used for simulation
_PCIE_BW_BPS = 64 * 1024**3


if _TRITON_AVAILABLE:
    @triton.jit
    def _copy_block_kernel(
        src_ptr,
        dst_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        Vectorized block copy using 128-bit (float4) loads.
        Each Triton program instance handles one tile of the block.
        """
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n_elements
        data = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, data, mask=mask)

    def copy_block_gpu(
        src: "torch.Tensor",
        dst: "torch.Tensor",
        stream: "torch.cuda.Stream | None" = None,
    ) -> None:
        """
        Copy src (GPU) -> dst (CPU pinned or GPU) using the Triton kernel.
        Both tensors must be float16 and contiguous.
        """
        assert src.dtype == dst.dtype == torch.float16
        n = src.numel()
        TILE = 1024
        grid = (triton.cdiv(n, TILE),)
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            _copy_block_kernel[grid](src, dst, n_elements=n, BLOCK=TILE)


# ---------------------------------------------------------------------------
# Simulation fallback (no CUDA required)
# ---------------------------------------------------------------------------

class SimBlockCopyResult:
    def __init__(self, n_bytes: int):
        self.n_bytes = n_bytes
        self.latency_s = n_bytes / _PCIE_BW_BPS
        self.bandwidth_gbs = _PCIE_BW_BPS / 1024**3

    def __repr__(self) -> str:
        return (
            f"SimBlockCopyResult(bytes={self.n_bytes/1024**2:.1f}MB, "
            f"latency={self.latency_s*1e6:.0f}µs, "
            f"bw={self.bandwidth_gbs:.0f}GB/s)"
        )


def simulate_block_evict(n_blocks: int) -> SimBlockCopyResult:
    """
    Simulate evicting n_blocks to CPU.  Returns timing estimate.
    No actual data movement — used for benchmarking and planning.
    """
    total_bytes = n_blocks * BLOCK_SIZE_BYTES
    return SimBlockCopyResult(total_bytes)


def simulate_block_restore(n_blocks: int) -> SimBlockCopyResult:
    """Simulate restoring n_blocks from CPU to GPU."""
    total_bytes = n_blocks * BLOCK_SIZE_BYTES
    return SimBlockCopyResult(total_bytes)


def numpy_block_copy(src: np.ndarray, dst: np.ndarray) -> float:
    """
    NumPy memcpy for CPU-only testing.
    Returns simulated PCIe latency in seconds (does not actually sleep).
    """
    np.copyto(dst, src)
    return src.nbytes / _PCIE_BW_BPS
