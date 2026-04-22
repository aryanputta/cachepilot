"""
kv_quantize.py — Triton kernel for INT8 KV cache quantization.

Runs on any CUDA GPU without a separate nvcc compilation step.
Falls back to a NumPy reference implementation when Triton is unavailable
(e.g., CPU-only machines or CI environments).

Usage:
    from cachepilot.kernels.kv_quantize import quantize_kv_block, dequantize_kv_block

    # fp16_kv: torch.Tensor [2, n_layers, n_heads, seq_len, head_dim] float16
    int8_kv, scales = quantize_kv_block(fp16_kv)
    fp16_restored   = dequantize_kv_block(int8_kv, scales)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel (GPU path)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _quantize_channel_kernel(
        fp16_ptr,
        int8_ptr,
        scale_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        One Triton program instance handles one (kv, layer, head) channel.
        Computes absmax, stores scale, then quantizes every element.
        """
        pid = tl.program_id(0)
        base = pid * n_elements

        # Pass 1: find absmax across the channel
        absmax = tl.zeros([1], dtype=tl.float32)
        for off in range(0, n_elements, BLOCK):
            idxs = base + off + tl.arange(0, BLOCK)
            mask = (off + tl.arange(0, BLOCK)) < n_elements
            vals = tl.load(fp16_ptr + idxs, mask=mask, other=0.0).to(tl.float32)
            absmax = tl.maximum(absmax, tl.abs(vals))

        scale = tl.maximum(absmax, 1e-6) / 127.0
        tl.store(scale_ptr + pid, scale)

        # Pass 2: quantize
        for off in range(0, n_elements, BLOCK):
            idxs = base + off + tl.arange(0, BLOCK)
            mask = (off + tl.arange(0, BLOCK)) < n_elements
            vals = tl.load(fp16_ptr + idxs, mask=mask, other=0.0).to(tl.float32)
            q = tl.clamp(tl.round(vals / scale), -127.0, 127.0).to(tl.int8)
            tl.store(int8_ptr + idxs, q, mask=mask)

    @triton.jit
    def _dequantize_channel_kernel(
        int8_ptr,
        scale_ptr,
        fp16_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base = pid * n_elements
        scale = tl.load(scale_ptr + pid)

        for off in range(0, n_elements, BLOCK):
            idxs = base + off + tl.arange(0, BLOCK)
            mask = (off + tl.arange(0, BLOCK)) < n_elements
            q = tl.load(int8_ptr + idxs, mask=mask, other=0).to(tl.float32)
            vals = (q * scale).to(tl.float16)
            tl.store(fp16_ptr + idxs, vals, mask=mask)

    def quantize_kv_block(
        kv: "torch.Tensor",
        block_size: int = 1024,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Quantize a KV cache tensor from FP16 to INT8 in-place using Triton.

        Parameters
        ----------
        kv : Tensor[2, n_layers, n_heads, seq_len, head_dim] float16
        block_size : Triton tile size (power of 2)

        Returns
        -------
        int8_kv : Tensor same shape, dtype=int8
        scales   : Tensor[2, n_layers, n_heads] float32 — one scale per channel
        """
        assert kv.dtype == torch.float16
        shape = kv.shape
        n_channels = shape[0] * shape[1] * shape[2]
        n_elements = shape[3] * shape[4]

        flat = kv.reshape(n_channels, n_elements).contiguous()
        int8_out = torch.empty_like(flat, dtype=torch.int8)
        scales = torch.empty(n_channels, dtype=torch.float32, device=kv.device)

        _quantize_channel_kernel[(n_channels,)](
            flat, int8_out, scales,
            n_elements=n_elements,
            BLOCK=block_size,
        )
        return int8_out.reshape(shape), scales.reshape(shape[:3])

    def dequantize_kv_block(
        int8_kv: "torch.Tensor",
        scales: "torch.Tensor",
        block_size: int = 1024,
    ) -> "torch.Tensor":
        shape = int8_kv.shape
        n_channels = shape[0] * shape[1] * shape[2]
        n_elements = shape[3] * shape[4]

        flat = int8_kv.reshape(n_channels, n_elements).contiguous()
        flat_scales = scales.reshape(n_channels)
        fp16_out = torch.empty_like(flat, dtype=torch.float16)

        _dequantize_channel_kernel[(n_channels,)](
            flat, flat_scales, fp16_out,
            n_elements=n_elements,
            BLOCK=block_size,
        )
        return fp16_out.reshape(shape)


# ---------------------------------------------------------------------------
# NumPy reference (CPU fallback — used in tests and CI)
# ---------------------------------------------------------------------------

def quantize_kv_numpy(
    kv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel symmetric INT8 quantization.

    kv : float16 array [..., seq_len, head_dim]
    Returns (int8_kv, scales) where scales has shape kv.shape[:-2].
    """
    orig_shape = kv.shape
    n_channels = int(np.prod(orig_shape[:-2]))
    n_elements = orig_shape[-2] * orig_shape[-1]

    flat = kv.astype(np.float32).reshape(n_channels, n_elements)
    absmax = np.abs(flat).max(axis=1, keepdims=True).clip(min=1e-6)
    scales = (absmax / 127.0).reshape(orig_shape[:-2])
    q = np.clip(np.round(flat / absmax * 127.0), -127, 127).astype(np.int8)
    return q.reshape(orig_shape), scales


def dequantize_kv_numpy(
    int8_kv: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    orig_shape = int8_kv.shape
    n_channels = int(np.prod(orig_shape[:-2]))
    n_elements = orig_shape[-2] * orig_shape[-1]

    flat = int8_kv.astype(np.float32).reshape(n_channels, n_elements)
    flat_scales = scales.reshape(n_channels, 1)
    return (flat * flat_scales).reshape(orig_shape).astype(np.float16)


# ---------------------------------------------------------------------------
# Unified API (dispatch to Triton or NumPy)
# ---------------------------------------------------------------------------

def quantize_kv(kv, **kwargs):
    if _TRITON_AVAILABLE and hasattr(kv, "device"):
        return quantize_kv_block(kv, **kwargs)
    return quantize_kv_numpy(np.asarray(kv))


def dequantize_kv(int8_kv, scales, **kwargs):
    if _TRITON_AVAILABLE and hasattr(int8_kv, "device"):
        return dequantize_kv_block(int8_kv, scales, **kwargs)
    return dequantize_kv_numpy(np.asarray(int8_kv), np.asarray(scales))
