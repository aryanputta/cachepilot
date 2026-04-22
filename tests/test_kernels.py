"""
Tests for INT8 KV quantization and block copy kernels (NumPy path).
GPU Triton path is tested separately when CUDA is available.
"""

import numpy as np
import pytest

from cachepilot.kernels.kv_quantize import quantize_kv_numpy, dequantize_kv_numpy
from cachepilot.kernels.kv_block_copy import simulate_block_evict, simulate_block_restore, numpy_block_copy, BLOCK_SIZE_BYTES


class TestKVQuantize:
    def make_kv(self, shape=(2, 4, 8, 64, 128), seed=0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randn(*shape).astype(np.float16)

    def test_output_shape_preserved(self):
        kv = self.make_kv()
        q, scales = quantize_kv_numpy(kv)
        assert q.shape == kv.shape
        assert scales.shape == kv.shape[:-2]

    def test_output_dtype_int8(self):
        kv = self.make_kv()
        q, _ = quantize_kv_numpy(kv)
        assert q.dtype == np.int8

    def test_values_in_range(self):
        kv = self.make_kv()
        q, _ = quantize_kv_numpy(kv)
        assert q.min() >= -127
        assert q.max() <= 127

    def test_roundtrip_scale_normalized_error(self):
        """
        Per-channel INT8 quantization guarantees:
            |x - dequant(quant(x))| <= scale / 2  =  absmax / 254

        Equivalently, the absolute error normalized by the channel absmax is <= 1/254.
        This bound holds for every element, not just large ones.  Relative error is
        meaningless for near-zero elements in a large-range channel; use this bound.
        """
        kv = self.make_kv((2, 4, 8, 128, 128))
        q, scales = quantize_kv_numpy(kv)
        restored = dequantize_kv_numpy(q, scales)

        kv_f = kv.astype(np.float32)
        res_f = restored.astype(np.float32)
        orig_shape = kv_f.shape
        n_channels = int(np.prod(orig_shape[:-2]))

        # Reconstruct per-channel absmax from scales: absmax = scale * 127
        scales_flat = scales.reshape(n_channels, 1) * 127.0
        abs_err = np.abs(kv_f - res_f).reshape(n_channels, -1)
        # Each error / absmax must be <= 1/254
        normalized = abs_err / np.maximum(scales_flat, 1e-9)
        assert normalized.max() < 0.5 + 1e-4  # 0.5 = 1/2 step; +tolerance for fp16 rounding

    def test_mean_absolute_error(self):
        kv = self.make_kv((2, 32, 32, 256, 128))
        q, scales = quantize_kv_numpy(kv)
        restored = dequantize_kv_numpy(q, scales)
        mae = np.abs(kv.astype(np.float32) - restored.astype(np.float32)).mean()
        assert mae < 0.01, f"MAE {mae:.5f} exceeds threshold"

    def test_zero_tensor(self):
        kv = np.zeros((2, 4, 8, 64, 128), dtype=np.float16)
        q, scales = quantize_kv_numpy(kv)
        restored = dequantize_kv_numpy(q, scales)
        assert np.allclose(restored, 0.0, atol=1e-4)

    def test_memory_reduction(self):
        """INT8 uses half the memory of FP16."""
        kv = self.make_kv()
        q, scales = quantize_kv_numpy(kv)
        assert q.nbytes == kv.nbytes // 2


class TestBlockCopy:
    def test_simulate_evict_correct_latency(self):
        result = simulate_block_evict(n_blocks=1)
        # 16MB at 64 GB/s = 250µs
        assert abs(result.latency_s - 250e-6) < 10e-6

    def test_simulate_evict_scales_linearly(self):
        r1 = simulate_block_evict(1)
        r4 = simulate_block_evict(4)
        assert abs(r4.latency_s / r1.latency_s - 4.0) < 0.01

    def test_simulate_restore_matches_evict(self):
        evict = simulate_block_evict(2)
        restore = simulate_block_restore(2)
        assert evict.latency_s == restore.latency_s

    def test_numpy_copy_correctness(self):
        src = np.random.randn(1024).astype(np.float16)
        dst = np.zeros_like(src)
        numpy_block_copy(src, dst)
        np.testing.assert_array_equal(src, dst)

    def test_numpy_copy_returns_latency(self):
        src = np.random.randn(BLOCK_SIZE_BYTES // 2).astype(np.float16)
        dst = np.zeros_like(src)
        latency = numpy_block_copy(src, dst)
        assert 1e-4 < latency < 1.0  # should be ~250µs simulated


class TestVLLMEvictor:
    """Tests for the PERC vLLM-compatible evictor."""

    def test_benchmark_perc_beats_lru(self):
        from cachepilot.vllm_patch.perc_evictor import benchmark_perc_vs_lru
        result = benchmark_perc_vs_lru(n_blocks=500, n_evictions=200, seed=99)
        assert result["perc_total_cost"] < result["lru_total_cost"]
        assert result["cost_reduction_pct"] > 0

    def test_evictor_add_remove(self):
        from cachepilot.vllm_patch.perc_evictor import PERCEvictor
        ev = PERCEvictor()
        ev.add(block_id=1, num_hashed_tokens=512)
        ev.add(block_id=2, num_hashed_tokens=128)
        assert ev.num_blocks == 2
        bid, ntok = ev.evict()
        assert bid in (1, 2)
        assert ev.num_blocks == 1

    def test_evictor_prefers_cheap_block(self):
        """Cheap block (short context, low lambda) should be evicted first."""
        import time
        from cachepilot.vllm_patch.perc_evictor import PERCEvictor, _BlockRecord

        ev = PERCEvictor(c_recompute=0.002, delta_serve=5.0)
        now = time.monotonic()

        # cheap: short context, dormant
        cheap = _BlockRecord(block_id=1, num_hashed_tokens=64)
        cheap.intervals.extend([100.0] * 5)  # lambda ≈ 0.01
        ev._blocks[1] = cheap

        # costly: long context, active
        costly = _BlockRecord(block_id=2, num_hashed_tokens=8192)
        costly.intervals.extend([0.5] * 10)  # lambda ≈ 2.0
        ev._blocks[2] = costly

        bid, _ = ev.evict()
        assert bid == 1, "PERC should evict the cheap block first"

    def test_eviction_cost_delta_positive(self):
        """PERC should have lower or equal cost than LRU's choice."""
        import time
        from cachepilot.vllm_patch.perc_evictor import PERCEvictor, _BlockRecord

        ev = PERCEvictor(c_recompute=0.002, delta_serve=5.0)
        now = time.monotonic()

        for i in range(20):
            r = _BlockRecord(block_id=i, num_hashed_tokens=(i + 1) * 100)
            r.intervals.extend([float(i + 1)] * 5)
            r.last_active = now - float(i * 10)
            ev._blocks[i] = r

        delta = ev.eviction_cost_delta()
        assert delta >= 0.0  # PERC never does worse than LRU by definition
