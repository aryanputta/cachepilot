"""
cost_model.py — Translate eviction events and VRAM usage into real dollar costs.

GPU pricing sourced from public provider listings (2025-Q2):
  Lambda Labs, CoreWeave, AWS, Vast.ai (spot median)

Cost of a KV cache eviction event = time to recompute that context × GPU $/hr.
PERC's provable improvement in expected eviction cost directly maps to
dollars saved per serving hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .quantization import KVPrecision, kv_bytes_per_token

# ---------------------------------------------------------------------------
# GPU catalog — VRAM, TFLOPS (BF16), and $/hr from public cloud listings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float
    bf16_tflops: float        # peak BF16 tensor core throughput
    cost_per_hr_usd: float    # on-demand cloud price, $/hr
    provider: str

GPU_CATALOG: Dict[str, GPU] = {
    "h100_sxm": GPU("H100 SXM5 80GB",  80.0, 1979.0, 2.49, "Lambda Labs"),
    "h100_pcie": GPU("H100 PCIe 80GB",  80.0, 1513.0, 2.06, "CoreWeave"),
    "a100_80":   GPU("A100 SXM4 80GB",  80.0,  312.0, 1.29, "Lambda Labs"),
    "a100_40":   GPU("A100 PCIe 40GB",  40.0,  312.0, 0.90, "CoreWeave"),
    "a10g":      GPU("A10G 24GB",       24.0,  125.0, 0.60, "AWS"),
    "rtx4090":   GPU("RTX 4090 24GB",   24.0,  165.3, 0.50, "Vast.ai spot"),
    "rtx3090":   GPU("RTX 3090 24GB",   24.0,   71.0, 0.25, "Vast.ai spot"),
    "l4":        GPU("L4 24GB",         24.0,  121.0, 0.80, "GCP"),
}


# ---------------------------------------------------------------------------
# Model catalog — parameter counts and KV cache sizing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float       # billions of parameters
    n_layers: int
    n_heads: int
    head_dim: int
    dtype_bytes: int = 2  # FP16 default

    def kv_bytes_per_token(self, precision: str | KVPrecision = KVPrecision.FP16) -> int:
        """2 (K+V) × layers × heads × head_dim × dtype_bytes."""
        return kv_bytes_per_token(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            precision=precision,
        )

    def vram_for_weights_gb(self) -> float:
        return self.params_b * 1e9 * self.dtype_bytes / 1024**3

MODEL_CATALOG: Dict[str, ModelSpec] = {
    "llama2_7b":  ModelSpec("LLaMA-2-7B",   7.0, 32, 32, 128),
    "llama2_13b": ModelSpec("LLaMA-2-13B", 13.0, 40, 40, 128),
    "llama2_70b": ModelSpec("LLaMA-2-70B", 70.0, 80, 64, 128),
    "llama3_8b":  ModelSpec("LLaMA-3-8B",   8.0, 32, 32, 128),
    "llama3_70b": ModelSpec("LLaMA-3-70B", 70.0, 80, 64, 128),
    "mistral_7b": ModelSpec("Mistral-7B",   7.0, 32, 32, 128),
    "mixtral_8x7b": ModelSpec("Mixtral-8×7B", 47.0, 32, 32, 128),
}


# ---------------------------------------------------------------------------
# Cost calculations
# ---------------------------------------------------------------------------

@dataclass
class CostReport:
    gpu: GPU
    model: ModelSpec
    total_context_tokens: int
    eviction_events: int
    total_eviction_cost_s: float    # expected recompute time from PERC score
    actual_recompute_cost_s: float  # wall-clock recompute based on GPU TFLOPS
    cost_per_hour_usd: float
    savings_vs_lru_s: float         # time saved vs LRU baseline
    savings_vs_lru_usd: float       # dollars saved vs LRU baseline
    sessions_saved_by_int8: int     # extra concurrent sessions from INT8 quant
    effective_vram_gb: float

    def summary(self) -> str:
        lines = [
            f"GPU:              {self.gpu.name}  (${self.gpu.cost_per_hr_usd:.2f}/hr)",
            f"Model:            {self.model.name}  ({self.model.params_b:.0f}B params)",
            f"Eviction events:  {self.eviction_events}",
            f"Recompute cost:   {self.actual_recompute_cost_s:.1f}s  "
            f"(${self.actual_recompute_cost_s / 3600 * self.gpu.cost_per_hr_usd:.4f})",
            f"PERC saves vs LRU: {self.savings_vs_lru_s:.1f}s  "
            f"(${self.savings_vs_lru_usd:.4f}/run, ${self.savings_vs_lru_usd * 3600:.2f}/hr extrapolated)",
            f"INT8 extra slots: +{self.sessions_saved_by_int8} concurrent sessions",
        ]
        return "\n".join(lines)


def compute_recompute_cost_s(
    eviction_cost_score_s: float,
    model: ModelSpec,
    gpu: GPU,
) -> float:
    """
    Convert PERC eviction cost score to actual wall-clock recompute seconds.

    The PERC score uses c_recompute = 0.002 s/token as a normalized unit.
    Here we compute the actual time based on GPU TFLOPS and model FLOPs per token.

    KV recompute FLOPs per token ≈ 4 × n_layers × n_heads × head_dim × seq_len
    (simplified attention FLOPs for one token attending to seq_len context)
    """
    # Approximate: recompute score in PERC units → actual seconds via TFLOPS
    # PERC cost = seq_len × 0.002.  Real cost = seq_len × (attn_flops / GPU_TFLOPS)
    # attn_flops_per_seq_token ≈ 4 × layers × heads × head_dim  (one attention pass)
    flops_per_token_per_context_token = (
        4 * model.n_layers * model.n_heads * model.head_dim
    )
    # TFLOPS = 10^12 FLOPs/s
    real_s_per_token_per_context = flops_per_token_per_context_token / (gpu.bf16_tflops * 1e12)

    # PERC score × (real_s / 0.002) = actual wall-clock cost
    return eviction_cost_score_s * (real_s_per_token_per_context / 0.002)


def compute_int8_session_gain(
    gpu: GPU,
    model: ModelSpec,
    avg_context_tokens: int = 512,
) -> int:
    """
    How many additional concurrent sessions fit when KV cache is INT8 vs FP16?

    INT8 halves the KV cache footprint, freeing space for more sessions.
    """
    kv_fp16 = model.kv_bytes_per_token() * avg_context_tokens
    kv_int8 = kv_fp16 // 2
    weight_bytes = int(model.vram_for_weights_gb() * 1024**3)
    usable_vram = int(gpu.vram_gb * 1024**3) - weight_bytes

    sessions_fp16 = max(1, usable_vram // kv_fp16)
    sessions_int8 = max(1, usable_vram // kv_int8)
    return sessions_int8 - sessions_fp16


def compute_kv_tier_session_gain(
    gpu: GPU,
    model: ModelSpec,
    avg_context_tokens: int = 512,
    precision: str | KVPrecision = KVPrecision.FP16,
) -> int:
    tier = KVPrecision.parse(precision)
    kv_baseline = model.kv_bytes_per_token(KVPrecision.FP16) * avg_context_tokens
    kv_tier = model.kv_bytes_per_token(tier) * avg_context_tokens
    weight_bytes = int(model.vram_for_weights_gb() * 1024**3)
    usable_vram = int(gpu.vram_gb * 1024**3) - weight_bytes

    sessions_baseline = max(1, usable_vram // kv_baseline)
    sessions_tier = max(1, usable_vram // max(kv_tier, 1))
    return sessions_tier - sessions_baseline


def full_cost_report(
    gpu_key: str,
    model_key: str,
    eviction_events: int,
    total_eviction_cost_s: float,
    lru_eviction_cost_s: float,
    avg_context_tokens: int = 512,
) -> CostReport:
    gpu = GPU_CATALOG[gpu_key]
    model = MODEL_CATALOG[model_key]

    actual_recompute = compute_recompute_cost_s(total_eviction_cost_s, model, gpu)
    lru_actual = compute_recompute_cost_s(lru_eviction_cost_s, model, gpu)
    savings_s = lru_actual - actual_recompute
    savings_usd = savings_s / 3600.0 * gpu.cost_per_hr_usd

    usable_vram = gpu.vram_gb - model.vram_for_weights_gb()

    return CostReport(
        gpu=gpu,
        model=model,
        total_context_tokens=eviction_events * avg_context_tokens,
        eviction_events=eviction_events,
        total_eviction_cost_s=total_eviction_cost_s,
        actual_recompute_cost_s=actual_recompute,
        cost_per_hour_usd=gpu.cost_per_hr_usd,
        savings_vs_lru_s=savings_s,
        savings_vs_lru_usd=savings_usd,
        sessions_saved_by_int8=compute_int8_session_gain(gpu, model, avg_context_tokens),
        effective_vram_gb=usable_vram,
    )
