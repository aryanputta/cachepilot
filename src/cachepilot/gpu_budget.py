"""
gpu_budget.py — GPU memory budget optimizer and session capacity planner.

Answers the question: "Given my GPU, model, and eviction policy,
how many concurrent sessions can I serve, and what does it cost?"

Designed for engineers who are VRAM-limited and need to either:
  (a) fit more sessions on the same hardware
  (b) justify a smaller GPU tier by showing PERC + INT8 closes the gap

Key insight:
  FP16 KV on LLaMA-2-7B uses 524 KB/token.
  INT8 KV uses 262 KB/token.
  PERC reduces eviction recompute waste.
  Combined: you can often drop from A100 to A10G without throughput regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .cost_model import (
    GPU,
    ModelSpec,
    GPU_CATALOG,
    MODEL_CATALOG,
    compute_int8_session_gain,
    compute_kv_tier_session_gain,
)
from .quantization import KVPrecision


@dataclass
class BudgetPlan:
    gpu: GPU
    model: ModelSpec
    avg_context_tokens: int

    # FP16 baseline
    fp16_concurrent_sessions: int
    fp16_kv_per_session_mb: float
    fp16_usable_vram_gb: float

    # INT8 improvement
    int8_concurrent_sessions: int
    int8_session_gain: int
    int8_kv_per_session_mb: float
    fp8_concurrent_sessions: int
    fp8_session_gain: int
    fp8_kv_per_session_mb: float

    # PERC eviction cost reduction
    perc_recompute_saved_pct: float     # % reduction vs LRU
    perc_effective_extra_sessions: float  # fractional: time saved ≈ N more sessions

    # Cost comparison
    cost_per_hour_usd: float
    cost_per_1k_tokens_usd: float       # at p50 throughput
    tokens_per_hour: int

    def print_report(self) -> None:
        print(f"\n{'='*62}")
        print(f"  GPU Budget Plan: {self.model.name} on {self.gpu.name}")
        print(f"{'='*62}")
        print(f"  GPU:              {self.gpu.name}")
        print(f"  VRAM:             {self.gpu.vram_gb:.0f} GB total")
        print(f"  Weights:          {self.model.vram_for_weights_gb():.1f} GB")
        print(f"  Usable for KV:    {self.fp16_usable_vram_gb:.1f} GB")
        print()
        print(f"  Context per session: {self.avg_context_tokens} tokens")
        print(f"  FP16 KV/session:  {self.fp16_kv_per_session_mb:.1f} MB")
        print(f"  INT8 KV/session:  {self.int8_kv_per_session_mb:.1f} MB")
        print()
        print(f"  Concurrent sessions:")
        print(f"    FP16 + LRU:     {self.fp16_concurrent_sessions}")
        print(f"    INT8 + LRU:     {self.int8_concurrent_sessions}  (+{self.int8_session_gain})")
        print(f"    FP8  + LRU:     {self.fp8_concurrent_sessions}  (+{self.fp8_session_gain})")
        print(f"    INT8 + PERC:    {self.int8_concurrent_sessions}  +{self.perc_effective_extra_sessions:.1f} effective")
        print(f"                    (via {self.perc_recompute_saved_pct:.1f}% recompute reduction)")
        print()
        print(f"  Cost:             ${self.cost_per_hour_usd:.2f}/hr")
        print(f"  Throughput:       ~{self.tokens_per_hour:,} tokens/hr")
        print(f"  Cost/1K tokens:   ${self.cost_per_1k_tokens_usd:.4f}")
        print(f"{'='*62}\n")


def plan_gpu_budget(
    gpu_key: str,
    model_key: str,
    avg_context_tokens: int = 512,
    tokens_per_second: float = 35.0,
    perc_recompute_savings_pct: float = 25.1,
) -> BudgetPlan:
    """
    Compute a full budget plan for a GPU + model combination.

    perc_recompute_savings_pct: measured PERC improvement vs LRU (default: 25.1%)
    tokens_per_second: sustained throughput estimate for this GPU/model pair.
    """
    gpu = GPU_CATALOG[gpu_key]
    model = MODEL_CATALOG[model_key]

    kv_fp16 = model.kv_bytes_per_token() * avg_context_tokens
    kv_int8 = kv_fp16 // 2
    kv_fp8 = model.kv_bytes_per_token(KVPrecision.FP8) * avg_context_tokens
    weight_bytes = int(model.vram_for_weights_gb() * 1024**3)
    usable_bytes = int(gpu.vram_gb * 1024**3) - weight_bytes
    usable_gb = usable_bytes / 1024**3

    fp16_sessions = max(1, usable_bytes // kv_fp16)
    int8_sessions = max(1, usable_bytes // kv_int8)
    fp8_sessions = max(1, usable_bytes // kv_fp8)
    int8_gain = int8_sessions - fp16_sessions
    fp8_gain = fp8_sessions - fp16_sessions

    tokens_per_hour = int(tokens_per_second * 3600)
    cost_per_1k = gpu.cost_per_hr_usd / (tokens_per_hour / 1000)

    # PERC saves perc_recompute_savings_pct % of recompute time.
    # That saved time can serve additional context.  At a session rate of
    # tokens_per_second, the fractional equivalent extra sessions is:
    # (recompute_savings × sessions) / 100
    perc_extra = fp16_sessions * perc_recompute_savings_pct / 100.0

    return BudgetPlan(
        gpu=gpu,
        model=model,
        avg_context_tokens=avg_context_tokens,
        fp16_concurrent_sessions=fp16_sessions,
        fp16_kv_per_session_mb=kv_fp16 / 1024**2,
        fp16_usable_vram_gb=usable_gb,
        int8_concurrent_sessions=int8_sessions,
        int8_session_gain=int8_gain,
        int8_kv_per_session_mb=kv_int8 / 1024**2,
        fp8_concurrent_sessions=fp8_sessions,
        fp8_session_gain=fp8_gain,
        fp8_kv_per_session_mb=kv_fp8 / 1024**2,
        perc_recompute_saved_pct=perc_recompute_savings_pct,
        perc_effective_extra_sessions=perc_extra,
        cost_per_hour_usd=gpu.cost_per_hr_usd,
        cost_per_1k_tokens_usd=cost_per_1k,
        tokens_per_hour=tokens_per_hour,
    )


def compare_gpu_tiers(
    model_key: str = "llama2_7b",
    avg_context_tokens: int = 512,
) -> List[BudgetPlan]:
    """Compare all catalog GPUs for a given model + context length."""
    plans = []
    for gpu_key in GPU_CATALOG:
        try:
            plan = plan_gpu_budget(gpu_key, model_key, avg_context_tokens)
            if plan.fp16_usable_vram_gb > 0:
                plans.append(plan)
        except Exception:
            continue
    return sorted(plans, key=lambda p: p.cost_per_hour_usd)


def downgrade_recommendation(
    current_gpu_key: str,
    model_key: str,
    required_sessions: int,
    avg_context_tokens: int = 512,
) -> Optional[BudgetPlan]:
    """
    Can we use a cheaper GPU tier by enabling INT8 + PERC?

    Returns the cheapest GPU plan that meets required_sessions
    with INT8 + PERC, or None if no cheaper option exists.
    """
    current = plan_gpu_budget(current_gpu_key, model_key, avg_context_tokens)
    candidates = []
    for gpu_key, gpu in GPU_CATALOG.items():
        if gpu.cost_per_hr_usd < current.cost_per_hour_usd:
            plan = plan_gpu_budget(gpu_key, model_key, avg_context_tokens)
            effective = plan.int8_concurrent_sessions + plan.perc_effective_extra_sessions
            if effective >= required_sessions:
                candidates.append(plan)
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.cost_per_hour_usd)
