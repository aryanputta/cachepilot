from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .cost_model import MODEL_CATALOG, ModelSpec
from .quantization import KVPrecision


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    name: str
    vram_gb: float
    fp16_tflops: float
    fp8_tflops: float | None
    memory_bandwidth_gbps: float
    l2_cache_mb: float | None
    interconnect_gbps: float | None
    cost_per_hr_usd: float | None
    provider: str
    availability: str

    def peak_tflops(self, precision: KVPrecision) -> float:
        if precision == KVPrecision.FP8 and self.fp8_tflops is not None:
            return self.fp8_tflops
        return self.fp16_tflops


OFFICIAL_GPU_CATALOG: dict[str, HardwareSpec] = {
    "l4": HardwareSpec(
        key="l4",
        name="NVIDIA L4 24GB",
        vram_gb=24.0,
        fp16_tflops=121.0,
        fp8_tflops=242.5,
        memory_bandwidth_gbps=300.0,
        l2_cache_mb=None,
        interconnect_gbps=64.0,
        cost_per_hr_usd=0.80,
        provider="Hugging Face Jobs / NVIDIA",
        availability="HF Jobs l4x1",
    ),
    "a10g": HardwareSpec(
        key="a10g",
        name="NVIDIA A10G 24GB",
        vram_gb=24.0,
        fp16_tflops=125.0,
        fp8_tflops=None,
        memory_bandwidth_gbps=600.0,
        l2_cache_mb=None,
        interconnect_gbps=64.0,
        cost_per_hr_usd=1.00,
        provider="Hugging Face Jobs / NVIDIA",
        availability="HF Jobs a10g-small",
    ),
    "a100_80": HardwareSpec(
        key="a100_80",
        name="NVIDIA A100 80GB",
        vram_gb=80.0,
        fp16_tflops=312.0,
        fp8_tflops=None,
        memory_bandwidth_gbps=1935.0,
        l2_cache_mb=40.0,
        interconnect_gbps=600.0,
        cost_per_hr_usd=2.50,
        provider="Hugging Face Jobs / NVIDIA",
        availability="HF Jobs a100-large",
    ),
    "h100_80": HardwareSpec(
        key="h100_80",
        name="NVIDIA H100 80GB",
        vram_gb=80.0,
        fp16_tflops=989.5,
        fp8_tflops=1979.0,
        memory_bandwidth_gbps=3350.0,
        l2_cache_mb=50.0,
        interconnect_gbps=900.0,
        cost_per_hr_usd=None,
        provider="NVIDIA official specs",
        availability="Not listed in current HF Jobs pricing",
    ),
    "h200_141": HardwareSpec(
        key="h200_141",
        name="NVIDIA H200 141GB",
        vram_gb=141.0,
        fp16_tflops=989.5,
        fp8_tflops=1979.0,
        memory_bandwidth_gbps=4800.0,
        l2_cache_mb=None,
        interconnect_gbps=900.0,
        cost_per_hr_usd=5.00,
        provider="Hugging Face Jobs / NVIDIA",
        availability="HF Jobs h200",
    ),
}


@dataclass(frozen=True)
class PrecisionRoofline:
    precision: KVPrecision
    kv_bytes_per_context_token: int
    decode_kv_bytes_per_output_token: int
    attention_flops_per_output_token: int
    arithmetic_intensity_flops_per_byte: float
    bandwidth_bound_tok_s: float
    compute_bound_tok_s: float
    roofline_tok_s: float
    memory_bound: bool
    cache_tokens_capacity: int
    sessions_at_context: int


@dataclass(frozen=True)
class HardwareScorecard:
    hardware: HardwareSpec
    model: ModelSpec
    avg_context_tokens: int
    usable_vram_gb: float
    ridge_point_flops_per_byte: float
    fp16: PrecisionRoofline
    int8: PrecisionRoofline
    fp8: PrecisionRoofline
    measured_tok_s: float | None = None

    def selected(self, precision: str | KVPrecision) -> PrecisionRoofline:
        tier = KVPrecision.parse(precision)
        if tier == KVPrecision.FP16:
            return self.fp16
        if tier == KVPrecision.INT8:
            return self.int8
        return self.fp8

    def roofline_efficiency(self, precision: str | KVPrecision = KVPrecision.FP16) -> float | None:
        if self.measured_tok_s is None:
            return None
        selected = self.selected(precision)
        return self.measured_tok_s / max(selected.roofline_tok_s, 1e-9)

    def tokens_per_dollar_hour(self, precision: str | KVPrecision = KVPrecision.FP16) -> float | None:
        if not self.hardware.cost_per_hr_usd:
            return None
        selected = self.selected(precision)
        return selected.roofline_tok_s * 3600.0 / self.hardware.cost_per_hr_usd

    def cache_tokens_per_dollar_hour(self, precision: str | KVPrecision = KVPrecision.FP16) -> float | None:
        if not self.hardware.cost_per_hr_usd:
            return None
        selected = self.selected(precision)
        return selected.cache_tokens_capacity / self.hardware.cost_per_hr_usd


def resolve_model_spec(
    model_key: str | None = None,
    *,
    name: str = "Custom Model",
    params_b: float | None = None,
    n_layers: int | None = None,
    n_heads: int | None = None,
    head_dim: int | None = None,
) -> ModelSpec:
    if model_key is not None:
        if model_key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from: {', '.join(MODEL_CATALOG)}.")
        return MODEL_CATALOG[model_key]

    if None in (params_b, n_layers, n_heads, head_dim):
        raise ValueError("Custom model requires params_b, n_layers, n_heads, and head_dim.")
    return ModelSpec(
        name=name,
        params_b=float(params_b),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        head_dim=int(head_dim),
    )


def _usable_vram_gb(hardware: HardwareSpec, model: ModelSpec) -> float:
    return max(hardware.vram_gb - model.vram_for_weights_gb(), 0.0)


def _precision_roofline(
    hardware: HardwareSpec,
    model: ModelSpec,
    *,
    avg_context_tokens: int,
    precision: KVPrecision,
) -> PrecisionRoofline:
    kv_bytes_per_context_token = model.kv_bytes_per_token(precision)
    decode_kv_bytes_per_output_token = max(avg_context_tokens, 1) * kv_bytes_per_context_token
    attention_flops_per_output_token = max(
        4 * model.n_layers * model.n_heads * model.head_dim * max(avg_context_tokens, 1),
        1,
    )
    arithmetic_intensity = attention_flops_per_output_token / max(decode_kv_bytes_per_output_token, 1)
    bandwidth_bound_tok_s = hardware.memory_bandwidth_gbps * 1e9 / decode_kv_bytes_per_output_token
    compute_bound_tok_s = hardware.peak_tflops(precision) * 1e12 / attention_flops_per_output_token
    roofline_tok_s = min(bandwidth_bound_tok_s, compute_bound_tok_s)
    memory_bound = bandwidth_bound_tok_s <= compute_bound_tok_s

    usable_vram_bytes = _usable_vram_gb(hardware, model) * 1024**3
    cache_tokens_capacity = int(usable_vram_bytes // max(kv_bytes_per_context_token, 1))
    sessions_at_context = int(cache_tokens_capacity // max(avg_context_tokens, 1))
    return PrecisionRoofline(
        precision=precision,
        kv_bytes_per_context_token=kv_bytes_per_context_token,
        decode_kv_bytes_per_output_token=decode_kv_bytes_per_output_token,
        attention_flops_per_output_token=attention_flops_per_output_token,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
        bandwidth_bound_tok_s=bandwidth_bound_tok_s,
        compute_bound_tok_s=compute_bound_tok_s,
        roofline_tok_s=roofline_tok_s,
        memory_bound=memory_bound,
        cache_tokens_capacity=cache_tokens_capacity,
        sessions_at_context=sessions_at_context,
    )


def build_hardware_scorecard(
    hardware_key: str,
    model: ModelSpec,
    *,
    avg_context_tokens: int = 2048,
    measured_tok_s: float | None = None,
) -> HardwareScorecard:
    if hardware_key not in OFFICIAL_GPU_CATALOG:
        raise ValueError(
            f"Unknown hardware key '{hardware_key}'. Choose from: {', '.join(OFFICIAL_GPU_CATALOG)}."
        )
    hardware = OFFICIAL_GPU_CATALOG[hardware_key]
    ridge_point = hardware.fp16_tflops * 1e12 / max(hardware.memory_bandwidth_gbps * 1e9, 1.0)
    return HardwareScorecard(
        hardware=hardware,
        model=model,
        avg_context_tokens=avg_context_tokens,
        usable_vram_gb=_usable_vram_gb(hardware, model),
        ridge_point_flops_per_byte=ridge_point,
        fp16=_precision_roofline(
            hardware,
            model,
            avg_context_tokens=avg_context_tokens,
            precision=KVPrecision.FP16,
        ),
        int8=_precision_roofline(
            hardware,
            model,
            avg_context_tokens=avg_context_tokens,
            precision=KVPrecision.INT8,
        ),
        fp8=_precision_roofline(
            hardware,
            model,
            avg_context_tokens=avg_context_tokens,
            precision=KVPrecision.FP8,
        ),
        measured_tok_s=measured_tok_s,
    )


def compare_hardware_scorecards(
    model: ModelSpec,
    *,
    avg_context_tokens: int = 2048,
    measured_tok_s: float | None = None,
) -> list[HardwareScorecard]:
    cards = [
        build_hardware_scorecard(
            key,
            model,
            avg_context_tokens=avg_context_tokens,
            measured_tok_s=measured_tok_s,
        )
        for key in OFFICIAL_GPU_CATALOG
    ]
    return sorted(cards, key=lambda card: card.fp8.roofline_tok_s, reverse=True)
