from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class KVPrecisionSpec:
    name: str
    bytes_per_scalar: float
    compression_vs_fp16: float
    description: str


class KVPrecision(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    FP8 = "fp8"

    @property
    def spec(self) -> KVPrecisionSpec:
        return _KV_SPECS[self]

    @property
    def bytes_per_scalar(self) -> float:
        return self.spec.bytes_per_scalar

    @property
    def compression_vs_fp16(self) -> float:
        return self.spec.compression_vs_fp16

    @classmethod
    def parse(cls, value: str | KVPrecision) -> KVPrecision:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unknown KV precision tier '{value}'. "
                f"Choose from: {', '.join(t.value for t in cls)}."
            ) from exc


_KV_SPECS = {
    KVPrecision.FP16: KVPrecisionSpec(
        name="FP16",
        bytes_per_scalar=2.0,
        compression_vs_fp16=1.0,
        description="Baseline half-precision KV cache.",
    ),
    KVPrecision.INT8: KVPrecisionSpec(
        name="INT8",
        bytes_per_scalar=1.0,
        compression_vs_fp16=2.0,
        description="Per-channel symmetric INT8 KV quantization.",
    ),
    KVPrecision.FP8: KVPrecisionSpec(
        name="FP8",
        bytes_per_scalar=1.0,
        compression_vs_fp16=2.0,
        description=(
            "Packed FP8 KV cache tier. This is a true 2x compression tier versus FP16; "
            "4x would require an INT4/NVFP4-style format instead."
        ),
    ),
}


def kv_bytes_per_token(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    precision: str | KVPrecision = KVPrecision.FP16,
) -> int:
    tier = KVPrecision.parse(precision)
    return int(2 * n_layers * n_heads * head_dim * tier.bytes_per_scalar)
