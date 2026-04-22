from __future__ import annotations

from typing import Iterable, List, Sequence

from .vllm_benchmark import (
    VLLMBenchmarkResult as ModelServeResult,
    benchmark_vllm_model,
    load_prompts,
)


def compare_vllm_models(
    models: Sequence[str],
    prompts: Sequence[str],
    max_tokens: int = 64,
    gpu_memory_utilization: float = 0.8,
    use_perc_evictor: bool = False,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
    prompt_source: str = "inline",
    prompt_schema: str = "prompt_list",
) -> List[ModelServeResult]:
    results: List[ModelServeResult] = []
    for model in models:
        results.append(
            benchmark_vllm_model(
                model=model,
                prompts=prompts,
                prompt_source=prompt_source,
                prompt_schema=prompt_schema,
                max_tokens=max_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                use_perc_evictor=use_perc_evictor,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                label=model,
            )
        )
    return results


def candidate_advantages(
    candidate: ModelServeResult,
    baselines: Iterable[ModelServeResult],
) -> List[str]:
    advantages: List[str] = []
    baseline_list = list(baselines)
    if not baseline_list:
        return advantages

    if all(candidate.tokens_per_second > baseline.tokens_per_second for baseline in baseline_list):
        advantages.append("higher generation throughput")
    if all(candidate.wall_time_s < baseline.wall_time_s for baseline in baseline_list):
        advantages.append("lower end-to-end latency")
    if candidate.generated_tokens >= max(b.generated_tokens for b in baseline_list):
        advantages.append("at least as much generated work completed")
    return advantages
