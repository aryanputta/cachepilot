import os

import pytest

from cachepilot.vllm_patch.perc_evictor import install_into_vllm


def _models():
    models = [("gpt2", os.getenv("CACHEPILOT_VLLM_GPT2_MODEL", "gpt2"))]
    llama = os.getenv("CACHEPILOT_VLLM_LLAMA_MODEL")
    if llama:
        models.append(("llama2", llama))
    return models


@pytest.mark.integration
@pytest.mark.parametrize(("label", "model_name"), _models())
def test_real_vllm_generate_smoke(label, model_name):
    pytest.importorskip("vllm")
    from vllm import LLM, SamplingParams

    install_into_vllm()
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=float(os.getenv("CACHEPILOT_VLLM_GPU_UTIL", "0.6")),
        disable_log_stats=True,
    )
    outputs = llm.generate(
        ["CachePilot keeps KV cache hot.", "Admission policy test prompt."],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    assert len(outputs) == 2
    assert all(output.outputs for output in outputs)
