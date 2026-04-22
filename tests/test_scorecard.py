from cachepilot.scorecard import build_hardware_scorecard, resolve_model_spec


def test_fp8_doubles_cache_capacity_and_bandwidth_bound():
    model = resolve_model_spec("llama3_8b")
    card = build_hardware_scorecard("a100_80", model, avg_context_tokens=2048)

    assert card.fp8.cache_tokens_capacity >= card.fp16.cache_tokens_capacity * 1.99
    assert card.fp8.roofline_tok_s >= card.fp16.roofline_tok_s * 1.99


def test_decode_attention_is_memory_bound_for_fp16():
    model = resolve_model_spec("llama3_8b")
    card = build_hardware_scorecard("h100_80", model, avg_context_tokens=2048)

    assert card.fp16.memory_bound is True
    assert card.fp16.arithmetic_intensity_flops_per_byte < card.ridge_point_flops_per_byte


def test_custom_model_resolution_requires_shape():
    model = resolve_model_spec(
        None,
        name="Custom 3B",
        params_b=3.0,
        n_layers=24,
        n_heads=24,
        head_dim=128,
    )

    assert model.name == "Custom 3B"
    assert model.n_layers == 24
