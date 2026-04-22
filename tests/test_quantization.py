from cachepilot.quantization import KVPrecision, kv_bytes_per_token


def test_fp8_is_half_the_size_of_fp16():
    fp16 = kv_bytes_per_token(32, 32, 128, KVPrecision.FP16)
    fp8 = kv_bytes_per_token(32, 32, 128, KVPrecision.FP8)
    assert fp8 * 2 == fp16


def test_parse_rejects_unknown_tier():
    try:
        KVPrecision.parse("bf16")
    except ValueError:
        return
    assert False, "Expected invalid tier to raise ValueError"
