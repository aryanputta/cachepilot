from cachepilot.tokenizer import count_tokens, heuristic_count_tokens


def test_heuristic_tokenizer_counts_punctuation():
    assert heuristic_count_tokens("hello, world!") == 4


def test_heuristic_tokenizer_splits_longer_words_more_realistically():
    assert heuristic_count_tokens("admissioncontroller") == 4


def test_heuristic_tokenizer_penalizes_camel_case_and_digits():
    assert heuristic_count_tokens("AdmissionPolicyV2") == 7


def test_count_tokens_falls_back_without_rust_binary():
    assert count_tokens("CachePilot admission policy", prefer_rust=False) >= 1
