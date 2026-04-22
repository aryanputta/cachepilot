import json

from cachepilot.vllm_benchmark import load_prompt_set, render_hf_vllm_uv_script


def test_load_prompt_set_from_prompt_file(tmp_path):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("alpha\n\nbeta\n")

    prompt_set = load_prompt_set(prompts_path=prompts_path)

    assert prompt_set.source == "file"
    assert prompt_set.schema == "prompt_file"
    assert prompt_set.prompts == ["alpha", "beta"]


def test_load_prompt_set_from_local_flat_dataset(tmp_path):
    dataset_path = tmp_path / "alpaca.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "instruction": "Summarize the system design",
                    "input": "CachePilot schedules KV memory",
                    "output": "Summary",
                }
            ]
        )
    )

    prompt_set = load_prompt_set(local_dataset=dataset_path)

    assert prompt_set.source == "local"
    assert prompt_set.schema == "flat_prompt_response"
    assert prompt_set.prompts == ["Summarize the system design\nCachePilot schedules KV memory"]


def test_load_prompt_set_from_local_conversation_dataset(tmp_path):
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "chat-1",
                    "conversations": [
                        {"from": "human", "value": "Explain FP8 KV caching."},
                        {"from": "gpt", "value": "It compresses cache state."},
                        {"from": "human", "value": "What is the tradeoff?"},
                    ],
                }
            ]
        )
    )

    prompt_set = load_prompt_set(local_dataset=dataset_path)

    assert prompt_set.schema == "conversation_list"
    assert prompt_set.prompts == ["Explain FP8 KV caching.\nWhat is the tradeoff?"]


def test_render_hf_vllm_uv_script_contains_config_and_patch():
    script = render_hf_vllm_uv_script(
        model="gpt2",
        hf_dataset="yahma/alpaca-cleaned",
        limit=8,
        compare_perc=True,
    )

    assert "CACHEPILOT_BENCHMARK_JSON_START" in script
    assert '"hf_dataset": "yahma/alpaca-cleaned"' in script
    assert "class PERCEvictor" in script
    assert "def install_into_vllm" in script
