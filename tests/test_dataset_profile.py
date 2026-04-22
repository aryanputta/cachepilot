import json

import pandas as pd

from cachepilot.dataset_profile import profile_local_dataset


def test_profile_local_instruction_json(tmp_path):
    path = tmp_path / "alpaca.json"
    payload = [
        {
            "instruction": "Summarize the article",
            "input": "A short article about GPUs.",
            "output": "The article explains GPU memory pressure.",
        },
        {
            "instruction": "Write a haiku",
            "input": "",
            "output": "Quiet cache at dawn.",
        },
    ]
    path.write_text(json.dumps(payload))

    result = profile_local_dataset(path)
    assert result.schema == "flat_prompt_response"
    assert result.rows_profiled == 2
    assert result.total_tokens.max > 0


def test_profile_local_turn_table_csv(tmp_path):
    path = tmp_path / "kaggle_turns.csv"
    frame = pd.DataFrame(
        [
            {"conversation_id": "c1", "role": "user", "message": "hello"},
            {"conversation_id": "c1", "role": "bot", "message": "hi there"},
            {"conversation_id": "c2", "role": "user", "message": "what is fp8"},
            {"conversation_id": "c2", "role": "bot", "message": "a compact floating point format"},
        ]
    )
    frame.to_csv(path, index=False)

    result = profile_local_dataset(path)
    assert result.schema == "turn_table"
    assert result.rows_profiled == 2
    assert result.response_tokens.mean > 0


def test_profile_local_conversation_jsonl(tmp_path):
    path = tmp_path / "sharegpt.jsonl"
    rows = [
        {
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "Explain KV cache"},
                {"from": "assistant", "value": "It stores attention keys and values."},
            ],
        }
    ]
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = profile_local_dataset(path)
    assert result.schema == "conversation_list"
    assert result.rows_profiled == 1
    assert result.prompt_tokens.max > 0
