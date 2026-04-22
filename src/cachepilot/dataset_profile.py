from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .tokenizer import count_tokens

PROMPT_COLUMNS = (
    "instruction",
    "input",
    "prompt",
    "question",
    "query",
    "context",
    "user_input",
)
RESPONSE_COLUMNS = (
    "output",
    "response",
    "answer",
    "completion",
    "assistant_response",
)
CONVERSATION_COLUMNS = ("conversations", "messages", "conversation", "chat", "dialogue")
ROLE_COLUMNS = ("role", "speaker", "author")
MESSAGE_COLUMNS = ("message", "content", "text", "utterance")
GROUP_COLUMNS = ("conversation_id", "dialogue_id", "chat_id", "thread_id", "session_id")

HF_DATASET_PRESETS: dict[str, dict[str, str]] = {
    "oasst1": {"dataset": "OpenAssistant/oasst1", "split": "train"},
    "alpaca": {"dataset": "yahma/alpaca-cleaned", "split": "train"},
    "sharegpt": {"dataset": "Aeala/ShareGPT_Vicuna_unfiltered", "split": "train"},
    "arena": {"dataset": "lmsys/lmsys-chat-1m", "split": "train"},
}

KAGGLE_DATASET_SUGGESTIONS: dict[str, str] = {
    "alpaca": "https://www.kaggle.com/datasets/thedevastator/alpaca-language-instruction-training",
    "arena": "https://www.kaggle.com/datasets/lmsysorg/chatbot-arena-conversations",
    "multi_turn": "https://www.kaggle.com/datasets/abhayayare/multi-turn-chatbot-conversation-dataset",
}


@dataclass
class TokenStats:
    mean: float
    p50: float
    p95: float
    p99: float
    max: int


@dataclass
class ProfileRow:
    row_id: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


@dataclass
class DatasetTokenProfile:
    source: str
    dataset_name: str
    schema: str
    rows_profiled: int
    rows_total: int | None
    prompt_tokens: TokenStats
    response_tokens: TokenStats
    total_tokens: TokenStats
    estimated_total_tokens: int | None
    top_rows: list[ProfileRow]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_rows"] = [asdict(row) for row in self.top_rows]
        return payload


def _stats(values: list[int]) -> TokenStats:
    if not values:
        return TokenStats(mean=0.0, p50=0.0, p95=0.0, p99=0.0, max=0)
    arr = pd.Series(values, dtype="int64")
    return TokenStats(
        mean=float(arr.mean()),
        p50=float(arr.quantile(0.50)),
        p95=float(arr.quantile(0.95)),
        p99=float(arr.quantile(0.99)),
        max=int(arr.max()),
    )


def _normalize_message(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(
            message.get("role")
            or message.get("from")
            or message.get("speaker")
            or message.get("author")
            or "unknown"
        ).strip().lower()
        if role in {"gpt", "assistant", "chatgpt", "model"}:
            role = "assistant"
        elif role in {"human", "user", "prompter"}:
            role = "user"
        content = str(
            message.get("content")
            or message.get("value")
            or message.get("text")
            or message.get("message")
            or ""
        )
        return role, content
    return "unknown", str(message)


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _extract_from_messages(messages: list[Any]) -> tuple[str, str]:
    prompt_parts: list[str] = []
    response_parts: list[str] = []
    for raw in messages:
        role, content = _normalize_message(raw)
        if not content:
            continue
        if "assistant" in role or role == "bot":
            response_parts.append(content)
        else:
            prompt_parts.append(content)
    return "\n".join(prompt_parts), "\n".join(response_parts)


def _row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "conversation_id", "dialogue_id", "chat_id", "thread_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return f"row-{fallback}"


def _profile_samples(
    samples: list[tuple[str, str, str]],
    *,
    source: str,
    dataset_name: str,
    schema: str,
    rows_total: int | None,
    top_k: int = 5,
) -> DatasetTokenProfile:
    prompt_counts: list[int] = []
    response_counts: list[int] = []
    total_counts: list[int] = []
    profile_rows: list[ProfileRow] = []

    for row_id, prompt_text, response_text in samples:
        prompt_tokens = count_tokens(prompt_text, prefer_rust=True)
        response_tokens = count_tokens(response_text, prefer_rust=True) if response_text else 0
        total_tokens = prompt_tokens + response_tokens
        prompt_counts.append(prompt_tokens)
        response_counts.append(response_tokens)
        total_counts.append(total_tokens)
        profile_rows.append(
            ProfileRow(
                row_id=row_id,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
            )
        )

    top_rows = sorted(profile_rows, key=lambda row: row.total_tokens, reverse=True)[:top_k]
    estimated_total = None
    if rows_total is not None and total_counts:
        estimated_total = int(sum(total_counts) / len(total_counts) * rows_total)

    return DatasetTokenProfile(
        source=source,
        dataset_name=dataset_name,
        schema=schema,
        rows_profiled=len(samples),
        rows_total=rows_total,
        prompt_tokens=_stats(prompt_counts),
        response_tokens=_stats(response_counts),
        total_tokens=_stats(total_counts),
        estimated_total_tokens=estimated_total,
        top_rows=top_rows,
    )


def _profile_flat_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    dataset_name: str,
    rows_total: int | None,
) -> DatasetTokenProfile:
    samples: list[tuple[str, str, str]] = []
    for idx, row in enumerate(rows):
        prompt = "\n".join(
            str(row[column]).strip()
            for column in PROMPT_COLUMNS
            if column in row and row[column] not in (None, "")
        ).strip()
        response = next(
            (
                str(row[column]).strip()
                for column in RESPONSE_COLUMNS
                if column in row and row[column] not in (None, "")
            ),
            "",
        )
        if prompt or response:
            samples.append((_row_id(row, idx), prompt, response))
    return _profile_samples(
        samples,
        source=source,
        dataset_name=dataset_name,
        schema="flat_prompt_response",
        rows_total=rows_total,
    )


def _profile_conversation_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    dataset_name: str,
    rows_total: int | None,
) -> DatasetTokenProfile:
    samples: list[tuple[str, str, str]] = []
    for idx, row in enumerate(rows):
        for column in CONVERSATION_COLUMNS:
            if column not in row:
                continue
            parsed = _parse_json_maybe(row[column])
            if isinstance(parsed, list):
                prompt, response = _extract_from_messages(parsed)
                samples.append((_row_id(row, idx), prompt, response))
                break
    return _profile_samples(
        samples,
        source=source,
        dataset_name=dataset_name,
        schema="conversation_list",
        rows_total=rows_total,
    )


def _profile_turn_table(
    frame: pd.DataFrame,
    *,
    source: str,
    dataset_name: str,
) -> DatasetTokenProfile:
    group_column = next((column for column in GROUP_COLUMNS if column in frame.columns), None)
    role_column = next((column for column in ROLE_COLUMNS if column in frame.columns), None)
    message_column = next((column for column in MESSAGE_COLUMNS if column in frame.columns), None)
    if group_column is None or role_column is None or message_column is None:
        raise ValueError("Turn-table schema requires group, role, and message columns.")

    samples: list[tuple[str, str, str]] = []
    for conversation_id, group in frame.groupby(group_column, sort=False):
        prompt_parts: list[str] = []
        response_parts: list[str] = []
        for _, row in group.iterrows():
            role = str(row[role_column]).strip().lower()
            content = str(row[message_column]).strip()
            if not content:
                continue
            if "assistant" in role or role == "bot":
                response_parts.append(content)
            else:
                prompt_parts.append(content)
        samples.append((str(conversation_id), "\n".join(prompt_parts), "\n".join(response_parts)))

    return _profile_samples(
        samples,
        source=source,
        dataset_name=dataset_name,
        schema="turn_table",
        rows_total=frame[group_column].nunique(),
    )


def profile_local_dataset(path: str | Path, limit: int | None = None) -> DatasetTokenProfile:
    target = Path(path)
    suffix = target.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(target, nrows=limit)
    elif suffix in {".jsonl", ".json"}:
        if suffix == ".jsonl":
            frame = pd.read_json(target, lines=True)
        else:
            payload = json.loads(target.read_text())
            frame = pd.DataFrame(payload)
        if limit is not None:
            frame = frame.head(limit)
    elif suffix == ".parquet":
        frame = pd.read_parquet(target)
        if limit is not None:
            frame = frame.head(limit)
    else:
        raise ValueError(f"Unsupported local dataset format: {suffix}")

    dataset_name = target.name
    records = frame.to_dict(orient="records")
    columns = set(frame.columns)

    if any(column in columns for column in CONVERSATION_COLUMNS):
        return _profile_conversation_rows(
            records,
            source="local",
            dataset_name=dataset_name,
            rows_total=len(frame),
        )
    if any(column in columns for column in GROUP_COLUMNS) and any(
        column in columns for column in ROLE_COLUMNS
    ):
        return _profile_turn_table(frame, source="local", dataset_name=dataset_name)
    return _profile_flat_rows(records, source="local", dataset_name=dataset_name, rows_total=len(frame))


def profile_hf_dataset(
    dataset: str,
    *,
    split: str = "train",
    config: str | None = None,
    limit: int = 1000,
    streaming: bool = True,
) -> DatasetTokenProfile:
    from datasets import load_dataset

    ds = load_dataset(dataset, name=config, split=split, streaming=streaming)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        rows.append(dict(row))
        if idx + 1 >= limit:
            break

    if not rows:
        raise ValueError(f"No rows found for dataset {dataset} split {split}.")

    columns = set(rows[0].keys())
    if any(column in columns for column in CONVERSATION_COLUMNS):
        return _profile_conversation_rows(
            rows,
            source="huggingface",
            dataset_name=dataset,
            rows_total=None,
        )
    if any(column in columns for column in GROUP_COLUMNS) and any(
        column in columns for column in ROLE_COLUMNS
    ):
        frame = pd.DataFrame(rows)
        return _profile_turn_table(frame, source="huggingface", dataset_name=dataset)
    return _profile_flat_rows(rows, source="huggingface", dataset_name=dataset, rows_total=None)
