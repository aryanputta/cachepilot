from __future__ import annotations

import inspect
import json
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .dataset_profile import (
    CONVERSATION_COLUMNS,
    GROUP_COLUMNS,
    HF_DATASET_PRESETS,
    MESSAGE_COLUMNS,
    PROMPT_COLUMNS,
    RESPONSE_COLUMNS,
    ROLE_COLUMNS,
)


@dataclass
class PromptSet:
    source: str
    label: str
    schema: str
    prompts: list[str]

    @property
    def prompt_count(self) -> int:
        return len(self.prompts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "schema": self.schema,
            "prompt_count": self.prompt_count,
        }


@dataclass
class VLLMBenchmarkResult:
    label: str
    model: str
    engine: str
    prompt_source: str
    prompt_schema: str
    prompt_count: int
    prompt_tokens: int
    generated_tokens: int
    wall_time_s: float
    max_tokens: int
    gpu_memory_utilization: float
    tensor_parallel_size: int

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / max(self.wall_time_s, 1e-6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "engine": self.engine,
            "prompt_source": self.prompt_source,
            "prompt_schema": self.prompt_schema,
            "prompt_count": self.prompt_count,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "wall_time_s": self.wall_time_s,
            "max_tokens": self.max_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tokens_per_second": self.tokens_per_second,
        }


def load_prompts(path: str | Path) -> list[str]:
    target = Path(path)
    if target.suffix == ".json":
        data = json.loads(target.read_text())
        return [str(item) for item in data if str(item).strip()]
    return [line.strip() for line in target.read_text().splitlines() if line.strip()]


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
        ).strip()
        return role, content
    return "unknown", str(message).strip()


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


def _row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "conversation_id", "dialogue_id", "chat_id", "thread_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return f"row-{fallback}"


def _extract_prompt_samples(rows: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    if not rows:
        raise ValueError("No rows available for prompt extraction.")

    columns = {column for row in rows for column in row.keys()}

    if any(column in columns for column in CONVERSATION_COLUMNS):
        prompts: list[str] = []
        for row in rows:
            for column in CONVERSATION_COLUMNS:
                if column not in row:
                    continue
                parsed = _parse_json_maybe(row[column])
                if not isinstance(parsed, list):
                    continue
                prompt_parts: list[str] = []
                for raw in parsed:
                    role, content = _normalize_message(raw)
                    if not content:
                        continue
                    if "assistant" in role or role == "bot":
                        continue
                    prompt_parts.append(content)
                prompt = "\n".join(prompt_parts).strip()
                if prompt:
                    prompts.append(prompt)
                break
        return "conversation_list", prompts

    if any(column in columns for column in GROUP_COLUMNS) and any(
        column in columns for column in ROLE_COLUMNS
    ):
        frame = pd.DataFrame(rows)
        group_column = next((column for column in GROUP_COLUMNS if column in frame.columns), None)
        role_column = next((column for column in ROLE_COLUMNS if column in frame.columns), None)
        message_column = next((column for column in MESSAGE_COLUMNS if column in frame.columns), None)
        if group_column and role_column and message_column:
            prompts = []
            for _, group in frame.groupby(group_column, sort=False):
                prompt_parts: list[str] = []
                for _, row in group.iterrows():
                    role = str(row[role_column]).strip().lower()
                    if "assistant" in role or role == "bot":
                        continue
                    content = str(row[message_column]).strip()
                    if content:
                        prompt_parts.append(content)
                prompt = "\n".join(prompt_parts).strip()
                if prompt:
                    prompts.append(prompt)
            return "turn_table", prompts

    prompts = []
    for idx, row in enumerate(rows):
        prompt = "\n".join(
            str(row[column]).strip()
            for column in PROMPT_COLUMNS
            if column in row and row[column] not in (None, "")
        ).strip()
        if not prompt:
            # Fallback for datasets that only expose a single string payload.
            fallback = next(
                (
                    str(value).strip()
                    for key, value in row.items()
                    if isinstance(value, str)
                    and key not in RESPONSE_COLUMNS
                    and value.strip()
                ),
                "",
            )
            prompt = fallback
        if prompt:
            prompts.append(prompt)
        elif any(column in row and row[column] not in (None, "") for column in RESPONSE_COLUMNS):
            prompts.append(f"Example {idx + 1}")
    return "flat_prompt_response", prompts


def _load_local_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
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
    return frame.to_dict(orient="records")


def _load_hf_rows(
    dataset: str,
    *,
    split: str = "train",
    config: str | None = None,
    limit: int = 128,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, name=config, split=split, streaming=streaming)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        rows.append(dict(row))
        if idx + 1 >= limit:
            break
    return rows


def load_prompt_set(
    *,
    prompts_path: str | Path | None = None,
    hf_dataset: str | None = None,
    local_dataset: str | Path | None = None,
    preset: str | None = None,
    split: str = "train",
    config: str | None = None,
    limit: int = 128,
) -> PromptSet:
    selected = [value is not None for value in (prompts_path, hf_dataset, local_dataset, preset)]
    if sum(selected) != 1:
        raise ValueError("Choose exactly one prompt source.")

    if prompts_path is not None:
        prompt_list = load_prompts(prompts_path)
        return PromptSet(
            source="file",
            label=str(prompts_path),
            schema="prompt_file",
            prompts=prompt_list,
        )

    if preset is not None:
        if preset not in HF_DATASET_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'.")
        preset_cfg = HF_DATASET_PRESETS[preset]
        hf_dataset = preset_cfg["dataset"]
        split = preset_cfg.get("split", split)

    if hf_dataset is not None:
        rows = _load_hf_rows(hf_dataset, split=split, config=config, limit=limit)
        schema, prompts = _extract_prompt_samples(rows)
        return PromptSet(
            source="huggingface",
            label=hf_dataset,
            schema=schema,
            prompts=prompts,
        )

    rows = _load_local_rows(local_dataset, limit=limit)
    schema, prompts = _extract_prompt_samples(rows)
    return PromptSet(
        source="local",
        label=str(local_dataset),
        schema=schema,
        prompts=prompts,
    )


def _count_prompt_tokens(tokenizer: Any, prompts: Sequence[str]) -> int:
    total = 0
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False)
        if hasattr(encoded, "input_ids"):
            token_ids = encoded.input_ids
        else:
            token_ids = encoded["input_ids"]
        total += len(token_ids)
    return total


def benchmark_vllm_model(
    *,
    model: str,
    prompts: Sequence[str],
    prompt_source: str = "inline",
    prompt_schema: str = "prompt_list",
    max_tokens: int = 64,
    gpu_memory_utilization: float = 0.8,
    use_perc_evictor: bool = False,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = True,
    disable_log_stats: bool = True,
    label: str | None = None,
) -> VLLMBenchmarkResult:
    if use_perc_evictor:
        from .vllm_patch.perc_evictor import install_into_vllm

        install_into_vllm()

    from vllm import LLM, SamplingParams

    prompt_list = [prompt for prompt in prompts if prompt.strip()]
    if not prompt_list:
        raise ValueError("No non-empty prompts available for benchmarking.")

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": enforce_eager,
        "disable_log_stats": disable_log_stats,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompt_tokens = _count_prompt_tokens(tokenizer, prompt_list)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    start = time.perf_counter()
    outputs = llm.generate(prompt_list, sampling_params)
    elapsed = time.perf_counter() - start
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs if output.outputs)
    engine = "vllm+perc" if use_perc_evictor else "vllm"
    return VLLMBenchmarkResult(
        label=label or f"{engine}:{model}",
        model=model,
        engine=engine,
        prompt_source=prompt_source,
        prompt_schema=prompt_schema,
        prompt_count=len(prompt_list),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        wall_time_s=elapsed,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )


def compare_vllm_backends(
    *,
    model: str,
    prompt_set: PromptSet,
    max_tokens: int = 64,
    gpu_memory_utilization: float = 0.8,
    compare_perc: bool = True,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
) -> list[VLLMBenchmarkResult]:
    results = [
        benchmark_vllm_model(
            model=model,
            prompts=prompt_set.prompts,
            prompt_source=prompt_set.label,
            prompt_schema=prompt_set.schema,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            use_perc_evictor=False,
            label=f"vllm:{model}",
        )
    ]
    if compare_perc:
        results.append(
            benchmark_vllm_model(
                model=model,
                prompts=prompt_set.prompts,
                prompt_source=prompt_set.label,
                prompt_schema=prompt_set.schema,
                max_tokens=max_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                use_perc_evictor=True,
                label=f"vllm+perc:{model}",
            )
        )
    return results


def render_hf_vllm_uv_script(
    *,
    model: str,
    hf_dataset: str,
    split: str = "train",
    config: str | None = None,
    limit: int = 64,
    max_tokens: int = 64,
    gpu_memory_utilization: float = 0.8,
    compare_perc: bool = True,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
) -> str:
    from .vllm_patch.perc_evictor import PERCEvictor, _BlockRecord, install_into_vllm

    cfg = {
        "model": model,
        "hf_dataset": hf_dataset,
        "split": split,
        "config": config,
        "limit": limit,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "compare_perc": compare_perc,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
    }
    block_record_src = inspect.getsource(_BlockRecord)
    perc_src = inspect.getsource(PERCEvictor)
    install_src = inspect.getsource(install_into_vllm)
    script = f"""\
import json
import math
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

from datasets import load_dataset
from huggingface_hub import whoami
from vllm import LLM, SamplingParams

CONFIG = {json.dumps(cfg, indent=2)}
CONVERSATION_COLUMNS = {json.dumps(list(CONVERSATION_COLUMNS))}
GROUP_COLUMNS = {json.dumps(list(GROUP_COLUMNS))}
ROLE_COLUMNS = {json.dumps(list(ROLE_COLUMNS))}
MESSAGE_COLUMNS = {json.dumps(list(MESSAGE_COLUMNS))}
PROMPT_COLUMNS = {json.dumps(list(PROMPT_COLUMNS))}
RESPONSE_COLUMNS = {json.dumps(list(RESPONSE_COLUMNS))}


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
        ).strip()
        return role, content
    return "unknown", str(message).strip()


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _extract_prompt_samples(rows: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    columns = {{column for row in rows for column in row.keys()}}
    if any(column in columns for column in CONVERSATION_COLUMNS):
        prompts: list[str] = []
        for row in rows:
            for column in CONVERSATION_COLUMNS:
                if column not in row:
                    continue
                parsed = _parse_json_maybe(row[column])
                if not isinstance(parsed, list):
                    continue
                prompt_parts: list[str] = []
                for raw in parsed:
                    role, content = _normalize_message(raw)
                    if content and not ("assistant" in role or role == "bot"):
                        prompt_parts.append(content)
                prompt = "\\n".join(prompt_parts).strip()
                if prompt:
                    prompts.append(prompt)
                break
        return "conversation_list", prompts

    if any(column in columns for column in GROUP_COLUMNS) and any(
        column in columns for column in ROLE_COLUMNS
    ):
        import pandas as pd

        frame = pd.DataFrame(rows)
        group_column = next((column for column in GROUP_COLUMNS if column in frame.columns), None)
        role_column = next((column for column in ROLE_COLUMNS if column in frame.columns), None)
        message_column = next((column for column in MESSAGE_COLUMNS if column in frame.columns), None)
        if group_column and role_column and message_column:
            prompts = []
            for _, group in frame.groupby(group_column, sort=False):
                prompt_parts: list[str] = []
                for _, row in group.iterrows():
                    role = str(row[role_column]).strip().lower()
                    if "assistant" in role or role == "bot":
                        continue
                    content = str(row[message_column]).strip()
                    if content:
                        prompt_parts.append(content)
                prompt = "\\n".join(prompt_parts).strip()
                if prompt:
                    prompts.append(prompt)
            return "turn_table", prompts

    prompts = []
    for idx, row in enumerate(rows):
        prompt = "\\n".join(
            str(row[column]).strip()
            for column in PROMPT_COLUMNS
            if column in row and row[column] not in (None, "")
        ).strip()
        if not prompt:
            fallback = next(
                (
                    str(value).strip()
                    for key, value in row.items()
                    if isinstance(value, str)
                    and key not in RESPONSE_COLUMNS
                    and value.strip()
                ),
                "",
            )
            prompt = fallback
        if prompt:
            prompts.append(prompt)
        elif any(column in row and row[column] not in (None, "") for column in RESPONSE_COLUMNS):
            prompts.append(f"Example {{idx + 1}}")
    return "flat_prompt_response", prompts


{block_record_src}


{perc_src}


{install_src}


def _count_prompt_tokens(tokenizer, prompts: Sequence[str]) -> int:
    total = 0
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False)
        if hasattr(encoded, "input_ids"):
            token_ids = encoded.input_ids
        else:
            token_ids = encoded["input_ids"]
        total += len(token_ids)
    return total


def _run_once(use_perc: bool, prompts: Sequence[str], schema: str) -> dict[str, Any]:
    if use_perc:
        install_into_vllm()
    llm_kwargs = {{
        "model": CONFIG["model"],
        "gpu_memory_utilization": CONFIG["gpu_memory_utilization"],
        "tensor_parallel_size": CONFIG["tensor_parallel_size"],
        "enforce_eager": True,
        "disable_log_stats": True,
    }}
    if CONFIG["max_model_len"] is not None:
        llm_kwargs["max_model_len"] = CONFIG["max_model_len"]
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompt_tokens = _count_prompt_tokens(tokenizer, prompts)
    start = time.perf_counter()
    outputs = llm.generate(
        list(prompts),
        SamplingParams(max_tokens=CONFIG["max_tokens"], temperature=0.0),
    )
    wall_time_s = time.perf_counter() - start
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs if output.outputs)
    engine = "vllm+perc" if use_perc else "vllm"
    return {{
        "label": f"{{engine}}:{{CONFIG['model']}}",
        "engine": engine,
        "model": CONFIG["model"],
        "prompt_source": CONFIG["hf_dataset"],
        "prompt_schema": schema,
        "prompt_count": len(prompts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "wall_time_s": wall_time_s,
        "tokens_per_second": generated_tokens / max(wall_time_s, 1e-6),
        "max_tokens": CONFIG["max_tokens"],
        "gpu_memory_utilization": CONFIG["gpu_memory_utilization"],
        "tensor_parallel_size": CONFIG["tensor_parallel_size"],
    }}


def main() -> None:
    try:
        identity = whoami()
        print("HF identity:", identity.get("name") or identity)
    except Exception as exc:
        print("HF identity unavailable:", exc)

    ds = load_dataset(
        CONFIG["hf_dataset"],
        name=CONFIG["config"],
        split=CONFIG["split"],
        streaming=True,
    )
    rows = []
    for idx, row in enumerate(ds):
        rows.append(dict(row))
        if idx + 1 >= CONFIG["limit"]:
            break
    if not rows:
        raise RuntimeError("Dataset returned no rows.")

    schema, prompts = _extract_prompt_samples(rows)
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        raise RuntimeError("No prompts extracted from dataset.")

    payload = {{
        "config": CONFIG,
        "schema": schema,
        "results": [_run_once(False, prompts, schema)],
    }}
    if CONFIG["compare_perc"]:
        payload["results"].append(_run_once(True, prompts, schema))

    print("CACHEPILOT_BENCHMARK_JSON_START")
    print(json.dumps(payload, indent=2))
    print("CACHEPILOT_BENCHMARK_JSON_END")


if __name__ == "__main__":
    main()
"""
    return textwrap.dedent(script)
