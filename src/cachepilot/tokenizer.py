from __future__ import annotations

import os
import subprocess
from pathlib import Path

ASCII_PUNCTUATION = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")


def _segment_class(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        if ch.isascii() and ch.isupper():
            return "upper"
        if ch.isascii() and ch.islower():
            return "lower"
        return "non_ascii"
    return "other"


def _chars_per_token(segment: str) -> int:
    if not segment:
        return 1
    if any(not ch.isascii() for ch in segment):
        return 2
    if segment.isdigit():
        return 3
    if any(ch.isdigit() for ch in segment) and any(ch.isalpha() for ch in segment):
        return 3
    if any(ch.isupper() for ch in segment) and any(ch.islower() for ch in segment):
        return 4
    return 5


def _count_segment_tokens(segment: str) -> int:
    if not segment:
        return 0

    tokens = 0
    run = [segment[0]]
    prev_class = _segment_class(segment[0])

    for ch in segment[1:]:
        cls = _segment_class(ch)
        boundary = False
        if prev_class == "lower" and cls == "upper":
            boundary = True
        elif {prev_class, cls} == {"digit", "lower"} or {prev_class, cls} == {"digit", "upper"}:
            boundary = True

        if boundary:
            chars_per_token = _chars_per_token("".join(run))
            tokens += max((len(run) + chars_per_token - 1) // chars_per_token, 1)
            run = [ch]
        else:
            run.append(ch)
        prev_class = cls

    chars_per_token = _chars_per_token("".join(run))
    tokens += max((len(run) + chars_per_token - 1) // chars_per_token, 1)
    return tokens


def heuristic_count_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 1

    tokens = 0
    current = []
    for ch in stripped:
        if ch.isspace() or (ch.isascii() and ch in ASCII_PUNCTUATION):
            if current:
                tokens += _count_segment_tokens("".join(current))
                current.clear()
            if ch.isascii() and ch in ASCII_PUNCTUATION:
                tokens += 1
        else:
            current.append(ch)
    if current:
        tokens += _count_segment_tokens("".join(current))
    return max(tokens, 1)


def default_rust_binary() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "rust"
        / "tokenizer"
        / "target"
        / "release"
        / "cachepilot-tokenizer"
    )


def rust_count_tokens(text: str, binary: str | Path | None = None) -> int:
    target = Path(binary) if binary else Path(os.getenv("CACHEPILOT_TOKENIZER_BIN", default_rust_binary()))
    if not target.exists():
        raise FileNotFoundError(target)

    proc = subprocess.run(
        [str(target)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return max(int(proc.stdout.strip()), 1)


def count_tokens(text: str, prefer_rust: bool = True) -> int:
    if prefer_rust:
        try:
            return rust_count_tokens(text)
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass
    return heuristic_count_tokens(text)
