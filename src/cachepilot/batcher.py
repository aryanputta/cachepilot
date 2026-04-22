from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from .scheduler import ScheduledRequest


class BatchMode(Enum):
    MAX_THROUGHPUT = "max_throughput"
    LOW_LATENCY = "low_latency"
    ADAPTIVE = "adaptive"


@dataclass
class Batch:
    requests: List[ScheduledRequest]
    mode: BatchMode

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_len for r in self.requests)


class DynamicBatcher:
    """
    Composes a batch from a list of pending requests.

    MAX_THROUGHPUT: fill up to max_batch_size or max_tokens_per_batch.
    LOW_LATENCY:    dispatch immediately with a single request.
    ADAPTIVE:       switch mode based on queue depth vs. recent latency.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 8192,
        latency_target_ms: float = 50.0,
        mode: BatchMode = BatchMode.ADAPTIVE,
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.latency_target_ms = latency_target_ms
        self.mode = mode

    def build_batch(
        self,
        pending: List[ScheduledRequest],
        queue_depth: int = 0,
        recent_latency_ms: float = 0.0,
    ) -> Batch:
        if not pending:
            return Batch([], self.mode)

        effective = self.mode
        if self.mode == BatchMode.ADAPTIVE:
            under_target = recent_latency_ms < self.latency_target_ms * 0.8
            high_queue = queue_depth > 10
            effective = (
                BatchMode.MAX_THROUGHPUT if (high_queue or under_target)
                else BatchMode.LOW_LATENCY
            )

        if effective == BatchMode.LOW_LATENCY:
            return Batch([pending[0]], effective)

        selected: List[ScheduledRequest] = []
        total = 0
        for req in pending:
            if len(selected) >= self.max_batch_size:
                break
            if total + req.prompt_len > self.max_tokens_per_batch:
                break
            selected.append(req)
            total += req.prompt_len

        return Batch(selected or [pending[0]], effective)
