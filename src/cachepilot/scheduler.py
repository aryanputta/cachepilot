from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Priority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class ScheduledRequest:
    _priority: int = field(compare=True)
    _tie: float = field(compare=True)  # arrival time as tiebreaker

    req_id: str = field(compare=False)
    prompt_len: int = field(compare=False)
    max_new_tokens: int = field(compare=False)
    workload: str = field(compare=False, default="chat")
    arrival_time: float = field(compare=False, default_factory=time.monotonic)
    sla_deadline: Optional[float] = field(compare=False, default=None)

    @classmethod
    def create(
        cls,
        req_id: str,
        prompt_len: int,
        max_new_tokens: int,
        workload: str = "chat",
        priority: Priority = Priority.NORMAL,
        sla_deadline: Optional[float] = None,
    ) -> "ScheduledRequest":
        now = time.monotonic()
        return cls(
            _priority=priority.value,
            _tie=now,
            req_id=req_id,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            workload=workload,
            arrival_time=now,
            sla_deadline=sla_deadline,
        )

    def promote(self) -> None:
        """Escalate to HIGH priority (e.g. when SLA deadline is near)."""
        self._priority = Priority.HIGH.value


class RequestQueue:
    """
    Min-heap priority queue with SLA deadline promotion.

    Requests nearing their deadline are promoted to HIGH priority
    on the next pop() call so they jump ahead of queued peers.
    """

    _SLA_PROMOTE_LEAD_S = 1.0  # promote if deadline < 1 s away

    def __init__(self) -> None:
        self._heap: List[ScheduledRequest] = []

    def push(self, req: ScheduledRequest) -> None:
        heapq.heappush(self._heap, req)

    def pop(self) -> Optional[ScheduledRequest]:
        if not self._heap:
            return None
        self._promote_sla()
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[ScheduledRequest]:
        return self._heap[0] if self._heap else None

    @property
    def depth(self) -> int:
        return len(self._heap)

    def _promote_sla(self) -> None:
        now = time.monotonic()
        dirty = False
        for req in self._heap:
            if (
                req.sla_deadline
                and req._priority > Priority.HIGH.value
                and req.sla_deadline - now < self._SLA_PROMOTE_LEAD_S
            ):
                req.promote()
                dirty = True
        if dirty:
            heapq.heapify(self._heap)
