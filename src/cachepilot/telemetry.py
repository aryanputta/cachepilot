from __future__ import annotations

import time
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class TokenEvent:
    req_id: str
    timestamp: float
    tokens: int
    latency_ms: float


@dataclass
class Snapshot:
    timestamp: float
    vram_util: float
    queue_depth: int
    active_sessions: int
    tokens_per_sec: float
    p50_latency_ms: float
    p95_latency_ms: float
    eviction_rate_per_s: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "cachepilot_vram_utilization_ratio": self.vram_util,
            "cachepilot_queue_depth": float(self.queue_depth),
            "cachepilot_active_sessions": float(self.active_sessions),
            "cachepilot_tokens_per_second": self.tokens_per_sec,
            "cachepilot_latency_p50_ms": self.p50_latency_ms,
            "cachepilot_latency_p95_ms": self.p95_latency_ms,
            "cachepilot_eviction_rate_per_second": self.eviction_rate_per_s,
        }


class TelemetryCollector:
    """
    Rolling-window metrics over the last WINDOW_S seconds.

    Tracks token throughput, latency distribution, VRAM utilisation,
    and eviction rate. Call snapshot() periodically to emit a Snapshot.
    """

    WINDOW_S: float = 60.0

    def __init__(self) -> None:
        self._events: deque[TokenEvent] = deque()
        self._eviction_ts: deque[float] = deque()
        self._snapshots: list[Snapshot] = []
        self._start = time.monotonic()

    def record_tokens(self, req_id: str, tokens: int, latency_ms: float) -> None:
        self._events.append(TokenEvent(req_id, time.monotonic(), tokens, latency_ms))
        self._trim()

    def record_eviction(self) -> None:
        now = time.monotonic()
        self._eviction_ts.append(now)
        cutoff = now - self.WINDOW_S
        while self._eviction_ts and self._eviction_ts[0] < cutoff:
            self._eviction_ts.popleft()

    def snapshot(
        self,
        vram_util: float,
        queue_depth: int,
        active_sessions: int,
    ) -> Snapshot:
        self._trim()
        now = time.monotonic()
        window = [e for e in self._events if e.timestamp > now - self.WINDOW_S]

        latencies = [e.latency_ms for e in window] or [0.0]
        total_tokens = sum(e.tokens for e in window)
        elapsed = max(
            now - window[0].timestamp if window else 1.0,
            1e-3,
        )
        tps = total_tokens / elapsed if window else 0.0

        s = Snapshot(
            timestamp=now,
            vram_util=vram_util,
            queue_depth=queue_depth,
            active_sessions=active_sessions,
            tokens_per_sec=tps,
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            eviction_rate_per_s=len(self._eviction_ts) / self.WINDOW_S,
        )
        self._snapshots.append(s)
        return s

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    def latest_snapshot(self) -> Snapshot | None:
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def latest_eviction_rate(self) -> float:
        snapshot = self.latest_snapshot()
        if snapshot is not None:
            return snapshot.eviction_rate_per_s
        now = time.monotonic()
        cutoff = now - self.WINDOW_S
        recent = [ts for ts in self._eviction_ts if ts >= cutoff]
        return len(recent) / self.WINDOW_S

    def time_since_last_eviction(self) -> float:
        if not self._eviction_ts:
            return self.WINDOW_S
        return max(time.monotonic() - self._eviction_ts[-1], 0.0)

    def summary(self) -> dict:
        if not self._snapshots:
            return {}
        tpss = [s.tokens_per_sec for s in self._snapshots]
        lats = [s.p95_latency_ms for s in self._snapshots]
        utils = [s.vram_util for s in self._snapshots]
        return {
            "mean_throughput_tok_s": float(np.mean(tpss)),
            "mean_p95_latency_ms": float(np.mean(lats)),
            "mean_vram_util": float(np.mean(utils)),
            "duration_s": time.monotonic() - self._start,
        }

    def to_prometheus(
        self,
        extra_metrics: Mapping[str, float] | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> str:
        snapshot = self.latest_snapshot()
        metrics = dict(snapshot.as_metrics() if snapshot is not None else {})
        if extra_metrics:
            metrics.update({k: float(v) for k, v in extra_metrics.items()})

        label_text = ""
        if labels:
            rendered = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
            label_text = f"{{{rendered}}}"

        lines = []
        for name, value in sorted(metrics.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{label_text} {value:.10g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _trim(self) -> None:
        cutoff = time.monotonic() - self.WINDOW_S * 2
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()
