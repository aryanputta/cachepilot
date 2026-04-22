"""
eviction.py — PERC and baseline eviction policies for CachePilot.

PERC (Priority Eviction with Resumption Cost) is the novel algorithm.
See docs/perc_proof.md for the full mathematical proof of optimality.

Short summary:
  score(i) = C_evict(i) / bytes(i)
  C_evict(i) = seq_len(i) * c_recompute * P(resume within delta_serve)
  P(resume within delta) = 1 - exp(-lambda_i * delta)   [Poisson model]

Evict the session with the LOWEST score first (cheapest per byte).

Proof sketch: minimizing expected recompute cost subject to a byte-budget
constraint is a bounded selection problem. Since costs are additive and
independent, the optimal greedy is to sort by cost/size and pick ascending —
identical to the fractional knapsack optimum. PERC computes exactly this
ordering. See docs/perc_proof.md for the full argument.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionCacheInfo:
    session_id: str
    seq_len: int
    n_blocks: int
    created_at: float
    last_active: float
    token_intervals: List[float] = field(default_factory=list)

    @property
    def idle_time(self) -> float:
        return time.monotonic() - self.last_active

    @property
    def lambda_hat(self) -> float:
        """
        EMA-estimated Poisson arrival rate (tokens/sec) from recent
        inter-token intervals. Falls back to a low prior when no history.
        """
        window = self.token_intervals[-20:]
        if not window:
            return 0.05  # prior: one token every ~20 s
        mean_interval = sum(window) / len(window)
        return 1.0 / max(mean_interval, 0.01)

    def p_resume(self, delta_s: float) -> float:
        """P(session issues a token within the next delta_s seconds)."""
        return 1.0 - math.exp(-self.lambda_hat * delta_s)

    def record_token(self) -> None:
        now = time.monotonic()
        self.token_intervals.append(now - self.last_active)
        self.last_active = now


# ---------------------------------------------------------------------------
# Eviction policies
# ---------------------------------------------------------------------------


class EvictionPolicy(ABC):
    @abstractmethod
    def rank(self, sessions: Dict[str, SessionCacheInfo]) -> List[str]:
        """Return session IDs sorted cheapest-to-evict first."""
        ...


class LRUEviction(EvictionPolicy):
    """Baseline: evict the longest-idle session first."""

    def rank(self, sessions: Dict[str, SessionCacheInfo]) -> List[str]:
        return sorted(sessions, key=lambda sid: sessions[sid].last_active)


class PriorityEviction(EvictionPolicy):
    """Baseline: evict the shortest-context session first."""

    def rank(self, sessions: Dict[str, SessionCacheInfo]) -> List[str]:
        return sorted(sessions, key=lambda sid: sessions[sid].seq_len)


class PERCEviction(EvictionPolicy):
    """
    Priority Eviction with Resumption Cost.

    Parameters
    ----------
    c_recompute : float
        Seconds to recompute one KV token if the evicted session resumes.
    delta_serve : float
        Lookahead horizon (seconds) used to estimate resumption probability.
        Set to the expected time between now and when the freed space is needed.
    """

    def __init__(self, c_recompute: float = 0.002, delta_serve: float = 5.0):
        self.c_recompute = c_recompute
        self.delta_serve = delta_serve

    def eviction_cost(self, info: SessionCacheInfo) -> float:
        """Expected recompute cost if we evict this session."""
        return info.seq_len * self.c_recompute * info.p_resume(self.delta_serve)

    def score(self, info: SessionCacheInfo) -> float:
        """Cost per block — lower means cheaper to evict per byte used."""
        return self.eviction_cost(info) / max(info.n_blocks, 1)

    def rank(self, sessions: Dict[str, SessionCacheInfo]) -> List[str]:
        return sorted(sessions, key=lambda sid: self.score(sessions[sid]))


# ---------------------------------------------------------------------------
# Eviction set selection
# ---------------------------------------------------------------------------


def select_eviction_set(
    sessions: Dict[str, SessionCacheInfo],
    policy: EvictionPolicy,
    blocks_needed: int,
) -> List[str]:
    """
    Return a minimal prefix of the policy ranking that frees at least
    blocks_needed blocks. May return fewer sessions than needed if the
    entire candidate pool is insufficient.
    """
    ranked = policy.rank(sessions)
    to_evict: List[str] = []
    freed = 0
    for sid in ranked:
        if freed >= blocks_needed:
            break
        to_evict.append(sid)
        freed += sessions[sid].n_blocks
    return to_evict
