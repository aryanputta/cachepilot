from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .eviction import (
    EvictionPolicy,
    PERCEviction,
    SessionCacheInfo,
    select_eviction_set,
)
from .memory import VRAMPool


@dataclass
class EvictionEvent:
    session_id: str
    seq_len: int
    blocks_freed: int
    expected_recompute_cost_s: float
    swapped_to_cpu: bool
    timestamp: float = field(default_factory=time.monotonic)


class KVCacheManager:
    """
    Manages GPU KV cache lifecycle across sessions.

    Responsibilities:
    - Allocate VRAM blocks when a session starts (prefill).
    - Grow allocations as the session generates tokens (decode).
    - Invoke the eviction policy when memory is tight.
    - Optionally track evicted sessions in a CPU shadow store so
      cost can be measured accurately in simulation.
    """

    def __init__(
        self,
        pool: VRAMPool,
        policy: Optional[EvictionPolicy] = None,
        cpu_offload: bool = True,
        c_recompute: float = 0.002,
        n_heads: int = 32,
        head_dim: int = 128,
        n_layers: int = 32,
    ):
        self._pool = pool
        self._policy = policy or PERCEviction(c_recompute=c_recompute)
        self._cpu_offload = cpu_offload
        self._c_recompute = c_recompute
        self._head_kwargs = dict(n_heads=n_heads, head_dim=head_dim, n_layers=n_layers)

        self._sessions: Dict[str, SessionCacheInfo] = {}
        self._cpu_cache: Dict[str, int] = {}  # session_id -> seq_len
        self._eviction_log: List[EvictionEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_session(self, session_id: str, prompt_len: int) -> bool:
        """
        Allocate VRAM for a new session.  Evicts others if needed.
        Returns True if the session was admitted.
        """
        needed = VRAMPool.blocks_needed(prompt_len, **self._head_kwargs)

        if not self._pool.allocate(session_id, needed):
            if not self._evict_for(needed):
                return False
            if not self._pool.allocate(session_id, needed):
                return False

        now = time.monotonic()
        self._sessions[session_id] = SessionCacheInfo(
            session_id=session_id,
            seq_len=prompt_len,
            n_blocks=needed,
            created_at=now,
            last_active=now,
        )
        return True

    def extend_session(self, session_id: str, new_tokens: int) -> bool:
        """
        Grow a session's KV cache as it generates tokens.
        Returns False if growth fails even after attempting eviction.
        """
        if session_id not in self._sessions:
            return False

        info = self._sessions[session_id]
        new_len = info.seq_len + new_tokens
        old_blocks = VRAMPool.blocks_needed(info.seq_len, **self._head_kwargs)
        new_blocks = VRAMPool.blocks_needed(new_len, **self._head_kwargs)
        extra = new_blocks - old_blocks

        if extra > 0:
            if not self._pool.allocate(session_id, extra):
                if not self._evict_for(extra):
                    return False
                if not self._pool.allocate(session_id, extra):
                    return False
            info.n_blocks = new_blocks

        info.seq_len = new_len
        info.record_token()
        return True

    def release_session(self, session_id: str) -> None:
        self._pool.deallocate(session_id)
        self._sessions.pop(session_id, None)
        self._cpu_cache.pop(session_id, None)

    def restore_session(self, session_id: str) -> bool:
        """Bring a CPU-offloaded session back to VRAM."""
        if session_id not in self._cpu_cache:
            return False
        seq_len = self._cpu_cache[session_id]
        needed = VRAMPool.blocks_needed(seq_len, **self._head_kwargs)

        if not self._pool.allocate(session_id, needed):
            if not self._evict_for(needed):
                return False
            if not self._pool.allocate(session_id, needed):
                return False

        now = time.monotonic()
        del self._cpu_cache[session_id]
        self._sessions[session_id] = SessionCacheInfo(
            session_id=session_id,
            seq_len=seq_len,
            n_blocks=needed,
            created_at=now,
            last_active=now,
        )
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def offloaded_sessions(self) -> int:
        return len(self._cpu_cache)

    @property
    def eviction_log(self) -> List[EvictionEvent]:
        return list(self._eviction_log)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_for(self, blocks_needed: int) -> bool:
        if not self._sessions:
            return False

        to_evict = select_eviction_set(self._sessions, self._policy, blocks_needed)
        if not to_evict:
            return False

        freed = 0
        for sid in to_evict:
            info = self._sessions[sid]
            self._eviction_log.append(
                EvictionEvent(
                    session_id=sid,
                    seq_len=info.seq_len,
                    blocks_freed=info.n_blocks,
                    expected_recompute_cost_s=(
                        info.seq_len * self._c_recompute * info.p_resume(5.0)
                    ),
                    swapped_to_cpu=self._cpu_offload,
                )
            )
            if self._cpu_offload:
                self._cpu_cache[sid] = info.seq_len
            self._pool.deallocate(sid)
            freed += info.n_blocks
            del self._sessions[sid]

        return freed >= blocks_needed
