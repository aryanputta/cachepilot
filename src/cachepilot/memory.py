from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

BLOCK_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB GPU memory pages


class BlockState(Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    PINNED = "pinned"  # model weights, never evicted


@dataclass
class MemoryBlock:
    block_id: int
    state: BlockState = BlockState.FREE
    session_id: Optional[str] = None


@dataclass
class VRAMStats:
    total_blocks: int
    free_blocks: int
    occupied_blocks: int
    pinned_blocks: int

    @property
    def used_bytes(self) -> int:
        return (self.occupied_blocks + self.pinned_blocks) * BLOCK_SIZE_BYTES

    @property
    def free_bytes(self) -> int:
        return self.free_blocks * BLOCK_SIZE_BYTES

    @property
    def utilization(self) -> float:
        denom = self.total_blocks - self.pinned_blocks
        return 1.0 - self.free_blocks / max(denom, 1)


class VRAMPool:
    """
    Simulates GPU VRAM as paged 16 MB blocks.

    Pinned blocks represent model weights and are never evicted.
    All public methods are thread-safe via a reentrant lock.
    """

    def __init__(self, total_gb: float = 24.0, pinned_gb: float = 8.0):
        total_bytes = int(total_gb * 1024**3)
        pinned_bytes = int(pinned_gb * 1024**3)

        self._total_blocks = total_bytes // BLOCK_SIZE_BYTES
        self._pinned_count = pinned_bytes // BLOCK_SIZE_BYTES

        self._blocks: List[MemoryBlock] = [
            MemoryBlock(
                i,
                BlockState.PINNED if i < self._pinned_count else BlockState.FREE,
            )
            for i in range(self._total_blocks)
        ]

        self._session_blocks: Dict[str, Set[int]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def stats(self) -> VRAMStats:
        with self._lock:
            counts: Dict[BlockState, int] = {s: 0 for s in BlockState}
            for b in self._blocks:
                counts[b.state] += 1
            return VRAMStats(
                total_blocks=self._total_blocks,
                free_blocks=counts[BlockState.FREE],
                occupied_blocks=counts[BlockState.OCCUPIED],
                pinned_blocks=counts[BlockState.PINNED],
            )

    def session_blocks(self, session_id: str) -> int:
        with self._lock:
            return len(self._session_blocks.get(session_id, set()))

    def bytes_for_session(self, session_id: str) -> int:
        return self.session_blocks(session_id) * BLOCK_SIZE_BYTES

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, session_id: str, n_blocks: int) -> bool:
        """Allocate n_blocks for session_id. Returns True on success."""
        with self._lock:
            free = [b for b in self._blocks if b.state == BlockState.FREE]
            if len(free) < n_blocks:
                return False
            for b in free[:n_blocks]:
                b.state = BlockState.OCCUPIED
                b.session_id = session_id
            self._session_blocks.setdefault(session_id, set()).update(
                b.block_id for b in free[:n_blocks]
            )
            return True

    def deallocate(self, session_id: str) -> int:
        """Free all blocks for session_id. Returns number of blocks freed."""
        with self._lock:
            block_ids = self._session_blocks.pop(session_id, set())
            for bid in block_ids:
                self._blocks[bid].state = BlockState.FREE
                self._blocks[bid].session_id = None
            return len(block_ids)

    # ------------------------------------------------------------------
    # Sizing helper
    # ------------------------------------------------------------------

    @staticmethod
    def blocks_needed(
        seq_len: int,
        n_heads: int = 32,
        head_dim: int = 128,
        n_layers: int = 32,
        dtype_bytes: int = 2,
    ) -> int:
        """
        KV cache footprint: 2 * layers * heads * head_dim * seq_len * dtype_bytes.
        (Factor-of-2 because we store both K and V tensors.)
        """
        kv_bytes = 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes
        return max(1, math.ceil(kv_bytes / BLOCK_SIZE_BYTES))
