"""
perc_evictor.py — PERC as a drop-in replacement for vLLM's block evictor.

vLLM's block manager (vllm/core/evictor_v2.py) defines an Evictor ABC with
a single `evict()` method.  This module implements that interface using PERC.

Installation:
    1. Copy this file into your vllm fork at:
       vllm/core/perc_evictor.py
    2. In vllm/core/block_manager_v2.py, replace:
       from vllm.core.evictor_v2 import LRUEvictor
       with:
       from vllm.core.perc_evictor import PERCEvictor as LRUEvictor
    3. Done.  No other changes needed.

Interface contract (from vLLM source):
    class Evictor(ABC):
        @abstractmethod
        def evict(self) -> Tuple[int, int]:  # (block_id, num_hashed_tokens)
            ...

        @property
        @abstractmethod
        def num_blocks(self) -> int: ...

    The evictor receives free_block() calls from the block manager and must
    track which blocks are evictable.  evict() returns the block_id and its
    associated token count.

PERC extension:
    vLLM's block manager passes us Block objects.  Each Block has:
        block.block_id     : int
        block.num_hashed_tokens : int  (sequence length associated with block)

    We extend this with:
        last_active[block_id] : float  (monotonic timestamp of last token)
        intervals[block_id]   : deque  (inter-token intervals for lambda_hat)

    These are maintained by recording a call to record_token(block_id) from
    the vLLM scheduler's step() loop (one line of instrumentation needed).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Standalone implementation that does NOT import vLLM so this file can be
# tested and benchmarked independently.
# ---------------------------------------------------------------------------


class _BlockRecord:
    __slots__ = ("block_id", "content_hash", "num_hashed_tokens", "last_active", "intervals")

    def __init__(self, block_id: int, content_hash: int, num_hashed_tokens: int):
        self.block_id = block_id
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_active: float = time.monotonic()
        self.intervals: Deque[float] = deque(maxlen=20)

    @property
    def lambda_hat(self) -> float:
        if not self.intervals:
            return 0.05
        return 1.0 / max(sum(self.intervals) / len(self.intervals), 0.01)

    def p_resume(self, delta: float) -> float:
        return 1.0 - math.exp(-self.lambda_hat * delta)

    def record_token(self) -> None:
        now = time.monotonic()
        self.intervals.append(now - self.last_active)
        self.last_active = now


class PERCEvictor:
    """
    vLLM-compatible block evictor using the PERC algorithm.

    Parameters
    ----------
    c_recompute : float
        Seconds to recompute one KV token if the evicted block is needed.
        Tune to your GPU: A100 ≈ 0.0015, H100 ≈ 0.0008, RTX 4090 ≈ 0.003.
    delta_serve : float
        Lookahead horizon (seconds).  Expected time between the eviction
        decision and when the freed block would next be needed.
        5–10 seconds is appropriate for most chat/API serving workloads.
    """

    def __init__(self, c_recompute: float = 0.002, delta_serve: float = 5.0):
        self.c_recompute = c_recompute
        self.delta_serve = delta_serve
        self._blocks: Dict[int, _BlockRecord] = {}

    # ------------------------------------------------------------------
    # vLLM Evictor interface
    # ------------------------------------------------------------------

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def add(
        self,
        block_id: int,
        content_hash: int,
        num_hashed_tokens: int,
        last_accessed: Optional[float] = None,
    ) -> None:
        """Called by vLLM block manager when a block becomes evictable."""
        record = _BlockRecord(block_id, content_hash, num_hashed_tokens)
        if last_accessed is not None:
            record.last_active = last_accessed
        self._blocks[block_id] = record

    def remove(self, block_id: int) -> _BlockRecord:
        """Called by vLLM when a block is re-allocated (taken off evict list)."""
        return self._blocks.pop(block_id)

    def update(self, block_id: int, last_accessed: float) -> None:
        if block_id not in self._blocks:
            raise ValueError("Attempting to update block that's not in the evictor")
        self._blocks[block_id].last_active = last_accessed

    def evict(self) -> Tuple[int, int]:
        """
        Return (block_id, content_hash) for the cheapest block to evict.

        PERC score per block:
            score = seq_len * c_recompute * P(resume within delta) / 1 block
        Since vLLM allocates one block per sequence (in the prefix-sharing sense),
        the denominator is 1 and score = eviction_cost directly.
        """
        if not self._blocks:
            raise ValueError("No evictable blocks available.")

        best_id = min(
            self._blocks,
            key=lambda bid: self._perc_score(self._blocks[bid]),
        )
        record = self._blocks.pop(best_id)
        return record.block_id, record.content_hash

    # ------------------------------------------------------------------
    # Instrumentation — call from vLLM scheduler step()
    # ------------------------------------------------------------------

    def record_token(self, block_id: int) -> None:
        """Record a token generation event for a block to update lambda_hat."""
        if block_id in self._blocks:
            self._blocks[block_id].record_token()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _perc_score(self, rec: _BlockRecord) -> float:
        cost = rec.num_hashed_tokens * self.c_recompute * rec.p_resume(self.delta_serve)
        return cost

    # ------------------------------------------------------------------
    # Comparison: LRU score for the same block (for benchmarking)
    # ------------------------------------------------------------------

    def lru_would_evict(self) -> Optional[int]:
        """Which block LRU would choose (for comparison logging)."""
        if not self._blocks:
            return None
        return min(self._blocks, key=lambda bid: self._blocks[bid].last_active)

    def eviction_cost_delta(self) -> float:
        """
        Expected cost saved by PERC vs LRU for the current eviction decision.

        A positive value means PERC chose a cheaper block to evict.
        A value of 0 means both algorithms agree on the target.
        """
        if len(self._blocks) < 2:
            return 0.0
        perc_choice = min(self._blocks, key=lambda b: self._perc_score(self._blocks[b]))
        lru_choice = self.lru_would_evict()
        if perc_choice == lru_choice:
            return 0.0
        return self._perc_score(self._blocks[lru_choice]) - self._perc_score(self._blocks[perc_choice])


def install_into_vllm() -> None:
    """
    Monkeypatch vLLM to use PERC for free-block eviction.

    This is intended for smoke/integration tests and local experiments where
    patching the import graph is easier than maintaining a custom vLLM fork.
    """
    import importlib

    evictor_mod = importlib.import_module("vllm.core.evictor")

    def _make_evictor(*args, **kwargs):
        return PERCEvictor()

    evictor_mod.PERCEvictor = PERCEvictor
    evictor_mod.LRUEvictor = PERCEvictor
    if hasattr(evictor_mod, "make_evictor"):
        evictor_mod.make_evictor = _make_evictor

    for module_name in ("vllm.core.block_manager", "vllm.core.block_manager_v2"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "make_evictor"):
            module.make_evictor = _make_evictor
        if hasattr(module, "LRUEvictor"):
            module.LRUEvictor = PERCEvictor


# ---------------------------------------------------------------------------
# Benchmark: PERC vs LRU on a synthetic block set
# ---------------------------------------------------------------------------

def benchmark_perc_vs_lru(n_blocks: int = 1000, n_evictions: int = 500, seed: int = 42) -> Dict:
    """
    Compare PERC vs LRU expected recompute cost on a synthetic workload.
    Returns cost statistics for both policies.
    """
    import random
    rng = random.Random(seed)

    perc_evictor = PERCEvictor(c_recompute=0.002, delta_serve=5.0)
    lru_costs: List[float] = []
    perc_costs: List[float] = []

    # Populate with heterogeneous blocks
    for i in range(n_blocks):
        seq_len = rng.randint(64, 8192)
        rec = _BlockRecord(i, i, seq_len)
        # Assign random lambda (activity rate)
        rec.intervals.extend([1.0 / max(rng.gauss(0.5, 0.8), 0.01)] * rng.randint(1, 20))
        rec.last_active = time.monotonic() - rng.uniform(0, 300)
        perc_evictor._blocks[i] = rec

    # Clone for LRU comparison
    import copy
    lru_blocks = copy.deepcopy(perc_evictor._blocks)

    for _ in range(min(n_evictions, n_blocks)):
        # PERC choice
        if perc_evictor._blocks:
            best = min(perc_evictor._blocks, key=lambda b: perc_evictor._perc_score(perc_evictor._blocks[b]))
            perc_costs.append(perc_evictor._perc_score(perc_evictor._blocks[best]))
            perc_evictor._blocks.pop(best)

        # LRU choice (on the same pool)
        if lru_blocks:
            oldest = min(lru_blocks, key=lambda b: lru_blocks[b].last_active)
            # cost that PERC would assign to LRU's choice
            rec = lru_blocks.pop(oldest)
            lru_costs.append(rec.num_hashed_tokens * 0.002 * rec.p_resume(5.0))

    return {
        "perc_total_cost": sum(perc_costs),
        "lru_total_cost": sum(lru_costs),
        "cost_reduction_pct": (1 - sum(perc_costs) / max(sum(lru_costs), 1e-9)) * 100,
        "n_evictions": len(perc_costs),
    }


if __name__ == "__main__":
    result = benchmark_perc_vs_lru(n_blocks=1000, n_evictions=500)
    print(f"PERC total cost:  {result['perc_total_cost']:.2f}s")
    print(f"LRU  total cost:  {result['lru_total_cost']:.2f}s")
    print(f"Cost reduction:   {result['cost_reduction_pct']:.1f}%")
