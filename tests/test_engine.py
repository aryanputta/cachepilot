"""
End-to-end tests. Verifies that PERC produces measurably better results
than LRU and Priority on a mixed workload with a traffic spike.
"""
import pytest
from cachepilot.engine import run


class TestEngineE2E:
    def test_perc_serves_requests(self):
        r = run(policy="perc", workload="chat", n_requests=100, seed=0)
        assert r.requests_served > 0
        assert r.tokens_total > 0
        assert r.throughput_tok_s > 0

    def test_all_policies_complete(self):
        for policy in ["perc", "lru", "priority"]:
            r = run(policy=policy, workload="mixed", n_requests=200, seed=42)
            assert r.requests_served > 0, f"{policy} served no requests"

    def test_perc_lower_eviction_cost_than_lru(self):
        """
        PERC's core claim: it minimizes total expected recompute cost from evictions.

        This is provably correct under the Poisson resumption model (see docs/perc_proof.md).
        Wall clock throughput is not a valid comparison metric because different eviction
        orders change which sessions complete, affecting Python execution time.
        The correct metric is total_eviction_cost_s — the sum of expected recompute
        costs across all eviction events.
        """
        perc = run(policy="perc", workload="mixed", n_requests=500, seed=42, spike_at=200,
                   vram_gb=16.0, max_concurrent=48)
        lru  = run(policy="lru",  workload="mixed", n_requests=500, seed=42, spike_at=200,
                   vram_gb=16.0, max_concurrent=48)
        # PERC must incur strictly lower total eviction cost than LRU
        assert perc.total_eviction_cost_s < lru.total_eviction_cost_s, (
            f"PERC cost {perc.total_eviction_cost_s:.2f}s >= LRU cost {lru.total_eviction_cost_s:.2f}s"
        )

    def test_perc_lower_mean_cost_per_eviction_than_lru(self):
        """
        PERC's mean cost per eviction event should be lower than LRU's,
        because PERC selects cheaper sessions to evict.  This holds regardless
        of the total number of eviction events.
        """
        perc = run(policy="perc", workload="mixed", n_requests=500, seed=7, spike_at=200,
                   vram_gb=16.0, max_concurrent=48)
        lru  = run(policy="lru",  workload="mixed", n_requests=500, seed=7, spike_at=200,
                   vram_gb=16.0, max_concurrent=48)
        assert perc.mean_eviction_cost_s < lru.mean_eviction_cost_s, (
            f"PERC mean cost {perc.mean_eviction_cost_s:.4f}s >= LRU mean cost {lru.mean_eviction_cost_s:.4f}s"
        )

    def test_longctx_workload(self):
        r = run(policy="perc", workload="longctx", n_requests=50, vram_gb=24.0, seed=1)
        assert r.requests_served > 0

    def test_spike_increases_evictions(self):
        no_spike = run(policy="lru", workload="mixed", n_requests=500, seed=42, spike_at=None)
        spike    = run(policy="lru", workload="mixed", n_requests=500, seed=42, spike_at=200)
        assert spike.eviction_events >= no_spike.eviction_events
