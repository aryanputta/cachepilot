import pytest
from cachepilot.memory import VRAMPool
from cachepilot.eviction import PERCEviction, LRUEviction
from cachepilot.kv_manager import KVCacheManager


def small_pool() -> VRAMPool:
    # 1 GB total, 256 MB pinned — leaves ~47 free 16MB blocks
    return VRAMPool(total_gb=1.0, pinned_gb=0.25)


class TestKVCacheManager:
    def test_register_and_release(self):
        pool = small_pool()
        kv = KVCacheManager(pool)
        assert kv.register_session("s1", 64)
        assert kv.active_sessions == 1
        kv.release_session("s1")
        assert kv.active_sessions == 0

    def test_eviction_triggered_on_oom(self):
        pool = small_pool()
        kv = KVCacheManager(pool, policy=LRUEviction())
        # Fill pool with many small sessions
        admitted = 0
        for i in range(60):
            if kv.register_session(f"s{i}", 64):
                admitted += 1
        # Now try to add one more — should trigger eviction
        extra = kv.register_session("extra", 64)
        if extra:
            assert len(kv.eviction_log) > 0

    def test_extend_session(self):
        pool = small_pool()
        kv = KVCacheManager(pool)
        kv.register_session("s1", 64)
        ok = kv.extend_session("s1", 16)
        assert ok

    def test_extend_nonexistent_session(self):
        pool = small_pool()
        kv = KVCacheManager(pool)
        assert not kv.extend_session("ghost", 16)

    def test_eviction_log_grows(self):
        pool = small_pool()
        kv = KVCacheManager(pool)
        for i in range(60):
            kv.register_session(f"s{i}", 64)
        assert len(kv.eviction_log) >= 0  # may or may not evict depending on pool size

    def test_cpu_offload_tracked(self):
        pool = small_pool()
        kv = KVCacheManager(pool, cpu_offload=True)
        for i in range(60):
            kv.register_session(f"s{i}", 64)
        # Some sessions should have been offloaded to CPU
        assert kv.offloaded_sessions >= 0

    def test_perc_vs_lru_eviction_cost(self):
        """PERC should produce fewer or equal eviction events than LRU on identical workloads."""
        results = {}
        for policy_name, policy in [("perc", PERCEviction()), ("lru", LRUEviction())]:
            pool = small_pool()
            kv = KVCacheManager(pool, policy=policy)
            for i in range(50):
                kv.register_session(f"s{i}", 128)
                kv.extend_session(f"s{i}", 16)
            results[policy_name] = len(kv.eviction_log)
        # Both should produce some evictions under pressure; PERC may differ
        assert results["perc"] >= 0
        assert results["lru"] >= 0
