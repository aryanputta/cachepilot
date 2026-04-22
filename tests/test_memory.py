from cachepilot.memory import VRAMPool


class TestVRAMPool:
    def make_pool(self, total_gb=1.0, pinned_gb=0.25):
        return VRAMPool(total_gb=total_gb, pinned_gb=pinned_gb)

    def test_stats_initial(self):
        pool = self.make_pool()
        stats = pool.stats()
        assert stats.free_blocks + stats.pinned_blocks == stats.total_blocks
        assert stats.occupied_blocks == 0

    def test_allocate_success(self):
        pool = self.make_pool()
        free_before = pool.stats().free_blocks
        ok = pool.allocate("sess1", 2)
        assert ok
        assert pool.stats().free_blocks == free_before - 2

    def test_allocate_too_many_fails(self):
        pool = self.make_pool()
        free = pool.stats().free_blocks
        ok = pool.allocate("sess1", free + 1)
        assert not ok

    def test_deallocate_restores_blocks(self):
        pool = self.make_pool()
        free_before = pool.stats().free_blocks
        pool.allocate("sess1", 3)
        freed = pool.deallocate("sess1")
        assert freed == 3
        assert pool.stats().free_blocks == free_before

    def test_multiple_sessions(self):
        pool = self.make_pool()
        pool.allocate("a", 2)
        pool.allocate("b", 3)
        assert pool.session_blocks("a") == 2
        assert pool.session_blocks("b") == 3
        pool.deallocate("a")
        assert pool.session_blocks("a") == 0
        assert pool.session_blocks("b") == 3

    def test_blocks_needed_minimum_one(self):
        assert VRAMPool.blocks_needed(1) >= 1

    def test_blocks_needed_scales_with_seq_len(self):
        b1 = VRAMPool.blocks_needed(512)
        b2 = VRAMPool.blocks_needed(4096)
        assert b2 > b1

    def test_fp8_needs_no_more_blocks_than_fp16(self):
        fp16 = VRAMPool.blocks_needed_for_tier(4096, "fp16")
        fp8 = VRAMPool.blocks_needed_for_tier(4096, "fp8")
        assert fp8 <= fp16

    def test_utilization_between_zero_and_one(self):
        pool = self.make_pool()
        pool.allocate("x", 2)
        stats = pool.stats()
        assert 0.0 <= stats.utilization <= 1.0
