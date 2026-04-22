from cachepilot.placement import DeviceState, NVLinkAwarePlacer, NVLinkTopology, PlacementRequest


def test_prefers_local_gpu_when_capacity_exists():
    topology = NVLinkTopology(n_gpus=2, links_gbps={(0, 1): 900.0})
    placer = NVLinkAwarePlacer(topology)
    devices = {
        0: DeviceState(gpu_id=0, total_blocks=100, free_blocks=40, active_sessions=2),
        1: DeviceState(gpu_id=1, total_blocks=100, free_blocks=90, active_sessions=0),
    }
    decision = placer.place(
        PlacementRequest(session_id="s1", blocks_needed=8, preferred_gpu=0),
        devices,
    )
    assert decision.gpu_id == 0
    assert not decision.remote


def test_prefers_nvlink_neighbor_over_pcie_peer():
    topology = NVLinkTopology(
        n_gpus=3,
        links_gbps={(0, 1): 900.0, (1, 2): 900.0},
        pcie_fallback_gbps=32.0,
    )
    placer = NVLinkAwarePlacer(topology)
    devices = {
        0: DeviceState(gpu_id=0, total_blocks=100, free_blocks=4, active_sessions=4),
        1: DeviceState(gpu_id=1, total_blocks=100, free_blocks=60, active_sessions=1),
        2: DeviceState(gpu_id=2, total_blocks=100, free_blocks=60, active_sessions=1),
    }
    decision = placer.place(
        PlacementRequest(session_id="s2", blocks_needed=16, preferred_gpu=0),
        devices,
    )
    assert decision.gpu_id == 1
    assert decision.remote
