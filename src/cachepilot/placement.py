from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class DeviceState:
    gpu_id: int
    total_blocks: int
    free_blocks: int
    active_sessions: int = 0

    @property
    def utilization(self) -> float:
        used = self.total_blocks - self.free_blocks
        return used / max(self.total_blocks, 1)


@dataclass(frozen=True)
class PlacementRequest:
    session_id: str
    blocks_needed: int
    preferred_gpu: int | None = None


@dataclass(frozen=True)
class PlacementDecision:
    session_id: str
    gpu_id: int
    score: float
    bandwidth_gbps: float
    free_blocks_after: int
    remote: bool


class NVLinkTopology:
    """
    Simple bandwidth graph used by the placement policy.

    Unlisted links default to a PCIe-like fallback bandwidth.
    """

    def __init__(
        self,
        n_gpus: int,
        links_gbps: Mapping[tuple[int, int], float] | None = None,
        pcie_fallback_gbps: float = 32.0,
    ) -> None:
        self.n_gpus = n_gpus
        self.pcie_fallback_gbps = pcie_fallback_gbps
        self._links: dict[tuple[int, int], float] = {}
        for (src, dst), bw in (links_gbps or {}).items():
            self._links[(src, dst)] = bw
            self._links[(dst, src)] = bw

    def bandwidth_gbps(self, src: int, dst: int) -> float:
        if src == dst:
            return float("inf")
        return self._links.get((src, dst), self.pcie_fallback_gbps)


class NVLinkAwarePlacer:
    """
    Placement policy that favors locality first, then fast interconnects.
    """

    def __init__(self, topology: NVLinkTopology) -> None:
        self._topology = topology

    def place(
        self,
        request: PlacementRequest,
        devices: Mapping[int, DeviceState],
    ) -> PlacementDecision:
        if request.preferred_gpu is not None:
            preferred = devices[request.preferred_gpu]
            if preferred.free_blocks >= request.blocks_needed:
                return PlacementDecision(
                    session_id=request.session_id,
                    gpu_id=preferred.gpu_id,
                    score=1e9,
                    bandwidth_gbps=float("inf"),
                    free_blocks_after=preferred.free_blocks - request.blocks_needed,
                    remote=False,
                )

        candidates: list[PlacementDecision] = []
        for gpu_id, device in devices.items():
            if device.free_blocks < request.blocks_needed:
                continue
            bw = self._topology.bandwidth_gbps(request.preferred_gpu or gpu_id, gpu_id)
            local_capacity = device.free_blocks / max(device.total_blocks, 1)
            load_penalty = device.active_sessions * 0.1
            bw_score = 1000.0 if bw == float("inf") else bw
            score = bw_score + local_capacity * 100.0 - load_penalty
            candidates.append(
                PlacementDecision(
                    session_id=request.session_id,
                    gpu_id=gpu_id,
                    score=score,
                    bandwidth_gbps=bw,
                    free_blocks_after=device.free_blocks - request.blocks_needed,
                    remote=request.preferred_gpu is not None and gpu_id != request.preferred_gpu,
                )
            )

        if not candidates:
            raise ValueError(
                f"No device has room for session '{request.session_id}' "
                f"({request.blocks_needed} blocks)."
            )

        return max(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.free_blocks_after,
                -candidate.gpu_id,
            ),
        )
