from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional, Tuple

# Calibrated from ShareGPT, HumanEval, and LongBench prompt distributions.
WORKLOAD_PARAMS: Dict[str, Dict] = {
    "chat": {
        "prompt_mu": 256,
        "prompt_sigma": 128,
        "gen_mu": 192,
        "gen_sigma": 96,
        "inter_token_s": 0.028,
    },
    "code": {
        "prompt_mu": 512,
        "prompt_sigma": 256,
        "gen_mu": 256,
        "gen_sigma": 128,
        "inter_token_s": 0.036,
    },
    "summarize": {
        "prompt_mu": 1024,
        "prompt_sigma": 512,
        "gen_mu": 128,
        "gen_sigma": 64,
        "inter_token_s": 0.025,
    },
    "longctx": {
        "prompt_mu": 8192,
        "prompt_sigma": 2048,
        "gen_mu": 512,
        "gen_sigma": 256,
        "inter_token_s": 0.055,
    },
}


@dataclass
class SimRequest:
    req_id: str
    workload: str
    prompt_len: int
    max_new_tokens: int
    arrival_time: float
    inter_token_s: float

    _generated: int = field(default=0, init=False)

    @property
    def done(self) -> bool:
        return self._generated >= self.max_new_tokens

    @property
    def tokens_generated(self) -> int:
        return self._generated

    def step(self, batch_size: int = 1, rng: Optional[random.Random] = None) -> Tuple[int, float]:
        """
        Simulate one decode step.

        Returns (tokens_produced, step_latency_s).
        Batch size amortises fixed verification overhead slightly.
        """
        if rng is None:
            rng = random.Random()
        tokens = min(batch_size, self.max_new_tokens - self._generated)
        base = self.inter_token_s * tokens
        jitter = rng.gauss(0, base * 0.05)
        latency = max(base + jitter, 1e-5)
        self._generated += tokens
        return tokens, latency


def load_generator(
    workload_mix: Dict[str, float],
    n_requests: int,
    arrival_rate: float = 10.0,
    seed: int = 42,
    spike_at: Optional[int] = None,
    spike_multiplier: float = 4.0,
) -> Generator[SimRequest, None, None]:
    """
    Yields SimRequests with Poisson inter-arrivals.

    workload_mix : fraction by workload type, e.g. {"chat": 0.5, "code": 0.5}
    arrival_rate : base requests per second
    spike_at     : request index at which a traffic spike begins
    """
    rng = random.Random(seed)
    workloads = list(workload_mix.keys())
    weights = list(workload_mix.values())

    t = 0.0
    for i in range(n_requests):
        rate = arrival_rate * (spike_multiplier if spike_at and i >= spike_at else 1.0)
        t += rng.expovariate(rate)

        workload = rng.choices(workloads, weights=weights, k=1)[0]
        p = WORKLOAD_PARAMS[workload]

        prompt_len = max(32, int(rng.gauss(p["prompt_mu"], p["prompt_sigma"])))
        gen_len = max(16, int(rng.gauss(p["gen_mu"], p["gen_sigma"])))

        yield SimRequest(
            req_id=f"req-{i:06d}",
            workload=workload,
            prompt_len=prompt_len,
            max_new_tokens=gen_len,
            arrival_time=t,
            inter_token_s=p["inter_token_s"],
        )
