"""
rl_policy.py — Lightweight neural policy for adaptive scheduling decisions.

Replaces hand-tuned heuristics for two decisions:
  1. Admission control: admit this request now, or defer until VRAM is freer?
  2. Lambda estimation: predict per-session activity rate from observable features.

Model: 2-layer MLP (64 hidden units).  Inference cost: ~0.1ms on CPU.
Training: behavioral cloning from PERC-oracle traces (imitation learning),
          optionally fine-tuned with policy gradient on goodput reward.

The model is intentionally tiny — it runs on the scheduler CPU thread,
not on the GPU, so it must not compete with the forward pass.

Inputs (normalized):
  - VRAM utilization [0, 1]
  - Queue depth [0, 1] (normalized to max_queue)
  - Prompt length [0, 1] (normalized to max_prompt_len)
  - Request priority [0, 1]
  - Estimated generation length [0, 1]
  - Recent eviction rate [0, 1]
  - Time since last eviction [0, 1]

Output:
  - Admission score in [0, 1]  (> 0.5 = admit now)
  - Lambda estimate in [0, inf) (token arrival rate for this session type)
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight MLP — pure NumPy, no PyTorch dependency
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class TinyMLP:
    """
    2-layer MLP: [input_dim → hidden → output_dim].
    Weights initialized with He initialization for ReLU layers.
    """

    def __init__(self, input_dim: int = 7, hidden: int = 64, output_dim: int = 2, seed: int = 0):
        rng = np.random.RandomState(seed)
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden)
        self.W1 = rng.randn(input_dim, hidden) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = _relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward_with_hidden(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = _relu(x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Returns (admission_score, lambda_estimate).
        admission_score in [0, 1]; lambda_estimate in [0, inf).
        """
        out = self.forward(features)
        admission = float(_sigmoid(out[0]))
        lambda_est = float(np.exp(np.clip(out[1], -5, 5)))  # softplus-like
        return admission, lambda_est

    def clone(self) -> TinyMLP:
        clone = TinyMLP(
            input_dim=self.W1.shape[0],
            hidden=self.W1.shape[1],
            output_dim=self.W2.shape[1],
        )
        clone.load_state_dict(self.state_dict())
        return clone

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.W1 = np.array(state["W1"], copy=True)
        self.b1 = np.array(state["b1"], copy=True)
        self.W2 = np.array(state["W2"], copy=True)
        self.b2 = np.array(state["b2"], copy=True)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        np.savez(target, **self.state_dict())

    @classmethod
    def load(cls, path: str | Path) -> TinyMLP:
        state = np.load(Path(path))
        model = cls(
            input_dim=state["W1"].shape[0],
            hidden=state["W1"].shape[1],
            output_dim=state["W2"].shape[1],
        )
        model.load_state_dict({key: state[key] for key in ("W1", "b1", "W2", "b2")})
        return model

    def update_weight(self, dW1, db1, dW2, db2, lr: float = 1e-3) -> None:
        """Gradient descent step (called by imitation learning trainer)."""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def policy_gradient_step(
        self,
        features: np.ndarray,
        action: bool,
        advantage: float,
        lr: float = 1e-3,
    ) -> None:
        """
        REINFORCE update for the admission Bernoulli head only.

        The lambda-estimation head is intentionally left untouched by the RL
        update so it can continue to be trained with supervised traces.
        """
        h, out = self.forward_with_hidden(features)
        prob = float(_sigmoid(out[0]))
        target = 1.0 if action else 0.0
        dlogit = (target - prob) * advantage

        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW2[:, 0] = h * dlogit
        db2[0] = dlogit

        dh = self.W2[:, 0] * dlogit
        dpre = dh * (h > 0).astype(float)
        dW1 = np.outer(features, dpre)
        db1 = dpre

        self.update_weight(dW1, db1, dW2, db2, lr=lr)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

@dataclass
class SchedulerState:
    vram_util: float        # [0, 1]
    queue_depth: int
    max_queue: int
    prompt_len: int
    max_prompt_len: int
    priority: int           # 1=HIGH, 2=NORMAL, 3=LOW
    est_gen_len: int
    max_gen_len: int
    eviction_rate: float    # evictions/second
    time_since_eviction: float  # seconds


def extract_features(state: SchedulerState) -> np.ndarray:
    return np.array([
        state.vram_util,
        state.queue_depth / max(state.max_queue, 1),
        state.prompt_len / max(state.max_prompt_len, 1),
        (state.priority - 1) / 2.0,
        state.est_gen_len / max(state.max_gen_len, 1),
        min(state.eviction_rate / 10.0, 1.0),
        min(state.time_since_eviction / 60.0, 1.0),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Imitation learning trainer
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    features: np.ndarray
    oracle_admit: float      # 1.0 = oracle said admit, 0.0 = defer
    oracle_lambda: float     # true lambda observed from this session


@dataclass
class AdmissionDecision:
    admit: bool
    confidence: float
    probability: float
    features: np.ndarray


@dataclass
class PolicyGradientSample:
    features: np.ndarray
    action: bool
    probability: float


@dataclass
class RLRunMetrics:
    reward: float
    throughput_tok_s: float
    tokens_total: int
    requests_dropped: int
    requests_deferred: int
    total_eviction_cost_s: float
    mean_vram_util: float


@dataclass
class FineTuneResult:
    episodes: int
    baseline_reward: float
    tuned_reward: float
    baseline_drop_rate: float
    tuned_drop_rate: float
    baseline_eviction_cost_s: float
    tuned_eviction_cost_s: float
    improvement_pct: float


class ILTrainer:
    """
    Imitation learning from PERC-oracle traces.

    The oracle is: admit if PERC score of cheapest eviction candidate
    is below a threshold (i.e., eviction would be cheap), otherwise defer.

    Training: mini-batch SGD with MSE loss on both outputs.
    """

    def __init__(self, model: TinyMLP, lr: float = 1e-3, batch_size: int = 32):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.buffer: list[Trace] = []
        self.losses: list[float] = []

    def add_trace(self, trace: Trace) -> None:
        self.buffer.append(trace)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        rng = np.random.RandomState()
        idxs = rng.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]

        X = np.stack([t.features for t in batch])           # [B, 7]
        y_admit = np.array([t.oracle_admit for t in batch])  # [B]
        y_lam   = np.log(np.array([max(t.oracle_lambda, 1e-4) for t in batch]))  # [B] log-space

        # Forward
        H = _relu(X @ self.model.W1 + self.model.b1)         # [B, 64]
        out = H @ self.model.W2 + self.model.b2               # [B, 2]
        pred_admit = _sigmoid(out[:, 0])
        pred_lam   = out[:, 1]

        # MSE loss
        loss_admit = np.mean((pred_admit - y_admit) ** 2)
        loss_lam   = np.mean((pred_lam - y_lam) ** 2)
        loss = loss_admit + loss_lam

        # Backprop
        B = self.batch_size
        d_out = np.zeros_like(out)
        d_out[:, 0] = 2 * (pred_admit - y_admit) * pred_admit * (1 - pred_admit) / B
        d_out[:, 1] = 2 * (pred_lam - y_lam) / B

        dW2 = H.T @ d_out
        db2 = d_out.sum(axis=0)
        dH  = d_out @ self.model.W2.T
        dH_relu = dH * (H > 0).astype(float)
        dW1 = X.T @ dH_relu
        db1 = dH_relu.sum(axis=0)

        self.model.update_weight(dW1, db1, dW2, db2, self.lr)
        self.losses.append(float(loss))
        return float(loss)


# ---------------------------------------------------------------------------
# Policy wrapper used by the scheduler
# ---------------------------------------------------------------------------

class AdmissionPolicy:
    """
    Wraps TinyMLP + heuristic fallback for the scheduler's admission decision.

    During warmup (first 100 requests) or when confidence is low, falls back
    to a simple threshold: admit if vram_util < 0.85.
    """

    WARMUP_N = 100
    ADMIT_THRESHOLD = 0.5

    def __init__(self, model: Optional[TinyMLP] = None, warmup_n: int = WARMUP_N):
        self.model = model or TinyMLP()
        self._n_decisions = 0
        self._warmup_n = warmup_n

    def decide(
        self,
        state: SchedulerState,
        sample: bool = False,
        rng: np.random.RandomState | None = None,
    ) -> AdmissionDecision:
        self._n_decisions += 1
        features = extract_features(state)

        if self._n_decisions <= self._warmup_n:
            admit = state.vram_util < 0.85
            return AdmissionDecision(
                admit=admit,
                confidence=0.5,
                probability=0.5,
                features=features,
            )

        score, _ = self.model.predict(features)
        if sample:
            sampler = rng or np.random.RandomState()
            admit = bool(sampler.rand() < score)
        else:
            admit = score > self.ADMIT_THRESHOLD

        return AdmissionDecision(
            admit=admit,
            confidence=abs(score - 0.5) * 2,
            probability=score,
            features=features,
        )

    def should_admit(self, state: SchedulerState) -> Tuple[bool, float]:
        """
        Returns (admit: bool, confidence: float).
        confidence in [0, 1] — how certain the model is.
        """
        decision = self.decide(state, sample=False)
        return decision.admit, decision.confidence

    def estimated_lambda(self, state: SchedulerState) -> float:
        """Return the model's prediction of this session's token arrival rate."""
        features = extract_features(state)
        _, lambda_est = self.model.predict(features)
        return lambda_est

    def reset(self) -> None:
        self._n_decisions = 0


class PolicyGradientTrainer:
    """
    Lightweight REINFORCE trainer for the admission controller.

    Reward is derived from whole-run serving metrics so the policy is pushed
    toward real operational outcomes: more useful tokens, fewer drops, and less
    recompute waste from evictions.
    """

    def __init__(
        self,
        policy: AdmissionPolicy,
        lr: float = 5e-4,
        baseline_momentum: float = 0.9,
    ) -> None:
        self.policy = policy
        self.lr = lr
        self.baseline_momentum = baseline_momentum
        self._reward_baseline = 0.0

    def make_sample(
        self,
        state: SchedulerState,
        rng: np.random.RandomState | None = None,
    ) -> tuple[AdmissionDecision, PolicyGradientSample]:
        decision = self.policy.decide(state, sample=True, rng=rng)
        sample = PolicyGradientSample(
            features=decision.features,
            action=decision.admit,
            probability=decision.probability,
        )
        return decision, sample

    def update_episode(
        self,
        samples: list[PolicyGradientSample],
        reward: float,
    ) -> None:
        advantage = np.clip(reward - self._reward_baseline, -25.0, 25.0)
        self._reward_baseline = (
            self.baseline_momentum * self._reward_baseline
            + (1.0 - self.baseline_momentum) * reward
        )
        for sample in samples:
            self.policy.model.policy_gradient_step(
                sample.features,
                sample.action,
                advantage,
                lr=self.lr,
            )


def reward_from_run(result) -> float:
    return (
        25.0 * result.requests_served
        + 0.01 * result.tokens_total
        - 30.0 * result.requests_dropped
        - 3.0 * result.requests_deferred
        - 4.0 * result.total_eviction_cost_s
        - 5.0 * max(result.vram_util_mean - 0.9, 0.0)
    )


def summarize_run(result) -> RLRunMetrics:
    return RLRunMetrics(
        reward=reward_from_run(result),
        throughput_tok_s=result.throughput_tok_s,
        tokens_total=result.tokens_total,
        requests_dropped=result.requests_dropped,
        requests_deferred=result.requests_deferred,
        total_eviction_cost_s=result.total_eviction_cost_s,
        mean_vram_util=result.vram_util_mean,
    )


def fine_tune_admission_policy(
    episodes: int = 12,
    workload: str = "mixed",
    n_requests: int = 400,
    arrival_rate: float = 10.0,
    vram_gb: float = 16.0,
    kv_tier: str = "fp16",
    seed: int = 42,
    max_concurrent: int = 48,
) -> FineTuneResult:
    from ..engine import run

    def _bootstrap_model(model: TinyMLP, bootstrap_seed: int) -> TinyMLP:
        trainer = ILTrainer(model, lr=5e-3, batch_size=64)
        rng = np.random.RandomState(bootstrap_seed)
        for _ in range(512):
            state = SchedulerState(
                vram_util=float(rng.uniform(0.1, 0.99)),
                queue_depth=int(rng.randint(0, 64)),
                max_queue=64,
                prompt_len=int(rng.randint(32, 8192)),
                max_prompt_len=8192,
                priority=int(rng.randint(1, 4)),
                est_gen_len=int(rng.randint(16, 2048)),
                max_gen_len=4096,
                eviction_rate=float(rng.uniform(0.0, 2.0)),
                time_since_eviction=float(rng.uniform(0.0, 60.0)),
            )
            trainer.add_trace(
                Trace(
                    features=extract_features(state),
                    oracle_admit=1.0 if state.vram_util < 0.85 else 0.0,
                    oracle_lambda=max(0.05, 1.2 - state.vram_util),
                )
            )
        for _ in range(64):
            trainer.train_step()
        return model

    baseline_model = _bootstrap_model(TinyMLP(seed=seed), seed)
    tuned_model = baseline_model.clone()

    baseline_policy = AdmissionPolicy(model=baseline_model.clone(), warmup_n=10**9)
    tuned_policy = AdmissionPolicy(model=tuned_model, warmup_n=0)
    trainer = PolicyGradientTrainer(tuned_policy, lr=1e-4)

    for episode in range(episodes):
        samples: list[PolicyGradientSample] = []
        tuned_policy.reset()

        original_decide = tuned_policy.decide

        def _recording_decide(state: SchedulerState, sample: bool = False, rng=None):
            decision = original_decide(state, sample=True, rng=rng)
            samples.append(
                PolicyGradientSample(
                    features=decision.features,
                    action=decision.admit,
                    probability=decision.probability,
                )
            )
            return decision

        tuned_policy.decide = _recording_decide  # type: ignore[method-assign]
        run_result = run(
            policy="perc",
            workload=workload,
            n_requests=n_requests,
            arrival_rate=arrival_rate,
            vram_gb=vram_gb,
            seed=seed + episode,
            kv_tier=kv_tier,
            max_concurrent=max_concurrent,
            admission_policy=tuned_policy,
            admission_sample=True,
        )
        tuned_policy.decide = original_decide  # type: ignore[method-assign]
        trainer.update_episode(samples, reward_from_run(run_result))

    baseline_rewards = []
    tuned_rewards = []
    baseline_drops = []
    tuned_drops = []
    baseline_costs = []
    tuned_costs = []

    for eval_seed in range(seed + 100, seed + 104):
        baseline_policy.reset()
        tuned_policy.reset()
        baseline_result = run(
            policy="perc",
            workload=workload,
            n_requests=n_requests,
            arrival_rate=arrival_rate,
            vram_gb=vram_gb,
            seed=eval_seed,
            kv_tier=kv_tier,
            max_concurrent=max_concurrent,
            admission_policy=baseline_policy,
            admission_sample=False,
        )
        tuned_result = run(
            policy="perc",
            workload=workload,
            n_requests=n_requests,
            arrival_rate=arrival_rate,
            vram_gb=vram_gb,
            seed=eval_seed,
            kv_tier=kv_tier,
            max_concurrent=max_concurrent,
            admission_policy=tuned_policy,
            admission_sample=False,
        )
        baseline_rewards.append(reward_from_run(baseline_result))
        tuned_rewards.append(reward_from_run(tuned_result))
        baseline_drops.append(baseline_result.requests_dropped / max(n_requests, 1))
        tuned_drops.append(tuned_result.requests_dropped / max(n_requests, 1))
        baseline_costs.append(baseline_result.total_eviction_cost_s)
        tuned_costs.append(tuned_result.total_eviction_cost_s)

    baseline_reward = float(np.mean(baseline_rewards))
    tuned_reward = float(np.mean(tuned_rewards))
    return FineTuneResult(
        episodes=episodes,
        baseline_reward=baseline_reward,
        tuned_reward=tuned_reward,
        baseline_drop_rate=float(np.mean(baseline_drops)),
        tuned_drop_rate=float(np.mean(tuned_drops)),
        baseline_eviction_cost_s=float(np.mean(baseline_costs)),
        tuned_eviction_cost_s=float(np.mean(tuned_costs)),
        improvement_pct=((tuned_reward - baseline_reward) / max(abs(baseline_reward), 1e-6)) * 100.0,
    )
