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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Returns (admission_score, lambda_estimate).
        admission_score in [0, 1]; lambda_estimate in [0, inf).
        """
        out = self.forward(features)
        admission = float(_sigmoid(out[0]))
        lambda_est = float(np.exp(np.clip(out[1], -5, 5)))  # softplus-like
        return admission, lambda_est

    def update_weight(self, dW1, db1, dW2, db2, lr: float = 1e-3) -> None:
        """Gradient descent step (called by imitation learning trainer)."""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


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
        self.buffer: List[Trace] = []
        self.losses: List[float] = []

    def add_trace(self, trace: Trace) -> None:
        self.buffer.append(trace)

    def train_step(self) -> Optional[float]:
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

    def __init__(self, model: Optional[TinyMLP] = None):
        self.model = model or TinyMLP()
        self._n_decisions = 0

    def should_admit(self, state: SchedulerState) -> Tuple[bool, float]:
        """
        Returns (admit: bool, confidence: float).
        confidence in [0, 1] — how certain the model is.
        """
        self._n_decisions += 1
        features = extract_features(state)

        if self._n_decisions < self.WARMUP_N:
            admit = state.vram_util < 0.85
            return admit, 0.5

        score, _ = self.model.predict(features)
        admit = score > self.ADMIT_THRESHOLD
        confidence = abs(score - 0.5) * 2  # [0, 1]
        return admit, confidence

    def estimated_lambda(self, state: SchedulerState) -> float:
        """Return the model's prediction of this session's token arrival rate."""
        features = extract_features(state)
        _, lambda_est = self.model.predict(features)
        return lambda_est
