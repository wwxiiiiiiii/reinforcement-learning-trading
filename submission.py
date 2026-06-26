"""
Student submission file.

Required interface:

    load_agent(model_dir)

The returned object must implement:

    act(obs)

Evaluation is always CPU-only. Do not add a device argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC


class SACAgent:
    """
    Wraps a trained SAC model for CPU-only inference.
    """

    def __init__(self, model_path: str | Path):
        self.model = SAC.load(str(model_path), device="cpu")

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class ZeroAgent:
    """Fallback agent — holds all positions."""

    def __init__(self, action_dim: int = 84):
        self.action_dim = action_dim

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


def load_agent(model_dir: str):
    model_dir = Path(model_dir)
    model_path = model_dir / "saved_agent.zip"

    if model_path.exists():
        print(f"[load_agent] Loading SAC model from: {model_path}")
        return SACAgent(model_path=model_path)

    print("[load_agent] WARNING: saved_agent.zip not found. Using ZeroAgent (hold).")
    return ZeroAgent(action_dim=84)
