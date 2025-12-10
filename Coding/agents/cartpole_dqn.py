"""
PyTorch DQN for CartPole (Gymnasium)
------------------------------------
- Online Q-network + Target Q-network (to stabilize training)
- Vectorized replay updates for efficiency
- ε-greedy exploration schedule
- Designed to be imported by train.py

Reading guide for students:
- Start from DQNSolver.__init__ to see how the agent is constructed.
- Then read act() (how actions are chosen), and the new step() method
  (how the agent internalizes experience and learns).
- experience_replay() contains the core optimization logic.
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Default Hyperparameters
# -----------------------------
# γ: discount factor for future rewards
GAMMA = 0.99
# Learning rate for Adam optimizer
LR = 1e-3
# Mini-batch size sampled from the replay buffer
BATCH_SIZE = 32
# Replay buffer capacity (number of transitions stored)
MEMORY_SIZE = 50_000
# Steps to warm up the buffer with random-ish actions before training
INITIAL_EXPLORATION_STEPS = 1_000
# ε schedule: start, final, multiplicative decay per update step
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
# How often (in steps) to hard-copy online -> target network
TARGET_UPDATE_STEPS = 500


class QNet(nn.Module):
    """
    A simple fully connected MLP to approximate Q(s, a).
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, act_dim), 
        )
        
        # original better NNet:
        # self.net = nn.Sequential(
        #     nn.Linear(obs_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, act_dim),
        # )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, obs_dim] → returns Q(s,·): [B, act_dim]
        return self.net(x)


class ReplayBuffer:
    """
    FIFO replay buffer storing transitions as numpy arrays.
    - We convert to torch.Tensor only when sampling a batch.
    - Each entry stores (s, a, r, s', mask) where mask = 0 if terminal else 1.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        s = np.asarray(s)
        s2 = np.asarray(s2)
        # Squeeze arrays that are [1, obs_dim] down to [obs_dim] for storage
        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)
        if s2.ndim == 2 and s2.shape[0] == 1:
            s2 = s2.squeeze(0)
        self.buf.append((s, a, r, s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        # Uniformly sample a mini-batch of transitions
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        # Shapes after stacking:
        #  s, s2: [B, obs_dim], a: [B], r: [B], m: [B]
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DQNConfig:
    """
    Configuration object to keep training hyperparameters tidy and visible.
    Students can pass a custom DQNConfig() to DQNSolver to experiment.
    """
    gamma: float = GAMMA
    lr: float = LR
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    initial_exploration: int = INITIAL_EXPLORATION_STEPS
    eps_start: float = EPS_START
    eps_end: float = EPS_END
    eps_decay: float = EPS_DECAY
    target_update: int = TARGET_UPDATE_STEPS
    # Auto-select CUDA if available; CPU is perfectly fine for CartPole
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DQNSolver:
    """
    PyTorch DQN agent with:
      - online (trainable) Q-network
      - target (slow-moving) Q-network
      - replay buffer
      - ε-greedy exploration

    Public API used by train.py:
      act(), step(), save(), load(), update_target()
    """

    def __init__(self, observation_space: int, action_space: int, cfg: DQNConfig | None = None):
        # Store dimensions and hyperparameters
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or DQNConfig()

        # Choose device (GPU if available, else CPU)
        self.device = torch.device(self.cfg.device)

        # Build online and target Q-networks
        self.online = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target = QNet(self.obs_dim, self.act_dim).to(self.device)
        # Initialize target to match online at the beginning
        self.update_target(hard=True)

        # Optimizer over online network parameters
        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        # Experience replay memory
        self.memory = ReplayBuffer(self.cfg.memory_size)

        # Global counters
        self.steps = 0
        self.exploration_rate = self.cfg.eps_start

    # -----------------------------
    # Acting & memory
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        ε-greedy action selection.
        - If evaluation_mode=True, always acts greedily (exploitation).
        - If evaluation_mode=False (training):
            - With probability ε: choose a random action (exploration).
            - Otherwise: choose argmax_a Q_online(s, a) (exploitation).
        Inputs:
          state_np: numpy array with shape [1, obs_dim]
        """
        # 1. Exploration (only during training)
        if not evaluation_mode and np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)

        # 2. Exploitation (greedy action)
        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]  # (1, obs_dim)
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            q = self.online(s)  # [1, act_dim]
            a = int(torch.argmax(q, dim=1).item())
        return a

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store a single transition (s, a, r, s', done) into replay buffer."""
        # The ReplayBuffer's push() method handles squeezing [1, obs_dim] arrays
        self.memory.push(state, action, reward, next_state, done)

    # -----------------------------
    # Learning from replay
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        This is the main "learning" hook called by train.py
        1. Store the transition (s,a,r,s',done) in the replay buffer.
        2. Trigger one learning step (experience_replay) which samples from the buffer.
        """
        self.remember(state, action, reward, next_state, done)
        self.experience_replay()

    def experience_replay(self):
        """
        One training step from replay buffer (vectorized).
        (This method is now 'private', only called by self.step())
        Steps:
          1) Wait until we have enough transitions and warmup steps.
          2) Sample a mini-batch and build tensors on the right device.
          3) Compute targets using target network:
               y = r + mask * γ * max_a' Q_target(s', a')
          4) Compute current Q(s,a) from online network and MSE loss to targets.
          5) Backprop + optimizer step.
          6) Decay ε and periodically hard-update target network.
        """
        # 1) Warmup and capacity check: skip updates if insufficient data
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._decay_eps()  # still decay a bit each step
            return

        # 2) Sample and convert to tensors
        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)               # [B, obs_dim]
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1) # [B, 1]
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1) # [B, 1]
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)               # [B, obs_dim]
        m_t  = torch.as_tensor(m,  dtype=torch.float32, device=self.device).unsqueeze(1) # [B, 1]; 0 if done else 1

        # 3) Q(s,a) from online network (gather picks the Q-value of the taken action)
        q_sa = self.online(s_t).gather(1, a_t)  # [B, 1]

        # Compute target values using the target network (no gradient)
        with torch.no_grad():
            q_next = self.target(s2_t).max(dim=1, keepdim=True)[0]  # [B, 1] = max_a' Q_target(s',a')
            target = r_t + m_t * self.cfg.gamma * q_next            # [B, 1]

        # 4) MSE loss between current Q(s,a) and the target
        loss = nn.functional.mse_loss(q_sa, target)

        # 5) Backpropagation step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 6) Exploration decay
        self._decay_eps()

        # Hard copy online → target every N steps
        if self.steps % self.cfg.target_update == 0:
            self.update_target(hard=True)

    def update_target(self, hard: bool = True, tau: float = 0.005):
        """
        Synchronize target network weights:
          - hard=True:  direct copy (common for DQN).
          - hard=False: Polyak averaging (soft update) with factor tau.
        """
        if hard:
            self.target.load_state_dict(self.online.state_dict())
        else:
            with torch.no_grad():
                for p_t, p in zip(self.target.parameters(), self.online.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str):
        """
        Save both online & target network weights plus config for reproducibility.
        Safe to version-control the small CartPole models.
        """
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        """
        Load weights from disk onto the correct device.
        Note: Only loads weights; if you serialized optim state, add it here.
        """
        # For untrusted files, consider torch.load(..., weights_only=True) in future PyTorch
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        # Optional: restore cfg from ckpt["cfg"] if you want to enforce same hyperparams

    # -----------------------------
    # Helpers
    # -----------------------------
    def _decay_eps(self):
        """Multiplicative ε decay with lower bound EPS_END; keep a global step counter."""
        self.exploration_rate = max(self.cfg.eps_end, self.exploration_rate * self.cfg.eps_decay)
        self.steps += 1