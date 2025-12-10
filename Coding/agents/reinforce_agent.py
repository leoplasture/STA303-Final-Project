import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------
# REINFORCE Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 0.001          # REINFORCE 结构简单，可以用稍大的学习率
ENTROPY_BETA = 0.0  # 纯 REINFORCE 通常不强求 Entropy，也可以设为 0.01

class PolicyNet(nn.Module):
    """
    纯策略网络 (只有 Actor，没有 Critic)
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class ReinforceConfig:
    gamma: float = GAMMA
    lr: float = LR
    entropy_beta: float = ENTROPY_BETA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ReinforceAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient)
    """
    def __init__(self, obs_dim, act_dim, cfg: ReinforceConfig = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or ReinforceConfig()
        self.device = torch.device(self.cfg.device)

        # 只有一个网络：Policy Network
        self.policy = PolicyNet(obs_dim, act_dim).to(self.device)
        self.optim = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

        # 存储一整局的数据
        self.log_probs = []
        self.rewards = []
        
        # 兼容 train.py
        self.exploration_rate = 0.0

    def act(self, state, evaluation_mode=False):
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.ndim == 1: state_t = state_t.unsqueeze(0)

        logits = self.policy(state_t)
        probs = F.softmax(logits, dim=-1)

        if evaluation_mode:
            return torch.argmax(probs, dim=1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # 存储 log_prob 用于后续计算梯度
            self.log_probs.append(dist.log_prob(action))
            return action.item()

    def step(self, state, action, reward, next_state, done):
        """
        REINFORCE 是回合更新算法 (Episodic)
        平时只存 reward，直到 done=True 才开始更新
        """
        self.rewards.append(reward)

        if done:
            self.update()

    def update(self):
        # 1. 计算蒙特卡洛回报 (Returns)
        # 从最后一步往前推，计算每一部能拿到的累计折扣奖励 G_t
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.cfg.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)

        # --- 关键技巧：回报归一化 (Return Normalization) ---
        # 如果没有这个，REINFORCE 在 CartPole 上极难收敛
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 2. 计算 Loss
        # Loss = - sum( log_prob * G_t )
        # 我们希望 log_prob 和 G_t 正相关 (奖励高的动作，概率要变大)
        loss = 0
        for log_prob, G_t in zip(self.log_probs, returns):
            loss += -log_prob * G_t

        # 3. 更新
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 4. 清空缓存 (为下一局做准备)
        self.log_probs = []
        self.rewards = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))