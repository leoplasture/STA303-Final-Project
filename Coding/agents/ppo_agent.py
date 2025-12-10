import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------
# PPO Hyperparameters (Episodic Version)
# -----------------------------
GAMMA = 0.99
LR = 0.0005
EPS_CLIP = 0.2
K_EPOCHS = 10
BATCH_EPISODES = 4      # 改为：每收集 4 局游戏更新一次
ENTROPY_BETA = 0.01

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)

        # Orthogonal Initialization
        for layer in [self.feature[0], self.actor, self.critic]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        feat = self.feature(x)
        return self.actor(feat), self.critic(feat)

    def evaluate(self, state, action):
        logits, value = self(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, value, dist_entropy

@dataclass
class PPOConfig:
    gamma: float = GAMMA
    lr: float = LR
    eps_clip: float = EPS_CLIP
    k_epochs: int = K_EPOCHS
    batch_episodes: int = BATCH_EPISODES  # 新参数
    entropy_beta: float = ENTROPY_BETA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent:
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)

        self.policy = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.exploration_rate = 0.0

        # 大 Batch 缓存
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_returns = [] # 直接存计算好的回报

        # 当前局缓存
        self.ep_rewards = []
        self.ep_states = []
        self.ep_actions = []
        self.ep_logprobs = []
        
        self.episode_counter = 0

    def act(self, state, evaluation_mode=False):
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.ndim == 1: state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.policy(state_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

        if evaluation_mode:
            return torch.argmax(probs, dim=1).item()
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # 暂存当前步数据
        self.ep_states.append(state_t.squeeze(0).cpu())
        self.ep_actions.append(action.item())
        self.ep_logprobs.append(action_logprob.item())

        return action.item()

    def step(self, state, action, reward, next_state, done):
        """
        改为回合制逻辑：收集完一局再结算
        """
        self.ep_rewards.append(reward)

        if done:
            # 1. 结算当前局：计算 Monte Carlo Returns
            discounted_reward = 0
            ep_returns = []
            for r in reversed(self.ep_rewards):
                discounted_reward = r + self.cfg.gamma * discounted_reward
                ep_returns.insert(0, discounted_reward)
            
            # 2. 存入大 Buffer
            self.buffer_states.extend(self.ep_states)
            self.buffer_actions.extend(self.ep_actions)
            self.buffer_logprobs.extend(self.ep_logprobs)
            self.buffer_returns.extend(ep_returns)

            # 3. 清空当前局缓存
            self.ep_rewards = []
            self.ep_states = []
            self.ep_actions = []
            self.ep_logprobs = []

            # 4. 检查是否凑够 Batch 进行更新
            self.episode_counter += 1
            if self.episode_counter % self.cfg.batch_episodes == 0:
                self.update()

    def update(self):
        # 准备数据
        old_states = torch.stack(self.buffer_states).to(self.device).detach()
        old_actions = torch.tensor(self.buffer_actions, dtype=torch.long).to(self.device).detach()
        old_logprobs = torch.tensor(self.buffer_logprobs, dtype=torch.float32).to(self.device).detach()
        rewards = torch.tensor(self.buffer_returns, dtype=torch.float32).to(self.device).detach()

        # 归一化 Rewards (非常重要)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # PPO Update Loop
        for _ in range(self.cfg.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Advantage = Returns - Values
            advantages = rewards - state_values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * F.mse_loss(state_values, rewards) - \
                   self.cfg.entropy_beta * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # 清空大 Buffer
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_returns = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))