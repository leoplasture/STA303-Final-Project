import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------
# A2C Hyperparameters (经过调优)
# -----------------------------
GAMMA = 0.99
LR = 5e-4
ENTROPY_BETA = 0.01
GAE_LAMBDA = 0.95
UPDATE_EVERY = 50
GRAD_CLIP = 0.5
VALUE_COEF = 0.5  # critic loss 系数
EPS = 1e-8

class ACNet(nn.Module):
    """
    Actor-Critic 共享特征网络
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        
        # Actor Head (策略头)
        self.actor = nn.Linear(128, act_dim)
        
        # Critic Head (价值头)
        self.critic = nn.Linear(128, 1)

        # 权重初始化 (Xavier Uniform)
        for layer in [self.feature[0], self.actor, self.critic]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value

@dataclass
class A2CConfig:
    gamma: float = GAMMA
    lr: float = LR
    entropy_beta: float = ENTROPY_BETA
    gae_lambda: float = GAE_LAMBDA
    update_every: int = UPDATE_EVERY
    grad_clip: float = GRAD_CLIP
    value_coef: float = VALUE_COEF
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class A2CAgent:
    def __init__(self, obs_dim, act_dim, cfg: A2CConfig = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or A2CConfig()
        self.device = torch.device(self.cfg.device)

        # 网络与优化器
        self.net = ACNet(obs_dim, act_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        # Rollout buffer (store python lists for efficiency, convert to tensor at update)
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        # 占位符以兼容外部代码
        self.exploration_rate = 0.0 

    def act(self, state, evaluation_mode=False):
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)  # [1, obs_dim]

        logits, _ = self.net(state_t)

        # 数值稳定：从 logits 减去 max 值
        logits = logits - logits.max(dim=-1, keepdim=True).values

        probs = F.softmax(logits, dim=-1)

        if evaluation_mode:
            action = torch.argmax(probs, dim=1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        
        return action

    def step(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
        if len(self.states) >= self.cfg.update_every or done:
            self.update(next_state, done)

    def update(self, last_next_state, last_done):
        if len(self.states) == 0:
            return

        # ====== 1. 转换为 tensor，并保证 dtype/device 一致 ======
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        # 有些环境会产生额外维度，尝试 squeeze
        if states_t.ndim == 3 and states_t.shape[1] == 1:
            states_t = states_t.squeeze(1)

        actions_t = torch.LongTensor(self.actions).to(self.device).view(-1, 1)  # [T,1]
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)  # [T]
        dones_t = torch.tensor(self.dones, dtype=torch.float32, device=self.device)      # [T]

        T = len(self.rewards)

        # ====== 2. 计算当前 states 的 logits 与 values ======
        logits, values = self.net(states_t)  # values: [T,1]
        # values 可能是 shape [T,1], 转为 [T,1] 保证后续拼接一致
        if values.ndim == 1:
            values = values.unsqueeze(-1)

        # ====== 3. 计算 bootstrap 的 last_value (确保 device/dtype 一致) ======
        with torch.no_grad():
            last_state_t = torch.FloatTensor(last_next_state).to(self.device)
            if last_state_t.ndim == 1:
                last_state_t = last_state_t.unsqueeze(0)
            _, last_value = self.net(last_state_t)
            # last_value 可能为 shape [1,1]
            last_value = last_value.to(values.dtype)
            # 如果 done，last_value 应该被置为 0
            last_value = last_value * (1.0 - float(last_done))

        # ====== 4. 计算 GAE（倒序） ======
        # all_values: [T+1, 1]
        all_values = torch.cat([values, last_value], dim=0)

        advantages = torch.zeros((T, 1), dtype=values.dtype, device=self.device)
        gae = torch.zeros(1, dtype=values.dtype, device=self.device)

        for t in range(T - 1, -1, -1):
            mask = 1.0 - dones_t[t]  # scalar tensor
            # delta: reward + gamma * V_{t+1} * mask - V_t
            delta = rewards_t[t] + self.cfg.gamma * all_values[t + 1].squeeze() * mask - all_values[t].squeeze()
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages[t, 0] = gae

        # 优势归一化（如果 T>1 才做）
        if T > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)  # small sample 更稳定
            advantages = (advantages - adv_mean) / (adv_std + EPS)

        # detach advantage (actor 用) 和构造 returns (critic 的 target)
        advantages_detached = advantages.detach()
        returns = values.detach() + advantages_detached  # [T,1]，target 独立于 critic 的梯度

        # ====== 5. 计算各类损失 ======
        # Critic loss (MSE between predicted values and returns)
        critic_loss = F.mse_loss(values, returns)

        # Actor loss
        # logits 做数值稳定处理
        logits = logits - logits.max(dim=-1, keepdim=True).values
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_t)  # [T,1]
        actor_loss = -(action_log_probs * advantages_detached).mean()

        # Entropy（鼓励探索）
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = -self.cfg.entropy_beta * entropy

        total_loss = actor_loss + self.cfg.value_coef * critic_loss + entropy_loss

        # ====== 6. 反向传播与梯度裁剪 ======
        self.optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optim.step()

        # ====== 7. 清空 buffer ======
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
