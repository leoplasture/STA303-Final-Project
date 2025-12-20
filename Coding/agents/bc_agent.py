import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass

# BC 超参数
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 50  # 这里的 Epoch 是指在数据集上训练多少轮

class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1), # 防止过拟合
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class BCConfig:
    lr: float = LR
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BCAgent:
    def __init__(self, obs_dim, act_dim, cfg: BCConfig = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or BCConfig()
        self.device = torch.device(self.cfg.device)

        self.policy = BCNet(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # 这里的 exploration_rate 只是为了兼容 train.py 的打印，BC 不需要探索
        self.exploration_rate = 0.0 

    def load_data_and_train(self, data_path="expert_data.npz"):
        """
        BC 特有的方法：一次性加载数据并进行全量训练
        """
        print(f"[BC] Loading data from {data_path}...")
        data = np.load(data_path)
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)

        dataset = TensorDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        print(f"[BC] Starting Supervised Learning ({self.cfg.epochs} Epochs)...")
        self.policy.train()
        for epoch in range(self.cfg.epochs):
            total_loss = 0
            for batch_s, batch_a in dataloader:
                self.optimizer.zero_grad()
                logits = self.policy(batch_s)
                loss = self.criterion(logits, batch_a)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.cfg.epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        print("[BC] Training Complete!")
        self.policy.eval()

    def act(self, state, evaluation_mode=True):
        # BC 始终像是 Evaluation 模式，直接输出概率最大的动作
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.ndim == 1: state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(state_t)
            action = torch.argmax(logits, dim=1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        # BC 是离线学习，不需要在交互过程中更新
        pass

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))