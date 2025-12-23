"""
bc_train.py
------------------------------------------------
Offline-to-Online 第一阶段：行为克隆 (Behavior Cloning)
功能：
1. 读取 expert_data.npz
2. 使用监督学习 (Supervised Learning) 训练 DQN 的网络结构
3. 保存预训练权重到 models/cartpole_bc.torch
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# 复用你现有的 DQN 结构，确保网络结构一致，方便后续加载
from agents.cartpole_dqn import DQNSolver, DQNConfig

DATA_PATH = "expert_data.npz"
SAVE_PATH = "models/cartpole_bc.torch"
BATCH_SIZE = 64
EPOCHS = 20  # 训练轮数，20轮足够学会了
LR = 1e-3    # BC 可以用大一点的学习率

def train_bc():
    # 1. 加载数据
    if not os.path.exists(DATA_PATH):
        print(f"[Error] {DATA_PATH} not found. Please run gen_data.py first.")
        return

    print(f"[Info] Loading expert data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    states = data['states']   # Shape: (N, 4)
    actions = data['actions'] # Shape: (N,)

    # 转换为 PyTorch Tensor
    tensor_x = torch.Tensor(states)
    tensor_y = torch.LongTensor(actions)

    # 制作 DataLoader
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 初始化模型
    # 我们直接实例化 DQNSolver，但只训练它的 online 网络
    # 注意：obs_dim=4, act_dim=2 是 CartPole 的标准
    agent = DQNSolver(observation_space=4, action_space=2)
    device = agent.device
    print(f"[Info] Training on {device}...")

    # 定义优化器和损失函数
    # BC 本质上是一个分类问题：输入状态 -> 分类动作 0 或 1
    optimizer = optim.Adam(agent.online.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 3. 开始训练
    agent.online.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_s, batch_a in dataloader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)

            # 前向传播：获取 Q值 (作为 Logits)
            q_values = agent.online(batch_s)

            # 计算 Loss：让专家动作对应的 Q 值最大
            loss = criterion(q_values, batch_a)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 4. 保存模型
    # 我们保存整个 agent 的状态，为了和 DQN 兼容，我们主要需要 online 的参数
    # 这里我们模拟 DQNSolver.save 的格式，只保存 state_dict 方便加载
    torch.save({
        "model": agent.online.state_dict(),
        "config": "BC_Pretrained"
    }, SAVE_PATH)
    print(f"[Done] BC Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_bc()