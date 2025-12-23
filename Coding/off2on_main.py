"""
off2on_main.py
------------------------------------------------
Offline-to-Online 第二阶段：在线微调 (Online Fine-tuning)
创新点：
- Jump-Start: 使用 BC 预训练权重初始化 DQN
- Low Exploration: 极低的初始 Epsilon，避免破坏学到的知识
- Smooth Transition: 将离线知识迁移到在线环境
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
import numpy as np
import torch
import os
import argparse
from scores.score_logger import ScoreLogger

# 引入 DQN
from agents.cartpole_dqn import DQNSolver, DQNConfig

ENV_NAME = "CartPole-v1"
BC_MODEL_PATH = "models/cartpole_bc.torch"
SAVE_PATH = "models/cartpole_off2on.torch"

def off2on_train(args):
    # 1. 环境设置
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 2. 配置 DQN (微调参数)
    # 这里的参数非常关键：学习率要低，Batch要大，Gamma要高
    cfg = DQNConfig(
        lr=args.lr,           # 建议 0.0001 (慢速微调)
        batch_size=128,       # 稳一点
        gamma=0.995,          # 保持远见
        memory_size=50000,
        target_update=100
    )
    
    agent = DQNSolver(obs_dim, act_dim, cfg=cfg)

    # ================= [核心创新点：加载 BC 权重] =================
    if os.path.exists(BC_MODEL_PATH):
        print(f"[Innovation] Loading BC pre-trained weights from {BC_MODEL_PATH}...")
        checkpoint = torch.load(BC_MODEL_PATH, map_location=agent.device)
        
        # 加载权重到 Online 网络
        agent.online.load_state_dict(checkpoint['model'], strict=False)
        # 别忘了把 Target 网络也同步成一样的！
        agent.target.load_state_dict(checkpoint['model'], strict=False)
        print("[Success] Weights loaded. Jump-start ready!")
        
        # [关键设置]：因为是高手，不需要从 1.0 开始瞎蒙
        # 我们把 epsilon 直接设为 0.1 或更低
        agent.exploration_rate = 0.05 
        print(f"[Config] Epsilon set to {agent.exploration_rate} (Skipping random exploration)")
    else:
        print(f"[Error] BC model not found at {BC_MODEL_PATH}! Running from scratch...")
    # ===============================================================

    print(f"[Info] Starting Off2On Fine-tuning for {args.episodes} episodes...")

    # 3. 训练循环 (和普通训练一样，但起点很高)
    for run in range(1, args.episodes + 1):
        state, _ = env.reset()
        state = np.reshape(state, (1, obs_dim))
        steps = 0
        
        while True:
            # Act
            action = agent.act(state) # 注意：这里依然有 epsilon 随机性，但很小(0.05)

            # Step
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state_raw, (1, obs_dim))
            
            # Learn
            # 即使是 BC 初始化的，我们也继续把新数据存入 memory 并训练
            # 这叫 "Online Fine-tuning"
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
            
            if done:
                print(f"Run: {run}, Score: {steps}, Epsilon: {agent.exploration_rate:.3f}")
                logger.add_score(steps, run)
                break
                
    env.close()
    agent.save(SAVE_PATH)
    print(f"[Done] Final Off2On model saved to {SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认给一个很低的学习率，因为只是微调
    parser.add_argument("--lr", type=float, default=0.0001, help="Fine-tuning learning rate")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes to run")
    args = parser.parse_args()
    
    off2on_train(args)