"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Trains a DQN/A2C/REINFORCE/PPO agent and logs scores via ScoreLogger (PNG + CSV)
- Supports command line arguments for hyperparameter tuning.
"""

from __future__ import annotations
import os
# 解决 OpenMP 冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import numpy as np
import gymnasium as gym
import torch

# 导入你的 Agent 和 Config
from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.a2c_agent import A2CAgent, A2CConfig
from agents.reinforce_agent import ReinforceAgent, ReinforceConfig
from agents.ppo_agent import PPOAgent, PPOConfig  # <--- [新增] 导入 PPO
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

def apply_overrides(cfg, args):
    """
    辅助函数：将命令行参数 args 中的值覆盖到 cfg 配置对象中
    """
    if args.lr is not None:
        cfg.lr = args.lr
        print(f"[Override] Learning Rate -> {cfg.lr}")
    
    if args.gamma is not None:
        cfg.gamma = args.gamma
        print(f"[Override] Gamma -> {cfg.gamma}")
    
    # 如果 Config 里有 batch_size 且命令行也传了，则覆盖
    if args.batch_size is not None and hasattr(cfg, 'batch_size'):
        cfg.batch_size = args.batch_size
        print(f"[Override] Batch Size -> {cfg.batch_size}")
        
    # 特殊处理：A2C/REINFORCE/PPO 的 entropy
    if args.entropy is not None and hasattr(cfg, 'entropy_beta'):
        cfg.entropy_beta = args.entropy
        print(f"[Override] Entropy Beta -> {cfg.entropy_beta}")

    return cfg

def train(num_episodes: int = 200, terminal_penalty: bool = True, algorithm: str = "dqn", args=None) -> object:
    """
    Main training loop.
    args: 命令行解析出来的参数对象，用于覆盖默认超参数
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 构造模型保存路径 (加上算法名，防止覆盖)
    model_path = os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")

    # Create CartPole environment
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- 实例化 Agent 并应用参数覆盖 ---
    if algorithm.lower() == "a2c":
        print(f"[Info] Training with A2C Agent...")
        cfg = A2CConfig()
        cfg = apply_overrides(cfg, args) # 覆盖参数
        agent = A2CAgent(obs_dim, act_dim, cfg=cfg)
    elif algorithm.lower() == "reinforce":
        print(f"[Info] Training with REINFORCE Agent...")
        cfg = ReinforceConfig()
        cfg = apply_overrides(cfg, args)
        agent = ReinforceAgent(obs_dim, act_dim, cfg=cfg)
    elif algorithm.lower() == "ppo": # <--- [新增] PPO 分支
        print(f"[Info] Training with PPO Agent...")
        cfg = PPOConfig()
        cfg = apply_overrides(cfg, args)
        agent = PPOAgent(obs_dim, act_dim, cfg=cfg)
    else:
        print(f"[Info] Training with DQN Agent...")
        cfg = DQNConfig()
        cfg = apply_overrides(cfg, args) # 覆盖参数
        agent = DQNSolver(obs_dim, act_dim, cfg=cfg)
    
    # 安全获取 device 属性
    device = getattr(agent, "device", "unknown")
    print(f"[Info] Using device: {device}")
    
    # 打印超参数 (兼容不同 Config 结构)
    lr = getattr(agent.cfg, "lr", "N/A")
    gamma = getattr(agent.cfg, "gamma", "N/A")
    print(f"[Info] Hyperparameters: LR={lr}, Gamma={gamma}")

    # Episode loop
    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            
            # 1. Action
            action = agent.act(state)

            # 2. Step
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0
            
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 3. Learning Step
            agent.step(state, action, reward, next_state, done)

            state = next_state

            if done:
                # 兼容不同 Agent 的日志打印
                eps_info = ""
                if hasattr(agent, "exploration_rate"):
                    eps_info = f", Epsilon: {agent.exploration_rate:.3f}"
                
                print(f"Run: {run}{eps_info}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()
    agent.save(model_path)
    print(f"[Train] Model saved to {model_path}")
    return agent


def evaluate(model_path: str | None = None,
             algorithm: str = "dqn",
             episodes: int = 5,
             render: bool = True,
             fps: int = 60):
    
    # 自动推断路径
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")

    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}. Train it first!")
        return

    print(f"[Eval] Using model: {model_path}")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- [修改] 增加 Agent 实例化逻辑 ---
    if algorithm.lower() == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm.lower() == "a2c":
        agent = A2CAgent(obs_dim, act_dim, cfg=A2CConfig())
    elif algorithm.lower() == "reinforce":
        agent = ReinforceAgent(obs_dim, act_dim, cfg=ReinforceConfig())
    elif algorithm.lower() == "ppo": # <--- [新增] PPO
        agent = PPOAgent(obs_dim, act_dim, cfg=PPOConfig())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model.")

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
            if dt > 0: time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores))
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Parameter Tuning")
    
    # 基础参数
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Run mode")
    # [修改] choices 加入 ppo
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "a2c", "reinforce", "ppo"], help="Algorithm")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    
    # 关键超参数 (默认值为 None，表示使用 Config 文件中的默认值)
    parser.add_argument("--lr", type=float, default=None, help="Learning Rate override")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma (Discount) override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--entropy", type=float, default=None, help="Entropy beta (A2C/PPO only)")

    args = parser.parse_args()

    if args.mode == "train":
        print(f"--- Starting Training ({args.algo.upper()}) ---")
        train(num_episodes=args.episodes, terminal_penalty=True, algorithm=args.algo, args=args)
        
    else:
        print(f"--- Starting Evaluation ({args.algo.upper()}) ---")
        evaluate(algorithm=args.algo, episodes=10, render=True, fps=60)