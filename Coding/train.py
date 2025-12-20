"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Trains DQN/A2C/REINFORCE/PPO/BC agents and logs scores via ScoreLogger.
- Supports command line arguments for algorithm switching and hyperparameter tuning.
"""

from __future__ import annotations
import os
# Fix OpenMP conflict error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import numpy as np
import gymnasium as gym
import torch

# Import Agents and Configs
from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.a2c_agent import A2CAgent, A2CConfig
from agents.reinforce_agent import ReinforceAgent, ReinforceConfig
from agents.ppo_agent import PPOAgent, PPOConfig
from agents.bc_agent import BCAgent, BCConfig  # <--- [修复] 导入 BC
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

def apply_overrides(cfg, args):
    """Helper: Override config parameters with command line arguments."""
    if args.lr is not None:
        cfg.lr = args.lr
        print(f"[Override] Learning Rate -> {cfg.lr}")
    
    if args.gamma is not None:
        cfg.gamma = args.gamma
        print(f"[Override] Gamma -> {cfg.gamma}")
    
    if args.batch_size is not None and hasattr(cfg, 'batch_size'):
        cfg.batch_size = args.batch_size
        print(f"[Override] Batch Size -> {cfg.batch_size}")
        
    if args.entropy is not None and hasattr(cfg, 'entropy_beta'):
        cfg.entropy_beta = args.entropy
        print(f"[Override] Entropy Beta -> {cfg.entropy_beta}")

    return cfg

def train(num_episodes: int = 200, terminal_penalty: bool = True, algorithm: str = "dqn", args=None) -> object:
    """Main training loop."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")

    # Create CartPole environment
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- Instantiate Agent based on algorithm ---
    if algorithm.lower() == "a2c":
        print(f"[Info] Training with A2C Agent...")
        cfg = A2CConfig()
        cfg = apply_overrides(cfg, args)
        agent = A2CAgent(obs_dim, act_dim, cfg=cfg)
        
    elif algorithm.lower() == "reinforce":
        print(f"[Info] Training with REINFORCE Agent...")
        cfg = ReinforceConfig()
        cfg = apply_overrides(cfg, args)
        agent = ReinforceAgent(obs_dim, act_dim, cfg=cfg)
        
    elif algorithm.lower() == "ppo":
        print(f"[Info] Training with PPO Agent...")
        cfg = PPOConfig()
        cfg = apply_overrides(cfg, args)
        agent = PPOAgent(obs_dim, act_dim, cfg=cfg)
        
    elif algorithm.lower() == "bc":  # <--- [修复] 加入 BC 分支逻辑
        print(f"[Info] Training with Behavior Cloning (BC) Agent...")
        cfg = BCConfig()
        cfg = apply_overrides(cfg, args)
        agent = BCAgent(obs_dim, act_dim, cfg=cfg)
        
        # BC is offline learning
        data_file = "expert_data.npz"
        if os.path.exists(data_file):
            agent.load_data_and_train(data_file)
            agent.save(model_path)
            print(f"[Train] BC Model saved to {model_path}")
            return agent
        else:
            raise FileNotFoundError(f"{data_file} not found! Please run gen_data.py first.")

    else: # Default to DQN
        print(f"[Info] Training with DQN Agent...")
        cfg = DQNConfig()
        cfg = apply_overrides(cfg, args)
        agent = DQNSolver(obs_dim, act_dim, cfg=cfg)
    
    # Print device and hyperparameters info
    device = getattr(agent, "device", "unknown")
    print(f"[Info] Using device: {device}")
    
    lr = getattr(agent.cfg, "lr", "N/A")
    gamma = getattr(agent.cfg, "gamma", "N/A")
    print(f"[Info] Hyperparameters: LR={lr}, Gamma={gamma}")

    # --- Main Training Loop ---
    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            
            # 1. Select Action
            action = agent.act(state)

            # 2. Step Environment
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0
            
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 3. Agent Learn Step
            agent.step(state, action, reward, next_state, done)

            state = next_state

            if done:
                # Log progress
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
    
    # Auto-infer model path if not provided
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")

    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}. Train it first!")
        return

    print(f"[Eval] Using model: {model_path}")

    # Setup environment
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- Instantiate Agent for Evaluation ---
    if algorithm.lower() == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm.lower() == "a2c":
        agent = A2CAgent(obs_dim, act_dim, cfg=A2CConfig())
    elif algorithm.lower() == "reinforce":
        agent = ReinforceAgent(obs_dim, act_dim, cfg=ReinforceConfig())
    elif algorithm.lower() == "ppo":
        agent = PPOAgent(obs_dim, act_dim, cfg=PPOConfig())
    elif algorithm.lower() == "bc":  # <--- [修复] 加入 BC 分支
        agent = BCAgent(obs_dim, act_dim, cfg=BCConfig())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load weights
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model.")

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    # Evaluation Loop
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
    
    # Core arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Run mode")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "a2c", "reinforce", "ppo", "bc"], help="Algorithm")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    # Hyperparameters
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
        evaluate(
            algorithm=args.algo, 
            episodes=args.episodes, 
            render=args.render, 
            fps=60
        )