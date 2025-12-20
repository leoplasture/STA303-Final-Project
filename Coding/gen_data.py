"""
生成专家数据 (Expert Data Collection)
------------------------------------
适配 REINFORCE 算法：加载训练好的 REINFORCE 模型，收集 (State, Action) 对。
"""
import numpy as np
import torch
import gymnasium as gym
import os

# 引入 REINFORCE Agent (因为你现在最好的模型是这个)
from agents.reinforce_agent import ReinforceAgent, ReinforceConfig

ENV_NAME = "CartPole-v1"
# 确保这里指向你刚才训练好的 REINFORCE 模型路径
MODEL_PATH = "models/cartpole_reinforce.torch"  
DATA_PATH = "expert_data.npz"
NUM_EPISODES = 20  

def generate_data():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Please train REINFORCE first!")
        return

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- 修改点：这里必须实例化 ReinforceAgent ---
    print(f"Loading expert model from {MODEL_PATH}...")
    cfg = ReinforceConfig()
    agent = ReinforceAgent(obs_dim, act_dim, cfg=cfg)
    
    # 加载权重
    agent.load(MODEL_PATH)
    print("Expert model loaded successfully!")

    expert_states = []
    expert_actions = []
    total_score = 0

    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0
        
        while not done:
            # 使用专家模型预测动作 (Evaluation Mode)
            action = agent.act(state, evaluation_mode=True)
            
            # 记录数据 (S, A)
            expert_states.append(state.squeeze())
            expert_actions.append(action)

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
        
        total_score += steps
        print(f"Episode {ep+1}: Score = {steps}")

    # 保存为 numpy 格式
    np.savez(DATA_PATH, states=np.array(expert_states), actions=np.array(expert_actions))
    print(f"\nSaved {len(expert_states)} samples to {DATA_PATH}")
    print(f"Average Expert Score: {total_score / NUM_EPISODES}")

if __name__ == "__main__":
    generate_data()