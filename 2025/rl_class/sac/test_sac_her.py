"""
    Probably you need to remove apirate parameter from the environments xml
    /Users/edoardo.fazzari/miniconda3/envs/MBUZAI_4students/lib/python3.13/site-packages/gymnasium_robotics/envs
"""

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import numpy as np
import time


# As we did for the FetchReach environment we need to set the render_mode
# to human for visualization
def make_eval_env():
    env = gym.make("HandManipulateBlock-v1", reward_type="dense", render_mode="human")
    return env

if __name__ == "__main__":
    # Paths
    log_dir = "results/"
    model_path = os.path.join(log_dir, "sac_model_dense.zip")
    vec_stats_path = os.path.join(log_dir, "vec_normalize_sac_dense.pkl")
    # Again you register the environments
    gym.register_envs(gymnasium_robotics)
    # Use a single environment for evaluation, DummyVecEnv is just fine
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize.load(vec_stats_path, eval_env)
    # Important: do not update stats at test time
    eval_env.training = False
    eval_env.norm_reward = False

    # --- Load the trained SAC model ---
    model = SAC.load(model_path, env=eval_env, device="mps")

    # --- Evaluate the model ---
    n_episodes = 10
    episode_rewards = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            eval_env.render()
            total_reward += reward
            step += 1
            time.sleep(0.01)
            if done.any():
                break
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1} reward: {total_reward}")

    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(episode_rewards)}")

    eval_env.close()
