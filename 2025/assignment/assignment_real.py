import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC, HerReplayBuffer
import numpy as np
import random
import torch
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def make_env():
    env = gym.make('FetchPickAndPlace-v4',
                   render_mode=None,
                   max_episode_steps=200,)
    env.reset()
    return env

def arg_parse():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='FetchPickAndPlace-v4', help='Environment Name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, default='PPO', help='Policy Model')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.05, help='Target network update rate')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, help='Buffer size')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    set_seed(args.seed)
    gym.register_envs(gymnasium_robotics)
    env = make_vec_env(make_env, n_envs=5, vec_env_cls=DummyVecEnv)
    if args.model == 'PPO':
        model = PPO('MultiInputPolicy', env, verbose=1)
    elif args.model == 'SAC':
        model = SAC('MultiInputPolicy',
                    env,
                    verbose=1,
                    batch_size=512,
                    learning_rate=1e-3,
                    buffer_size=args.buffer_size,
                    tau=args.tau,
                    gamma=args.gamma,
                    device='mps' if torch.mps.is_available() else torch.device('cpu'),
                    seed=args.seed,
                    )
    elif args.model == 'SAC_HER':
        pass
    else:
        raise ValueError('Invalid model')

    model.learn(total_timesteps=1_000_000)
    model.save(f'{args.model}.pkl')




