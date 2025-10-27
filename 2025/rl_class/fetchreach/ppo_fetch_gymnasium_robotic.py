"""
    INSTALLATION:
        1. For mac: brew install mujoco
        2. export MUJOCO_PATH=/opt/homebrew/opt/mujoco
        3. source ~/.zshrc
        4. pip install gymnasium
        5. pip install mujoco-py gymnasium-robotics sb3-contrib stable-baselines3
"""
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


if __name__ == "__main__":
    # When using gymnasium robotics the environment available in the package
    # are not present in gymnasium hence we need to register them to have them
    # available when using gymnasium make function
    gym.register_envs(gymnasium_robotics)
    # Now that the environment are loaded to gymnasium we can make them using
    # the environment name provided by the official documentation https://robotics.farama.org/index.html
    # Since all these environment are based on MuJoCo, so please ensure to have it
    # installed and having the latest version since newer environment version are
    # sometimes not compatible with previous versions of MuJoCo (everything described in the documentation!)
    env = gym.make('FetchReachDense-v4', max_episode_steps=3000)
    # The environment defines how the observation space, action space and everything that
    # needs to make it work.
    # When starting working with a new environment it mandatory to understand how its
    # observation and action space work before starting to use it.
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)
    # As seen in the slides we can wrap the environment to make it suitable for
    # our specific needs.
    # For example, the observation space is now a Dict something that can be handled,
    # however we can simply it making it an array by using the wrapper FlattenObservation
    # which will handle the transformation from Dict to Box (type of array of gymnasium)
    env = FlattenObservation(env)
    print('Observation space:', env.observation_space)
    # Single we want to vectorize the environment to benefit training we can use
    # some wrappers from stable_baseline3. (Note: also gymnasium has its own vector environments,
    # but since we are working the stable_baseline3 and not implementing our DRL algorithms it is
    # better to use those from stable_baseline3)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Now that we have vectorize, we can create are PPO algorithm object. Most of the hyperparameters
    # have the default values suggested by the original paper. We are using a MlpPolicy since our
    # observation space is just a vector.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # how many steps before each update
        batch_size=64,  # mini-batch size during optimization
        n_epochs=10,  # number of epochs per update
        gamma=0.98,  # discount factor
        gae_lambda=0.95,  # GAE smoothing
        clip_range=0.2,  # PPO clipping
        ent_coef=0.0,  # entropy bonus (encourages exploration)
        device="cpu", # if you want you can train on cuda or mps also
    )
    # To start training is simple
    model.learn(total_timesteps=100_000)
    # You can save normalization stats and model
    model.save("ppo_fetchreach")
    env.save('vecnormalize_fetchreach.pkl')


