import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


if __name__ == "__main__":
    # Make the environment just like before, except that now we want to
    # visualize it. To do that set the render_mode to human
    gym.register_envs(gymnasium_robotics)
    env = gym.make('HandManipulateBlock-v4', render_mode='human')
    env = FlattenObservation(env)
    print('Observation space:', env.observation_space)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load('results/vecnormalize_fetchreach.pkl', env)
    env.training = False
    env.norm_rewards = False
    # For loading the trained model you can use the load method and
    # load the zip file created by save
    model = PPO.load('results/ppo_fetchreach.zip')

    # For testing you need to interact with the environment
    # We test it for 10 epochs
    for _ in range(10):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Be aware that gymnasium returns 5: obs, reward, done, truncated, info
            # however gymnasium-robotics still use the 4-parameter schema of gym (deprecated)
            # which integrated done together with truncated.
            obs, reward, done, info = env.step(action)
            # Since we set the render mode we can show what is going on with render
            env.render()
            total_reward += reward
        print('Total reward:', total_reward)
    # When you are done with your environment you must close it otherwise an error will occur
    # if you terminate without safely closing it.
    env.close()
