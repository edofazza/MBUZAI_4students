import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


# Instead of using a lambda as in the case of the FetchReach code
# we can define a function. Usually, this is much better since we
# can include in the function additional wrappers before vectorizing
# it and having everything in just one place.
def make_env():
    env = gym.make('HandManipulateBlock-v1', reward_type='dense')
    return env


if __name__ == '__main__':
    # Again, you need to register the environments
    gym.register_envs(gymnasium_robotics)
    # Now, differently from before we want to make use of parallel environments
    # running at the same time. When doing this we need to define if we want to
    # use DummyVecEnv or SubprocVecEnv. Usually, the former is faster however
    # the latter is recommended for heavy environments like robotics
    env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)
    # When wrapping with VecNormalize do not use norm_reward in this case since
    # we use HER that modifies rewards internally and normalization can cause
    # distortions
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200.)
    # Now you need to set the parameters for HER before instantiating SAC.
    # HER as multiple configuration parameters but two are the most important
    # the number of virtual goals to sample and the strategy for sampling goals.
    # The most commonly used and most effective sampling strategy is future, which
    # is also the default value. So most of the type your problem is more about
    # defining the number of sampled goals.
    replay_buffer_kwargs = {
        'n_sampled_goal': 4,
        'goal_selection_strategy': 'future'
    }
    # SAC configuration with HER requires to add the replay_buffer_class and the
    # replay_buffer_kwargs that we have created. Furthermore, differently from PPO
    # SAC uses a replay buffer for storing trajectories so we can take advantage of
    # this buffer for wait before learning until we reach a certain number of iterations
    # this is done to reduce the problem of training the model on unlucky trajectories.
    # What is the architecture? Different from before we kept the Dict observation space
    # which has three parts. Hence, we cannot use MlpPolicy but MultiInputPolicy which
    # builds one small encoder per key in the dictionary, then concatenates all embeddings
    # before feeding them to the main policy and value networks
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        learning_starts=5_000,
        gamma=0.98,
        learning_rate=3e-4,
        tau=0.02,
        batch_size=512,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
        device='mps'
    )
    # Start training
    model.learn(total_timesteps=5_000_000)
    # Save model and normalization stats
    model.save('sac_model_dense')
    env.save('vec_normalize_sac_dense.pkl')
