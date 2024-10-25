import os 
import torch.nn as nn
import datetime
import numpy as np

from lunar_lander import LunarLander

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor







def train(algorithm, train_env, evaluation_env, algorithm_kwargs, num_of_runs=5, log_folder=None, train_steps=1_000_000, evaluation_episodes=10):
    """
    Train and evaluate a reinforcement learning algorithm over multiple runs.

    Parameters:
    - algorithm: The RL algorithm class to be trained (e.g., PPO, DQN).
    - train_env: The training environment to use.
    - evaluation_env: The environment to use for evaluation.
    - algorithm_kwargs: Additional keyword arguments for initializing the algorithm.
    - num_of_runs: Number of independent training runs (default: 5).
    - log_folder: Directory to store logs and trained model weights (default: None [date and time]).
    - train_steps: Number of training timesteps per run (default: 1,000,000).
    - evaluation_episodes: Number of episodes for evaluation (default: 10).

    Returns:
    - means: List of mean rewards obtained in each run.
    - stds: List of standard deviations of rewards obtained in each run.
    """
    
    if log_folder is None:
        log_folder = str(datetime.datetime.now()).replace(" ", "_")

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    means = []
    stds = []

    for run in range(1, num_of_runs + 1):
        model = algorithm(env = train_env, tensorboard_log=os.path.join(log_folder, "run_{}_tensorboard".format(run)), **algorithm_kwargs) # init
        model.learn(total_timesteps = train_steps, progress_bar=True) # learn

        rewards_mean, rewards_std = evaluate_policy(model, evaluation_env, n_eval_episodes=evaluation_episodes) # evaluate
        print("RUN {}| reward_mean - {:.3f}\treward_std - {:.3f}".format(run, rewards_mean, rewards_std))

        means.append(rewards_mean)
        stds.append(rewards_std)
        model.save(os.path.join(log_folder, "run_{}_weights".format(run))) # save

    print("Mean reward from {} runs: {}".format(num_of_runs, np.mean(means)))
    print("Std of mean reward from {} runs: {}".format(num_of_runs, np.std(means)))
    return means, stds




if __name__ == "__main__":

    CURRENT_RUN_FOLDER = "./test_run"
    if not os.path.exists(CURRENT_RUN_FOLDER):
        os.mkdir(CURRENT_RUN_FOLDER)
    env_kwargs = {
        "render_mode" : "None",
    }

    train_env = make_vec_env(LunarLander, n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs, seed=2) # Make 8 parallel training envs

    evaluation_env =  Monitor(LunarLander(**env_kwargs), filename=os.path.join(CURRENT_RUN_FOLDER, "eval_env_monitor.log")) # Wrap eval environment into monitor to keep results


    algorithm_kwargs = {
        "policy" : "MlpPolicy",
        "policy_kwargs": {
            "net_arch" : dict(pi=[16,16], vf=[16,16]),
            "activation_fn" : nn.ReLU,
        },
        "learning_rate" : 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "gae_lambda": 0.98,
        "gamma": 0.999,
        "n_epochs": 4,
        "ent_coef": 0.01,
    }

    train(PPO, train_env, evaluation_env, algorithm_kwargs, log_folder=CURRENT_RUN_FOLDER)
    
