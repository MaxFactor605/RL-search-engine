import os
import numpy as np
import torch.nn as nn

from lunar_lander import LunarLander
from train import train

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor








if __name__ == "__main__":
    designs = {
        'A': {'main_engine_power': 13.0, 'side_engine_power': 0.6},
        'B': {'main_engine_power': 5.0, 'side_engine_power': 2.0},
        'C': {'main_engine_power': 25.0, 'side_engine_power': 0.1},
    }


    best_design = None
    best_reward = float("-inf")


    for design_name, design in designs.items():
    
        log_folder = "Design_{}_log".format(design_name)
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        env_kwargs = {
            "render_mode" : "None",
            "main_engine_power" : design["main_engine_power"],
            "side_engine_power" : design["side_engine_power"],
        }

        train_env = make_vec_env(LunarLander, n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs, seed=2) # Make 8 parallel training envs

        evaluation_env =  Monitor(LunarLander(**env_kwargs), filename=os.path.join(log_folder, "eval_env_monitor.log")) # Wrap eval environment into monitor to keep results

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
        
        print("Training design {} with main_engine_power - {} and side_engine_power - {}".format(design_name, design["main_engine_power"], design["side_engine_power"]))
        means, stds = train(PPO, train_env, evaluation_env, algorithm_kwargs, log_folder=log_folder)
        print("Design {} | mean_reward {}\tstd_beatween_runs {}".format(design_name, np.mean(means), np.std(means)))
        if np.mean(means) > best_reward:
            best_design = design_name
            best_reward = np.mean(means)


    print("Best design - {} with reward - {}".format(best_design, best_reward))
    os.rename("Design_{}_log".format(best_design), "Design_{}_log_best".format(best_design)) # Mark best design
