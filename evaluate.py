import torch.nn as nn
import sys
import os
from stable_baselines3 import PPO
from lunar_lander import LunarLander







if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate.py log_dir run_number")
        sys.exit()
    env = LunarLander(render_mode="human", main_engine_power=25.0, side_engine_power=0.1)
    model = PPO.load(os.path.join(sys.argv[1],"run_{}_weights.zip".format(sys.argv[2])))


    obs, info = env.reset()
    step = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, trunc, info = env.step(action)
        
        if term or trunc:
            break

        step += 1
  

    

