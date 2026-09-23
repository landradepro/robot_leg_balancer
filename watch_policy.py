import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

from leg_env import LegBalanceEnv

policy = PPO.load(sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_leg", device="cpu")
env = LegBalanceEnv()
obs, _ = env.reset(seed=0)

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        action, _ = policy.predict(obs, deterministic=True)
        with viewer.lock():
            obs, _, fell, done, _ = env.step(action)
            if fell or done:
                obs, _ = env.reset()
        viewer.sync()
        time.sleep(max(0.0, env.dt - (time.time() - t0)))
