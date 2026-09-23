import sys

import numpy as np
from stable_baselines3 import PPO

from leg_env import LegBalanceEnv

policy = PPO.load(sys.argv[1], device="cpu") if len(sys.argv) > 1 else None
env = LegBalanceEnv()


def survives(force, seed):
    obs, _ = env.reset(seed=seed, options={"push_force": force, "push_time": 1.0})
    while True:
        action = np.zeros(3) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, _, fell, done, _ = env.step(action)
        if fell:
            return False
        if done:
            return True


def survives_all_seeds(force):
    return all(survives(force, seed) for seed in range(3))


print("Policy:", sys.argv[1] if policy else "PD baseline (zero action)")
for direction, name in [(+1, "forward"), (-1, "backward")]:
    last_ok = 0
    for f in range(10, 500, 10):
        if survives_all_seeds(direction * f):
            last_ok = f
        else:
            break
    print(f"  survives a {last_ok} N push {name}")
