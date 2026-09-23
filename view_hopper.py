import time
import gymnasium as gym

env = gym.make("Hopper-v5", render_mode="human")
env.reset(seed=0)
dt = env.unwrapped.dt  # seconds of simulated time per step

try:
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        time.sleep(dt)  # slow to real time
        if terminated or truncated:
            env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()
