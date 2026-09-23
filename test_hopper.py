import os
os.environ["MUJOCO_GL"] = "egl"  # headless rendering; use "osmesa" if egl fails
import gymnasium as gym
import imageio

env = gym.make("Hopper-v5", render_mode="rgb_array")
env.reset(seed=0)
frames = []
for _ in range(300):
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    frames.append(env.render())
    if terminated or truncated:
        env.reset()
imageio.mimsave("hopper_random.mp4", frames, fps=30)
print("saved hopper_random.mp4")
env.close()
