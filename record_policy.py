import os
import sys

os.environ["MUJOCO_GL"] = "egl"  # headless rendering; use "osmesa" if egl fails

import imageio
import mujoco
from stable_baselines3 import PPO

from leg_env import LegBalanceEnv

path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_leg_push70"
max_push = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out = sys.argv[4] if len(sys.argv) > 4 else "leg_balance"

policy = PPO.load(path, device="cpu")
env = LegBalanceEnv(max_push=max_push)
renderer = mujoco.Renderer(env.model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -8
cam.lookat[:] = [0.0, 0.0, 0.55]

obs, _ = env.reset(seed=seed)
frames = []
for _ in range(500):  # 10 s at 50 Hz
    action, _ = policy.predict(obs, deterministic=True)
    obs, _, fell, done, _ = env.step(action)
    cam.lookat[0] = env.data.qpos[0]  # follow the leg if it drifts
    renderer.update_scene(env.data, camera=cam)
    frames.append(renderer.render())
    if fell or done:
        obs, _ = env.reset()

fps = round(1 / env.dt)
imageio.mimsave(f"{out}.mp4", frames, fps=fps)
imageio.mimsave(f"{out}.gif", frames[::5], fps=fps // 5, loop=0)
renderer.close()
print(f"saved {out}.mp4 and {out}.gif ({len(frames)} frames)")
