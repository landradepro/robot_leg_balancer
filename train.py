import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from leg_env import LegBalanceEnv

if __name__ == "__main__":
    max_push = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000
    name = f"runs/ppo_leg_push{int(max_push)}"

    os.makedirs("runs", exist_ok=True)
    env = make_vec_env(
        LegBalanceEnv, n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"max_push": max_push}
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu",
        verbose=1,
        tensorboard_log="runs/tb",
    )
    try:
        model.learn(total_timesteps=steps)
    finally:
        model.save(name)
        print(f"saved {name}.zip")
