import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from leg_env import LegBalanceEnv

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_leg_push40"
    max_push = float(sys.argv[2]) if len(sys.argv) > 2 else 70.0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1_500_000
    name = f"runs/ppo_leg_push{int(max_push)}"

    env = make_vec_env(
        LegBalanceEnv, n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"max_push": max_push}
    )
    model = PPO.load(base, env=env, device="cpu")
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
    finally:
        model.save(name)
        print(f"saved {name}.zip")
