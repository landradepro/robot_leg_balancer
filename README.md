# Robot Leg Balance

A simulated robot leg (ankle, knee, hip and torso) that **learns to stay upright when you push it**. No robotics experience is needed to follow this guide. You'll install everything, watch the leg in 3D, and train it yourself.

![demo](media/leg_balance.gif)

Full video: [media/leg_balance.mp4](media/leg_balance.mp4)

## What you'll learn

- How a robot is described to a physics simulator (MuJoCo).
- How a "reinforcement learning" agent (PPO) learns a skill by trial and error.
- How to measure whether it actually learned something.

## The idea in 60 seconds

1. **The leg.** A physics simulator ([MuJoCo](https://mujoco.org/)) models a foot, shank, thigh and torso with three motors (ankle, knee, hip).
2. **The baseline.** With no learning, the motors just hold the joints straight. This tips over from a tiny push (10 N forward, 0 N backward).
3. **The learner.** A neural network watches the leg's angles and speeds 50 times per second and nudges the motors. It's trained with PPO ([Stable-Baselines3](https://stable-baselines3.readthedocs.io/)). Each second alive earns reward; falling ends the episode.
4. **The pushes.** During training, random shoves hit the torso, so the network has to learn to recover.

After training, the leg survives **60 N pushes in both directions**.

## Step 1: What you need

- A computer with **Windows 11 + WSL2 (Ubuntu 22.04)**, or any Linux machine. No special GPU is required: training runs on the CPU.
- About 5 GB of free disk space (most of it is PyTorch).
- A GitHub account is only needed if you want to save your own copy.

**Windows only: install WSL2 with Ubuntu.** Open PowerShell as administrator:
```powershell
wsl --install -d Ubuntu-22.04
```
Restart if asked, then open **Ubuntu 22.04** from the Start menu. Every command below runs inside that Ubuntu terminal. Windows 11 shows the simulator windows on your desktop automatically (this feature is called WSLg).

## Step 2: Install the system packages

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git libegl1 libgl1 libosmesa6 ffmpeg
```

## Step 3: Download the project

```bash
git clone https://github.com/landradepro/robot_leg_balancer.git
cd robot_leg_balancer
```

## Step 4: Create a Python environment and install the libraries

A "virtual environment" keeps this project's libraries separate from everything else on your computer.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "gymnasium[mujoco]" stable-baselines3 matplotlib imageio imageio-ffmpeg tensorboard
```
This takes a few minutes. **Every time you open a new terminal, go back to the project folder and run `source .venv/bin/activate` again.** Your prompt starts with `(.venv)` when it's active.

Check that it worked:
```bash
python -c "import mujoco, gymnasium, stable_baselines3; print('all good')"
```

## Step 5: Watch the untrained leg

```bash
python view_leg.py
```
A 3D window opens with the leg standing. To push it:
1. Double-click the blue torso.
2. Hold **Ctrl** and **right-drag**.

It should tip over from even gentle pushes. Close the window when done.

## Step 6: Watch the trained leg

```bash
python watch_policy.py models/ppo_leg_push70
```
Random pushes hit it every 1 to 3 seconds and it recovers each time. Push it yourself the same way as before, and compare with Step 5.

## Step 7: Measure how good it is

```bash
python eval_policy.py
python eval_policy.py models/ppo_leg_push70
```
The first line tests the untrained baseline. The second tests the trained policy. Each test applies a single 0.1 s shove to the torso, then checks whether the leg is still standing 9 seconds later. It raises the force in 10 N steps until the leg falls, and repeats each test with 3 different random starts.

My results:

| Model | Trained on pushes up to | Survives forward | Survives backward |
|---|---|---|---|
| No learning (baseline) | none | 10 N | 0 N |
| `models/ppo_leg_push40.zip` | 40 N | 40 N | 50 N |
| `models/ppo_leg_push70.zip` | 70 N (fine-tuned from the 40 N model) | 60 N | 60 N |

## Step 8: Train your own

```bash
python train.py 40 2000000
```
- `40` is the maximum push force in newtons during training.
- `2000000` is how many simulation steps to train for. This takes roughly 8 to 10 minutes on a laptop CPU.

Watch the table it prints. **`ep_len_mean`** is how many steps the leg stays up per episode, and **500 is the maximum** (10 seconds without falling). It should climb toward 500. The model is saved as `runs/ppo_leg_push40.zip`, and it's also saved if you stop early with Ctrl+C.

Then test it:
```bash
python eval_policy.py runs/ppo_leg_push40
python watch_policy.py runs/ppo_leg_push40
```

**Make it tougher (curriculum training).** Trying to learn from very hard pushes right away fails, so start easy and fine-tune the successful model on harder pushes:
```bash
python finetune.py runs/ppo_leg_push40 70 1500000
python eval_policy.py runs/ppo_leg_push70
```

**Optional: training graphs.** In a second terminal (with the environment activated), run this and open http://localhost:6006 in your browser:
```bash
tensorboard --logdir runs/tb
```

## Step 9: Record a video

```bash
python record_policy.py models/ppo_leg_push70 60 0 leg_balance
```
This creates `leg_balance.mp4` and `leg_balance.gif`. The numbers are the model, the maximum push, and the random seed. Change the seed to get different pushes.

## Things to try

- **Make the foot smaller** in `leg.xml` (`size="0.06 0.05 0.03"` on the foot) and retrain. It gets much harder to balance.
- **Make the ankle weaker** by lowering `forcerange` on the ankle motor.
- **Change the reward** in `leg_env.py` (the `reward = ...` lines in `step`) and see how the behaviour changes.

## Project files

| File | What it does |
|---|---|
| `leg.xml` | The robot: bodies, joints, motors (MuJoCo format) |
| `leg_env.py` | The "game" the learner plays: observations, actions, reward, pushes |
| `train.py` | Trains a policy from scratch |
| `finetune.py` | Continues training an existing policy on harder pushes |
| `eval_policy.py` | Measures the largest push a policy survives |
| `watch_policy.py` | Live 3D view of a trained policy |
| `view_leg.py` | Live 3D view of the untrained baseline |
| `push_test.py` | Push test for the untrained baseline (no learning involved) |
| `record_policy.py` | Saves a video and GIF of a policy |
| `test_hopper.py`, `view_hopper.py` | Setup checks using Gymnasium's built-in Hopper |
| `models/` | Trained policies |

## Troubleshooting

- **No window opens on Windows.** In PowerShell run `wsl --update`, then `wsl --shutdown`, and reopen Ubuntu. Check that `echo $DISPLAY` prints something like `:0`.
- **A graphics error when opening a window.** Run `sudo apt install -y libglfw3 libxkbcommon-x11-0 mesa-utils`. Or try `LIBGL_ALWAYS_SOFTWARE=1 python watch_policy.py models/ppo_leg_push70`.
- **EGL error when recording.** Change `"egl"` to `"osmesa"` near the top of `record_policy.py`.
- **A harmless traceback mentioning `EGLError` at the very end of recording.** If the files were saved, ignore it.
- **`ModuleNotFoundError`.** You probably forgot `source .venv/bin/activate`.
- **Training fails with "start a new process before the current process has finished its bootstrapping phase".** Your training script code needs to be inside an `if __name__ == "__main__":` block, as in `train.py`.

## Lessons from building this

- Defining a "fall" only by torso height was a bug: it counted a normal crouch as falling, so the policy could never learn to crouch into a push. Using the torso tilt angle fixed it.
- Training directly on very hard pushes (up to 100 N) learned nothing useful. Starting at 40 N and increasing the difficulty worked.

