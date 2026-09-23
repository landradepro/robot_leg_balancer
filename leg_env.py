import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leg.xml")


class LegBalanceEnv(gym.Env):
    """Planar hip-knee-ankle leg that must stay upright while being pushed."""

    PUSH_STEPS = 5  # 5 control steps = 0.1 s
    ACTION_SCALE = np.array([0.5, 0.6, 0.8])  # rad of correction: ankle, knee, hip

    def __init__(self, max_push=100.0, episode_seconds=10.0, random_pushes=True):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.torso = self.model.body("torso").id
        self.frame_skip = 10  # 10 x 0.002 s = 50 Hz control
        self.dt = self.frame_skip * self.model.opt.timestep
        self.max_steps = int(episode_seconds / self.dt)
        self.max_push = max_push
        self.random_pushes = random_pushes
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

    def _obs(self):
        q, qd = self.data.qpos, self.data.qvel
        torso_pitch = q[2:6].sum()  # all hinges share the y axis, so angles add up
        return np.concatenate(
            [[q[1], torso_pitch], q[2:6], qd, self.last_action]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.uniform(-0.02, 0.02, 3)
        self.data.qpos[3] = noise[0]  # ankle
        self.data.qpos[4] = abs(noise[1])  # knee (cannot go negative)
        self.data.qpos[5] = noise[2]  # hip
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.push_left = 0
        self.push_now = 0.0
        self.was_pushing = False
        self.last_action = np.zeros(3)
        self.fixed_force = options.get("push_force")  # used for evaluation
        if self.fixed_force is not None:
            self.next_push_step = int(options.get("push_time", 1.0) / self.dt)
        elif self.random_pushes:
            self.next_push_step = int(self.np_random.integers(25, 100))
        else:
            self.next_push_step = -1
        return self._obs(), {}

    def _push_force(self):
        force = 0.0
        if self.push_left > 0:
            force = self.push_now
            self.push_left -= 1
            if self.push_left == 0 and self.fixed_force is None and self.random_pushes:
                self.next_push_step = self.step_count + int(self.np_random.integers(50, 150))
        elif 0 <= self.next_push_step <= self.step_count:
            if self.fixed_force is not None:
                self.push_now = self.fixed_force
            else:
                self.push_now = self.np_random.uniform(-self.max_push, self.max_push)
            force = self.push_now
            self.push_left = self.PUSH_STEPS - 1
            self.next_push_step = -1
        return force

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = np.clip(self.ACTION_SCALE * action, self.ctrl_low, self.ctrl_high)

        force = self._push_force()
        if force != 0.0 or self.was_pushing:
            self.data.xfrc_applied[self.torso, 0] = force
        self.was_pushing = force != 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        q, qd = self.data.qpos, self.data.qvel
        torso_pitch = q[2:6].sum()
        fell = abs(torso_pitch) > 0.8 or self.data.xpos[self.torso, 2] < 0.4
        # fallen = tipped past ~46 degrees, or genuinely collapsed (not just crouched)

        reward = (
            1.0
            - 2.0 * torso_pitch**2
            - 1.0 * q[2] ** 2  # foot pitch: keep the foot flat
            - 0.1 * qd[0] ** 2  # do not slide away
            - 0.01 * np.sum(action**2)
            - 0.05 * np.sum((action - self.last_action) ** 2)
        )
        if fell:
            reward -= 5.0
        self.last_action = action

        truncated = self.step_count >= self.max_steps
        return self._obs(), float(reward), bool(fell), truncated, {}
