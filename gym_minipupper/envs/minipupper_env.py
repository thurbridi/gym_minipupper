from cmath import inf
import numpy as np
import gym

from gym_minipupper.utils import get_project_root

from gym import spaces
from pathlib import Path
from itertools import product

import pybullet as p
import pybullet_data
import stable_baselines3 as sb3


class MinipupperEnv(gym.Env):
    def __init__(self, render=True):
        JOINT_FRICTION = 0.1
        self.MAX_MOTOR_FORCE = 5000

        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10.0)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf")
        self.pupper = p.loadURDF(
            (
                get_project_root()
                / Path("gym_minipupper/minipupper_description/urdf/minipupper.urdf")
            )
            .absolute()
            .__str__()
        )

        self.pupper_joint_names = [
            "".join(joint_name)
            for joint_name in product(
                ["r", "l"],
                ["f_", "h_"],
                ["hip_joint", "upper_leg_joint", "lower_leg_joint"],
            )
        ]
        self.pupper_joint_ids = [
            id
            for id in range(p.getNumJoints(self.pupper))
            if p.getJointInfo(self.pupper, id)[1].decode("utf8")
            in self.pupper_joint_names
        ]

        self.action_space = spaces.Box(
            low=np.array([-np.pi / 2] * len(self.pupper_joint_names)),
            high=np.array([np.pi / 2] * len(self.pupper_joint_names)),
            dtype=np.float32,
        )
        # TODO: Use gym.spaces.Dict
        self.observation_space = spaces.Box(
            low=np.array([-inf, -inf, -inf, -np.pi, -np.pi, -np.pi]),
            high=np.array([inf, inf, inf, np.pi, np.pi, np.pi]),
            dtype=np.float32,
        )

        p.setJointMotorControlArray(
            self.pupper,
            self.pupper_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=np.zeros(len(self.pupper_joint_ids)),
            forces=np.zeros(len(self.pupper_joint_ids), dtype=np.float32)
            + JOINT_FRICTION,
        )

    def step(self, action):
        p.setJointMotorControlArray(
            self.pupper,
            self.pupper_joint_ids,
            p.POSITION_CONTROL,
            targetPositions=action,
            forces=np.zeros(len(self.pupper_joint_ids)) + self.MAX_MOTOR_FORCE,
        )
        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = bool(np.any(np.abs(obs[-3:-1]) > np.pi / 2, axis=None))
        info = self._get_info()

        return obs, reward, done, info

    def reset(self):

        startPos = [0, 0, 0.1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.pupper, startPos, startOrientation)

        obs = self._get_obs()

        return obs

    def render(self):
        raise NotImplementedError

    def close(self):
        p.disconnect()

    def _get_obs(self):
        pos, angle = p.getBasePositionAndOrientation(self.pupper)
        angle = p.getEulerFromQuaternion(angle)

        return np.array([*pos, *angle], dtype=np.float32)

    def _get_info(self):
        return dict()

    def _get_reward(self, state):
        reward = state[0]

        return reward


if __name__ == "__main__":
    env = MinipupperEnv()

    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            break

    env.close()
