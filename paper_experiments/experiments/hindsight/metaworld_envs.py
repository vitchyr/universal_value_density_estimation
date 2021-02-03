import os
import pickle
from collections import OrderedDict

import gym
import metaworld
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.envs.mujoco.sawyer_xyz import v2
import mujoco_py
import numpy as np
from gym.envs.registration import register


def register_metaworld_envs():
    register(
        id='SawyerWindow-v0',
        entry_point='paper_experiments.experiments.hindsight.metaworld_envs:SawyerWindow',
        max_episode_steps=100,
    )
    register(
        id='SawyerFaucet-v0',
        entry_point='paper_experiments.experiments.hindsight.metaworld_envs:SawyerFaucet',
        max_episode_steps=100,
    )


class CustomMetaWorldEnv(MujocoEnv):
    """Hack to add some hooks in.

    Inherit Mujoco to be a friend and override some of the methods."""
    def get_image(self, width=84, height=84, camera_name=None):
        if len(self.sim.render_contexts) == 0:
            if 'gpu_id' in os.environ:
                device_id =int(os.environ['gpu_id'])
            else:
                device_id = - 1
            viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=device_id)
            # for off-screen rendering
            self.camera_init(viewer.cam)
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )[::-1,:,:]

    @staticmethod
    def camera_init(camera):
        # defaults
        distance = 2.342908832125616
        elevation = -45
        azimuth = 90
        look_at = [0, 0.59, 0.098]
        trackbodyid = -1

        # behind the shoulder
        distance = 1.75
        rotation_angle = 135
        elevation = -30
        # look_at = [0, 0.59, 0.098]
        look_at = [0., 0.5, 0.05]

        for i in range(3):
            camera.lookat[i] = look_at[i]
        camera.distance = distance
        camera.elevation = elevation
        camera.azimuth = rotation_angle
        camera.trackbodyid = trackbodyid

    def viewer_setup(self):
        # for GUI rendering
        self.camera_init(self.viewer.cam)

    def sample_goals(self, batch_size):
        return {
            'desired_goal': np.tile(
                self.fixed_goal,
                (batch_size, 1)
            )
        }

    @staticmethod
    def state_to_goal(states):
        return states

    @property
    def observation_space(self):
        old_obs_space = super().observation_space
        new_obs_space = gym.spaces.Box(
            low=old_obs_space.low[:6],
            high=old_obs_space.high[:6],
        )
        new_goal_space = gym.spaces.Box(
            low=np.hstack((old_obs_space.low[:3], self.goal_space.low)),
            high=np.hstack((old_obs_space.high[:3], self.goal_space.high)),
        )
        return gym.spaces.Dict({
            'observation': new_obs_space,
            'desired_goal': new_goal_space,
            'achieved_goal': new_goal_space,
        })

    def reset(self):
        obs = super().reset()
        return self._create_dict_obs(obs)

    def _create_dict_obs(self, obs):
        obs = obs[:6]
        return dict(
            observation=obs,
            desired_goal=self.fixed_goal.copy(),
            achieved_goal=self.state_to_goal(obs),
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)
        new_obs = self._create_dict_obs(obs)
        new_reward = np.linalg.norm(new_obs['achieved_goal'] - new_obs['desired_goal'])
        new_info = self._update_info(new_obs, info)
        return new_obs, new_reward, done, new_info

    def _update_info(self, obs, info):
        new_info = info.copy()
        achieved = obs['achieved_goal']
        goal = obs['desired_goal']
        difference = achieved - goal
        hand_difference = difference[..., :3]
        obj_difference = difference[..., 3:]

        new_info['distance'] = np.linalg.norm(difference)
        new_info['distance/hand'] = np.linalg.norm(hand_difference)
        new_info['distance/obj'] = np.linalg.norm(obj_difference)
        return new_info


class SawyerWindow(CustomMetaWorldEnv, v2.SawyerWindowCloseEnvV2):
    """Wrapper for the sawyer_window task."""

    def __init__(
            self,
            fixed_goal_offset=(0, 0, 0),
            fixed_goal=(
                    0.32,
                    0.66,
                    0.07,
                    0.11542878, 0.69000003, 0.16),
    ):
        super(SawyerWindow, self).__init__()
        self.random_init = False
        data = {
            'env_cls': SawyerWindow,
            'rand_vec': np.array(fixed_goal_offset),
            'partially_observable': True,
        }
        task = metaworld.Task(
            env_name='drawer-open-v3',
            data=pickle.dumps(data),
        )
        self.set_task(task)
        self.fixed_goal = np.array(fixed_goal)

    def reset_model(self):
        self._reset_hand()

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()

        self._target_pos = self.fixed_goal[-3:]
        # put it out of frame for now
        # self._target_pos = self.fixed_goal[-3:] + np.array([0, 0, 99])

        self.sim.model.body_pos[self.model.body_name2id(
            'window'
        )] = self.obj_init_pos
        self.data.set_joint_qpos('window_slide', 0.2)

        return self._get_obs()


class SawyerFaucet(CustomMetaWorldEnv, v2.SawyerFaucetOpenEnvV2):
    """Wrapper for the sawyer_faucet task."""

    def __init__(
            self,
            random_init=False,
            fixed_goal=(
                    0., 0.625, 0.125,
                    0.175, 0.8, 0.125,
            ),
    ):
        super(SawyerFaucet, self).__init__()
        self.random_init = random_init
        data = {
            'env_cls': SawyerFaucet,
            'rand_vec': np.array(fixed_goal),
            'partially_observable': True,
        }
        task = metaworld.Task(
            env_name='unused',
            data=pickle.dumps(data),
        )
        self.set_task(task)
        self.fixed_goal = np.array(fixed_goal)

    def reset_model(self):
        self._reset_hand()

        # Compute faucet position
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        self.sim.model.body_pos[self.model.body_name2id(
            'faucetBase'
        )] = self.obj_init_pos

        self._target_pos = self.fixed_goal[-3:]

        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()
