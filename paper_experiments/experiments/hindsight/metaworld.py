from typing import Any

import gym
import gym.spaces.box
import numpy as np
import sacred
import torch
import tqdm

from algorithms import environment
from algorithms.agents.hindsight import her_td3
from generative import rnvp
from workflow import reporting

NUM_ITERS = int(3e6)
MAX_PATH_LEN = 100
GOAL_DIM = 6
NUM_TRAJS_PER_EPOCH = 50


class NoNormalizer:
    def normalize_state(self, state):
        return state

    def denormalize_goal(self, goal):
        return goal


class DensityEstimator(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, reward_factor: float,
                 num_bijectors: int):
        super().__init__()
        self._reward_factor = reward_factor
        self._model = rnvp.SimpleRealNVP(GOAL_DIM, state_dim + action_dim, 300,
                                         num_bijectors)

    def forward(self, goal: torch.Tensor, states: torch.Tensor,
                actions: torch.Tensor):
        goal = torch.squeeze(goal, dim=1)[:, :GOAL_DIM]
        states = torch.squeeze(states, dim=1)
        actions = torch.squeeze(actions, dim=1)
        context = torch.cat([states, actions], dim=1)
        # predict relative to current state, similar to author's Fetch code
        # first GOAL_DIM are the goal, next GOAL_DIM are the achieved
        goal = goal - states[:, GOAL_DIM:]
        goal_log_pdf = self._model(goal, context).sum(dim=1)
        return goal_log_pdf

    def reward(self, goal: torch.Tensor, states: torch.Tensor,
               actions: torch.Tensor):
        # noinspection PyCallingNonCallable
        return self(goal, states, actions).exp() * self._reward_factor


class MetaworldEnv(environment.GymEnv):
    goal_dim = GOAL_DIM

    def __init__(
            self, env_name: str, progressive_noise: bool, small_goal: bool,
            small_goal_size: float = 0.005
    ):
        from paper_experiments.experiments.hindsight.metaworld_envs import (
            register_metaworld_envs
        )
        from gym.envs.registration import registry
        if env_name not in registry.env_specs:
            register_metaworld_envs()
        self._env = gym.make(env_name)
        self._normalizer = NoNormalizer()
        self._env.seed(np.random.randint(10000) * 2)
        self._progressive_noise = progressive_noise
        super().__init__(self._env)
        self._obs_space = gym.spaces.box.Box(low=-np.inf, high=np.inf,
                                             shape=(12,))

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        state = super().reset()
        state = self._normalizer.normalize_state(state)
        self._state = np.concatenate(
            [state['desired_goal'], state['observation']])
        return self._state

    def replace_goals(self, transition_sequence: her_td3.HerTransitionSequence,
                      goals: torch.Tensor,
                      replacement_probability: float):
        replace_indices = torch.rand(transition_sequence.states.shape[0],
                                     1) < replacement_probability
        replace_indices = transition_sequence.states.new_tensor(replace_indices)
        transition_sequence.states[:, 0,
        :self.goal_dim] = replace_indices * goals + (
                    1 - replace_indices) * transition_sequence.states[:, 0,
                                           :self.goal_dim]
        transition_sequence.next_states[:, 0,
        :self.goal_dim] = replace_indices * goals + (
                    1 - replace_indices) * transition_sequence.next_states[:, 0,
                                           :self.goal_dim]
        transition_sequence.rewards[:, 0] = transition_sequence.rewards.new(
            self._env.compute_reward(
                self._normalizer.denormalize_goal(goals.cpu().numpy()),
                self._normalizer.denormalize_goal(
                    transition_sequence.states[:, 0,
                    :self.goal_dim].cpu().numpy()), None))
        return her_td3.HerTransitionSequence(
            states=transition_sequence.states.detach(),
            actions=transition_sequence.actions.detach(),
            rewards=transition_sequence.rewards.detach(),
            next_states=transition_sequence.next_states.detach(),
            timeout_weight=transition_sequence.timeout_weight.detach(),
            terminal_weight=transition_sequence.terminal_weight.detach(),
            action_log_prob=transition_sequence.action_log_prob.detach(),
            time_left=transition_sequence.time_left.detach(),
            achieved_goal=transition_sequence.achieved_goal.detach()
        )

    def step(self, action):
        if self._progressive_noise:
            state, original_reward, is_terminal, info = super().step(
                action + (action ** 2).mean() * np.random.randn(
                    self.action_dim) * np.exp(-1))
        else:
            state, original_reward, is_terminal, info = super().step(action)

        # if original_reward > -1:
        # print(state['achieved_goal'])
        # print(state['desired_goal'])
        state = self._normalizer.normalize_state(state)
        info['achieved_goal'] = state['achieved_goal']
        self._state = np.concatenate(
            [state['desired_goal'], state['observation']])

        return self._state, original_reward, is_terminal, info


def make_env(env_name: str, progressive_noise: bool, small_goal: bool,
             small_goal_size: float = 0.005) -> MetaworldEnv:
    return MetaworldEnv(env_name, progressive_noise, small_goal, small_goal_size)


def train_metaworld(experiment: sacred.Experiment, agent: Any, eval_env: MetaworldEnv, progressive_noise: bool, small_goal: bool):
    reporting.register_field("eval_final_success")
    reporting.register_field("eval_success_rate")
    keys = [
        'distance',
        'distance/obj',
        'distance/hand',
    ]
    for k in keys:
        new_k = "eval_{}".format(k.replace('/', '_'))
        reporting.register_field(new_k + '_mean')
        reporting.register_field(new_k + '_final')
    reporting.register_field("action_norm")
    reporting.finalize_fields()
    trange = tqdm.trange(NUM_ITERS, position=0, leave=True)
    for iteration in trange:
        if iteration % 10000 == 0:
            action_norms = []
            success_rate = 0
            final_success = 0
            distances = {k: 0 for k in keys}
            final_distances = {k: 0 for k in keys}
            n_steps_total = 0
            for i in range(NUM_TRAJS_PER_EPOCH):
                state = eval_env.reset()
                t = 0
                final_dist = 0
                while not eval_env.needs_reset:
                    action = agent.eval_action(state)
                    action_norms.append(np.linalg.norm(action))
                    state, reward, is_terminal, info = eval_env.step(action)
                    n_steps_total += 1
                    for k in keys:
                        distances[k] += info[k]
                        final_dist = info[k]
                    final_success = 0
                    t += 1
                    if reward > -1. or t == MAX_PATH_LEN:
                        success_rate += 1
                        final_success = 1
                        break
                final_distances[k] += final_dist
            reporting.iter_record("eval_final_success", final_success)
            reporting.iter_record("eval_success_rate", success_rate)
            for k in keys:
                new_k = "eval_{}".format(k.replace('/', '_'))
                reporting.iter_record(
                    new_k + '_mean',
                    distances[k] / n_steps_total
                )
                reporting.iter_record(
                    new_k + '_final',
                    final_distances[k] / NUM_TRAJS_PER_EPOCH
                )
            reporting.iter_record("action_norm", np.mean(action_norms).item())
        if iteration % 20000 == 0:
            policy_path = f"/tmp/policy_{iteration}"
            with open(policy_path, 'wb') as f:
                torch.save(agent.freeze_policy(torch.device('cpu')), f)
            experiment.add_artifact(policy_path)

        agent.update()
        reporting.iterate()
        trange.set_description(f"{iteration} -- " + reporting.get_description(
            ["return", "td_loss", "env_steps"]))


def show_ant():
    pass


if __name__ == '__main__':
    show_fetch()