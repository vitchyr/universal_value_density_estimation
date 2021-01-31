import functools

import sacred
import torch
import torch.nn.functional as f

from algorithms.agents.hindsight import uvd
from easy_launcher.core import run_experiment
from easy_logger import logger
from generative import rnvp
from workflow import reporting
from workflow import util

from paper_experiments.experiments.hindsight import fetch


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim = 500
        self._h1 = torch.nn.Linear(state_dim + action_dim, hdim)
        self._h2 = torch.nn.Linear(hdim, hdim)

        self._v_out = torch.nn.Linear(hdim, 1)

    def forward(self, states: torch.Tensor, actions: torch.tensor):
        x = f.leaky_relu(self._h1(torch.cat((states, actions), dim=1)))
        x = f.leaky_relu(self._h2(x))
        return self._v_out(x)


# noinspection PyUnresolvedReferences
class DensityEstimator(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, reward_factor: float, num_bijectors: int):
        super().__init__()
        self._reward_factor = reward_factor
        self._model = rnvp.SimpleRealNVP(2, state_dim + action_dim, 300, num_bijectors)

    def forward(self, goal: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        goal = torch.squeeze(goal, dim=1)[:, :2]
        states = torch.squeeze(states, dim=1)
        actions = torch.squeeze(actions, dim=1)
        context = torch.cat([states, actions], dim=1)
        # noinspection PyCallingNonCallable
        goal = goal - states[:, 6:8]
        goal_log_pdf = self._model(goal, context).sum(dim=1)
        return goal_log_pdf

    def reward(self, goal: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        # noinspection PyCallingNonCallable
        return self(goal, states, actions).exp() * self._reward_factor

def exp(variant):
    log_dir = logger.get_snapshot_dir()
    print('log_dir', log_dir)
    experiment = sacred.Experiment(
        name="Fetch - UVD",
        interactive=True,
        save_git_info=True,
        base_dir=log_dir,
    )

    # noinspection PyUnusedLocal
    @experiment.config
    def _config():
        discount_factor = 0.98
        policy_learning_rate = 8e-4
        critic_learning_rate = 8e-4
        density_learning_rate = 2e-4
        burnin = 10000
        batch_size = 512
        min_replay_size = 1000
        replay_size = 1500000
        density_replay_size = 50000
        target_update_step = 0.01
        exploration_noise = -2.3
        target_action_noise = 0.0
        env_name = 'FetchPush-v1'
        progressive_noise = False
        small_goal = False
        shuffle_goals = False
        sequence_length = 4
        num_bijectors = 6
        reward_factor = 0.02
        step_limit = 1.0
        # this is scaled for HER and UVD due to different computational speed. env_steps/iteration should be comparable
        num_envs = 1
        reward_factor = 0.1
        num_bijectors = 5
        policy_learning_rate = 2e-4
        critic_learning_rate = 2e-4
        small_goal_size = 0.005

    # noinspection DuplicatedCode
    @experiment.main
    def run(env_name: str, progressive_noise: bool, reward_factor: float, small_goal: bool, small_goal_size: float, num_bijectors: int, _config):
        device = torch.device('cuda:0')
        # target_dir = "/home/vitchyr/mnt2/log2/uvd/generated_data/algorithms"
        target_dir = log_dir
        reporting.register_global_reporter(experiment, target_dir)
        eval_env = fetch.make_env(env_name, progressive_noise, small_goal, small_goal_size)
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.shape[0]
        q1 = QNetwork(state_dim, action_dim).to(device)
        q2 = QNetwork(state_dim, action_dim).to(device)
        density_model = DensityEstimator(state_dim, action_dim, reward_factor, num_bijectors).to(device)
        policy = fetch.PolicyNetwork(state_dim, action_dim).to(device)
        params_parser = util.ConfigParser(uvd.UVDParams)
        params = params_parser.parse(_config)
        agent = uvd.UVDTD3(functools.partial(fetch.make_env, env_name, progressive_noise, small_goal), device, density_model, q1, q2,
                           policy, params)
        fetch.train_fetch(experiment, agent, eval_env, progressive_noise, small_goal)

    experiment.run()


import pythonplusplus.machine_learning.hyperparameter as hyp

def main():
    variant = dict(
        env_name='FetchPush-v1',
        progressive_noise=False,
        reward_factor=0.02,
        config={
            "discount_factor": 0.98,
            "policy_learning_rate": 8e-4,
            "critic_learning_rate": 8e-4,
            "density_learning_rate": 2e-4,
            "burnin": 10000,
            "batch_size": 512,
            "min_replay_size": 1000,
            "replay_size": 1500000,
            "density_replay_size": 50000,
            "target_update_step": 0.01,
            "exploration_noise": -2.3,
            "target_action_noise": 0.0,
            "env_name": "FetchPush-v1",
            "progressive_noise": False,
            "small_goal": False,
            "shuffle_goals": False,
            "sequence_length": 4,
            "num_bijectors": 6,
            "reward_factor": 0.02,
            "step_limit": 1.0
        },
    )
    n_seeds = 1
    # mode = 'local'
    mode = 'here_no_doodad'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'sss'
    # exp_name = __file__.split('/')[-1].split('.')[0].replace('_', '-')
    # print('exp_name', exp_name)

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['exp_id'] = exp_id
        for _ in range(n_seeds):
            run_experiment(
                exp,
                mode=mode,
                variant=variant,
                use_gpu=True,
                prepend_date_to_exp_prefix=True,
            )


if __name__ == '__main__':
    main()
