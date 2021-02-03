import functools
import os

import sacred
from sacred.observers import file_storage
import torch
import torch.nn.functional as f

from algorithms.agents.hindsight import uvd
from easy_launcher.core import setup_logger
from easy_logger import logger
from generative import rnvp
from workflow import reporting
from workflow import util

from paper_experiments.experiments.hindsight import sawyer
import doodad as dd

import matplotlib

matplotlib.use('agg')


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim1 = 400
        hdim2 = 300
        self._h1 = torch.nn.Linear(state_dim + action_dim, hdim1)
        self._h2 = torch.nn.Linear(hdim1, hdim2)

        self._v_out = torch.nn.Linear(hdim2, 1)

    def forward(self, states: torch.Tensor, actions: torch.tensor):
        x = f.leaky_relu(self._h1(torch.cat((states, actions), dim=1)))
        x = f.leaky_relu(self._h2(x))
        return self._v_out(x)


# noinspection PyUnresolvedReferences
class DensityEstimator(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, reward_factor: float,
                 num_bijectors: int):
        super().__init__()
        self._reward_factor = reward_factor
        self._model = rnvp.SimpleRealNVP(2, state_dim + action_dim, 300,
                                         num_bijectors)

    def forward(self, goal: torch.Tensor, states: torch.Tensor,
                actions: torch.Tensor):
        goal = torch.squeeze(goal, dim=1)[:, :2]
        states = torch.squeeze(states, dim=1)
        actions = torch.squeeze(actions, dim=1)
        context = torch.cat([states, actions], dim=1)
        # noinspection PyCallingNonCallable
        goal = goal - states[:, 6:8]
        goal_log_pdf = self._model(goal, context).sum(dim=1)
        return goal_log_pdf

    def reward(self, goal: torch.Tensor, states: torch.Tensor,
               actions: torch.Tensor):
        # noinspection PyCallingNonCallable
        return self(goal, states, actions).exp() * self._reward_factor


args_dict = dd.get_args()
method_call = args_dict['method_call']
run_settings = args_dict['run_experiment_kwargs']
exp_name = run_settings['exp_name']
variant = run_settings['variant']
exp_id = run_settings['exp_id']
seed = run_settings['seed']
snapshot_mode = run_settings['snapshot_mode']
snapshot_gap = run_settings['snapshot_gap']
git_infos = run_settings['git_infos']
script_name = run_settings['script_name']
trial_dir_suffix = run_settings['trial_dir_suffix']
base_log_dir = run_settings['base_log_dir']
use_gpu = run_settings['use_gpu']
gpu_id = run_settings['gpu_id']


exp_dir = os.path.join(base_log_dir, exp_name)
experiment = sacred.Experiment(
    name=exp_name,
    interactive=True,
    save_git_info=True,
    base_dir=exp_dir,
)

# noinspection PyUnusedLocal
@experiment.config
def _config():
    # this is scaled for HER and UVD due to different computational speed. env_steps/iteration should be comparable
    num_envs = 1
    reward_factor = 0.1
    num_bijectors = 5
    policy_learning_rate = 2e-4
    critic_learning_rate = 2e-4
    small_goal_size = 0.005

experiment.add_config(variant)
observer = file_storage.FileStorageObserver(exp_dir)
experiment.observers.append(observer)


# noinspection DuplicatedCode
@experiment.automain
def run(env_name: str, progressive_noise: bool, reward_factor: float,
        small_goal: bool, small_goal_size: float, num_bijectors: int, _config):
    actual_log_dir = setup_logger(
        exp_name=exp_name,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=script_name,
        log_dir=observer.dir
    )
    print('exp dir', actual_log_dir)
    log_dir = logger.get_snapshot_dir()
    device = torch.device('cuda:{}'.format(gpu_id))
    target_dir = log_dir
    reporting.register_global_reporter(experiment, target_dir)
    eval_env = sawyer.make_env(env_name, progressive_noise, small_goal, small_goal_size)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)
    density_model = DensityEstimator(state_dim, action_dim, reward_factor, num_bijectors).to(device)
    policy = sawyer.PolicyNetwork(state_dim, action_dim).to(device)
    params_parser = util.ConfigParser(uvd.UVDParams)
    params = params_parser.parse(_config)
    agent = uvd.UVDTD3(functools.partial(sawyer.make_env, env_name, progressive_noise, small_goal), device, density_model, q1, q2,
                       policy, params)
    sawyer.train_sawyer(experiment, agent, eval_env, progressive_noise, small_goal)



