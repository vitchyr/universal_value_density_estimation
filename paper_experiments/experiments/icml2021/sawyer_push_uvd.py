import pythonplusplus.machine_learning.hyperparameter as hyp
from easy_launcher.core import run_experiment


def main():
    variant = dict(
        small_goal=False,
        small_goal_size=0.005,
        discount_factor=0.98,
        policy_learning_rate=8e-4,
        critic_learning_rate=8e-4,
        density_learning_rate=2e-4,
        burnin=10000,
        # burnin=10,
        batch_size=256,
        min_replay_size=1000,
        # min_replay_size=10,
        replay_size=1000000,
        density_replay_size=50000,
        target_update_step=0.01,
        exploration_noise=-2.3,
        target_action_noise=0.0,
        progressive_noise=False,
        shuffle_goals=False,
        sequence_length=4,
        num_bijectors=6,
        reward_factor=0.02,
        step_limit=1.0,
    )
    n_seeds = 1
    mode = 'local'
    # mode = 'here_no_doodad'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    target_script = '/home/vitchyr/git/universal_value_density_estimation/paper_experiments/experiments/icml2021/sawyer_push_script.py'

    n_seeds = 4
    mode = 'sss'
    exp_name = 'icml2021--' + __file__.split('/')[-1].split('.')[0].replace('_', '-')
    print('exp_name', exp_name)

    search_space = {
        'env_name': [
            'SawyerPush-FixedInit-FixedGoal-x0p15-y0p7-v0',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['exp_id'] = exp_id
        if mode != 'local':
            target_script = target_script.replace('/home/vitchyr/', '/global/home/users/vitchyr/')
        for _ in range(n_seeds):
            run_experiment(
                None,
                target_script=target_script,
                exp_name='dev-uvd',
                mode=mode,
                variant=variant,
                use_gpu=True,
                gpu_id=1,
                prepend_date_to_exp_name=True,
                time_in_mins=3*24*60 - 1,
            )


if __name__ == '__main__':
    main()
