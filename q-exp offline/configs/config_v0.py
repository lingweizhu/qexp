from best_setting import BEST_AGENT
from utils import write_job_scripts, add_seed_scripts#, add_param_scripts, policy_evolution_scripts

def sweep():
    sweep = {
        " --pi_lr ": [3e-3, 1e-3, 3e-4, 1e-4],
        " --q_lr_prob ": [1.],
        " --info ": ["reproduce"],
    }
    target_agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medrep", "medium"]
    target_distributions = ["HTqGaussian", "SGaussian", "Gaussian", "Beta", "Student"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, num_runs=5, comb_num_base=0, prev_file=0, line_per_file=1)


def more_seeds_for_the_best():
    sweep = {
        " --q_lr_prob ": [1.],
        " --info ": ["reproduce"],
    }
    target_agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medrep", "medium"]
    target_distributions = ["HTqGaussian", "SGaussian", "Gaussian", "Beta", "Student"]
    add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions,
                     defined_param=BEST_AGENT,
                     num_runs=5, run_base=5, comb_num_base=0, prev_file=5100, line_per_file=1)

if __name__ == "__main__":
    sweep()
    more_seeds_for_the_best()