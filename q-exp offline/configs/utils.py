import os
import copy
import itertools
from pathlib import Path
from default import DEFAULT_AGENT, DEFAULT_ENV
from best_setting import BEST_AGENT

def write_to_file(cmds, prev_file=0, line_per_file=1):
    curr_dir = os.getcwd()
    cmd_file_path = os.path.join(curr_dir, "scripts/tasks_{}.sh")

    cmd_file = Path(cmd_file_path.format(int(prev_file)))
    cmd_file.parent.mkdir(exist_ok=True, parents=True)

    count = 0
    print("First script:", cmd_file)
    file = open(cmd_file, 'w')
    for cmd in cmds:
        file.write(cmd + "\n")
        count += 1
        if count % line_per_file == 0:
            file.close()
            prev_file += 1
            cmd_file = Path(cmd_file_path.format(int(prev_file)))
            file = open(cmd_file, 'w')
    if not file.closed:
        file.close()
    print("Last script:", cmd_file_path.format(str(prev_file)), "\n")


def generate_cmd(flag_collection, base="python run_ac_offline.py "):
    cmd = base
    for k, v in flag_collection.items():
        cmd += "{} {}".format(k, v)
    cmd += "\n"
    return cmd

# def generate_flag_combinations(sweep_parameters):
def write_job_scripts(sweep_params, target_agents, target_envs, target_datasets, target_distributions, num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1):
    agent_parameters = copy.deepcopy(DEFAULT_AGENT)
    cmds = []
    aedd_comb = list(itertools.product(target_agents, target_envs, target_datasets, target_distributions))
    for aedd in aedd_comb:
        agent, env, dataset, dist = aedd
        kwargs = {}
        kwargs[" --agent "] = agent
        kwargs[" --env_name "] = env
        kwargs[" --dataset "] = dataset
        kwargs[" --distribution "] = dist
        kwargs.update(DEFAULT_ENV[env])

        default_params = agent_parameters["{}-{}".format(env, dataset)][agent]
        settings = {**sweep_params, **default_params}

        keys, values = zip(*settings.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            kwargs[" --param "] = comb_num + comb_num_base
            for (k, v) in param_comb.items():
                kwargs[k] = v
            for run in list(range(run_base, run_base+num_runs)):
                kwargs[" --seed "] = run
                cmds.append(generate_cmd(kwargs))
    write_to_file(cmds, prev_file=prev_file, line_per_file=line_per_file)

def add_seed_scripts(sweep_params, target_agents, target_envs, target_datasets, target_distributions,
                     defined_param=BEST_AGENT, num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1):
    agent_parameters = copy.deepcopy(defined_param)
    cmds = []
    aedd_comb = list(itertools.product(target_agents, target_envs, target_datasets, target_distributions))
    for aedd in aedd_comb:
        agent, env, dataset, dist = aedd
        kwargs = {}
        kwargs[" --agent "] = agent
        kwargs[" --env_name "] = env
        kwargs[" --dataset "] = dataset
        kwargs[" --distribution "] = dist
        kwargs.update(DEFAULT_ENV[env])

        default_params = agent_parameters["{}-{}-{}".format(env, dataset, dist)][agent]
        settings = {**sweep_params, **default_params}

        keys, values = zip(*settings.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            kwargs[" --param "] = comb_num + comb_num_base
            for (k, v) in param_comb.items():
                kwargs[k] = v
            for run in list(range(run_base, run_base+num_runs)):
                kwargs[" --seed "] = run
                cmds.append(generate_cmd(kwargs))
    write_to_file(cmds, prev_file=prev_file, line_per_file=line_per_file)
