import os
import argparse

import core.environment.env_factory as environment
from core.utils import torch_utils, logger, run_funcs
from core.agent.in_sample import InSampleAC
from core.agent.iql import IQL
from core.agent.tsallis_awac import TKLPolicyInAC
from core.agent.td3_bc import TD3BC
from core.agent.awac import AWAC


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--load_network', default='', type=str)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--param', default=0, type=int)
    parser.add_argument('--info', default='test_v0', type=str)
    parser.add_argument('--env_name', default='Hopper', type=str)
    parser.add_argument('--dataset', default='medium', type=str)
    parser.add_argument('--discrete_control', default=0, type=int)
    parser.add_argument('--state_dim', default=11, type=int)
    parser.add_argument('--action_dim', default=3, type=int)
    parser.add_argument('--action_min', default=-1., type=float)
    parser.add_argument('--action_max', default=1., type=float)

    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--log_interval', default=10000, type=int)
    parser.add_argument('--pi_lr', default=3e-4, type=float)
    parser.add_argument('--q_lr_prob', default=1., type=float)
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--timeout', default=1000, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--use_target_network', default=1, type=int)
    parser.add_argument('--target_network_update_freq', default=1, type=int)
    parser.add_argument('--polyak', default=0.995, type=float)
    parser.add_argument('--evaluation_criteria', default='return', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--test_mode', default=0, type=int)

    parser.add_argument('--distribution', default='Gaussian', type=str)
    parser.add_argument('--distribution_param', default=2., type=float)

    parser.add_argument('--agent', default='IQL', type=str)
    parser.add_argument('--tsallis_q', default=2, type=int)
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--expectile', default=0.8, type=float)


    cfg = parser.parse_args()
    torch_utils.set_one_thread()
    torch_utils.random_seed(cfg.seed)
    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = "data/output/{}/{}/{}/{}/{}/{}_param/{}_run".format(cfg.info, cfg.env_name, cfg.dataset, cfg.agent,
                                                          cfg.distribution, cfg.param, cfg.seed)
    cfg.exp_path = os.path.join(project_root, exp_path)
    torch_utils.ensure_dir(cfg.exp_path)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)
    cfg.offline_data = run_funcs.load_testset(cfg.env_name, cfg.dataset, cfg.seed)
    cfg.q_lr = cfg.pi_lr * cfg.q_lr_prob

    # Setting up the logger
    cfg.logger = logger.Logger(cfg, cfg.exp_path)
    logger.log_config(cfg)

    # Initializing the agent and running the experiment
    if cfg.agent == "IQL":
        agent_obj = IQL(cfg)
    elif cfg.agent == "InAC":
        agent_obj = InSampleAC(cfg)
    elif cfg.agent == "TAWAC":
        agent_obj = TKLPolicyInAC(cfg)
    elif cfg.agent == "TD3BC":
        agent_obj = TD3BC(cfg)
    elif cfg.agent == "AWAC":
        agent_obj = AWAC(cfg)
    else:
        raise NotImplementedError

    run_funcs.run_steps(agent_obj, cfg.max_steps, cfg.log_interval, exp_path)
