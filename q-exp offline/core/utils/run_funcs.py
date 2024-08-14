import pickle
import time
import copy
import numpy as np

import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import gzip
import torch
import matplotlib.pyplot as plt
from core.utils import torch_utils
from mpl_toolkits.mplot3d.axes3d import Axes3D

EARLYCUTOFF = "EarlyCutOff"

formal_dataset_name = {
    'medexp': 'Medium-Expert',
    'medium': 'Medium',
    'medrep': 'Medium-Replay',
}
formal_distribution_name = {
    "HTqGaussian": "Heavy-Tailed q-Gaussian",
    "SGaussian": "Squashed Gaussian",
    "Gaussian": "Gaussian",
    "Student": "Student's t",
    "Beta": "Beta"
}

def load_testset(env_name, dataset, id):
    path = None
    if env_name == 'HalfCheetah':
        if dataset == 'expert':
            path = {"env": "halfcheetah-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "halfcheetah-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "halfcheetah-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "halfcheetah-medium-replay-v2"}
    elif env_name == 'Walker2d':
        if dataset == 'expert':
            path = {"env": "walker2d-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "walker2d-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "walker2d-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "walker2d-medium-replay-v2"}
    elif env_name == 'Hopper':
        if dataset == 'expert':
            path = {"env": "hopper-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "hopper-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "hopper-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "hopper-medium-replay-v2"}
    elif env_name == 'Ant':
        if dataset == 'expert':
            path = {"env": "ant-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "ant-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "ant-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "ant-medium-replay-v2"}
    
    elif env_name == 'Acrobot':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/acrobot/transitions_50k/train_40k/{}_run.pkl".format(id)}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/acrobot/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    elif env_name == 'LunarLander':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/lunar_lander/transitions_50k/train_500k/{}_run.pkl".format(id)}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/lunar_lander/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    elif env_name == 'MountainCar':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/mountain_car/transitions_50k/train_60k/{}_run.pkl".format(id)}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/mountain_car/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    
    assert path is not None
    testsets = {}
    for name in path:
        if name == "env":
            env = gym.make(path['env'])
            try:
                data = env.get_dataset()
            except:
                env = env.unwrapped
                data = env.get_dataset()
            testsets[name] = {
                'states': data['observations'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'next_states': data['next_observations'],
                'terminations': data['terminals'],
            }
        else:
            pth = path[name]
            with open(pth.format(id), 'rb') as f:
                testsets[name] = pickle.load(f)
        
        return testsets
    else:
        return {}

def run_steps(agent, max_steps, log_interval, eval_pth):
    t0 = time.time()
    evaluations = []
    agent.populate_returns(initialize=True)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            agent.save(timestamp="_{}".format(agent.total_steps))
            mean, median, min_, max_ = agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=True)
            evaluations.append(mean)
            t0 = time.time()
        if max_steps and agent.total_steps >= max_steps:
            agent.save(timestamp="_{}".format(agent.total_steps))
            break
        agent.step()
    np.save(eval_pth+"/evaluations.npy", np.array(evaluations))

def policy_evolution(cfg, agent, num_samples=500):
    # rng = np.random.RandomState(1024)
    # rnd_idx = rng.choice(agent.offline_data['env']['states'].shape[0], size=10)
    # rnd_states = agent.offline_data['env']['states'][rnd_idx]
    # states = torch_utils.tensor(agent.state_normalizer(rnd_states), agent.device)

    init_state = agent.env.reset().reshape((1, -1))
    state = torch_utils.tensor(agent.state_normalizer(init_state), agent.device)
    state = state.repeat_interleave(num_samples, dim=0)
    with torch.no_grad():
        on_policy, _ = agent.ac.pi.sample(state)
        on_policy = on_policy.cpu().detach().numpy()
    for i in range(0, agent.action_dim):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300, subplot_kw={'projection': '3d'})
        zeros0 = np.zeros((num_samples, i))
        zeros1 = np.zeros((num_samples, cfg.action_dim - 1 - i))
        # zeros0 = on_policy[:, :i]
        # zeros1 = on_policy[:, i+1:]
        xs = np.linspace(-1, 1, num=num_samples).reshape((num_samples, 1))
        actions = np.concatenate([zeros0, xs, zeros1], axis=1)
        actions = torch_utils.tensor(actions, agent.device)
        # time_color = ["#C2E4EF", "#98CAE1", "#6EA6CD", "#4A7BB7", "#364B9A"]
        time_color = {
            "HTqGaussian": ["#729bba", "#5192c4", "#2d81c2", "#1069ad", "#014f8a"],
            "Student": ["#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"],
            "Gaussian": ["#bf8888", "#b85f5f", "#b53c3c", "#a81e1e", "#8c0303"],
        }

        xticks = [-1, -0.5, 0, 0.5, 1]
        dfs = []
        plot_xs = []
        plot_ys = []
        plot_gys = []
        # timestamps = list(range(0, 250000, 50000))
        timestamps = list(range(0, 500, 100))

        for idx, timestamp in enumerate(timestamps):
            agent.load(cfg.load_network, "_{}".format(int(timestamp)))
            # with torch.no_grad():
            #     a, _ = agent.ac.pi.sample(rnd_state, deterministic=False)
            # a = torch_utils.to_np(a)
            with torch.no_grad():
                dist, mean, shape, dfx = agent.ac.pi.distribution(state, dim=i)
                density = torch.exp(dist.log_prob(actions[:, i:i+1])).detach().cpu().numpy()
            # ax.plot(xs.flatten(), ys, zs=idx, zdir='y', color=time_color[idx], alpha=0.5)
            plot_xs.append(xs.flatten())
            plot_ys.append(density.flatten())
            if cfg.distribution == "Student":
                assert dfx.shape[1] == 1
                dfx = dfx.detach().cpu().numpy()
                dfs.append(dfx[0, 0])
                # if i == 1:
                #     ax.text(xs.flatten()[0]-1.1, idx, 1.2, "DoF={:.2f}".format(dfx[0, 0]), c=time_color[idx])
                # else:
                #     ax.text(xs.flatten()[0]-0.6, idx, 1.2, "DoF={:.2f}".format(dfx[0, 0]), c=time_color[idx])
                ax.text(xs.flatten()[0]-0.6, idx, 0.8, "DoF={:.2f}".format(dfx[0, 0]), c=time_color[cfg.distribution][idx])

            assert mean.shape[1] == 1
            if cfg.distribution == "Student":
                shape = shape ** 2
            normal = torch.distributions.Normal(mean, shape)
            normal_density = torch.exp(normal.log_prob(actions[:, i:i + 1])).detach().cpu().numpy()
            # ax.plot(xs.flatten(), normal_density.flatten(), zs=idx, zdir='y', color=time_color[idx], alpha=0.4, linestyle='--')
            plot_gys.append(normal_density.flatten())

        plot_ys = np.asarray(plot_ys)
        plot_gys = np.asarray(plot_gys)
        # ysmin, ysmax = plot_ys.min(), plot_ys.max()
        # gysmin, gysmax = plot_gys.min(), plot_gys.max()
        for idx, timestamp in enumerate(timestamps):
            xs, ys, gys = plot_xs[idx], plot_ys[idx], plot_gys[idx]
            # # ys = (ys - min(ysmin, gysmin)) / (max(ysmax, gysmax) - min(ysmin, gysmin))
            # # gys = (gys - min(ysmin, gysmin)) / (max(ysmax, gysmax) - min(ysmin, gysmin))
            # ys = ys / ys.max()
            # gys = gys / gys.max()

            largers = np.argpartition(gys, -1)[-1:]
            gys[largers] = np.nan

            ax.plot(xs, ys, zs=idx, zdir='y', color=time_color[cfg.distribution][idx], alpha=0.8)
            ax.plot(xs, gys, zs=idx, zdir='y', color=time_color["Gaussian"][idx], alpha=0.8, linestyle='--')
        ax.plot([], [], c=time_color[cfg.distribution][-1], linestyle='-', label=formal_distribution_name[cfg.distribution], alpha=0.8)
        ax.plot([], [], c=time_color["Gaussian"][-1], linestyle='--', label="Gaussian", alpha=0.8)


        trail = "st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=0)
        ax.set_zlim(0, 1.)
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticklabels(['0', '', '0.5', '', '1'], rotation=0, ha='center')
        ax.set_yticks(np.arange(len(timestamps)))
        ax.set_yticklabels(["{}".format(t//100) for t in timestamps], rotation=-20, ha='center', va='center')
        ax.set_ylabel(r'Steps (x$10^2$)')
        ax.set_zlabel(r'Density')
        ax.set_xlabel(r'Action')

        ax.set_title("{} Policy Evolution\n{}{} Dimension".format(formal_distribution_name[cfg.distribution], i + 1, trail))
        if cfg.distribution == "Student":
            ax.annotate('', xytext=(0.81, -0.01), xy=(1.065, 0.25), xycoords='axes fraction',
                        arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
        elif cfg.distribution == "HTqGaussian":
            # ax.view_init(elev=0, azim=-40, roll=0)
            # ax.annotate('', xytext=(0.65, 0.135), xy=(1.02, 0.155), xycoords='axes fraction',
            #             arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
            # ax.view_init(elev=20, azim=-20, roll=0)
            # ax.annotate('', xytext=(0.51, -0.01), xy=(0.9, 0.05), xycoords='axes fraction',
            #             arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
            ax.view_init(elev=20, azim=-20, roll=0)
            ax.annotate('', xytext=(0.51, -0.01), xy=(0.85, 0.05), xycoords='axes fraction',
                        arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
        plt.legend(loc='lower left', bbox_to_anchor=(-0.3, 0.83), prop={'size': 10}, ncol=1, frameon=False)
        plt.tight_layout()
        plt.savefig(cfg.exp_path+"/{}_vis_dim{}.png".format(cfg.distribution, i), dpi=300)
        # plt.show()

def policy_evolution_multipolicy(cfg, agent_objs, time_color, num_samples=500, alpha=0.8):
    init_state = agent_objs[0].env.reset().reshape((1, -1))
    state = torch_utils.tensor(agent_objs[0].state_normalizer(init_state), agent_objs[0].device)
    state = state.repeat_interleave(num_samples, dim=0)
    for i in range(0, agent_objs[0].action_dim):
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300, subplot_kw={'projection': '3d'})
        zeros0 = np.zeros((num_samples, i))
        zeros1 = np.zeros((num_samples, cfg.action_dim - 1 - i))
        xs = np.linspace(-1, 1, num=num_samples).reshape((num_samples, 1))
        actions = np.concatenate([zeros0, xs, zeros1], axis=1)
        actions = torch_utils.tensor(actions, agent_objs[0].device)

        xticks = [-1, -0.5, 0, 0.5, 1]
        dfs = []
        plot_ys = {}
        for agent in agent_objs:
            plot_ys[agent.cfg.distribution] = []
        timestamps = list(range(0, 500, 100))

        for idx, timestamp in enumerate(timestamps):
            for agent in agent_objs:
                agent.load(agent.cfg.load_network, "_{}".format(int(timestamp)))
                with torch.no_grad():
                    dist, mean, shape, dfx = agent.ac.pi.distribution(state, dim=i)
                    density = torch.exp(dist.log_prob(actions[:, i:i+1])).detach().cpu().numpy()
                plot_ys[agent.cfg.distribution].append(density.flatten())
                if agent.cfg.distribution == "Student":
                    assert dfx.shape[1] == 1
                    dfx = dfx.detach().cpu().numpy()
                    dfs.append(dfx[0, 0])
                    ax.text(xs.flatten()[0]-0.6, idx, 0.9, "DoF={:.2f}".format(dfx[0, 0]), c=time_color[agent.cfg.distribution][idx])
                    # ax.zaxis.get_offset_text().set_rotation(60)
        for idx, timestamp in enumerate(timestamps):
            for agent in agent_objs:
                plot_ys_dist = np.asarray(plot_ys[agent.cfg.distribution])
                ys = np.asarray(plot_ys[agent.cfg.distribution][idx])
                if cfg.density_normalization:
                    # ys = ys / ys.max()
                    ys = (ys - plot_ys_dist.min()) / (plot_ys_dist.max() - plot_ys_dist.min())
                    # ys = (ys - ys.min()) / (ys.max() - ys.min())
                # largers = np.argpartition(gys, -1)[-1:]
                # gys[largers] = np.nan
                ax.plot(xs, ys, zs=idx, zdir='y', color=time_color[agent.cfg.distribution][idx], alpha=alpha, zorder=len(timestamps)-idx)

        for agent in agent_objs:
            ax.plot([], [], c=time_color[agent.cfg.distribution][-1], linestyle='-', label=formal_distribution_name[agent.cfg.distribution], alpha=alpha)

        trail = "st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=0)
        ax.set_zlim(0, 1.)
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticklabels(['0', '', '0.5', '', '1'], rotation=0, ha='center')
        ax.set_yticks(np.arange(len(timestamps)))
        ax.set_yticklabels(["{}".format(t//100) for t in timestamps], rotation=-20, ha='center', va='center')
        ax.set_ylabel(r'Steps (x$10^2$)')
        ax.set_zlabel(r'Density')
        ax.set_xlabel(r'Action')
        plt.subplots_adjust(left=-0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None)
        # ax.set_title("{} Policy Evolution on {}\n{}{} Dimension".format(cfg.agent, cfg.env_name, i + 1, trail))
        fig.text(0.01, 0.95, "{} Policy Evolution on {} {}".format(cfg.agent, cfg.env_name, formal_dataset_name[cfg.dataset]), fontsize=12)
        fig.text(0.4, 0.9, "{}{} Dimension".format(i + 1, trail), fontsize=12)
        ax.view_init(elev=30, azim=-30, roll=0)
        ax.annotate('', xytext=(0.61, 0.09), xy=(0.95, 0.2), xycoords='axes fraction',
                    arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1.5]))
        ax.grid(False)
        plt.legend(loc='lower left', bbox_to_anchor=(-0, 0.75), prop={'size': 10}, ncol=1, frameon=False)
        plt.tight_layout()
        plt.savefig(cfg.exp_path+"/vis_dim{}.png".format(i), dpi=300)
