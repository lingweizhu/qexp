from core.policy.gaussian import MLPCont, SquashedGaussian, Gaussian
from core.policy.qGaussian import qMultivariateGaussian, qHeavyTailedGaussian
from core.policy.student import Student
from core.policy.beta import Beta


def get_continuous_policy(name, cfg):
    if name == "GaussianD":
        return MLPCont(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2)
    elif name == "Gaussian":
        return Gaussian(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2, cfg.action_min, cfg.action_max)
    elif name == "SGaussian":
        return SquashedGaussian(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2, cfg.action_min, cfg.action_max)
    elif name == "qGaussian":
        return qMultivariateGaussian(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units]*2, cfg.action_min, cfg.action_max, entropic_index=cfg.distribution_param)
    elif name == "HTqGaussian":
        return qHeavyTailedGaussian(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units]*2, cfg.action_min, cfg.action_max, entropic_index=cfg.distribution_param)
    elif name == "Student":
        return Student(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2, cfg.action_min, cfg.action_max, df=cfg.distribution_param)
    elif name == "Beta":
        return Beta(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2, cfg.action_min, cfg.action_max)
