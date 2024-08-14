import torch
from torch.optim import SGD, Adam

def get_optimizer(cfg: dict, params: torch.Tensor) -> torch.optim.Optimizer:
    """
    cfg contains:
        - optimizer
        - lr
        - weight_decay
        - betas: [b1,b2]
        - eps

    """
    match cfg.name:
        case "SGD":
            return SGD(params=params,
                       lr=cfg.lr,
                       weight_decay=cfg.weight_decay)
        case "Adam":
            return Adam(params=params,
                        lr=cfg.lr,
                        betas=tuple(cfg.betas),
                        eps=cfg.eps,
                        weight_decay=cfg.weight_decay)
        case _:
            raise NotImplementedError
