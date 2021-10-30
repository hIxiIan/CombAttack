import torch


class ARGS:
    def __init__(self, cmd, targets=None, splits=None,

                 wl_limit=0.5, hops=2, prob=0.8, eps=1e-4):
        # 通用
        self.seed = cmd.seed
        self.verbose = cmd.verbose
        self.device = "gpu" if cmd.device == "gpu" and torch.cuda.is_available() else "cpu"
        self.dataset = cmd.dataset
        self.us = False if cmd.subgraph_type in ['sga'] else True

        # attack
        self.subgraph_type = cmd.subgraph_type
        self.sample_ratio = cmd.sample_ratio
        self.targets = targets
        self.splits = splits
        self.direct_attack = not cmd.indirect_attack

        # dw
        self.p = cmd.p
        self.q = cmd.q
        self.wl_limit = wl_limit

        # spreader
        self.hops = hops
        self.prob = prob

        # ppr
        self.alpha = cmd.alpha
        self.eps = eps
