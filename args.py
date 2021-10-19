class ARGS:
    def __init__(self, cmd, targets=None, splits=None,

                 seed=0, verbose=0, device="cpu",

                 subgraph_type="dw", sample_ratio=0.05, direct_attack=True, with_w_label=False,

                 p=2.0, q=0.25, is_purity_matrix=False, is_wl_matrix=False, is_ce_matrix=False, wl_limit=0.5,

                 hops=2, keep_hops=False, prob=0.5,

                 alpha=0.25, eps=1e-4):
        # 通用
        self.seed = cmd.seed
        self.verbose = cmd.verbose
        self.device = cmd.device
        self.dataset = cmd.dataset
        self.us = not cmd.n_us

        # attack
        self.subgraph_type = cmd.subgraph_type
        self.sample_ratio = cmd.sample_ratio
        self.targets = targets
        self.splits = splits
        self.direct_attack = not cmd.indirect_attack
        self.with_w_label = with_w_label

        # dw
        self.p = p
        self.q = q
        self.is_purity_matrix = is_purity_matrix
        self.is_wl_matrix = is_wl_matrix
        self.is_ce_matrix = is_ce_matrix
        self.wl_limit = wl_limit

        # spreader
        self.hops = hops
        self.keep_hops = keep_hops
        self.prob = prob

        # ppr
        self.alpha = alpha
        self.eps = eps
