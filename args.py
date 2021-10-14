class ARGS:
    def __init__(self, targets,

                 seed=0, verbose=0, device="cpu",

                 subgraph_type="dw", sample_ratio=0.05, direct_attack=True, with_w_label=False,

                 p=2.0, q=0.25, is_purity_matrix=False, is_wl_matrix=False, is_ce_matrix=False, level_limit=0,

                 hops=2, keep_hops=False, prob=0.5,

                 alpha=0.25, eps=1e-4):
        # 通用
        self.seed = seed
        self.verbose = verbose
        self.device = device

        # attack
        self.subgraph_type = subgraph_type
        self.sample_ratio = sample_ratio
        self.targets = targets
        self.direct_attack = direct_attack
        self.with_w_label = with_w_label

        # dw
        self.p = p
        self.q = q
        self.is_purity_matrix = is_purity_matrix
        self.is_wl_matrix = is_wl_matrix
        self.is_ce_matrix = is_ce_matrix
        self.level_limit = level_limit

        # spreader
        self.hops = hops
        self.keep_hops = keep_hops
        self.prob = prob

        # ppr
        self.alpha = alpha
        self.eps = eps
