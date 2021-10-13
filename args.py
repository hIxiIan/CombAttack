class ARGS:
    def __init__(self, targets, seed=0, verbose=0, subgraph_type="dw", sample_ratio=0.2,
                 direct_attack=True, prob=0.5, with_w_label=False, hops=2, keep_hops=False, device="cpu",
                 alpha=0.25, eps=1e-4, p=2.0, q=0.25):
        self.seed = seed
        self.verbose = verbose
        self.subgraph_type = subgraph_type
        self.sample_ratio = sample_ratio
        self.targets = targets
        self.direct_attack = direct_attack
        self.p = p
        self.q = q
        self.with_w_label = with_w_label
        self.device = device
        self.alpha = alpha
        self.eps = eps
        self.hops = hops
        self.keep_hops = keep_hops
        self.prob = prob
