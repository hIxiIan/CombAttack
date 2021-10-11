class ARGS:
    def __init__(self, targets, seed=0, verbose=0, subgraph_type="dw", sample_ratio=0.2,
                 direct_attack=True, with_w_label=False, device="cpu", p=2.0, q=0.25):
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
