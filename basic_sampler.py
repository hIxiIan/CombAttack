class BasicSampler:
    def __init__(self, G, seed):
        self.G = G.G
        self.seed = seed

    def sample(self, sample_length, start_node):
        raise NotImplementedError

    def simulate_samples(self, start_nodes, walk_length, num_walks=None):
        raise NotImplementedError