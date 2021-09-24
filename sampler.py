import numpy as np
import random

from utils import to_list
from spreader import Spreader
from walker import Walker
from ppr import PPRer


class Sampler:
    def __init__(self, adj_matrix, p=1.0, q=1.0, seed=123, reset=True):
        self.seed = seed
        self.adj_matrix = adj_matrix
        self.p = p
        self.q = q
        self.walker = Walker(self.adj_matrix, self.p, self.q)
        if self.p != 1.0 or self.q != 1.0:
            self.walker.preprocess_transition_probs()
        self.spreader = Spreader(self.adj_matrix)
        self.PPRer = PPRer(self.adj_matrix)
        # if reset:
        #     self.reset()

    def reset(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def random_sample(self, targets, sample_nums, dw=False):
        targets = to_list(targets)
        if dw:
            return self.deepwalk_sample(targets, sample_nums)

        return self.node2vec_sample(targets, sample_nums)

    def deepwalk_sample(self, targets, sample_nums):
        targets = to_list(targets)
        edges, nodes = self.walker.deepwalk_sample(targets, sample_nums)
        return edges, nodes

    def node2vec_sample(self, targets, sample_nums):
        targets = to_list(targets)
        edges, nodes = self.walker.node2vec_sample(targets, sample_nums)
        return edges, nodes

    def spread_sample(self, targets, prob, hops=2, hop_mode=False):
        targets = to_list(targets)
        if not hop_mode:
            hops = 2
        edges, nodes = self.spreader.spread_sample(targets, prob, hops, hop_mode)
        return edges, nodes

    def ppr_sample(self, targets, alpha, esp):
        targets = to_list(targets)
        edges, nodes = self.PPRer.ppr_sample(targets, alpha, esp)
        return edges, nodes

    def ppr_topk_sample(self, targets, alpha, esp, topk):
        targets = to_list(targets)
        edges, nodes = self.PPRer.ppr_topk_sample(targets, alpha, esp, topk)
        return edges, nodes

    # def ppr_topk_sample_parallel(self, targets, alpha, esp, topk):
    #     targets = to_list(targets)
    #     edges, nodes = self.PPRer.ppr_topk_sample_parallel(targets, alpha, esp, topk)
    #     return edges, nodes