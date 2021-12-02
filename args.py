import os

import torch
from utils import DATASET_BLOCKCHAIN


class ClusterParms:
    def __init__(self, max_iter=300, n_init=40, seed=2020, topk_cluster=3, random=False, is_het=False):
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        self.topk_cluster = topk_cluster
        self.random = random
        self.is_het = is_het


class ARGS:
    def __init__(self, cmd, targets=None, splits=None,

                 wl_limit=0.5, hops=2, prob=0.8, eps=1e-4,
                 node_attr=None, node_label=None):
        # 通用
        self.seed = cmd.seed
        self.verbose = cmd.verbose
        self.device = cmd.device if cmd.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"
        self.dataset = cmd.dataset
        self.us = False if cmd.subgraph_type in ['sga'] else True
        self.cluster = True if cmd.subgraph_type == "cluster" else False
        self.embed_type = cmd.embed_type
        self.is_phi = True if cmd.is_phi == "true" else False

        # attack
        self.subgraph_type = cmd.subgraph_type
        self.sample_ratio = cmd.sample_ratio
        self.targets = targets
        self.splits = splits
        self.direct_attack = True if cmd.direct_attack == "true" else False
        self.atk_model_type = cmd.atk_model_type
        self.blockchain = True if cmd.dataset in DATASET_BLOCKCHAIN else False

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

        # blockchain
        self.node_attr = node_attr
        self.node_label = node_label

        random = True if cmd.random == "true" else False
        is_het = True if cmd.dataset in ["chameleon", "squirrel"] else False # 无效
        is_het = True
        self.cluster_parms = ClusterParms(cmd.max_iter,
                                          cmd.n_init,
                                          self.seed,
                                          cmd.topk_cluster,
                                          random,
                                          is_het)
        # test_parms
        self.hids = cmd.hids
        self.acts = cmd.acts
        self.weight_decay = cmd.weight_decay
        self.lr = cmd.lr
