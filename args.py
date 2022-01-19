import os
import pandas as pd
import torch
from utils import DATASET_BLOCKCHAIN


class ClusterParms:
    def __init__(self, max_iter=300, n_init=40, seed=2020, topk_cluster=3,
                 random=False, is_het=False, lay_act="layer", lay_act_cnt=1, distance_type="euclidean",
                 mix_cluster="false", mix_types="MLP,SGC2",
                 test_mode="-1", deg_limit=2, sur_label_pro_limit=0.9):
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        self.topk_cluster = topk_cluster
        self.random = random
        self.is_het = is_het
        self.lay_act = lay_act
        self.lay_act_cnt = lay_act_cnt
        self.distance_type = distance_type
        self.mix_cluster = True if mix_cluster == "true" else False
        self.mix_types = mix_types.split(',')
        self.test_mode = test_mode
        self.deg_limit = deg_limit
        self.sur_label_pro_limit= sur_label_pro_limit


class ARGS:
    def __init__(self, cmd, targets=None, splits=None,

                 wl_limit=0.5, hops=2, prob=0.8, eps=1e-4,
                 graph=None):
        # 通用
        self.seed = cmd.seed
        self.verbose = cmd.verbose
        self.device = cmd.device if cmd.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"
        self.dataset = cmd.dataset
        self.us = False if cmd.subgraph_type in ['sga'] else True
        self.cluster = True if cmd.subgraph_type == "cluster" else False
        self.embed_type = cmd.embed_type
        self.is_phi = True if cmd.is_phi == "true" else False
        self.is_topk = True if cmd.is_topk == "true" else False
        self.edge_flips = True if cmd.edge_flips == "true" else False
        self.bmbc_mode = True if cmd.bmbc_mode == "true" else False
        self.graph = graph

        # attack
        self.subgraph_type = cmd.subgraph_type
        self.sample_ratio = cmd.sample_ratio
        self.targets = targets
        self.target_nums = cmd.target_nums
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
        self.adj_matrix = graph.adj_matrix
        self.node_attr = graph.node_attr
        self.node_label = graph.node_label
        if self.dataset in ["tedge", "trans2vec"]:
            nodes_to_keep = pd.read_csv('dataset/phishing/tedge_nodes_to_keep.csv').values.ravel()
            path = 'result/test_tedge/' + cmd.features_file
            if 'csv' not in path:
                path += '.csv'
            node_attr = pd.read_csv(path).values[nodes_to_keep]
            graph.node_attr = node_attr
            self.node_attr = graph.node_attr
            self.train_size = cmd.train_size # make use of the target nodes selection
            self.features_file = path
            self.nodes_to_keep = nodes_to_keep
            self.trans2vec_model = cmd.trans2vec_model

        random = True if cmd.random == "true" else False
        is_het = True if cmd.dataset in ["chameleon", "squirrel"] else False # 无效
        is_het = True
        self.cluster_parms = ClusterParms(cmd.max_iter,
                                          cmd.n_init,
                                          self.seed,
                                          cmd.topk_cluster,
                                          random,
                                          is_het,
                                          cmd.lay_act,
                                          cmd.lay_act_cnt,
                                          cmd.distance_type,
                                          cmd.mix_cluster,
                                          cmd.mix_types,
                                          cmd.test_mode,
                                          cmd.deg_limit,
                                          cmd.sur_label_pro_limit)
        # test_parms
        self.hids = cmd.hids
        self.acts = cmd.acts
        self.weight_decay = cmd.weight_decay
        self.lr = cmd.lr
