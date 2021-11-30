import os
from time import strftime, localtime

import graphgallery as gg
import pandas as pd
import argparse

from utils import save_test, get_datasets, get_embed_types, get_attacked_types, RES_COLUMNS, RES_ERRORS
from ca import run

n_classes_dict = {
        'cora': 7,
        'cora_full': 70,
        'citeseer': 6,
        'ogbn-arxiv': 40,
        'reddit': 41,

        'coauthor_phy': 5,

        'chameleon': 5,
        'squirrel': 5,
        'blockchain30000': 2,
        'blockchain40000': 2,
        'blockchain50000': 2,
    }


def do_run(res, key, st, cmd, filename, prefix=""):
    print()
    print('====={}: {} start====='.format(prefix, key))
    try:
        rsps = run(st, cmd=cmd, verbose=False)
        for rsp in rsps:
            hids = "" if cmd.hids is None else str(len(cmd.hids))
            weight_decay = "" if cmd.weight_decay is None else str(cmd.weight_decay)
            lr = "" if cmd.lr is None else str(cmd.lr)
            res.loc['_'.join([key, rsp[0], hids, weight_decay, lr])] = rsp[1:]
        res.to_csv(filename)
    except Exception as e:
        res.loc[key] = RES_ERRORS[1:]
        print('=====ASSERT_ERROR:'.format(repr(e)))
    print('====={}: {}   end====='.format(prefix, key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="cluster", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-da", "--direct_attack", default="true", type=str, help="direct attack")
    parser.add_argument("-tn", "--target_nums", default=50, type=int, help="target nums")

    parser.add_argument("--dataset", default="", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="", type=str)
    parser.add_argument('-ip', '--is_phi', default="true", type=str)
    parser.add_argument('-atk', '--atk_model_type', default="", type=str)

    parser.add_argument('--max_iter', default=300, type=int)
    parser.add_argument('--n_init', default=40, type=int)
    parser.add_argument('--times', default=1, type=int)
    parser.add_argument('-tc', '--topk_cluster', default=3, type=int)
    parser.add_argument('-r', '--random', default="false", type=str)
    parser.add_argument('--run_sga', default="true", type=str)
    cmd = parser.parse_args()
    gg.set_backend("th")

    rootdir = "result/test_cluster"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    personal = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    dataset_ = get_datasets(cmd.dataset)
    embed_types_ = get_embed_types(cmd.embed_type)
    atked_types_ = get_attacked_types(cmd.atk_model_type)
    print(dataset_)
    print(embed_types_)
    print(atked_types_)

    hids_ = [[512, 256, 128, 64],
            [512, 256, 128, 64, 32],
            [512, 256, 128, 64, 32, 16]]
    acts_ = [['relu', 'relu', 'relu', 'relu'],
            ['relu', 'relu', 'relu', 'relu', 'relu'],
            ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']]
    weight_decay_ = [5e-2, 5e-3, 5e-4, 5e-5]
    lr_ = [0.05, 0.01, 0.005, 0.001]

    for dataset in dataset_:
        if dataset in ["cora_full", "ogbn-arxiv", "reddit"]:
            cmd.topk_cluster = int(n_classes_dict[dataset] / 2)
        cmd.dataset = dataset
        _prefix = rootdir + os.sep + "_".join([cmd.dataset, personal])
        seeds = [2022, 2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
        times = min(cmd.times, len(seeds))
        for i in range(times):
            res = pd.DataFrame(columns=RES_COLUMNS[1:])
            cmd.seed = seeds[i]
            filename = "_".join([_prefix, str(i)]) + '.csv'
            print(filename)
            if cmd.run_sga == "true":
                cmd.hids = None
                cmd.acts = None
                cmd.weight_decay = None
                cmd.lr = None
                key = '_'.join(['sga'])
                do_run(res, key, 'sga', cmd, filename)

            for embed_type in embed_types_:
                cmd.embed_type = embed_type
                for i, hids in enumerate(hids_):
                    cmd.hids = hids
                    cmd.acts = acts_[i]
                    for weight_decay in weight_decay_:
                        cmd.weight_decay = weight_decay
                        for lr in lr_:
                            cmd.lr = lr
                            key = '_'.join([embed_type])
                            do_run(res, key, 'cluster', cmd, filename, embed_type)

        save_test(_prefix, times, seeds[:times])