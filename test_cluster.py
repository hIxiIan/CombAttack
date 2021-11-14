import os
from time import strftime, localtime

import graphgallery as gg
import pandas as pd
import argparse

from utils import save_test, get_datasets, get_embed_types, RES_COLUMNS, RES_ERRORS
from ca import run


def do_run(res, key, st, cmd, filename, prefix=""):
    print()
    print('====={}: {} start====='.format(prefix, key))
    try:
        res.loc[key] = run(st, cmd=cmd, verbose=False)
        res.to_csv(filename)
    except Exception as e:
        res.loc[key] = RES_ERRORS
        print('=====ASSERT_ERROR:'.format(repr(e)))
    print('====={}: {}   end====='.format(prefix, key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="cluster", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")
    parser.add_argument("-tn", "--target_nums", default=100, type=int, help="target nums")

    parser.add_argument("--dataset", default="", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    parser.add_argument("-p", default=7.0, type=float)
    parser.add_argument("-q", default=0.25, type=float)
    parser.add_argument("-a", "--alpha", default=0.25, type=float)
    parser.add_argument("-et", "--embed_type", default="", type=str)
    parser.add_argument('-ip', '--is_phi', default="true", type=str)

    parser.add_argument('--max_iter', default=300, type=int)
    parser.add_argument('--n_init', default=40, type=int)
    parser.add_argument('--times', default=1, type=int)
    cmd = parser.parse_args()
    gg.set_backend("th")

    rootdir = "result/test_cluster"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    personal = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    dataset_ = get_datasets(cmd.dataset)
    embed_types_ = get_embed_types(cmd.embed_type)
    print(dataset_)
    print(embed_types_)

    for dataset in dataset_:
        cmd.dataset = dataset
        _prefix = rootdir + os.sep + "_".join([cmd.dataset, personal])
        subgraph_type = "cluster"
        seeds = [2022, 2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
        times = min(cmd.times, len(seeds))
        for i in range(times):
            res = pd.DataFrame(columns=RES_COLUMNS)
            cmd.seed = seeds[i]
            filename = "_".join([_prefix, str(i)]) + '.csv'
            print(filename)
            do_run(res, 'sga', 'sga', cmd, filename)
            # us
            for embed_type in embed_types_:
                cmd.embed_type = embed_type
                key = '_'.join([embed_type])
                do_run(res, key, subgraph_type, cmd, filename, embed_type)

        save_test(_prefix, times)



