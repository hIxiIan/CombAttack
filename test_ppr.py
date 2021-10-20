import os
from time import strftime, localtime

import graphgallery as gg
import pandas as pd
import argparse
from ca import run
from utils import save_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-st", "--subgraph_type", default="dw_wl", type=str, help="sample method")
    parser.add_argument("-sr", "--sample_ratio", default=0.05, type=float, help="ratio of sampled nodes")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")

    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    cmd = parser.parse_args()
    gg.set_backend("th")

    res = pd.DataFrame(columns=['acc', 'wlacc', 'cost'])
    rootdir = "result/" + strftime("%Y_%m_%d_%H_%M_%S", localtime())
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    personal = "test_ppr"
    prefix = "_".join([cmd.dataset, personal])
    _prefix = rootdir + os.sep + prefix

    times = 3
    seeds = [2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
    for i in range(times):
        cmd.seed = seeds[i]
        filename = "_".join([_prefix, str(i)]) + '.csv'
        print(filename)
        # sga
        try:
            acc, wlacc, cost = run("sga", cmd=cmd, us=False, verbose=False)
            res.loc['sga'] = [acc, wlacc, cost]
            print('-------sga')
            res.to_csv(filename)
        except Exception as e:
            res.loc['sga'] = [-1, -1, -1]
            print('##################################error', repr(e))

        # us
        # no, yes, yes, yes, no
        subgraph_types = ['ppr', 'ppr_nums', 'ppr_topk_des', 'ppr_topk_asc',
                          'ppr_wl_limit', 'ppr_wl_limit_nums', 'ppr_wl', 'ppr_wl_limit_wl'
                          'ppr_wl_topk_des', 'ppr_wl_topk_asc']

        # ppr
        for subgraph_type in subgraph_types:
            p = 1.0
            q = 1.0
            for alpha in [0.5, 0.25, 0.1, 0.05, 0.01]:
                key = '_'.join([subgraph_type, str(alpha)])
                try:
                    acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, alpha=alpha, verbose=False)
                    res.loc[key] = [acc, wlacc, cost]
                except Exception as e:
                    res.loc[key] = [-1, -1, -1]
                    print('##################################error', repr(e))
                print('-------subgraph:{}, alpha={}'.format(subgraph_type, alpha))
                res.to_csv(filename)

    print(_prefix)
    save_test(_prefix, times)

