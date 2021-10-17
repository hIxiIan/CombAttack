import os
import graphgallery as gg
import pandas as pd
import argparse
from ca import run


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

    rootdir = "result" + os.sep
    prefix = cmd.dataset + "_new_wl_"
    times = 5
    seeds = [2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
    for i in range(times):
        cmd.seed = seeds[i]
        cmd.add_wl = True
        # filename = "result/result" + strftime("%Y_%m_%d_%H_%M_%S", localtime()) + ".csv"
        filename = rootdir + prefix + str(i) + '.csv'
        print(filename)
        # sga
        acc, wlacc, cost = run("dw", cmd=cmd, us=False, verbose=False)
        res.loc['sga'] = [acc, wlacc, cost]
        print('-------sga')
        res.to_csv(filename)

        # us
        subgraph_types = ['dw', 'dw_purity', 'dw_wl', 'dw_kh', 'dw_ce', 'n2v', 'n2v_purity', 'n2v_wl', 'n2v_ce',
                          'spread_random_wl', 'spread_random_wl_keep_hops', 'spread_random_ce',
                          'spread_random_ce_keep_hops', 'spread_ce', 'ppr', 'ppr_wl']
        # 'ppr_topk_des', 'ppr_topk_asc', 'ppr_wl_'

        # dw
        for subgraph_type in subgraph_types[:5]:
            for level_limit in [0, 1, 2]:
                key = '_'.join([subgraph_type, str(level_limit)])
                p = 1.0
                q = 1.0
                try:
                    acc, wlacc, cost = run(subgraph_type, cmd=cmd,p=p, q=q, level_limit=level_limit, verbose=False)
                    res.loc[key] = [acc, wlacc, cost]
                except Exception as e:
                    res.loc[key] = [-1, -1]
                    print('##################################error', repr(e))
                print('-------subgraph:{}, level_limit:{}'.format(subgraph_type, level_limit))
                res.to_csv(filename)

        # n2v
        for subgraph_type in subgraph_types[5:9]:
            for p in [0.5, 2.0]:
                for q in [0.25, 2.0]:
                    key = '_'.join([subgraph_type, str(p), str(q)])
                    try:
                        acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
                        res.loc[key] = [acc, wlacc, cost]
                    except Exception as e:
                        res.loc[key] = [-1, -1]
                        print('##################################error', repr(e))
                    print('-------subgraph:{}, p={}, q={}'.format(subgraph_type, p, q))
                    res.to_csv(filename)

        # spread
        for subgraph_type in subgraph_types[9:14]:
            key = '_'.join([subgraph_type])
            p = 1.0
            q = 1.0
            try:
                acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, verbose=False)
                res.loc[key] = [acc, wlacc, cost]
            except Exception as e:
                res.loc[key] = [-1, -1]
                print('##################################error', repr(e))
            print('-------subgraph:{}'.format(subgraph_type))
            res.to_csv(filename)

        # ppr
        for subgraph_type in subgraph_types[14:15]:
            p = 1.0
            q = 1.0
            for alpha in [0.5, 0.25, 0.1, 0.05, 0.01]:
                key = '_'.join([subgraph_type, str(alpha)])
                try:
                    acc, wlacc, cost = run(subgraph_type, cmd=cmd, p=p, q=q, alpha=alpha, verbose=False)
                    res.loc[key] = [acc, wlacc, cost]
                except Exception as e:
                    res.loc[key] = [-1, -1]
                    print('##################################error', repr(e))
                print('-------subgraph:{}, alpha={}'.format(subgraph_type, alpha))
                res.to_csv(filename)
    tdf = None
    for i in range(times):
        filename = rootdir + prefix + str(i) + '.csv'
        df = pd.read_csv(filename, index_col=0)
        if i == 0:
            tdf = df
        else:
            tdf += df
    tdf /= times
    tdf.to_csv(rootdir + prefix + '_total.csv')

