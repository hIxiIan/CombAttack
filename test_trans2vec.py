import argparse
import os
from trans2vec import run_trans2vec, random_seed
from time import strftime, localtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-tt", "--tedge_type", default="TBS+WBS", type=str)
    parser.add_argument("--run_emb", default="true", type=str)
    parser.add_argument("--run_nc", default="true", type=str)
    parser.add_argument("-f", "--filename", default="", type=str)
    parser.add_argument("-d", "--dimensions", default=64, type=int)  # 128
    parser.add_argument("--num_walks", default=20, type=int)  # 4
    parser.add_argument("--walk_length", default=5, type=int)  # 10
    parser.add_argument("--window_size", default=10, type=int)  # 4
    parser.add_argument("--workers", default=1, type=int)  # 8
    parser.add_argument("--train_size", default=0.8, type=float)
    parser.add_argument('--times', default=1, type=int)
    parser.add_argument("--trans2vec_model", default="OCSVM", type=str)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--gf_alias_mode", default="false", type=str)
    parser.add_argument("--gf", default="true", type=str)
    args = parser.parse_args()
    args.gf = True if args.gf == "true" else False
    args.gf_alias_mode = True if args.gf and args.gf_alias_mode == "true" else False

    outputdir = "result/test_trans2vec"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    args.outputdir = outputdir
    args.curtime = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    print(args.curtime)
    seeds = [2022, 2012, 1997, 5018, 2413, 97, 21, 32, 56, 44, 94]
    times = min(args.times, len(seeds))

    count = 0
    total = times
    for i in range(times):
        args.i = i
        args.seed = seeds[i]
        count += 1
        print(args.seed)
        print('times:{}, count: {}/{}'.format(i, count, total))
        run_trans2vec(args)