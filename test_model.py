import argparse
import os
import pandas as pd
import torch
import graphgallery as gg
import graphgallery.functional as gf
from ca import get_model, get_dr_model
from graphgallery.datasets import NPZDataset
from time import strftime, localtime

ds = ["GCN", "SGC", "SimPGCN", "GCN_Jaccard", "FastGCN"]


def get_models(models):
    if len(models) <= 0:
        return ds
    return models.split(",")


def print_args(args):
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="gpu", type=str, help="code environment")
    parser.add_argument("-m", "--model", default="GCN", type=str, help="model")
    parser.add_argument("--dataset", default="cora", type=str, help="dataset")
    parser.add_argument("--is_gf", default="false", type=str, help="graphgallery / deeprobust")
    parser.add_argument("--test", default="true", type=str, help="test parms mode")
    args = parser.parse_args()
    print_args(args)
    args.device = args.device if args.device in ["gpu", "cuda:0", "cuda:1"] and torch.cuda.is_available() else "cpu"

    gg.set_backend("th")
    data = NPZDataset(args.dataset,
                      root="~/GraphData/datasets/",
                      verbose=False,
                      transform="standardize")
    graph = data.graph
    splits = data.split_nodes(random_state=15)
    args.splits = splits
    gf.random_seed(args.seed, gg.backend())

    models = get_models(args.model)
    if args.test == "false":
        for model_name in models:
            if args.is_gf == "true":
                model = get_model(model_name, args, graph)
                model.fit(splits.train_nodes, splits.val_nodes, verbose=args.verbose, epochs=200)
                results = model.evaluate(splits.test_nodes, verbose=args.verbose)
                print(f'Model: {model_name}, Test accuracy {results.accuracy:.2%}')
            else:
                model, acc = get_dr_model(model_name, args, graph)
                print(f'Model: {model_name}, Test accuracy {acc:.2%}')
    else:

        hids_ = [[32],
                 [64],
                 [128],
                 [256],
                 [512]]
        weight_decay_ = [5e-5, 5e-4, 5e-3, 5e-2]
        lr_ = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        print(hids_)
        print(weight_decay_)
        print(lr_)
        count = 0
        total = len(models) * len(hids_) * len(weight_decay_) * len(lr_)

        rootdir = "result/test_model"
        if not os.path.exists(rootdir):
            os.mkdir(rootdir)
        filename = rootdir + os.sep + "_".join([args.dataset, strftime("%Y_%m_%d_%H_%M_%S", localtime())]) + '.csv'
        df = pd.DataFrame(columns=["acc"])
        for model_name in models:
            for hids in hids_:
                args.hids = hids
                args.acts = ['relu' for _ in args.hids]
                for weight_decay in weight_decay_:
                    args.weight_decay = weight_decay
                    for lr in lr_:
                        args.lr = lr
                        count += 1
                        print('{}/{}'.format(count, total))
                        if args.is_gf == "true":
                            model = get_model(model_name, args, graph, is_model=True)
                            model.fit(splits.train_nodes, splits.val_nodes, verbose=args.verbose, epochs=200)
                            results = model.evaluate(splits.test_nodes, verbose=args.verbose)
                            print('hids:{}, wd:{}, lr:{}'.format(args.hids, args.weight_decay, args.lr))
                            print(f'Model: {model_name}, Test accuracy {results.accuracy:.2%}')
                            key = "_".join([model_name, str(args.hids[0]), str(args.weight_decay), str(arhs.lr)])
                            df.loc[key] = np.array([results.accuracy])
                            df.to_csv(filename)
                        else:
                            assert False, "invalid dr test parms"
                            model, acc = get_dr_model(model_name, args, graph)
                            print(f'Model: {model_name}, Test accuracy {acc:.2%}')