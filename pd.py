import gc
import os
import argparse
import networkx as nx
import pickle
import numpy as np
import lightgbm as lgb
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy as dcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from scipy.sparse import coo_matrix
from tqdm import tqdm


def load_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


"""常用函数及lgb模型的定义"""


def read_embeds(fname):
    with open(fname, 'r') as f:
        """skip the first row， first col"""
        data = f.readlines()
        npdata = np.loadtxt(data, float, delimiter=' ')
        argsort = np.argsort(npdata[:, 0])
        npdata = npdata[argsort].tolist()
        npdata = np.delete(npdata, 0, axis=1)
        return npdata


"""
根据返回特定下标ilis对应的lis中的元素
"""


def lid(lis, ilis):
    return [lis[i] for i in ilis]


def eval_f(y_pred, y_true):
    y_pred = y_pred.reshape((2, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.label
    score = metrics.f1_score(y_true, y_pred, pos_label=1)
    return 'F1_score: ', score, True


def lgb_train_model(train_x, train_y, random_seed):
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    lgb_paras = {
        'objective': 'multiclass',
        'learning_rate': 0.03,
        'num_leaves': 50,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'seed': random_seed,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'metric': 'multi_logloss',
        'num_threads': 8,
        'num_class': 2,
        'force_col_wise': True,
        'min_data_in_leaf': 20,
        'verbosity': -1, # 控制训练过程是否输出
        #         'scale_pos_weight':100,
    }

    auc, recall, precision, f1, result_proba = [], [], [], [], []
    for tr_i, val_i in skfold.split(train_x, train_y):
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_i], train_y.iloc[tr_i], train_x.iloc[val_i], train_y.iloc[val_i]
        train_set = lgb.Dataset(tr_x, tr_y)
        val_set = lgb.Dataset(val_x, val_y)
        lgb_model = lgb.train(lgb_paras,
                              train_set,
                              #                               num_boost_round=100
                              valid_sets=[val_set],
                              # early_stopping_rounds=60,
                              feval=eval_f,
                              callbacks=[lgb.log_evaluation(0)]
                              )

        val_pred = np.argmax(lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration), axis=1)
        auc_score = metrics.roc_auc_score(val_y, val_pred)
        recall_score = metrics.recall_score(val_y, val_pred, pos_label=1)
        precision_score = metrics.precision_score(val_y, val_pred, pos_label=1)
        f1_score = metrics.f1_score(val_y, val_pred, pos_label=1)
        auc.append(auc_score)
        recall.append(recall_score)
        precision.append(precision_score)
        f1.append(f1_score)
    res = [np.mean(auc), np.mean(recall), np.mean(precision), np.mean(f1)]
    return res


def get_lgb_model(train_x, train_y, random_seed):
    lgb_paras = {
        'objective': 'multiclass',
        'learning_rate': 0.03,
        'num_leaves': 50,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'seed': random_seed,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'metric': 'multi_logloss',
        'num_threads': 8,
        'num_class': 2,
        'force_col_wise': True,
        'min_data_in_leaf': 20,
        'verbosity': -1, # 控制训练过程是否输出
        #         'scale_pos_weight':100,
    }
    tr_x, test_x, tr_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=random_seed) #  stratify=train_y.values.ravel()

    auc, recall, precision, f1, result_proba = [], [], [], [], []
    train_set = lgb.Dataset(tr_x, tr_y)
    # test_set = lgb.Dataset(test_x, test_y)
    lgb_model = lgb.train(lgb_paras,
                          train_set,
                          #                               num_boost_round=100
                          # valid_sets=[val_set],
                          # early_stopping_rounds=60,
                          feval=eval_f,
                          callbacks=[lgb.log_evaluation(0)]
                          )
    all_predict = np.argmax(lgb_model.predict(train_x, num_iteration=lgb_model.best_iteration), axis=1)
    y_pred = np.argmax(lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration), axis=1)
    lgb_model.attacked_models_acc = [(np.array(test_y) == y_pred).mean()]
    auc_score = metrics.roc_auc_score(test_y, y_pred)
    recall_score = metrics.recall_score(test_y, y_pred, pos_label=1)
    precision_score = metrics.precision_score(test_y, y_pred, pos_label=1)
    f1_score = metrics.f1_score(test_y, y_pred, pos_label=1)
    auc.append(auc_score)
    recall.append(recall_score)
    precision.append(precision_score)
    f1.append(f1_score)

    test_res = [np.mean(auc), np.mean(recall), np.mean(precision), np.mean(f1)]
    return test_res, all_predict, lgb_model


def print_res(model_name, res):
    print(model_name[:4], 'auc', res[0], 'recall', res[1], 'precision', res[2], 'f1', res[3])


def fea_tree():
    fea_res, rcnt = [0, 0, 0, 0], 10
    for i in tqdm(range(rcnt)):
        lgb_res = lgb_train_model(train_x.copy(), train_y.copy(), random.randint(0, 10000))
    return lgb_res


def dw_tree():
    global train_x, train_y, DATA_PATH, RANDOM_SEED
    dw_res, rcnt = [0, 0, 0, 0], 5
    for i in range(rcnt):
        node_feas, labels = dcopy(train_x.values), dcopy(train_y.values)
        embe_feas = read_embeds(DATA_PATH + '/embeds_dw_%d.dat' % i)
        np_fea_lab = np.hstack((node_feas, embe_feas, labels))
        columns_name = ['f%02d' % i for i in range(np_fea_lab.shape[1] - 1)] + ['label']
        df_fea_lab = pd.DataFrame(data=np_fea_lab, columns=columns_name, dtype=float)
        df_fea_lab['label'] = df_fea_lab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fea_lab.columns if x not in y_cols_name]

        df_x = df_fea_lab[x_cols_name]
        df_y = df_fea_lab[y_cols_name]

        lgb_res = lgb_train_model(df_x, df_y, RANDOM_SEED)
        print(lgb_res)
        for i in range(len(dw_res)):
            dw_res[i] += lgb_res[i]
    dw_res = [i / rcnt for i in dw_res]
    gc.collect()
    return dw_res


def n2v_tree():
    global train_x, train_y
    n2v_res, rcnt = [0, 0, 0, 0], 5
    for i in tqdm(range(rcnt)):
        node_feas, labels = dcopy(train_x.values), dcopy(train_y.values)
        embe_feas = read_embeds(DATA_PATH + '/embeds_n2v_%d.dat' % i)
        np_fealab = np.hstack((node_feas, embe_feas, labels))
        columns_name = ['f%02d' % i for i in range(np_fealab.shape[1] - 1)] + ['label']
        df_fealab = pd.DataFrame(data=np_fealab, columns=columns_name, dtype=float)
        df_fealab['label'] = df_fealab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fealab.columns if x not in y_cols_name]

        n2vdf_x = df_fealab[x_cols_name]
        n2vdf_y = df_fealab[y_cols_name]

        lgb_res = lgb_train_model(n2vdf_x, n2vdf_y, RANDOM_SEED)
        for i in range(len(n2v_res)):
            n2v_res[i] += lgb_res[i]

    n2v_res = [i / rcnt for i in n2v_res]
    gc.collect()
    return n2v_res


def line_tree():
    line_res, rcnt = [0, 0, 0, 0], 5
    for i in tqdm(range(rcnt)):
        embe_feas = read_embeds(DATA_PATH + '/embeds_line_%d.dat' % i)
        node_feas, labels = dcopy(train_x.values), dcopy(train_y.values)
        np_fealab = np.hstack((node_feas, embe_feas, labels))
        columns_name = ['f%02d' % i for i in range(np_fealab.shape[1] - 1)] + ['label']
        df_fealab = pd.DataFrame(data=np_fealab, columns=columns_name, dtype=float)
        df_fealab['label'] = df_fealab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fealab.columns if x not in y_cols_name]

        linedf_x = df_fealab[x_cols_name]
        linedf_y = df_fealab[y_cols_name]

        lgb_res = lgb_train_model(linedf_x, linedf_y, RANDOM_SEED)
        for i in range(len(line_res)):
            line_res[i] += lgb_res[i]
    line_res = [i / rcnt for i in line_res]
    gc.collect()
    return line_res


"""无监督gcn做embedding"""


def normalize(A):
    lena = A.shape[0]
    row = [i for i in range(lena)]
    col, data = row.copy(), [1 for i in range(lena)]
    eye_mat = coo_matrix((data, (row, col)), shape=(lena, lena))
    A = A + eye_mat # A + I
    d = np.power(A.sum(1), -0.5) # D^(-1/2)
    d = np.ravel(d) # 一维
    i = [j for j in range(lena)]
    D = coo_matrix((d, (i, i)), shape=(lena, lena))

    scipy_mat = (D * A * D).tocoo() # coo形式的D * A * D
    return scipy_mat


def scipy_tensor(scipy_mat):
    row, col, data = scipy_mat.row, scipy_mat.col, scipy_mat.data
    lena = scipy_mat.shape[0]
    indice, data = torch.LongTensor([row, col]), torch.FloatTensor(data)
    torch_mat = torch.sparse.FloatTensor(indice, data, torch.Size([lena, lena]))
    return torch_mat


class GCN(nn.Module):
    def __init__(self, DAD, dim_in, dim_out, random_seed):
        super(GCN, self).__init__()
        torch.manual_seed(random_seed)
        self.DAD = DAD
        self.fc1 = nn.Linear(dim_in, dim_out, bias=False)

    #         self.fc2 = nn.Linear(dim_in, dim_out, bias=False)
    #         self.fc3 = nn.Linear(dim_in ,dim_out,bias=False)
    #         self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
    #         self.fc2 = nn.Linear(dim_in ,dim_in,bias=False)
    #         self.fc3 = nn.Linear(dim_in ,dim_out,bias=False)
    #         self.fc4 = nn.Linear(dim_in ,dim_in,bias=False)
    #         self.fc5 = nn.Linear(dim_in ,dim_out,bias=False)

    def forward(self, X, flag=1):
        X = torch.tanh(self.fc1(self.DAD.mm(X)))
        #         X = F.tanh(self.fc2(self.DAD.mm(X)))
        #         X = F.tanh(self.fc3(self.DAD.mm(X)))
        #         X = F.tanh(self.fc2(self.DAD.mm(X)))
        #         X = F.tanh(self.fc3(self.DAD.mm(X)))
        #         X = F.tanh(self.fc4(self.DAD.mm(X)))
        #         X = F.tanh(self.fc5(self.DAD.mm(X)))

        return X


def get_input_vars():
    global train_x, train_y, scipy_adj_matrix
    adj_matrix = dcopy(scipy_adj_matrix)
    A_normed = normalize(adj_matrix)
    A_normed = scipy_tensor(A_normed)
    adj_mat = scipy_tensor(adj_matrix)
    X, Y = dcopy(train_x), dcopy(train_y)
    X, Y = torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).long()
    return X, Y, A_normed, adj_mat


def prob_res(pred):
    pred_np = pred.detach().numpy()
    return np.argmax(pred_np, axis=1)


def gcn_train(X, Y, A_normed, A, epoch, lr, weight_decay, esize, random_seed):
    dim_n, dim_f = X.shape[0], X.shape[1]
    gcn_model = GCN(A_normed, dim_f, esize, random_seed=random_seed)
    optim = torch.optim.Adam(gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epoch):
        Z = gcn_model(X, flag=1)
        adj_dec = Z.mm(Z.t())

        loss = torch.norm(adj_dec - A, p='fro')
        loss = torch.pow(loss, 2) / (dim_n)
        print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return Z


def gcn_tree(epoch=10, lr=0.02, weight_decay=2e-6, esize=8, random_seed=2022):
    global train_x, train_y
    gcn_res, rcnt = [0, 0, 0, 0], 5
    X_new, Y_new, A_normed_new, adj_mat_new = get_input_vars()

    for i in range(rcnt):
        X, Y, A_normed, adj_mat = dcopy(X_new), dcopy(Y_new), dcopy(A_normed_new), dcopy(adj_mat_new)
        embed_gcn = gcn_train(X, Y, A_normed, adj_mat, epoch=epoch, lr=lr, weight_decay=weight_decay, esize=esize,
                              random_seed=random_seed + i)
        embe_feas = embed_gcn.detach().numpy()
        cur_x, cur_y = dcopy(train_x.values), dcopy(train_y.values)
        np_fealab = np.hstack((cur_x, embe_feas, cur_y))
        #         np_fealab = np.hstack((embe_feas, dcopy(train_y.values)))
        columns_name = ['f%02d' % i for i in range(np_fealab.shape[1] - 1)] + ['label']
        df_fealab = pd.DataFrame(data=np_fealab, columns=columns_name, dtype=float)
        df_fealab['label'] = df_fealab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fealab.columns if x not in y_cols_name]

        gcndf_x = df_fealab[x_cols_name]
        gcndf_y = df_fealab[y_cols_name]

        lgb_res = lgb_train_model(gcndf_x, gcndf_y, RANDOM_SEED)
        #         coprint.coprint(str(i) + '  ' + str(random_seed+i) + ' ' +  str(lgb_res), 'cyan')
        for i in range(len(gcn_res)):
            gcn_res[i] += lgb_res[i]
        gc.collect()

    gcn_res = [round(i / rcnt, 4) for i in gcn_res]
    gc.collect()
    return gcn_res
    # gcn_res = gcn_tree(epoch=6, lr=0.0035, weight_decay=1e-6, esize=8, random_seed=7)


class ARGS:
    def __init__(self, cmd):
        self.seed = cmd.seed
        self.verbose = cmd.verbose
        self.device = cmd.device
        self.sample_size = cmd.sample_size
        self.direct_attack = not cmd.indirect_attack
        self.us = not cmd.n_us

        self.PUBLICDATA_PATH = '/home/whx/GraphData/datasets/jiaying/publicdata/'
        self.PUBLICDATA_PATH = 'C://Users/pc/GraphData/datasets/jiaying/publicdata/'
        self.SAMPLE_MULGS_PATH = os.path.join(self.PUBLICDATA_PATH, 'graph_%d/SP_MulGs.pkl' % self.sample_size)
        self.FEATURES_PATH = os.path.join(self.PUBLICDATA_PATH, 'graph_%d/features.dat' % self.sample_size)
        self.DATA_PATH = os.path.join(self.PUBLICDATA_PATH, 'graph_%d' % self.sample_size)
        self.lr = cmd.learning_rate
        self.epoch = cmd.epoch
        self.weight_decay = cmd.weight_decay
        self.cuda = torch.cuda.is_available() and cmd.device == "gpu"

# gpu实现
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--verbose", default=0, type=int, help="print details")
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"], help="code environment")

    parser.add_argument("-ss", "--sample_size", default=30000, type=int, help="sample size")
    parser.add_argument("-in_da", "--indirect_attack", action="store_true", help="indirect attack")
    parser.add_argument("-lr", "--learning_rate", default=0.005, type=float, help="learning rate")
    parser.add_argument("--epoch", default=6, type=int)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--n_us", action="store_true", help="run sga model")
    cmd = parser.parse_args()
    args = ARGS(cmd)

    DATA_PATH = args.DATA_PATH
    RANDOM_SEED = args.seed

    df = load_pickle(args.FEATURES_PATH)
    sp_mulG = load_pickle(args.SAMPLE_MULGS_PATH)

    y_cols_name = ['label']
    x_cols_name = [x for x in df.columns if x not in y_cols_name]
    train_x = dcopy(df[x_cols_name])
    train_y = dcopy(df[y_cols_name])
    scipy_adj_matrix = nx.convert_matrix.to_scipy_sparse_matrix(sp_mulG, format='coo')

    gcn_res = gcn_tree(epoch=args.epoch, lr=args.lr, weight_decay=args.weight_decay, esize=8, random_seed=args.seed)
    print(gcn_res)