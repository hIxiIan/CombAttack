import os
import re
import numpy as np
import networkx as nx
import random
import torch
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.model_selection import ShuffleSplit
import scipy.io as io
import sys
import sklearn
import numba

def calculate_predictive_loss(target, predictions):
    prediction_loss = torch.nn.functional.nll_loss(predictions, target)
    return  prediction_loss

def calculate_reward(target, prediction):
    prediction = torch.argmax(prediction,dim=1)
    # if target.item() == prediction.item():
    #     reward = 1.0
    # else:
    #     reward = -1.0
    #reward = ((prediction == target).float() - 0.5) * 2
    acc = (prediction == target).int().sum()
    return acc, prediction

def create_batches(id_train, batch_size, cuda):
    batches = [id_train[i:i + batch_size] for i in range(0, len(id_train), batch_size)]
    for x in range(len(batches)):
        if cuda:
            batches[x] = torch.LongTensor(batches[x]).cuda()
        else:
            batches[x] = torch.LongTensor(batches[x])
    return batches

def update_log(self):
    average_loss = self.epoch_loss/self.nodes_processed
    self.logs["losses"].append(average_loss)
    
def create_logs(args):
    log = dict()
    log["losses"] = []
    log["params"] = vars(args)
    return log


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def  loadRedditFromNPZ(dataset_dir):
    adj  =  sp.load_npz(dataset_dir+"reddit_adj.npz")
    data  =  np.load(dataset_dir+"reddit.npz")

    return  adj,  data['feats'],  data['y_train'],  data['y_val'],  data['y_test'],  data['train_index'],  data['val_index'],  data['test_index']

def  loadEtherFromNPZ(dataset_dir):
    adj  =  sp.load_npz(dataset_dir+"ether_degree_adj_degrees_distribution.npz")
#     adj = coo_matrix((1402220,1402220), dtype=np.float64)
    data  =  np.load(dataset_dir+"ether_fastGCN.npz")
    np.nan_to_num(data['feats'])

    return  adj,  data['feats'],  data['y_train'],  data['y_val'],  data['y_test'],  data['train_index'],  data['val_index'],  data['test_index'],data['train_target']

def  Origin_load_ether_data(data_path="data/ether/",  normalization="AugNormAdj",  cuda=True):
#     adj,  features,  y_train,  y_val,  y_test,  train_index,  val_index,  test_index,train_target  =  loadEtherFromNPZ(data_path)
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index, train_target = loadEtherFromNPZ(data_path)
    in_transaction_value = np.array(features[:, 0])
    in_transaction_count = np.array(features[:, 1])
    out_transaction_value = np.array(features[:, 3])
    out_transaction_count = np.array(features[:, 4])
    in_transaction_value = np.reshape(in_transaction_value, newshape=(in_transaction_value.shape[0], 1))
    in_transaction_count = np.reshape(in_transaction_count, newshape=(in_transaction_count.shape[0], 1))
    out_transaction_value = np.reshape(out_transaction_value, newshape=(out_transaction_value.shape[0], 1))
    out_transaction_count = np.reshape(out_transaction_count, newshape=(out_transaction_count.shape[0], 1))
    features = np.concatenate(
        (in_transaction_value, in_transaction_count, out_transaction_value, out_transaction_count), axis=1)
    print("features = {} type = {}".format(features.shape, type(features)))
    # -------------------------------------------------------------
    labels  =  np.zeros(adj.shape[0])
    labels[train_index]    =  y_train
    labels[val_index]    =  y_val
    labels[test_index]    =  y_test
    ###################################code for exam with same label in adj##################################
#     adj  =  adj  +  adj.T - adj.T
#     target = []
#     for i in  range(816):
#         target.append(i)
#     adj_target = adj[target,  :][:,  target].todense().getA()
#     row = []
#     col = []
#     for i in range(816):
#         for j in range(816):
#             if adj_target[i][j] == 1 and labels[i] == labels[j]:
#                 row.append(i)
#                 col.append(j)
#     data = []
#     for i in range(len(row)):
#         data.append(1)
#     row = np.array(row)
#     col = np.array(col)
#     data = np.array(data)
#     adj = coo_matrix((data,(row,col)),shape=(1402220,1402220)) 
    #######################################end##############################################################
    sparse_adj  =  adj  +  adj.T - adj.T
    sparse_adj_train  =  sparse_adj[train_index,  :][:,  train_index]
    sparse_adj_train_all  =  sparse_adj[train_target,  :][:,  train_target]
    features = np.nan_to_num(features)
    features  =  torch.FloatTensor(np.array(features))
    features  =  (features-features.mean(dim=0))/features.std(dim=0)
    adj  =  sparse_mx_to_torch_sparse_tensor(sparse_adj).float()
    train_adj  =  sparse_mx_to_torch_sparse_tensor(sparse_adj_train).float()
    train_adj_all = sparse_mx_to_torch_sparse_tensor(sparse_adj_train_all).float()
    labels  =  torch.LongTensor(labels)
    if  cuda:
            adj =  adj.cuda()
            train_adj  =  train_adj.cuda()
            train_adj_all = train_adj_all.cuda()
            features  =  features.cuda()
            labels  =  labels.cuda()
    train_feature  =  features[train_index,  :]
    train_feature_all  =  features[train_target,  :]
    train_labels  =  labels[train_index]
    print('train_index', len(list(train_index)))
    print('val_index', len(list(val_index)))
    print('test_index', len(list(test_index)))
    print('adj', adj.size())
    print('features', features.size())
    train_index = list(range(len(list(train_index))))
    
    return sparse_adj, sparse_adj_train,sparse_adj_train_all, features, train_feature,train_feature_all, labels,  train_labels, train_index, val_index, test_index, labels.unique().size()[0]

def  Origin_load_reddit_data(data_path="data/reddit/",  normalization="AugNormAdj",  cuda=True):
    adj,  features,  y_train,  y_val,  y_test,  train_index,  val_index,  test_index  =  loadRedditFromNPZ(data_path)
    labels  =  np.zeros(adj.shape[0])
    labels[train_index]    =  y_train
    labels[val_index]    =  y_val
    labels[test_index]    =  y_test
    sparse_adj  =  adj  +  adj.T 
    sparse_adj_train  =  sparse_adj[train_index,  :][:,  train_index]

    
    features  =  torch.FloatTensor(np.array(features))
    features  =  (features-features.mean(dim=0))/features.std(dim=0)
    adj  =  sparse_mx_to_torch_sparse_tensor(sparse_adj).float()
    train_adj  =  sparse_mx_to_torch_sparse_tensor(sparse_adj_train).float()
    labels  =  torch.LongTensor(labels)
    if  cuda:
            adj =  adj.cuda()
            train_adj  =  train_adj.cuda()
            features  =  features.cuda()
            labels  =  labels.cuda()
    #return  adj,  train_adj,  features,  labels,  train_index,  val_index,  test_index
    train_feature  =  features[train_index,  :]
    train_labels  =  labels[train_index]
    print('train_index', len(list(train_index)))
    print('val_index', len(list(val_index)))
    print('test_index', len(list(test_index)))
    print('adj', adj.size())
    print('features', features.size())
    train_index = list(range(len(list(train_index))))
    
    return sparse_adj, sparse_adj_train, features, train_feature, labels,  train_labels, train_index, val_index, test_index, labels.unique().size()[0]

def create_batches_forWalk(walks, batch_size):
    batches = [walks[i:i + batch_size, : ] for i in range(0, len(walks), batch_size)]
    return batches

def create_batches_forList(id_train, batch_size, cuda):
    batches = [id_train[i:i + batch_size] for i in range(0, len(id_train), batch_size)]
    for x in range(len(batches)):
        if cuda:
            batches[x] = torch.LongTensor(batches[x]).cuda()
        else:
            batches[x] = torch.LongTensor(batches[x])
    return batches
def pre_sample_perbatch(walk_times,adj, train_index):
    nodes_num = len(train_index)
    walks = torch.zeros(nodes_num, walk_times+1).cuda().long()
    degrees = adj.sum(1)
    candi_node = 0
    
    walks[: , 0] = torch.tensor(train_index).cuda()
    candi_node = adj[train_index]
    chosen_node = torch.zeros(len(train_index), adj.shape[0]).cuda()
    for id in range(len(train_index)):
        chosen_node[id][train_index[id]] = 1.
    candi_node = ((candi_node - chosen_node)>= 1.0).float()
    for walk_id in range(walk_times):
        for x in range(candi_node.shape[0]):
            if candi_node[x].sum()==0:
                #print('error')
                candi_node[x][train_index[x]] = 1.
        p = candi_node * degrees
        p = p / (p.sum(1).unsqueeze(1))
        
        m = torch.distributions.categorical.Categorical(p)
        new_node = m.sample()
        walks[:, walk_id+1] = new_node
        for id in range(len(train_index)):
            chosen_node[id][new_node[id]] = 1.
        candi_node = candi_node + adj[new_node] - chosen_node
        candi_node = (candi_node>= 1.0).float()
    return walks
    
def pre_sample(walk_times,adj, train_index, batch_size, save_name, way, do_walk):
    if do_walk:
        nodes_num = len(train_index)
        walks = torch.zeros(nodes_num, walk_times+1).cuda().long()
        batches = create_batches_forList(train_index, batch_size, True)
        i=0
        degrees = adj.sum(1)
        candi_node = 0
        for batch in batches:
            walks[i*batch_size : i*batch_size + len(batch), 0] = batch
            candi_node = adj[batch]
            chosen_node = torch.zeros(len(batch), adj.shape[0]).cuda()
            for id in range(len(batch)):
                chosen_node[id][batch[id]] = 1.
            candi_node = ((candi_node - chosen_node)>= 1.0).float()
            for walk_id in range(walk_times):
                for x in range(candi_node.shape[0]):
                    if candi_node[x].sum()==0:
                        #print('error')
                        candi_node[x][batch[x]] = 1.
                p = candi_node * degrees
                p = p / (p.sum(1).unsqueeze(1))
                #p == 1
                #print(p,(candi_node.sum(1)==0).sum())
                
                m = torch.distributions.categorical.Categorical(p)
                new_node = m.sample()
                walks[i*batch_size : i*batch_size + len(batch), walk_id+1] = new_node
                for id in range(len(batch)):
                    chosen_node[id][new_node[id]] = 1.
                candi_node = candi_node + adj[new_node] - chosen_node
                candi_node = (candi_node>= 1.0).float()
                
            i+=1
        torch.cuda.empty_cache()
        #result1 = np.array(walks.cpu())
        #io.savemat('walks_'+save_name+'.mat',{save_name:result1})
        return walks
    else:
        walks = io.loadmat(save_name+'.mat')
#         walks = io.loadmat('walks_b/'+save_name+'.mat')
        return torch.tensor(walks[save_name]).cuda()
    
@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())

@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals

def pre_sample_with_pprgo(walk_times,adj, train_index, batch_size, save_name, way, do_walk,alpha,epsilon):
    topk = walk_times+1
    adj_matrix, attr_matrix, labels,train_idx, val_idx, test_idx = load_data_heterophily_for_sample("chameleon")
    train_idx = np.array(train_idx) 
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
#     print("shape of train_idx:",train_idx.shape)
#     print("shape of val_idx:",val_idx.shape)
#     print("shape of test_idx:",test_idx.shape)
    nodes_num = len(train_index)
    if len(train_index) == 1092:
        save_name = './walks_chameleon_pprgo/train_'+'seed='+str(0)+'_walk='+str(topk-1)
        nodes = train_idx
    elif len(train_index) == 729:
        save_name = './walks_chameleon_pprgo/val_'+'seed='+str(0)+'_walk='+str(topk-1)
        nodes = val_idx
    elif len(train_index) == 456:
        save_name = './walks_chameleon_pprgo/test_'+'seed='+str(0)+'_walk='+str(topk-1)
        nodes = test_idx
    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]
    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk)
    walks = np.zeros((nodes_num, topk))  
    for i in range(len(neighbors)):
        for j in range(topk):
            if len(neighbors[i])-j-1 < 0:
#                 print(str(i)+' did not have top'+str(j)+' neighbors, so we sample itself '+str(neighbors[i][len(neighbors[i])-1]))
                walks[i,j] = neighbors[i][len(neighbors[i])-1]
            else:
#                 print(str(i)+' did have top'+str(j))
#                 print(str(neighbors[i][-j-1])+' is the number')
                walks[i,j] = neighbors[i][-j-1]
   
    result1 = walks
#     print("result1",result1)
#     print("shape of result1",result1.shape)
#     print("type of result1",type(result1))
#     print("dtype of result1",result1.dtype)
    result1 = result1.astype(np.int64)
    io.savemat(save_name+'.mat',{save_name:result1})
    walks = io.loadmat(save_name+'.mat')
    return torch.tensor(walks[save_name]).cuda()
   

def pre_sample2(walk_times,adj, train_index, batch_size, save_name, way, do_walk):
    if do_walk:
        nodes_num = len(train_index)
        walks = torch.zeros(nodes_num, walk_times+1).cuda().long()
        batches = create_batches_forList(train_index, batch_size, True)
        i=0
        candi_node = 0
        for batch in batches:
            walks[i*batch_size : i*batch_size + len(batch), 0] = batch
            candi_node = adj[batch]
            for walk_id in range(walk_times):
                p = candi_node / (candi_node.sum(1).unsqueeze(1))
                p == 1
                #print(p,(candi_node.sum(1)==0).sum())
                m = torch.distributions.categorical.Categorical(p)
                new_node = m.sample()
                walks[i*batch_size : i*batch_size + len(batch), walk_id+1] = new_node
                if way == 'both':
                    candi_node = candi_node + adj[new_node]
                    candi_node = (candi_node>= 1.0).float()
                elif way == 'depth':
                    candi_node = adj[new_node]
                elif way == 'width':
                    pass
                else:
                    print('error')
            i+=1
        torch.cuda.empty_cache()
        #result1 = np.array(walks.cpu())
        #io.savemat('walks_'+save_name+'.mat',{save_name:result1})
        return walks
    else:
        walks = io.loadmat('walks_'+save_name+'.mat')
        return torch.tensor(walks[save_name]).cuda()
    
def sparse_index_select(list_of_rows, idxs):
    #print(batch_size, num_of_nodes)
    res = torch.zeros(idxs.shape[0], len(list_of_rows), dtype=torch.bool, device = torch.device('cuda', 0))
    #print(len(list_of_rows))
    for i in range(idxs.shape[0]):
        #print(idxs[i])
        res[i][list_of_rows[idxs[i]]] = 1
    #res =  res.requires_grad_(True)
    return res

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    coo = sparse_mx.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def load_data_heterophily_for_sample(dataset_name, train_percentage=0.6, val_percentage=0.2):
    graph_adjacency_list_file_path = os.path.join('data/new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('data/new_data', dataset_name,
                                                            f'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    assert (train_percentage is not None and val_percentage is not None)
    assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

    train_and_val_index, test_index = next(
        ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
            np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]

    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    dic_id2label = labels
    dic_id2label = np.array(dic_id2label)
    id_test = test_index
    id_train = list(train_index)
    id_valid = list(val_index)
    
    return adj, features, dic_id2label, id_train, id_valid, id_test


def load_data_heterophily(dataset_name, train_percentage=0.6, val_percentage=0.2):
    graph_adjacency_list_file_path = os.path.join('data/new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('data/new_data', dataset_name,
                                                            f'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    assert (train_percentage is not None and val_percentage is not None)
    assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

    train_and_val_index, test_index = next(
        ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
            np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]

    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    graph = torch.FloatTensor(adj.todense().getA()).cuda()
    isolate = (graph.sum(1)==0).nonzero()
    graph[isolate,isolate] = 1.
    dic_id2feature = preprocess_features(features)
    dic_id2feature = torch.FloatTensor(dic_id2feature).cuda()
    dic_id2label = labels
    dic_id2label = torch.LongTensor(np.array(dic_id2label)).cuda()
    id_test = test_index
    id_train = list(train_index)
    id_valid = list(val_index)
    np.set_printoptions(threshold=1000000)
#     print("id_train:",id_train)
#     print("id_valid:",id_valid)
#     print("id_test:",id_test)
#     print("graph:",graph.cpu().numpy())
#     print("dic_id2feature:",dic_id2feature.shape)
#     print("dic_id2label:",dic_id2label.shape)
#     print("dic_id2feature:",dic_id2feature.cpu().numpy())
#     print("dic_id2label:",dic_id2label.cpu().numpy())
#     print("features:",features)
#     print("labels:",labels.shape)
    return graph, dic_id2feature, dic_id2label, id_train, id_valid, id_test , num_labels


def load_data2(dataset_str, cuda): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    if dataset_str == 'reddit':
        return Origin_load_reddit_data(cuda=cuda)
    if dataset_str == 'ether':
        return Origin_load_ether_data(cuda=cuda)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
        
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer_gcn':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #features = normalize_features(features)
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #adj = adj + sp.eye(adj.shape[0])
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))
    print(len(list(idx_train)))
    print(len(list(idx_val)))
    print(len(list(idx_test)))
    
    dic_id2feature = preprocess_features(features).getA()
    #dic_id2feature = features.todense().getA()
    dic_id2feature = torch.FloatTensor(dic_id2feature).cuda()
    id_test = idx_test
    id_train = list(idx_train)
    id_valid = list(idx_val)
    num_labels = labels.shape[1]
    dic_id2label = [np.argmax(one_hot) for one_hot in labels]
    dic_id2label = torch.LongTensor(np.array(dic_id2label)).cuda()
    #dic_id2label = torch.FloatTensor(np.array(dic_id2label))
    
    #graph = nx.Graph(adj)
    #graph = torch.FloatTensor(adj.todense().getA())
    graph = torch.FloatTensor(adj.todense().getA()).cuda()
    isolate = (graph.sum(1)==0).nonzero()
    graph[isolate,isolate] = 1.
    print(adj.shape)
    print(features.shape)
    print(type(graph))

    return graph, dic_id2feature, dic_id2label, id_train, id_valid, id_test , num_labels

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
#     return features


