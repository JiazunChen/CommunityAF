import pathlib
import collections
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
from graph import Graph


class PretrainComDataset(Dataset):
    def __init__(self, com, train_size):
        self.com = com
        self.n_com = train_size

    def __len__(self):
        return self.n_com

    def __getitem__(self, idx):
        return torch.tensor(idx, dtype=torch.long)

def load_dataset(root, name):
    root = pathlib.Path(root)
    prefix = f'{name}-1.90'
    with open(root / f'{prefix}.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')
        edges = np.array([[int(i) for i in x.split()] for x in edges])
    with open(root / f'{prefix}.cmty.txt') as fh:
        comms = fh.read().strip().split('\n')
        comms = [[int(i) for i in x.split()] for x in comms]
    if (root / f'{prefix}.nodefeat.txt').exists():
        with open(root / f'{prefix}.nodefeat.txt') as fh:
            nodefeats = [x.split() for x in fh.read().strip().split('\n')]
            nodefeats = {int(k): [int(i) for i in v] for k, *v in nodefeats}
    else:
        nodefeats = None
    print('edge', len(edges))
    graph = Graph(edges)
    nxg = nx.Graph()
    nxg.add_edges_from(edges)
    return graph, comms, nodefeats, prefix,nxg

def split_comms(graph, comms, train_size,community_min):
    train_comms, valid_comms = comms[:train_size], comms[train_size:]
    test_comms = []
    print(f'blen {len(train_comms)} {len(valid_comms)}')
    train_comms = [list(x) for nodes in train_comms for x in graph.connected_components(nodes) if
                   len(x) >= community_min]
    valid_comms = [list(x) for nodes in valid_comms for x in graph.connected_components(nodes) if
                   len(x) >= community_min]
    print(f'alen {len(train_comms)} {len(valid_comms)}')
    max_size = max(len(x) for x in train_comms + valid_comms + test_comms)
    return train_comms, valid_comms, test_comms, max_size

def choose_seed(mode,x):
    if mode =='max':
        return max(x)
    elif mode =='min':
        return min(x)
    else:
        return np.random.choice(x)

def load_data(args):
    graph, comms, nodefeats, ds_name, nxg = load_dataset(args.root, args.dataset)
    train_comms, valid_comms, test_comms, max_size = split_comms(graph, comms, args.train_size,args.community_min)
    args.ds_name = ds_name
    args.max_size = max_size

    eval_seeds = [choose_seed(args.eval_seed_mode,x) for x in valid_comms] #[max(x) for x in valid_comms] #  [min(x) for x in valid_comms] or  [np.random.choice(x) for x in valid_comms]
    print(f'[{ds_name}] # Nodes: {graph.n_nodes} ', flush=True)
    print(f'[# comms] Train: {len(train_comms)} Valid: {len(valid_comms)} Test: {len(test_comms)}', flush=True)
    return graph, train_comms, valid_comms, test_comms, eval_seeds, nodefeats, ds_name, nxg


def connected_components(g, nodes):

    remaining = set(nodes)
    ccs = []
    cc = set()
    queue = collections.deque()
    while len(remaining) or len(queue):
        # print(queue, remaining)
        if len(queue) == 0:
            if len(cc):
                ccs.append(cc)
            v = remaining.pop()
            cc = {v}
            queue.extend(set(g.neighbors(v)) & remaining)
            remaining -= {v}
            remaining -= set(g.neighbors(v))
        else:
            v = queue.popleft()
            queue.extend(set(g.neighbors(v)) & remaining)
            cc |= (set(g.neighbors(v)) & remaining) | {v}
            remaining -= set(g.neighbors(v))
    if len(cc):
        ccs.append(cc)
    return ccs


def preprocess_nodefeats(conv, nodefeats, hidden_size=64):
    ind = np.array([[i, j] for i, js in nodefeats.items() for j in js])
    sp_feats = sp.csr_matrix((np.ones(len(ind)), (ind[:, 0], ind[:, 1])))
    convolved_feats = conv(sp_feats)
    svd = TruncatedSVD(hidden_size, 'arpack')
    x = svd.fit_transform(convolved_feats)
    x = (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True)
    return x

def myshortest_path(nxg, seed):
    path_map = {}
    vis = [seed]
    path_map[seed] = [seed]
    cnt = 0
    while cnt < len(vis):
        u = vis[cnt]
        cnt += 1
        if len(path_map[u]) > 5:
            continue
        for v in nxg.neighbors(u):
            if v in path_map.keys():
                continue
            else:
                path_map[v] = path_map[u] + [v]
                vis.append(v)
    return path_map

def get_augment(seeds,graph,nxg,conv):
    bs = len(seeds)
    feat = np.zeros((bs, graph.n_nodes), dtype=np.float32)
    for i, seed in enumerate(seeds):
        shortest_path = myshortest_path(nxg, seed)
        len_path = [(key, 1 / len(shortest_path[key])) for key in shortest_path.keys() if len(shortest_path[key]) < 5]
        data = [v for k, v in len_path]
        ind = [k for k, v in len_path]
        feat[i][ind] = data
    return conv(sp.csr_matrix(feat.T)).T





def get_data(idx,args,roll_mapper,pre_dataset):
    result = {
        "batch_neighbor": [],
        "batch_next_onehot": [],
        "batch_z_seed": [],
        "batch_z_node": [],
        "batch_com": [],
        "batch_labels": []
    }
    if args.rankingloss:
        result["batch_neg"] = []
        result["batch_neg_neighbors"] = []
        result["batch_neg_z_seed"] = []
        result["batch_neg_z_node"] = []
        result["batch_neg_z_augment"] = []

    if args.augment:
        result["batch_z_augment"] = []
    for j in idx:

        for i in roll_mapper[j]:
            result["batch_neighbor"].extend(pre_dataset[i]['neighbor'])
            result["batch_next_onehot"].extend(pre_dataset[i]['next_onehot'])
            result["batch_z_seed"].extend(pre_dataset[i]['z_seed'])
            result["batch_z_node"].extend(pre_dataset[i]['z_node'])
            result["batch_com"].extend(pre_dataset[i]['now_com'])
            result["batch_labels"].extend(pre_dataset[i]['label'])
            if args.rankingloss:
                result["batch_neg"].append(pre_dataset[i]['neg'])
                result["batch_neg_neighbors"].append(pre_dataset[i]['neg_neighbors'])
                result["batch_neg_z_seed"].append(pre_dataset[i]['neg_z_seed'])
                result["batch_neg_z_node"].append(pre_dataset[i]['neg_z_node'])
                result["batch_neg_z_augment"].append(pre_dataset[i]['neg_z_augment'])


            if args.augment:
                result["batch_z_augment"].extend(pre_dataset[i]['z_augment'])
    return result

def make_single_node_encoding(new_node, graph,conv):
    bs = len(new_node)
    n_nodes = graph.n_nodes
    ind = np.array([[v, i] for i, v in enumerate(new_node) if v is not None], dtype=np.int64).T
    if len(ind):
        data = np.ones(ind.shape[1], dtype=np.float32)
        x_nodes = conv(sp.csc_matrix((data, ind), shape=[n_nodes, bs])).T
    else:
        x_nodes = conv(sp.csc_matrix((n_nodes, bs), dtype=np.float32)).T
    return x_nodes

def make_nodes_encoding(new_node, graph, conv):
    bs = len(new_node)
    n_nodes = graph.n_nodes
    ind = [[v, i] for i, vs in enumerate(new_node) for v in vs]
    if len(ind):
        ind = np.asarray(ind, dtype=np.int64).T
        data = np.ones(ind.shape[1], dtype=np.float32)
        x_nodes = conv(sp.csc_matrix((data, ind), shape=[n_nodes, bs])).T
    else:
        x_nodes = conv(sp.csc_matrix((n_nodes, bs), dtype=np.float32)).T
    return x_nodes


