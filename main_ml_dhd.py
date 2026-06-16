import argparse
import copy
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)
from torch_sparse import SparseTensor
import time

import nxmetis
import networkx as nx
import numpy as np
from torch_geometric.utils import to_undirected
import pickle

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label.cpu(), args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if args.dataset not in {'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])



nx_G = nx.Graph()
node_list = list(range(n))
nx_G.add_nodes_from(node_list)
edge_list = list(zip(
    dataset.graph['edge_index'][0].numpy(),
    dataset.graph['edge_index'][1].numpy()
))
nx_G.add_edges_from(edge_list)

if args.method == 'hd':
    # ====== Original single-level coarsening ======
    print(args.num_centroids)
    _, communities = nxmetis.partition(nx_G, nparts=args.num_centroids)
    print(args.num_centroids, len(communities))

    nodes_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            nodes_to_community[node] = i

    # Build super graph
    super_G = nx.Graph()
    super_G.add_nodes_from(list(range(args.num_centroids)))
    for u, v in nx_G.edges():
        if nodes_to_community[u] != nodes_to_community[v]:
            super_G.add_edge(nodes_to_community[u], nodes_to_community[v])

    # Shortest path distance
    super_distances = dict(nx.all_pairs_shortest_path_length(super_G))
    max_dist = 30
    distance_matrix = np.zeros((n, args.num_centroids)) + max_dist
    for node_id in range(n):
        comm = nodes_to_community[node_id]
        for target_comm in super_G.nodes():
            if target_comm in super_distances.get(comm, {}):
                distance_matrix[node_id, target_comm] = super_distances[comm][target_comm]

    dataset.graph['distance_matrix'] = torch.tensor(
        distance_matrix, dtype=torch.float32).to(device)

    nodes_to_community_list = [nodes_to_community[i] for i in range(n)]
    dataset.graph['nodes_to_community_tensor'] = torch.tensor(
        nodes_to_community_list, dtype=torch.long).to(device)

    print(distance_matrix.shape, dataset.graph['nodes_to_community_tensor'].shape)

elif args.method == 'ml_hd':
    # ====== Multi-level coarsening ======
    num_centroids_list = args.num_centroids_list
    print(f"\nBuilding {len(num_centroids_list)}-level hierarchy: {num_centroids_list}")

    hierarchy_data = {
        'distance_matrices': [],
        'nodes_to_community': [],
        'num_levels': len(num_centroids_list),
        'num_centroids_list': num_centroids_list,
    }

    current_G = nx_G
    current_mapping = list(range(n))  # original node -> current level node

    for level, num_centroids in enumerate(num_centroids_list):
        actual_centroids = min(num_centroids, current_G.number_of_nodes())
        if actual_centroids < 2:
            print(f"  Level {level}: Too few nodes, stopping.")
            break

        print(f"\n  Level {level}: {current_G.number_of_nodes()} nodes -> {actual_centroids} communities")

        # # METIS partition
        _, communities = nxmetis.partition(current_G, nparts=actual_centroids)

        # # current node -> community mapping
        current_node_to_comm = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                current_node_to_comm[node] = comm_id

        # original node -> current level community mapping
        orig_to_comm = torch.zeros(n, dtype=torch.long)
        for orig_node in range(n):
            current_node = current_mapping[orig_node]
            orig_to_comm[orig_node] = current_node_to_comm.get(current_node, 0)

        hierarchy_data['nodes_to_community'].append(orig_to_comm.to(device))

        #Build super graph
        super_G = nx.Graph()
        super_G.add_nodes_from(list(range(actual_centroids)))
        for u, v in current_G.edges():
            comm_u = current_node_to_comm.get(u, 0)
            comm_v = current_node_to_comm.get(v, 0)
            if comm_u != comm_v:
                super_G.add_edge(comm_u, comm_v)

        # Shortest path distance
        super_distances = dict(nx.all_pairs_shortest_path_length(super_G))
        max_dist = 30
        distance_matrix = np.zeros((n, actual_centroids)) + max_dist
        for orig_node in range(n):
            comm_of_node = orig_to_comm[orig_node].item()
            for target_comm in super_G.nodes():
                if target_comm in super_distances.get(comm_of_node, {}):
                    distance_matrix[orig_node, target_comm] = \
                        super_distances[comm_of_node][target_comm]

        hierarchy_data['distance_matrices'].append(
            torch.tensor(distance_matrix, dtype=torch.float32).to(device)
        )

        print(f"    Super graph: {super_G.number_of_nodes()} nodes, {super_G.number_of_edges()} edges")
        print(f"    Distance matrix: {distance_matrix.shape}")

        # Update mapping and graph
        current_mapping = [orig_to_comm[i].item() for i in range(n)]
        current_G = super_G

    #Update actual levels
    hierarchy_data['num_levels'] = len(hierarchy_data['distance_matrices'])
    hierarchy_data['num_centroids_list'] = num_centroids_list[:hierarchy_data['num_levels']]

    dataset.graph['hierarchy_data'] = hierarchy_data

    print(f"\n=== Hierarchy summary ===")
    for l in range(hierarchy_data['num_levels']):
        dm = hierarchy_data['distance_matrices'][l]
        nc = hierarchy_data['nodes_to_community'][l]
        print(f"  Level {l}: dist_matrix {dm.shape}, num_communities {nc.unique().shape[0]}")
        


dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args.method, args, c, d, device)

# using rocauc as the eval function
if args.dataset in ('deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()

### Training loop ###
patience = 0
optimizer = torch.optim.Adam(
    model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        train_time=time.time()
        model.train()
        optimizer.zero_grad()
        emb = None
        out= model(dataset)
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(
                    dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        
        loss.backward()
        optimizer.step()
        # print(time.time()-train_time)
        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


file_name = f'{args.dataset}_{args.method}'

file_name += '.txt'
out_path = os.path.join(out_folder, file_name)
with open(out_path, 'a+') as f:
    print_str = make_print(args.method)
    f.write(print_str)
    f.write(results)
    f.write('\n\n')
