"""
   Deep learning toolkit
"""
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

"""
   general toolkit
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import networkx as nx
from tqdm import tqdm  # show a smart progress meter

"""
   GNN toolkit
"""
import dgl
from dgl.nn.pytorch import GraphConv


"""
IMPORTING CUSTOM MODULES/METHODS
"""
from data_test_case import case33_tieline, case33_tieline_DG
from NN_model import DNN_TieLine, DNN_VarCon
from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset

"""
==================== Graph dataset preparation =================
"""
# ====================== build a graph based on the system ===============
# define the graph by the source nodes and destination nodes
grid = case33_tieline()
src = grid["line"][:, 1] - 1  # source nodes in array format
dst = grid["line"][:, 2] - 1   # destination nodes in array format

# Edges are directional in DGL; Make them bi-directional.
u = np.concatenate([src, dst])
v = np.concatenate([dst, src])

# build graph using DGL
g = dgl.DGLGraph((u, v))
print('We have %d nodes.' % g.number_of_nodes())
print('We have %d edges.' % g.number_of_edges())

# Since the actual graph is undirected, we convert it for visualization purpose.
nx_G = g.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

"""
# ======================= add features to the nodes and edges =====================
"""
dt = pd.read_csv("trajectory_BC__2021_01_05_14_36.csv", converters={'line': eval, 'load': eval, 'action': eval})
n_sample = dt.shape[0]
feat_e = np.reshape(dt['line'].iloc[0], (1, 37))
feat_n = np.reshape(dt['load'].iloc[0], (1, 33))
# inverse one-hot encoding
y = np.reshape(dt['action'].iloc[0].index(1), (1, 1)) # find the index of the one element as label
for i in range(1, n_sample):
    feat_e = np.append(feat_e, np.reshape(dt['line'].iloc[i], (1, 37)), axis=0)
    feat_n = np.append(feat_n, np.reshape(dt['load'].iloc[i], (1, 33)), axis=0)
    y = np.append(y, np.reshape(dt['action'].iloc[i].index(1), (1, 1)), axis=0)
feat_e = pd.DataFrame(feat_e)
feat_n = pd.DataFrame(feat_n)


"""
============== Define a graph convolutional neural networks =================
"""
use_gpu = False; gpu_id = -1; device = None # CPU

MODEL_NAME = 'GatedGCN'

n_heads = -1
edge_feat = False
pseudo_dim_MoNet = -1
kernel = -1
gnn_per_block = -1
embedding_dim = -1
pool_ratio = -1
n_mlp_GIN = -1
gated = False
self_loop = False
# self_loop = True
max_time = 12

if MODEL_NAME == 'GatedGCN':
    seed = 41;
    epochs = 1000;
    batch_size = 5;
    init_lr = 5e-5;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    hidden_dim = 70;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'mean'

# generic new_params
net_params = {}
net_params['device'] = device
net_params['gated'] = gated  # for mlpnet baseline
net_params['in_dim'] = feat_n.shape[0]  # each node has three feature values representing the RGB
net_params['in_dim_edge'] = feat_e.shape[0] # each node has one feature value representing the Euclidean length
net_params['residual'] = True
net_params['hidden_dim'] = hidden_dim
net_params['out_dim'] = out_dim
num_classes = len(np.unique(np.array(y)))
net_params['n_classes'] = num_classes
net_params['n_heads'] = n_heads
net_params['L'] = L  # min L should be 2
net_params['readout'] = "sum"
net_params['layer_norm'] = True
net_params['batch_norm'] = True
net_params['in_feat_dropout'] = 0.0
net_params['dropout'] = 0.0
net_params['edge_feat'] = edge_feat
net_params['self_loop'] = self_loop

model = gnn_model(MODEL_NAME, net_params)


"""
=========================== Training =======================
"""
# (1) Pytorch Dataset is based on Python typing.generic
# (2) Pytorch DataLoader retrieve the data based on the index, that is, x[i]
# (3) So, it is important to define __getitem__(self, index) so that DataLoader can retrieve the correct samples
# (4) a Python iterator object must implement two special methods, __iter__() and __next__(), collectively called the iterator protocol
from torch.utils.data import Dataset
class torch_data_def(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

# # ================= test code for the Pytorch Dataset and DataLoader ===================
# X = np.array([[1, 2], [2, 3], [6, 7], [3, 5], [6, 9], [3, 5], [6, 7]])
# Y = np.array([[114], [203], [678], [345], [123], [423], [523]])
# # using both Dataset and DataLoader
# torch_data = torch_data_def(X, Y)
# a = DataLoader(torch_data, batch_size=2)
# for batch_ndx, (x, y, idx) in enumerate(a):
#     print(batch_ndx)
#     print(x)
#     print(y)
#     print(idx)
# # using only DataLoader
# b = DataLoader(X, batch_size=2)
# for batch_ndx, sample in enumerate(b):
#     print(batch_ndx)
#     print(sample)
# # ================= test code for list ===================
a = [1, 2, 3, 4, 5]
for i in a:
    print(i)


# # ================= create Pytorch DataLoader for training and testing data ===================
# a = DataLoader(feat_e.to_numpy(), batch_size=1)
# for batch_ndx, sample in enumerate(a):
#     print(sample)

# train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
# val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
# test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

# parameters
params = {}
params['seed'] = seed
params['epochs'] = epochs
params['batch_size'] = batch_size
params['init_lr'] = init_lr
params['lr_reduce_factor'] = lr_reduce_factor
params['lr_schedule_patience'] = lr_schedule_patience
params['min_lr'] = min_lr
params['weight_decay'] = weight_decay
params['print_epoch_interval'] = 5
params['max_time'] = max_time
optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

# =============== begine training =============
model.train()
epoch_loss = 0
epoch_train_acc = 0
nb_data = 0
gpu_mem = 0


# for iter, (batch_graphs, batch_labels) in enumerate(train_loader):
#     batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
#     batch_e = batch_graphs.edata['feat'].to(device)
#     batch_labels = batch_labels.to(device)
#     optimizer.zero_grad()
#
#     batch_scores = model.forward(batch_graphs, batch_x, batch_e)
#     loss = model.loss(batch_scores, batch_labels)
#     loss.backward()
#     optimizer.step()
#     epoch_loss += loss.detach().item()
#     epoch_train_acc += accuracy(batch_scores, batch_labels)
#     nb_data += batch_labels.size(0)
# epoch_loss /= (iter + 1)
# epoch_train_acc /= nb_data

# model.eval()
# epoch_test_loss = 0
# epoch_test_acc = 0
# nb_data = 0
# with torch.no_grad():
#     for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
#         batch_x = batch_graphs.ndata['feat'].to(device)
#         batch_e = batch_graphs.edata['feat'].to(device)
#         batch_labels = batch_labels.to(device)
#
#         batch_scores = model.forward(batch_graphs, batch_x, batch_e)
#         loss = model.loss(batch_scores, batch_labels)
#         epoch_test_loss += loss.detach().item()
#         epoch_test_acc += accuracy(batch_scores, batch_labels)
#         nb_data += batch_labels.size(0)
#     epoch_test_loss /= (iter + 1)
#     epoch_test_acc /= nb_data