"""
    IMPORTING LIBS
"""
# The work is based on Pytorch and Deep Graph Library (DGL)
# Deep Graph Library (DGL) is a Python package built for easy implementation of graph neural network model family,
# on top of existing DL frameworks (e.g. PyTorch, MXNet, Gluon etc.).

# The raw graph data is based on numpy sparse matrix, which can be reviewed by a.toarray()


import numpy as np
import os
import socket
import time
import random
import glob
import argparse
import json

import dgl  # Deep Graph Library (DGL)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.CSL_graph_classification.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []
    avg_epochs = []

    t0 = time.time()
    per_epoch_time = []

    dataset = LoadData(DATASET_NAME)

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    if net_params['pos_enc']:
        print("[!] Adding graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])  #TODO

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for split_number in range(5):

            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])

            model = gnn_model(MODEL_NAME, net_params)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []

            # batching exception for Diffpool
            drop_last = True if MODEL_NAME == 'DiffPool' else False
            # drop_last = False

            if MODEL_NAME in ['RingGNN', '3WLGNN']:
                # import train functions specific for WL-GNNs
                from train.train_CSL_graph_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network
                from functools import partial  # util function to pass pos_enc flag to collate function

                train_loader = DataLoader(trainset, shuffle=True, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
                val_loader = DataLoader(valset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
                test_loader = DataLoader(testset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))

            else:
                # import train functions for all other GCNs
                from train.train_CSL_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

                train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
                val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
                test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

            with tqdm(range(params['epochs'])) as t:
                for epoch in t:

                    t.set_description('Epoch %d' % epoch)

                    start = time.time()

                    if MODEL_NAME in ['RingGNN', '3WLGNN']:  # since different batch training function for dense GNNs
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                    else:  # for all other models common train function
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

                    # epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)

                    _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)

                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    epoch_train_acc = 100. * epoch_train_acc
                    epoch_test_acc = 100. * epoch_test_acc

                    t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)

                    per_epoch_time.append(time.time() - start)

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch - 1:
                            os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break

                    # Stop training after params['max_time'] hours
                    if time.time() - t0_split > params['max_time'] * 3600 / 10:  # Dividing max_time by 10, since there are 10 runs in TUs
                        print('-' * 89)
                        print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time'] / 10))
                        break

            _, test_acc = evaluate_network(model, device, test_loader, epoch)
            _, train_acc = evaluate_network(model, device, train_loader, epoch)
            avg_test_acc.append(test_acc)
            avg_train_acc.append(train_acc)
            avg_epochs.append(epoch)

            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time() - t0) / 3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""" \
          .format(np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}""" \
          .format(np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.3f}\n with test acc s.d. {:.3f}\nTRAIN ACCURACY averaged: {:.3f}\n with train s.d. {:.3f}\n\n
    Convergence Time (Epochs): {:.3f}\nTotal Time Taken: {:.3f} hrs\nAverage Time Per Epoch: {:.3f} s\n\n\nAll Splits Test Accuracies: {}\n\nAll Splits Train Accuracies: {}""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100,
                        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100, np.mean(np.array(avg_epochs)),
                        (time.time() - t0) / 3600, np.mean(per_epoch_time), avg_test_acc, avg_train_acc))


def main(config=None):

    # parameters
    params = config['params']

    # dataset
    DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)

    # device
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    out_dir = config['out_dir']

    # GNN model
    MODEL_NAME = config['model']

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    # CSL
    net_params['num_node_type'] = dataset.all.num_node_type
    net_params['num_edge_type'] = dataset.all.num_edge_type
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes

    # RingGNN
    if MODEL_NAME == 'RingGNN':
        num_nodes_train = [dataset.train[0][i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[0][i][0].number_of_nodes() for i in range(len(dataset.test))]
        num_nodes = num_nodes_train + num_nodes_test
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

    # RingGNN, 3WLGNN
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        if net_params['pos_enc']:
            net_params['in_dim'] = net_params['pos_enc_dim']
        else:
            net_params['in_dim'] = 1

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime(
        '%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime(
        '%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime(
        '%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)

# ==================== setup parameters ==============
# select GPU or CPU
#use_gpu = True; gpu_id = 0; device = None # default GPU
use_gpu = False; gpu_id = -1; device = None # CPU

# """
#     USER CONTROLS
# """
# MODEL_NAME = 'GatedGCN'
# MODEL_NAME = 'MoNet'
MODEL_NAME = 'GCN'
# MODEL_NAME = 'GAT'
# MODEL_NAME = 'GraphSage'
# MODEL_NAME = 'DiffPool'
# MODEL_NAME = 'MLP'
# MODEL_NAME = '3WLGNN'

DATASET_NAME = 'CSL'
out_dir = 'out/CSL_graph_classification/'
root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')

print("[I] Loading data (notebook) ...")
dataset = LoadData(DATASET_NAME)
trainset, valset, testset = dataset.train, dataset.val, dataset.test
print("[I] Finished loading.")


# """
#     PARAMETERS
# """
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
max_time = 48
residual = True
layer_norm = True
batch_norm = True
pos_enc_dim = 20

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
    readout = 'sum'
    init_lr = 5e-4;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 1;
    pos_enc_dim = 20;
    batch_size = 5;  # v1
    init_lr = 5e-4;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 10;
    pos_enc_dim = 20;
    batch_size = 5;  # v2

if MODEL_NAME == 'GCN':
    seed = 41;
    epochs = 1000;
    batch_size = 5;
    init_lr = 5e-5;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    hidden_dim = 146;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'sum'

if MODEL_NAME == 'GAT':
    seed = 41;
    epochs = 1000;
    batch_size = 50;
    init_lr = 5e-5;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    n_heads = 8;
    hidden_dim = 18;
    out_dim = n_heads * hidden_dim;
    dropout = 0.0;
    readout = 'sum'
    print('True hidden dim:', out_dim)

if MODEL_NAME == 'GraphSage':
    seed = 41;
    epochs = 1000;
    batch_size = 50;
    init_lr = 5e-5;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    hidden_dim = 90;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'sum'

if MODEL_NAME == 'MLP':
    seed = 41
    epochs = 10
    batch_size = 50
    init_lr = 5e-4
    lr_reduce_factor = 0.5
    lr_schedule_patience = 25
    min_lr = 1e-6
    weight_decay = 0
    L = 4
    hidden_dim = 145
    out_dim = hidden_dim
    dropout = 0.0
    readout = 'sum'

if MODEL_NAME == 'GIN':
    seed = 41;
    epochs = 1000;
    batch_size = 50;
    init_lr = 5e-4;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    hidden_dim = 110;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'sum'
    n_mlp_GIN = 2;
    learn_eps_GIN = True;
    neighbor_aggr_GIN = 'sum'

if MODEL_NAME == 'MoNet':
    seed = 41;
    epochs = 1000;
    batch_size = 50;
    init_lr = 5e-4;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    hidden_dim = 90;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'sum'
    pseudo_dim_MoNet = 2;
    kernel = 3;

if MODEL_NAME == 'RingGNN':
    seed = 41;
    epochs = 1000;
    batch_size = 1;
    init_lr = 5e-5;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    # L=4; hidden_dim=145; out_dim=hidden_dim; dropout=0.0; readout='mean'
    L = 2;
    hidden_dim = 37;
    out_dim = hidden_dim;
    dropout = 0.0;
    edge_feat = False
    residual = False;
    layer_norm = False;
    batch_norm = False

if MODEL_NAME == '3WLGNN':
    seed = 41;
    epochs = 1000;
    batch_size = 1;
    init_lr = 5e-5;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    # L=4; hidden_dim=145; out_dim=hidden_dim; dropout=0.0; readout='mean'
    L = 3;
    hidden_dim = 78;
    out_dim = hidden_dim;
    dropout = 0.0;
    edge_feat = False
    residual = False;
    layer_norm = False;
    batch_norm = False

# DEV
# epochs=10

# generic new_params
net_params = {}
net_params['device'] = device
net_params['num_node_type'] = dataset.all.num_node_type
net_params['num_edge_type'] = dataset.all.num_edge_type
net_params['gated'] = False  # for mlpnet baseline
net_params['residual'] = residual
net_params['hidden_dim'] = hidden_dim
net_params['out_dim'] = out_dim
num_classes = len(np.unique(dataset.all.graph_labels))
net_params['n_classes'] = num_classes
net_params['n_heads'] = n_heads
net_params['L'] = L  # min L should be 2
net_params['readout'] = "mean"
net_params['layer_norm'] = layer_norm
net_params['batch_norm'] = batch_norm
net_params['in_feat_dropout'] = 0.0
net_params['dropout'] = 0.0
net_params['edge_feat'] = edge_feat
net_params['self_loop'] = self_loop

# specific for MoNet
net_params['pseudo_dim_MoNet'] = pseudo_dim_MoNet
net_params['kernel'] = kernel

# specific for GIN
net_params['n_mlp_GIN'] = n_mlp_GIN
net_params['learn_eps_GIN'] = True
net_params['neighbor_aggr_GIN'] = 'sum'

# specific for graphsage
net_params['sage_aggregator'] = 'meanpool'
net_params['sage_aggregator'] = 'maxpool'

# specific for RingGNN
net_params['radius'] = 2
run = 0
num_nodes_train = [trainset[run][i][0].number_of_nodes() for i in range(len(trainset))]
num_nodes_test = [testset[run][i][0].number_of_nodes() for i in range(len(testset))]
num_nodes = num_nodes_train + num_nodes_test
net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
net_params['in_dim'] = pos_enc_dim

# specific for 3WLGNN
net_params['depth_of_mlp'] = 2
net_params['in_dim'] = pos_enc_dim

# specific for pos_enc_dim
net_params['pos_enc'] = True
net_params['pos_enc_dim'] = pos_enc_dim

view_model_param(MODEL_NAME, net_params)

# ==================== run the script ===============
config = {}
# gpu config
gpu = {}
gpu['use'] = use_gpu
gpu['id'] = gpu_id
config['gpu'] = gpu
# GNN model, dataset, out_dir
config['model'] = MODEL_NAME
config['dataset'] = DATASET_NAME
config['out_dir'] = out_dir
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
config['params'] = params
# network parameters
config['net_params'] = net_params

# run the main function
main(config)






