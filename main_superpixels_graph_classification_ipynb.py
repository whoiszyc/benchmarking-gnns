




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

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
from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
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

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
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
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # import train functions specific for WL-GNNs
        from train.train_superpixels_graph_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network

        train_loader = DataLoader(trainset, shuffle=True, collate_fn=dataset.collate_dense_gnn)
        val_loader = DataLoader(valset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
        test_loader = DataLoader(testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)

    else:
        # import train functions for all other GCNs
        from train.train_superpixels_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

        train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                else:   # for all other models common train function
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

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

                
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)    

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_acc = evaluate_network(model, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(test_acc))*100, np.mean(np.array(train_acc))*100, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
               




def main(config):
    """
        USER CONTROLS
    """

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
        
    # Superpixels
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        max_num_nodes_train = max([dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))])
        max_num_nodes_test = max([dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))])
        max_num_node = max(max_num_nodes_train, max_num_nodes_test)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']
        
    if MODEL_NAME == 'RingGNN':
        num_nodes_train = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))]
        num_nodes = num_nodes_train + num_nodes_test
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
        
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)



# ==================== setup parameters ==============
# select GPU or CPU
#use_gpu = True; gpu_id = 0; device = None # default GPU
use_gpu = False; gpu_id = -1; device = None # CPU

# MODEL_NAME = '3WLGNN'
# MODEL_NAME = 'RingGNN'
MODEL_NAME = 'GatedGCN'
# MODEL_NAME = 'MoNet'
# MODEL_NAME = 'GCN'
# MODEL_NAME = 'GAT'
# MODEL_NAME = 'GraphSage'
# MODEL_NAME = 'DiffPool'
# MODEL_NAME = 'MLP'
# MODEL_NAME = 'GIN'

DATASET_NAME = 'MNIST'
# DATASET_NAME = 'CIFAR10'

out_dir = 'out/superpixels_graph_classification/'
root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')

print("[I] Loading data (notebook) ...")
dataset = LoadData(DATASET_NAME)
trainset, valset, testset = dataset.train, dataset.val, dataset.test
print("[I] Finished loading.")

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
    readout = 'mean'

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
    hidden_dim = 19;
    out_dim = n_heads * hidden_dim;
    dropout = 0.0;
    readout = 'mean'
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
    hidden_dim = 108;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'mean'

if MODEL_NAME == 'MLP':
    seed = 41;
    epochs = 1000;
    batch_size = 50;
    init_lr = 5e-4;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    gated = False;  # MEAN
    L = 4;
    hidden_dim = 168;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'mean'
    gated = True;  # GATED
    L = 4;
    hidden_dim = 150;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'mean'

if MODEL_NAME == 'DiffPool':
    seed = 41;
    epochs = 1000;
    batch_size = 50;
    init_lr = 5e-4;
    lr_reduce_factor = 0.5;
    lr_schedule_patience = 25;
    min_lr = 1e-6;
    weight_decay = 0
    L = 4;
    hidden_dim = 32;
    out_dim = hidden_dim;
    dropout = 0.0;
    readout = 'mean'
    n_heads = 8;
    gnn_per_block = 3;
    batch_size = 128;
    pool_ratio = 0.15
    embedding_dim = 32;  # MNIST
    # embedding_dim=16; # CIFAR10

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
    readout = 'mean'
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
    readout = 'mean'
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
    L = 4;
    hidden_dim = 25;
    out_dim = hidden_dim;
    dropout = 0.0;

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
    hidden_dim = 80;
    out_dim = hidden_dim;
    dropout = 0.0;

# generic new_params
net_params = {}
net_params['device'] = device
net_params['gated'] = gated  # for mlpnet baseline
net_params['in_dim'] = trainset[0][0].ndata['feat'][0].size(0)  # each node has three feature values representing the RGB
net_params['in_dim_edge'] = trainset[0][0].edata['feat'][0].size(0) # each node has one feature value representing the Euclidean length
net_params['residual'] = True
net_params['hidden_dim'] = hidden_dim
net_params['out_dim'] = out_dim
num_classes = len(np.unique(np.array(trainset[:][1])))
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

# for MLPNet
net_params['gated'] = gated

# specific for MoNet
net_params['pseudo_dim_MoNet'] = pseudo_dim_MoNet
net_params['kernel'] = kernel

# specific for GIN
net_params['n_mlp_GIN'] = n_mlp_GIN
net_params['learn_eps_GIN'] = True
net_params['neighbor_aggr_GIN'] = 'sum'

# specific for graphsage
net_params['sage_aggregator'] = 'meanpool'

# specific for diffpoolnet
net_params['data_mode'] = 'default'
net_params['gnn_per_block'] = gnn_per_block
net_params['embedding_dim'] = embedding_dim
net_params['pool_ratio'] = pool_ratio
net_params['linkpred'] = True
net_params['num_pool'] = 1
net_params['cat'] = False
net_params['batch_size'] = batch_size

# specific for RingGNN
net_params['radius'] = 2
num_nodes_train = [trainset[i][0].number_of_nodes() for i in range(len(trainset))]
num_nodes_test = [testset[i][0].number_of_nodes() for i in range(len(testset))]
num_nodes = num_nodes_train + num_nodes_test
net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

# specific for 3WLGNN
net_params['depth_of_mlp'] = 2

# calculate assignment dimension: pool_ratio * largest graph's maximum
# number of nodes  in the dataset
max_num_nodes_train = max(num_nodes_train)
max_num_nodes_test = max(num_nodes_test)
max_num_node = max(max_num_nodes_train, max_num_nodes_test)
net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

view_model_param(MODEL_NAME, net_params)

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


main(config)
















