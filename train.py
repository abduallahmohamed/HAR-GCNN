#!/usr/bin/env python
# coding: utf-8

import os
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import argparse
from dataset import HARGCNNDataset
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as skmetrics

parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--nfeat', type=int, default=275)
parser.add_argument('--nhid', type=int, default=51)
parser.add_argument('--nodes_cnt', type=int, default=3)
parser.add_argument('--model', default='hargcnn',
                    help='hargcnn,cnn,lstm')


# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=30,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')
parser.add_argument('--data_per', type=float, default=1)
parser.add_argument('--tag', default='run_',
                    help='personal tag for the model ')

# Data specific parameters
parser.add_argument('--normalization', default='abduallahs',
                    help='abduallahs,kipfs')
parser.add_argument('--fet_vec_size', type=int, default=224)
parser.add_argument('--label_vec_size', type=int, default=51)
parser.add_argument('--miss_thr', type=float, default=0.5)
parser.add_argument('--noise_thr', type=float, default=0.5)
# Can be the version too like 0,1,2
parser.add_argument('--randomseed', type=int, default=100)
parser.add_argument('--dataset', default='ExtraSensory',
                    help='ExtraSensory,PAMAP')
parser.add_argument('--test', action="store_true", default=False,
                    help='Set to only test the model')
args = parser.parse_args()
print(args)

if args.dataset == "ExtraSensory":
    # Load the selected model
    if args.model == "hargcnn":
        from modelsExtraSensory.hargcnn import HARGCNN as Net
    elif args.model == "cnn":
        raise("Not implemented error")
    elif args.model == "lstm":
        raise("Not implemented error")

elif args.dataset == "PAMAP":
    # Load the selected model
    if args.model == "hargcnn":
        from modelsPAMAP.hargcnn import HARGCNN as Net
    elif args.model == "cnn":
        raise("Not implemented error")
    elif args.model == "lstm":
        raise("Not implemented error")


# Reproducability
torch.manual_seed(args.randomseed)
np.random.seed(seed=args.randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Model
model = Net(nfeat=args.nfeat, nhid=args.nhid, nadjf=args.nhid,
            args=args).cuda()  # All models should have the same signature


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Params: ", count_parameters(model))

# Data set
# Load the data
if args.dataset == "ExtraSensory":
    with open("./dataset/ExtraSensory.pkl", "rb") as f:
        esdswf = pickle.load(f)
    dset_train = HARGCNNDataset(esdswf["X_train"], esdswf["Y_train"],
                                nodes_count=args.nodes_cnt, miss_thr=args.miss_thr, noise_thr=args.noise_thr,
                                randomseed=args.randomseed, normalization=args.normalization,
                                fet_vec_size=args.fet_vec_size, label_vec_size=args.label_vec_size, datatype_=args.dataset, test_train="train")

    dset_val = HARGCNNDataset(esdswf["X_test"], esdswf["Y_test"],
                              nodes_count=args.nodes_cnt, miss_thr=args.miss_thr, noise_thr=args.noise_thr,
                              randomseed=args.randomseed, normalization=args.normalization,
                              fet_vec_size=args.fet_vec_size, label_vec_size=args.label_vec_size, datatype_=args.dataset, test_train="test")
else:
    with open("./dataset/PAMAP.pkl", "rb") as f:
        esdswf = pickle.load(f)
    dset_train = HARGCNNDataset(esdswf["X_train"], esdswf["Y_train_onehot"], _single_label=esdswf["Y_train"],
                                nodes_count=args.nodes_cnt, miss_thr=args.miss_thr, noise_thr=args.noise_thr,
                                randomseed=args.randomseed, normalization=args.normalization,
                                fet_vec_size=args.fet_vec_size, label_vec_size=args.label_vec_size, datatype_=args.dataset, test_train="train")

    dset_val = HARGCNNDataset(esdswf["X_test"], esdswf["Y_test_onehot"], _single_label=esdswf["Y_test"],
                              nodes_count=args.nodes_cnt, miss_thr=args.miss_thr, noise_thr=args.noise_thr,
                              randomseed=args.randomseed, normalization=args.normalization,
                              fet_vec_size=args.fet_vec_size, label_vec_size=args.label_vec_size, datatype_=args.dataset, test_train="test")


loader_train = DataLoader(
    dset_train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0)

loader_val = DataLoader(
    dset_val,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=1)


# Create_all_args_tag
args.tag += str(args.dataset) + "_" + str(args.model)+"_" + \
    str(args.nodes_cnt)+"_"+str(args.data_per)+"_"+str(args.normalization)+"_"
args.tag += str(args.miss_thr)+"_"+str(args.noise_thr)+"_"+str(args.randomseed)

# Create check point
checkpoint_dir = './trainedmodels/'+args.tag+'/'

if args.test:
    checkpoint = torch.load(checkpoint_dir+'val_best.pth')
    model.load_state_dict(checkpoint)
else:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tensorboard_dir = './runs/'+args.tag+'/'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    with open(checkpoint_dir+'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    writer = SummaryWriter(tensorboard_dir)


if args.dataset == "ExtraSensory":
    bce_loss = nn.BCELoss().cuda()

    def graph_loss(V_pred, V_target):
        #         print(V_pred.shape,V_target.shape)
        return bce_loss(V_pred.squeeze(), V_target[:, :, args.fet_vec_size:].squeeze())
else:
    cse_loss = nn.CrossEntropyLoss().cuda()

    def graph_loss(V_pred, V_target):
        # Vatrget = batch,node,class --> batch, node as class  dim =  1
        V_target = V_target.squeeze()

        return cse_loss(V_pred, V_target)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)
print('Tensorboard dir:', checkpoint_dir)


print('*'*30)
print("Initiating ....")

# Training settings

optimizer = optim.Adadelta(model.parameters())
metrics = {'train_loss': [],  'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}


# Training

def train(epoch):
    global metrics, loader_train
    model.train()
    loss_batch = 0
    batch_count = 0
    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        if len(batch) == 3:
            V, A, Vcrr = batch
        else:
            V, A, Vcrr, Slabel = batch

        optimizer.zero_grad()
        V_pred = model(Vcrr.squeeze(), A.squeeze())

        if len(batch) == 3:
            l = graph_loss(V_pred, V.squeeze())
        else:
            l = graph_loss(V_pred.squeeze(), Slabel)

        l.backward()

        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        # Metrics
        loss_batch += l.item()
        if batch_count % 300 == 0:
            print('TRAIN:', '\t Epoch:', epoch,
                  '\t Loss:', loss_batch/batch_count)

        metrics['train_loss'].append(loss_batch/batch_count)
        writer.add_scalar('Loss/train', loss_batch/batch_count, epoch)


def vald(epoch):
    global metrics, loader_val, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    with torch.no_grad():

        for cnt, batch in enumerate(loader_val):
            batch_count += 1

            # Get data
            batch = [tensor.cuda() for tensor in batch]
            if len(batch) == 3:
                V, A, Vcrr = batch
            else:
                V, A, Vcrr, Slabel = batch

            V_pred = model(Vcrr.squeeze(), A.squeeze())

#             l = graph_loss(V_pred,V.squeeze())
            if len(batch) == 3:
                l = graph_loss(V_pred, V.squeeze())
            else:
                l = graph_loss(V_pred.squeeze(), Slabel)

            # Metrics
            loss_batch += l.item()
            if batch_count % 300 == 0:

                print('VALD:', '\t Epoch:', epoch,
                      '\t Loss:', loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)
    writer.add_scalar('Loss/test', loss_batch/batch_count, epoch)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir+'val_best.pth')  # OK


def test():
    global metrics, loader_val, constant_metrics
    model.eval()
    batch_count = 0
    _pred = []
    _target = []
    with torch.no_grad():

        for cnt, batch in enumerate(loader_val):
            batch_count += 1

            # Get data
            batch = [tensor.cuda() for tensor in batch]
            if len(batch) == 3:
                V, A, Vcrr = batch
            else:
                V, A, Vcrr, Slabel = batch

            V_pred = model(Vcrr.squeeze(), A.squeeze())
            if args.dataset == "PAMAP":
                V_pred = V_pred.transpose(1, 2)
                V_pred = F.softmax(V_pred, dim=-1)

            # Metrics
            # Collect pred for later procc
            V_pred_np = V_pred.data.cpu().numpy().squeeze()
            V_target_np = V.data.cpu().numpy().squeeze()
            B = V_pred_np.shape[0]
            # print(V_pred_np.shape, V_target_np.shape)
            for b in range(B):
                # V.shape torch.Size([128, 3, 51] = batch,Nodes, labels
                for n in range(args.nodes_cnt):
                    _pred.append(V_pred_np[b, n, :].squeeze())
                    _target.append(
                        V_target_np[b, n, args.fet_vec_size:].squeeze())

    _pred = np.asarray(_pred)
    _target = np.asarray(_target)

    if args.dataset == "ExtraSensory":
        _pred[_pred >= 0.5] = 1
        _pred[_pred < 0.5] = 0

        _f1 = skmetrics.f1_score(_target, _pred, average="macro")

        _acc = 0
        for i in range(51):
            _acc += skmetrics.accuracy_score(_target[:, i], _pred[:, i])
        _acc /= 51

        print("Macro F1 score: ", _f1, "| Accuracy: ", _acc)
    else:
        _f1 = skmetrics.f1_score(np.argmax(_target, axis=1), np.argmax(
            _pred, axis=1), average="micro")
        _acc = skmetrics.accuracy_score(
            np.argmax(_target, axis=1), np.argmax(_pred, axis=1))

        print("F1 score: ", _f1, "| Accuracy: ", _acc)


if args.test:
    print('Testing started ...')
    test()

else:
    # Training loop
    print('Training started ...')
    for epoch in range(args.num_epochs):
        train(epoch)
        vald(epoch)

        print('*'*30)
        print('Epoch:', args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*'*30)

        with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)
