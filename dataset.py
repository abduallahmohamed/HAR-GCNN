import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np

import random

from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm
import pickle


class HARGCNNDataset(Dataset):
    """Dataloader for the Trajectory datasets"""

    def __init__(
            self, fea_, lbls_, _single_label=None, nodes_count=3, miss_thr=0.5, noise_thr=0.5, randomseed=11, normalization="abduallahs", fet_vec_size=224, label_vec_size=51, datatype_="extraSens", test_train="train"):

        super(HARGCNNDataset, self).__init__()
        # Reproducability

        stored_name = "./store_" + str(nodes_count)+str(miss_thr)+str(noise_thr)+str(randomseed)+str(
            normalization)+str(fet_vec_size)+str(label_vec_size)+str(datatype_)+str(test_train)+".pkl"
        print("Store name:", stored_name)
        if os.path.isfile(stored_name):
            print("Stored name is found")
            with open(stored_name, 'rb') as handle:
                data_dict = pickle.load(handle)
                self.v = data_dict["v"]
                self.A = data_dict["A"]
                if "SingleLabel" in data_dict:
                    self.SingleLabel = data_dict["SingleLabel"]
                self.v_cour = data_dict["v_cour"]
        else:

            random.seed(a=randomseed)
            torch.manual_seed(randomseed)
            np.random.seed(seed=randomseed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            dslen = len(fea_)

            _crpt_max = int(2*nodes_count/3)
            self.v = []  # Holds the vertices
            self.A = []  # Holds the adjacency
            self.v_cour = []  # Holds the noisy or missed labels
            self.SingleLabel = []  # holds the single labels targets not the one hot encoded format
            sliding_ranges = [range(i, i+nodes_count)
                              for i in range(dslen) if i+nodes_count <= dslen]
            # Convert to Graphs for each sequence of nodes_count,
            # So, we will have a graph of nodes_count and its adjacency matrix
            pbar = tqdm(total=len(sliding_ranges))
            print("Creating train/test data....")
            for s in sliding_ranges:
                pbar.update(1)
                V = torch.zeros(nodes_count, fet_vec_size +
                                label_vec_size).type(torch.float)
                V_noise = torch.zeros(
                    nodes_count, fet_vec_size+label_vec_size).type(torch.float)
                if _single_label is not None:
                    _sl = torch.zeros(nodes_count, 1).type(torch.long)
                _cnt = 0
                _crpt_cnt = 0
                for i in s:
                    V[_cnt, :] = torch.cat(
                        [fea_[i, :, :], lbls_[i, :, :]], axis=1).squeeze()
                    if _single_label is not None:
                        _sl[_cnt, 0] = int(_single_label[i, :, :].squeeze())

                    if random.random() >= miss_thr and _crpt_cnt <= _crpt_max:  # Then miss the labels
                        V_noise[_cnt, :] = torch.cat(
                            [fea_[i, :, :], torch.zeros_like(lbls_[i, :, :])], axis=1).squeeze()
                        _crpt_cnt += 1
                    elif random.random() >= noise_thr and _crpt_cnt <= _crpt_max:  # make noisy featuers
                        V_noise[_cnt, :] = torch.cat([fea_[i, :, :].add_(torch.randn(
                            fea_[i, :, :].size())/10.0), torch.zeros_like(lbls_[i, :, :])], axis=1).squeeze()
                    else:  # Keep
                        V_noise[_cnt, :] = torch.cat(
                            [fea_[i, :, :], lbls_[i, :, :]], axis=1).squeeze()

                    _cnt += 1
                    # loop over the data and construct a graph as following

                if normalization == "abduallahs":
                    A_hat = np.ones((nodes_count, nodes_count))
                    G = nx.from_numpy_matrix(A_hat)  # create a graph
                    A = torch.from_numpy(nx.normalized_laplacian_matrix(
                        G).toarray()).type(torch.float)  # to normalize
                elif normalization == "kipfs":
                    A_hat = np.ones((nodes_count, nodes_count))
                    # Fully connected graph degree matrix of A_hat
                    D_hat = np.eye(nodes_count)*nodes_count
                    D_half_inv = np.linalg.inv(np.sqrt(D_hat))
                    A = torch.from_numpy(
                        np.matmul(np.matmul(D_half_inv, A_hat), D_half_inv)).type(torch.float)
                else:
                    print("Error, unknown normalization selected", normalization)

                self.v.append(V)
                self.A.append(A)
                self.v_cour.append(V_noise)
                if _single_label is not None:
                    self.SingleLabel.append(_sl)

            data_dict = {}
            data_dict["v"] = self.v
            data_dict["A"] = self.A
            data_dict["v_cour"] = self.v_cour
            if _single_label is not None:
                data_dict["SingleLabel"] = self.SingleLabel
            print("Saving data pkl...")
            with open(stored_name, 'wb') as handle:
                pickle.dump(data_dict, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
                print("Saved data at:", stored_name)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, index):
        if hasattr(self, 'SingleLabel'):
            return self.v[index], self.A[index], self.v_cour[index], self.SingleLabel[index]
        return self.v[index], self.A[index], self.v_cour[index]
