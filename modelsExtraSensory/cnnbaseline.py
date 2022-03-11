
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn

import argparse

import torch.nn.functional as F


class CNNBaseLine(nn.Module):
    def __init__(self, nfeat, nhid, nadjf=None, args=None):
        super(CNNBaseLine, self).__init__()

        int_dim = 14
        self.gc_in = nn.Conv1d(args.nfeat, int_dim, 3, padding=1)
        self.pre_in = nn.PReLU()
        self.gc1 = nn.Conv1d(int_dim, int_dim, 3, padding=1)
        self.pre1 = nn.PReLU()
        self.gc2 = nn.Conv1d(int_dim, int_dim, 3, padding=1)
        self.pre2 = nn.PReLU()
        self.gc3 = nn.Conv1d(int_dim, int_dim, 3, padding=1)
        self.pre3 = nn.PReLU()

        self.cnn_label = nn.Conv1d(int_dim, args.nhid, 3, padding=1)

        self.fet_vec_size = args.fet_vec_size
        self.label_vec_size = args.label_vec_size
        self.sigmd = nn.Sigmoid()
        # This is a gray area, depending on the data representaion

    def forward(self, V):
        # Process the graph
        #         print("Input:",V.shape)
        V = V.transpose(1, 2)
        x = self.pre_in(self.gc_in(V))

        x = self.pre1(self.gc1(x))+x
        x = self.pre2(self.gc2(x))+x
        x = self.pre2(self.gc3(x))+x

        x_label = self.sigmd(self.cnn_label(x))
#         print("Output:",x_label.shape)

        return x_label
