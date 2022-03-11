import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn

import argparse


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 but modified to be batched
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape, self.weight.shape)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class HARGCNN(nn.Module):
    def __init__(self, nfeat, nhid, nadjf=None, args=None):
        super(HARGCNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.pre1 = nn.PReLU()
        self.prec1 = nn.PReLU()
        self.prec2 = nn.PReLU()
        self.prec3 = nn.PReLU()

        self.cnn1 = nn.Conv2d(1, 3, 3, padding=1)
        self.cnn2 = nn.Conv2d(3, 3, 3, padding=1)
        self.cnn3 = nn.Conv2d(3, 3, 3, padding=1)
#         self.cnn_feat = nn.Conv2d(3,1,3,padding=1)
        self.cnn_label = nn.Conv2d(3, 1, 3, padding=1, bias=False)

        self.fet_vec_size = args.fet_vec_size
        self.label_vec_size = args.label_vec_size
        self.sigmd = nn.Sigmoid()
        # This is a gray area, depending on the data representaion

    def forward(self, V, A):
        # Process the graph
        x = self.pre1(self.gc1(V, A))
        x = x.unsqueeze(1)
        # print(x.shape)
        # x = x.view(x.sape[0], 1, x.shape[0], x.shape[1])
        # x = x.transpose(1, 2)
        # Now process the info inside it
        x = self.prec1(self.cnn1(x))
        x = self.prec2(self.cnn2(x))
        x = self.prec3(self.cnn3(x))

#         x_feat  = self.cnn_feat(x[:,:,:,:self.fet_vec_size])
        x_label = self.sigmd(self.cnn_label(x))

        return x_label
