import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class LSTMBaseLine(nn.Module):

    def __init__(self, input_dim=52+12, hidden_dim=10, num_layers=1, args=None):
        super(LSTMBaseLine, self).__init__()

        self.tmodel = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=True)

        #ConvLSTM( input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)
        self.conv_dim = nn.Conv1d(hidden_dim, 32, 3, padding=1)
        self.conv_pre = nn.PReLU()
        self.conv_output = nn.Conv1d(32, 12, 3, padding=1)
        self.sigm = nn.Sigmoid()

        self.num_layers = num_layers
        self.hidden_size = hidden_dim

    def forward(self, x):
        # x = ([128, 3, 275]) -- >  =  B, T, F
        # _batch = x.shape[0]
        # h_0 = torch.zeros(self.num_layers,_batch,self.hidden_size).cuda()
        # c_0 = torch.zeros(self.num_layers,_batch,self.hidden_size).cuda()
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero. Pytorch doc

        v, (h, c) = self.tmodel(x)
        # v =[128, 3, 12]
        x = v.transpose(1, 2)  # x = [128,12,3]
        x = self.conv_pre(self.conv_dim(x))  # x = [128,32,3]
        x = self.conv_output(x)
        return x  # torch.Size([128, 3, 51])
