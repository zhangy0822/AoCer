import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from utils.util_C_GCN import *


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, which shared the weight between two separate graphs
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        Graph Conv function
        :param input: input signal
        :param adj: adj graph dict [OPC, OMC, all]
        :param conv_mode: choose which graph to make convolution (separate graphs or whole graph)
        '''

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



class C_GCN(nn.Module):

    def __init__(self, num_classes, in_channel=300, embed_size=300, t=0, adj_array=None, glove_array = None, norm_func='sigmoid', adj_gen_mode='C_GCN', opt=None):
        super(C_GCN, self).__init__()

        self.cfg = opt
        self.num_classes = num_classes
        self.embed_size = embed_size
        if self.cfg.n_gcn == 2:
            self.gc1 = GraphConvolution(in_channel, embed_size)
            self.gc2 = GraphConvolution(embed_size,  in_channel)
        elif self.cfg.n_gcn == 1:
            self.gc1 = GraphConvolution(in_channel, embed_size)
        elif self.cfg.n_gcn == 3:
            self.gc1 = GraphConvolution(in_channel, embed_size)
            self.gc2 = GraphConvolution(embed_size,  embed_size)
            self.gc3 = GraphConvolution(embed_size,  in_channel)
        elif self.cfg.n_gcn == 4:
            self.gc1 = GraphConvolution(in_channel, embed_size)
            self.gc2 = GraphConvolution(embed_size,  embed_size)
            self.gc3 = GraphConvolution(embed_size,  embed_size)
            self.gc4 = GraphConvolution(embed_size,  in_channel)
        self.relu = nn.LeakyReLU(0.2)

        self.adj = adj_array
        self.glove_vector = glove_array
        if self.cfg.cuda:
            self.glove_vector, self.adj = self.glove_vector.cuda(), self.adj.cuda()


    def gen_rescale(self, t, adj):
        adj = rescale_adj_matrix(adj_mat=adj, t=20, p=0.0001).float()
        adj[adj < t] = 0
        adj[adj >= t] = 1
        adj = adj + torch.eye(self.num_classes).float()
        return adj



    def forward(self):
        adj_all = gen_adj(self.adj)
        if self.cfg.n_gcn == 2:
            x = self.gc1(self.glove_vector, adj_all)
            x = self.relu(x)
            x = self.gc2(x, adj_all)
            concept_feature = l2norm(x)
            return concept_feature
        elif self.cfg.n_gcn == 1:
            x = self.gc1(self.glove_vector, adj_all)
            concept_feature = l2norm(x)
            return concept_feature
        elif self.cfg.n_gcn == 3:
            x = self.gc1(self.glove_vector, adj_all)
            x = self.relu(x)
            x = self.gc2(x, adj_all)
            x = self.relu(x)
            x = self.gc3(x, adj_all)
            concept_feature = l2norm(x)
            return concept_feature  
        elif self.cfg.n_gcn == 4:
            x = self.gc1(self.glove_vector, adj_all)
            x = self.relu(x)
            x = self.gc2(x, adj_all)
            x = self.relu(x)
            x = self.gc3(x, adj_all)
            x = self.relu(x)
            x = self.gc4(x, adj_all)
            concept_feature = l2norm(x)
            return concept_feature   


    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]




