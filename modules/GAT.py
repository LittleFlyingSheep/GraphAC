import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.activation = nn.LeakyReLU(self.alpha)
        # self.leakyrelu = nn.ReLU()

    def forward(self, h, adj):

        Wh = torch.matmul(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        # e = self._prepare_attentional_mechanism_input(Wh)
        e = self._prepare_attentional_mechanism_input_repeat(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # meaningless, if we don't have a clear initial graph structure, this sentence can be ignored.
        
        # normalize to get the weight of the conneciton between nodes.
        attention = F.softmax(attention, dim=2)  # softmax on the last N dimension

        
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input_repeat(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (B, 2 * out_feature, 1)
        # e.shape (B, N, N)

        N = Wh.shape[1]
        a_input = torch.cat([Wh.repeat(1, 1, N).view(Wh.shape[0], N*N, -1), Wh.repeat(1, N, 1)], dim=2).view(Wh.shape[0], Wh.shape[1], Wh.shape[1], 2*self.out_features)  # (B, N, N, 2*out_features)
        
        e = torch.matmul(a_input, self.a).squeeze(3)  # (B, N, N)
        
        return self.activation(e)
    
    def plot(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input_repeat(Wh)

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # softmax on the last N dimension
        attention = F.dropout(attention, self.dropout, training=self.training)  # (B, N, N)
        
        return attention 

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    
class GraphAttentionLayer_topk(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, k_num=5):
        super(GraphAttentionLayer_topk, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.k_num = k_num  # the number of the top-k adjacent nodes.

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = torch.matmul(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input_repeat(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # normalize to get the weight of the conneciton between nodes.
        attention = F.softmax(attention, dim=2)  # softmax on the last N dimension (recorvery for non-residual)
        attention = F.dropout(attention, self.dropout, training=self.training)  # (B, N, N)
        
        # top-k attention matrix
        topks, _ = torch.topk(attention, k=self.k_num, dim=2)
        indexs_to_remove = attention < topks[:, :, -1].view(attention.shape[0], attention.shape[1], 1)
        attention[indexs_to_remove] = 0  # masking the values less than top-k. 
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input_repeat(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (B, 2 * out_feature, 1)
        # e.shape (B, N, N)
        N = Wh.shape[1]
        a_input = torch.cat([Wh.repeat(1, 1, N).view(Wh.shape[0], N*N, -1), Wh.repeat(1, N, 1)], dim=2).view(Wh.shape[0], Wh.shape[1], Wh.shape[1], 2*self.out_features)  # (B, N, N, 2*out_features)
        
        e = torch.matmul(a_input, self.a).squeeze(3)  # (B, N, N)
        
        return self.leakyrelu(e)
    
    def plot(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input_repeat(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # softmax on the last N dimension
        attention = F.dropout(attention, self.dropout, training=self.training)  # (B, N, N)
        
        # top-k attention matrix
        topks, _ = torch.topk(attention, k=self.k_num, dim=2)
        indexs_to_remove = attention < topks[:, :, -1].view(attention.shape[0], attention.shape[1], 1)

        attention = attention - attention.min()

        attention[indexs_to_remove] = 0  # masking the values less than top-k.
        
        return attention 

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

