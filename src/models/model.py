import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool


class GAT_G(torch.nn.Module):
    """
    GAT model for Graph Classification
    """
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid, concat=False,
                             heads=self.out_head, dropout=0.6)

        self.fc1 = nn.Linear(self.hid+10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x, edge_index, batch, eig, stats):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = torch.cat((x, stats.reshape(x.shape[0], 10)), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    
class GCN_G(torch.nn.Module):
    """
    GCN model for Graph Classification
    """
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels, dropout=0.6)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, dropout=0.6)
        self.fc1 = nn.Linear(hidden_channels+10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x, edge_index, batch, eig, stats):


        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = torch.cat((x, stats.reshape(x.shape[0], 10)), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    
    
class GIN_G(torch.nn.Module):
    """
    GIN model for Graph Classification
    """
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), dropout=0.6)
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), dropout=0.6)

        self.lin1 = Linear(dim_h*2 + 10, dim_h*3)
        self.lin2 = Linear(dim_h*3, 5)

    def forward(self, x, edge_index, batch, eig, stats):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, stats.reshape(h1.shape[0], 10)), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.lin2(h)
        
        return F.log_softmax(h, dim=1)
    
    
    
    
    
    