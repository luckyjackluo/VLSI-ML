import torch
import numpy as np
from torch_geometric.data import Data
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from torch_geometric.data import Data, InMemoryDataset, download_url
import networkx as nx

class netlist(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['netlist.gpickle']

    @property
    def processed_file_names(self):
        return ['netlist.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        paths = np.load("./data/paths.npy")

        path_lst = []
        for f in paths:
            path_lst.append(f.split("_")[:-1])

        path_lst = np.array([lst[1] for lst in path_lst]).reshape(-1, 1)
        encoder = OneHotEncoder()
        labels = encoder.fit_transform(path_lst).toarray()
        labels = torch.tensor(labels, dtype=torch.float)

        X_eig = np.load("./data/eig_all.npy")
        X_new = np.load("./data/new_X.npy")

        X_eig = StandardScaler().fit_transform(X_eig)
        X_new = StandardScaler().fit_transform(X_new)

        for index in range(len(paths)):
            path = paths[index]
            netlist_name = path.split(".")[0]
            g_path = f"./data/gps/{netlist_name}.gpickle"
            graph = nx.read_gpickle(g_path)
            nodelist = list(graph.nodes())
            nodedict = {nodelist[idx]:idx for idx in range(len(nodelist))}
            X = torch.tensor([[graph.in_degree()[node], graph.out_degree()[node]] for node in nodelist], dtype=torch.float)
            y = labels[index]
            edge_index = torch.tensor([[nodedict[tp[0]], nodedict[tp[1]]] for tp in list(graph.edges())], dtype=torch.long).T
            gp_data = Data(x=X, y=y, edge_index=edge_index, eig=torch.tensor(X_eig[index], dtype=torch.float), stats=torch.tensor(X_new[index], dtype=torch.float))

            data_list.append(gp_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
