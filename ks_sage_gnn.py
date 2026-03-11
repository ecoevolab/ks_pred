import os
import glob
import pandas as pd
import networkx as nx
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Dataset


# Read data to train SAGEconv model.
# I have multiple networks, each with its own adjacency
# matrix, node features, and target values. 
# The data is stored in separate files for each network and
# type of data (adjacency, features, targets).
# Edge data is in the form of adjacency matrix where
# columns are source nodes and rows are target nodes,
# values are weights of edges. Self loos should be ignored.
# I have node features as a matrix where the first row are
# feature names and the rest are feature values.
# Node-level target values are in a separate file where
# ther first row are target names and the rest are target valuees.

class dataset_loader(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.adj_files = sorted(glob.glob(os.path.join(root_dir,"networks", '*_net.tsv')))
        self.node_files = sorted(glob.glob(os.path.join(root_dir,"nodes", '*_nodes.tsv')))
        self.target_files = sorted(glob.glob(os.path.join(root_dir,"targets", '*_target.tsv')))

        if len(self.adj_files) != len(self.node_files) or len(self.adj_files) != len(self.target_files):
            raise ValueError("Mismatch in number of adjacency, node feature, and target files")


    def len(self):
        return len(self.graph_files)

    def get(self, ii):

        # Read adjacency matrix and convert to edge index and edge weight
        adj_file = self.adj_files[ii]
        A = pd.read_csv(adj_file, header=None, sep='\t').values
        A = torch.tensor(A, dtype=torch.float64)
        A.fill_diagonal_(0)
        targets,sources = torch.nonzero(A, as_tuple=True)

        # edge weights
        edge_weight = A[targets, sources]

        # edge index
        edge_index = torch.stack([sources, targets], dim=0)

        # Read target value
        tgt_file = self.target_files[ii]
        targets = pd.read_csv(tgt_file, sep='\t').values
        y = torch.tensor(targets[:,2], dtype=torch.float64)

        # Read node features
        nd_file = self.node_files[ii]
        features = pd.read_csv(nd_file, sep='\t').values
        x = torch.tensor(features[:,0:15], dtype=torch.float64)
        # Remove nan values !!!!!!!
        x = torch.nan_to_num(x, nan=0.0)

        # Check dimensins
        assert A.shape[0] == A.shape[1] == len(y) == x.shape[0], "Mismatch between adjacency and targets"


        if torch.isnan(y).any():
            raise ValueError(f"NaN target values in {base_name}")
        if torch.isnan(edge_weight).any():
            raise ValueError(f"NaN edge weights in {base_name}")
        if torch.isnan(x).any():
            raise ValueError(f"NaN node features in {base_name}")

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)












