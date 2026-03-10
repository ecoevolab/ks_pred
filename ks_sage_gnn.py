import os
import glob
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

        if lel(self.adj_files) != len(self.node_files)


    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        adj_path = self.graph_files[idx]
        base_name = os.path.basename(adj_path).replace('_adj.csv', '')
        target_path = os.path.join(self.root_dir, f"{base_name}_targets.csv")
        feature_path = os.path.join(self.root_dir, f"{base_name}_features.csv")

        adj = pd.read_csv(adj_path, header=None).values
        target = pd.read_csv(target_path, header=None).values.squeeze()

        assert adj.shape[0] == adj.shape[1] == len(target), "Mismatch between adjacency and targets"

        row, col = np.nonzero(adj)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_weight = torch.tensor(adj[row, col], dtype=torch.float32)

        # Handles optional node level features
        if os.path.exists(feature_path):
            x = pd.read_csv(feature_path, header=None).values
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = torch.ones((adj.shape[0], 1))

        y = torch.tensor(target, dtype=torch.float32)

        if torch.isnan(y).any():
            raise ValueError(f"NaN target values in {base_name}")
        if torch.isnan(edge_weight).any():
            raise ValueError(f"NaN edge weights in {base_name}")
        if torch.isnan(x).any():
            raise ValueError(f"NaN node features in {base_name}")

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)




def read_data(network_name):
   # Read adjacency matrix
    adj_matrix = torch.load(f'{network_name}_adj.pt')
    
    # Read node features
    features = torch.load(f'{network_name}_features.pt')
    
    # Read target values
    targets = torch.load(f'{network_name}_targets.pt')
    
    return adj_matrix, features, targets



torch.load("/home/sur/lab/exp/2026/today2/sims/networks/088e4b1149acd16f282c_net.tsv")
