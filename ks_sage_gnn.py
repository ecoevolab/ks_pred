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
        return len(self.adj_files)

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

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, num_nodes=x.shape[0])

# Metric for accuracy (RMSE)
def rmse(pred_y, y):
    """Calculate RMSE."""
    return torch.sqrt(torch.mean((pred_y - y) ** 2)).item()

def sse(pred_y, y):
    """Calculate SSE"""
    return ((pred_y - y) ** 2).item()



# Defiuine SAGE nn model (from Labonne)
class SAGE_ks(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_h)
        self.sage3 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        h = torch.relu(h)
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage3(h, edge_index)
        return h

    def fit(self, loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        self.train()
        for epoch in range(epochs+1):
            total_loss = 0
            SSE = 0
            n = 0
            val_loss = 0
            val_SSE = 0
            val_n = 0

            # Train on batches
            for batch in loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss.item()
                SSE += sse(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                n += 
                loss.backward()
                optimizer.step()

                # Validation
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_RMSE += rmse(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])

            # Print metrics every 10 epochs
            if epoch % 20 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {loss/len(loader):.3f} | Train RMSE: {RMSE:>6.2f}% | Val Loss: {val_loss/len(train_loader):.2f} | Val RMSE: {val_RMSE:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        RMSE = rmse(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc





data = dataset_loader("/home/sur/lab/exp/2026/today2/sims")
data[0]
data[1]
data[2]
# Create training, validation, and test masks. We have multiple networks, so we will create masks for each network separately. We will use an 80/10/10 split for train/val/test sets.
for ii in range(len(data)):
    print(f"Processing network {ii+1}/{len(data)}") 
    





















for data in data:
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Randomly assign nodes to train, val, and test sets (80/10/10 split)
    indices = torch.randperm(num_nodes)
    train_mask[indices[:int(0.8 * num_nodes)]] = True
    val_mask[indices[int(0.8 * num_nodes):int(0.9 * num_nodes)]] = True
    test_mask[indices[int(0.9 * num_nodes):]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask













