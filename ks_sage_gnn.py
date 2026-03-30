import os
import glob
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


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
        A = torch.tensor(A, dtype=torch.float32)
        A.fill_diagonal_(0)
        targets,sources = torch.nonzero(A, as_tuple=True)

        # edge weights
        edge_weight = A[targets, sources]

        # edge index
        edge_index = torch.stack([sources, targets], dim=0)

        # Read target value
        tgt_file = self.target_files[ii]
        targets = pd.read_csv(tgt_file, sep='\t').values
        y = torch.tensor(targets[:,2], dtype=torch.float32)

        # Read node features
        nd_file = self.node_files[ii]
        features = pd.read_csv(nd_file, sep='\t').values
        x = torch.tensor(features[:,0:15], dtype=torch.float32)
        # Remove nan values !!!!!!!
        x = torch.nan_to_num(x, nan=0.0)

        # Check dimensins
        assert A.shape[0] == A.shape[1] == len(y) == x.shape[0], "Mismatch between adjacency and targets"


        if torch.isnan(y).any():
            raise ValueError(f"NaN target values in {self.root_dir}")
        if torch.isnan(edge_weight).any():
            raise ValueError(f"NaN edge weights in {self.root_dir}")
        if torch.isnan(x).any():
            raise ValueError(f"NaN node features in {self.root_dir}")

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, num_nodes=x.shape[0])

def sse(pred_y, y):
    """Calculate SSE"""
    return ((pred_y - y) ** 2).sum()


# Define SAGE nn model (from Labonne)
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
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage3(h, edge_index)
        return h
    
    def fit(self, loader, val_loader, epochs):
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
        self.train()
        for epoch in range(epochs+1):
            train_loss = 0
            val_loss = 0
            train_rmse = 0
            val_rmse = 0
            train_sse = 0
            val_sse = 0
            train_n = 0
            val_n = 0
            
            # Train on batches
            for batch in loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                #return out, batch.y
                loss = criterion(out.squeeze(-1), batch.y)
                train_loss += loss.item() / len(batch.y) / len(batch.y)
                train_sse += sse(out.argmax(dim=1), batch.y)
                train_n += len(batch.y) # Only 1 dimensional target values
                loss.backward()
                optimizer.step()
            
            # For can be removed if I ensure that val_loader has only one batch.
            for batch in val_loader:
                out = self(batch.x, batch.edge_index)
                val_loss += criterion(out.squeeze(-1), batch.y) / len(batch.y)
                val_sse += sse(out.argmax(dim=1), batch.y)
                val_n += len(batch.y) # Only 1 dimensional target values
            
            # Calculate rmse
            train_rmse = torch.sqrt(train_sse / train_n).item()
            val_rmse = torch.sqrt(val_sse / val_n).item()

            # Print metrics every 10 epochs
            if epoch % 20 == 0:
                print(f'Epoch {epoch} | Train Loss: {train_loss} | Train RMSE: {train_rmse} | Val Loss: {val_loss} | Val RMSE: {val_rmse}')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        sse_test = sse(out.argmax(dim=1), data.y)
        rmse_test = sse_test / len(data.y)
        return rmse_test.item()

# Read data
data = dataset_loader("/home/sur/lab/exp/2026/2026-03-09.sim_glv/sims")
# data[0]
# data[1]
# data[2]

# Create training, validation, and test datasets. We have multiple networks,
# so each network will only be included in one of the masks.
# Use 80/10/10 split for train/val/test.
# Could be added to the data loader, but I will do it here for now.
n_networks = len(data)
r_ii = torch.randperm(n_networks)

train_mask = torch.zeros(n_networks, dtype=torch.bool)
train_mask[ r_ii[ :int(0.8 * n_networks) ] ] = True

val_mask = torch.zeros(n_networks, dtype=torch.bool)
val_mask[ r_ii [ int(0.8 * n_networks):int(0.9 * n_networks) ] ] = True

test_mask = torch.zeros(n_networks, dtype=torch.bool)
test_mask[ r_ii [ int(0.9 * n_networks): ] ] = True

train_dataset = data[train_mask]
val_dataset = data[val_mask]
test_dataset = data[test_mask]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


# Initialize model
gnn_sage_ks = SAGE_ks(dim_in = 15, dim_h = 8, dim_out = 1)
print(gnn_sage_ks)

# Train
gnn_sage_ks.fit(train_loader, val_loader, epochs=100)

# Test
rmse = gnn_sage_ks.test(test_loader.dataset[0])
print(f'GraphSAGE test RMSE: {rmse}')

# Plot
# Get obseved and predicted values for model
obs = test_loader.dataset[0].y
preds = gnn_sage_ks.forward(test_loader.dataset[0].x, test_loader.dataset[0].edge_index)

# Make a scatter plot of observed vs predicted values
plt.scatter(obs, preds.detach().numpy())
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('GraphSAGE Predictions vs Observed')
plt.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'r--') # Add a diagonal line for reference
plt.show()

