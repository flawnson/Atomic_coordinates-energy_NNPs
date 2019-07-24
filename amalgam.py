import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pre

from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

"""IMPORTS AND VARIABLES"""
import trait
import unitary

num_epochs = 1000
batch_size = 128
lr = 0.005

"""DATA PREPROCESSING"""

"""GRAPH CONVOLUTIONAL ARCHITECTURE"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(pre.train_data.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.7)
        self.conv2 = GCNConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.7)
        self.conv3 = GCNConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.7)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()        
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x

"TRAINING INITIALIZATION"
def train():
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = data.y.to(device)
        loss = torch.sqrt(criterion(output, target))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(pre.train_data)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss
loader = DataLoader(pre.train_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    train()