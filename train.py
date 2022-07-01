import os
from pprint import pprint
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, DataLoader

from models.lorentz import LorentzNet
from models.dataset import GraphDataset


if __name__ == '__main__':
    data_dir = f"hitgraphs/"
    graph_files = np.array([os.listdir(data_dir)])
    train_set = GraphDataset(graph_files=graph_files)
    train_loader = DataLoader(train_set)
    for data in train_loader:
        
        #pprint(data.x, data.edge_index, data.edge_attr)