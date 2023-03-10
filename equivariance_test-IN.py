import os
from statistics import mean
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from models.interaction import InteractionNetwork
from models.dataset import GraphDataset

def R(theta):
    rot_mat = torch.tensor([
            [np.cos(theta), np.sin(theta), 0.],              
            [-np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]
    ], dtype=torch.float32)
    return rot_mat

# Load the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
home_dir = ""
indir = f"hitgraphs/geometric/1p5/"
graph_files = np.array(os.listdir(indir))
graph_files = np.array(
    [os.path.join(indir, graph_file) for graph_file in graph_files]
)
graph_paths = [os.path.join(indir, filename) for filename in graph_files]
n_graphs = len(graph_files)

IDs = np.arange(n_graphs)
partition = {
    "train": graph_files[IDs[:1000]],
    "test": graph_files[IDs[1000:1400]],
    "val": graph_files[IDs[1400:1500]],
}
params = {"batch_size": 1, "shuffle": True, "num_workers": 6}
test_set = GraphDataset(graph_files=partition["test"])
test_loader = DataLoader(test_set, **params)
print("...Successfully loaded test graphs")

model = InteractionNetwork(
    hidden_size=16,
    n_layers=3
).to(device)


model_paths = [
    "./trained_models/IN_geometric_h16_1.5GeV_angle25.pt",
    "./trained_models/IN_geometric_h16_1.5GeV_angle50.pt",
    "./trained_models/IN_geometric_h16_1.5GeV_angle75.pt",
    "./trained_models/IN_geometric_h16_1.5GeV_angle100.pt",
]

all_metrics = []

for num in range(4):
    print(model_paths[num])
    model.load_state_dict(torch.load(model_paths[num]))
    model.eval()
    metrics = []
    thld = 0.268313725490
    with torch.no_grad():
        for theta in torch.linspace(0, torch.pi, 10):
            aucs = []
            rot_mat = R(theta).to(device)
            for data in test_loader:
                data = data.to(device)
                out = model(torch.matmul(data.x, rot_mat), data.edge_index, data.edge_attr)
                TP = torch.sum((data.y == 1).squeeze() & (out > thld).squeeze()).item()
                TN = torch.sum((data.y == 0).squeeze() & (out < thld).squeeze()).item()
                FP = torch.sum((data.y == 0).squeeze() & (out > thld).squeeze()).item()
                FN = torch.sum((data.y == 1).squeeze() & (out < thld).squeeze()).item()
                acc = (TP + TN) / (TP + TN + FP + FN)
                auc = roc_auc_score(data.y.cpu(), (out > thld).float().cpu())
                aucs.append(auc)
            metrics.append(mean(aucs))
            print(f"[Theta: {theta}] ROC AUC: {mean(aucs)}")
    all_metrics.append(metrics)

np.save(f"IN-angle-sweep-data.npy", all_metrics)