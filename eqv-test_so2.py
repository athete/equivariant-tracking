import os
from statistics import mean
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from models.euclidean_so2 import EuclidNet
from models.dataset import GraphDataset

def R(theta):
    return torch.tensor([
        [torch.cos(theta), torch.sin(theta), 0.],
        [-torch.sin(theta), torch.cos(theta), 0.],
        [0., 0., 1.]
    ])

# Load the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
home_dir = ""
indir = f"hitgraphs/geometric/2p0/"
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

model = EuclidNet(
    n_scalar=2,
    n_input=2,
    n_hidden=32,
    n_layers=3,
    n_output=1,
    c_weight=1,
    group="SO(2)"
).to(device)
model.load_state_dict(torch.load("./trained_models/so2/EN_geometric_epoch100_L3_2.0GeV.pt"))

model.eval()
metrics = {"accs": [], "aucs": []}
thld = 0.268313725490
with torch.no_grad():
    for theta in torch.linspace(0, 2*torch.pi, 100):
        accs, aucs = [], []
        for data in test_loader:
            data = data.to(device)
            out = model(torch.matmul(data.x, R(theta).to(device)), data.edge_index)
            TP = torch.sum((data.y == 1).squeeze() & (out > thld).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (out < thld).squeeze()).item()
            FP = torch.sum((data.y == 0).squeeze() & (out > thld).squeeze()).item()
            FN = torch.sum((data.y == 1).squeeze() & (out < thld).squeeze()).item()
            acc = (TP + TN) / (TP + TN + FP + FN)
            auc = roc_auc_score(data.y.cpu(), (out > thld).float().cpu())
            aucs.append(auc)
            accs.append(acc)
        metrics["aucs"].append(mean(aucs))
        metrics["accs"].append(mean(accs))
        print(f"[Theta: {theta}] Accuracy: {mean(accs)}\t ROC AUC: {mean(aucs)}")

np.save("so2-equv-test.npy", metrics)