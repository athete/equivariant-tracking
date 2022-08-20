import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch_geometric.data import DataLoader

from models.euclidean_eq import EuclidNet
#from models.interaction import InteractionNetwork
from models.dataset import GraphDataset


def calc_dphi(phi1, phi2):
    dphi = phi2 - phi2
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < np.pi] += 2 * np.pi
    return dphi


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2))


model_dir = "trained_models/"
models = os.listdir(model_dir)
model_paths = [model for model in models if "epoch100" and "EQ" in model]
models_by_pt = {model.split("_")[3].strip("GeV")[:3]: model for model in model_paths}

# initial discriminants
thlds = {
    "2.0": 0.283,
    "1.5": 0.284529,
    "1.3": 0.21129,
    "1.0": 0.16107,
    "0.9": 0.0447,
    "0.8": 0.03797,
    "0.7": 0.02275,
    "0.6": 0.01779,
}
device = "cpu"

pt_min = sys.argv[1]
model = models_by_pt[pt_min]
print("...evaluating ", model)
thld = thlds[pt_min]
euclid_net = EuclidNet(n_input=3, n_hidden=40, n_layers=1, n_output=1, c_weight=1e-3)
#in_model = InteractionNetwork(40).to(device)
#in_model.load_state_dict(
#    torch.load(os.path.join(model_dir, model), map_location=torch.device(device))
#)
euclid_net.to(device)
euclid_net.eval()

construction = "geometric"
graph_indir = f"hitgraphs/{construction}/{pt_min.replace('.', 'p')}"
print("...sampling graphs from ", os.path.join(graph_indir, construction, pt_min.replace('.', 'p')))
graph_files = np.array(os.listdir(graph_indir))
graph_files = np.array([os.path.join(graph_indir, f) for f in graph_files])
n_graphs = len(graph_files)

# create a test dataloader
IDs = np.arange(n_graphs)
params = {"batch_size": 1, "shuffle": True, "num_workers": 6}
test_set = GraphDataset(graph_files=graph_files[IDs[1000:1400]])
test_loader = DataLoader(test_set, **params)

pt_bins = np.array(
    [
        0.6,
        0.8,
        1,
        1.2,
        1.5,
        1.8,
        2.1,
        2.5,
        3,
        3,
        3.5,
        4,
        5,
        6,
        8,
        10,
        15
    ]
)
pt_bin_centers = (pt_bins[1:] + pt_bins[:-1]) / 2
effs_by_pt = {"perfect": [], "dm": [], "lhc": []}

eta_bins = np.linspace(-4, 4, 28)
eta_bin_centers = (eta_bins[1:] + eta_bins[:-1]) / 2
effs_by_eta = {"perfect": [], "dm": [], "lhc": []}

lhc_effs, perfect_effs, dm_effs = [], [], []
with torch.no_grad():
    counter = 0
    found_by_pt = {
        "perfect": np.zeros(len(pt_bin_centers)),
        "dm": np.zeros(len(pt_bin_centers)),
        "lhc": np.zeros(len(pt_bin_centers)),
    }
    missed_by_pt = {
        "perfect": np.zeros(len(pt_bin_centers)),
        "dm": np.zeros(len(pt_bin_centers)),
        "lhc": np.zeros(len(pt_bin_centers)),
    }
    found_by_eta = {
        "perfect": np.zeros(len(eta_bin_centers)),
        "dm": np.zeros(len(eta_bin_centers)),
        "lhc": np.zeros(len(eta_bin_centers)),
    }
    missed_by_eta = {
        "perfect": np.zeros(len(eta_bin_centers)),
        "dm": np.zeros(len(eta_bin_centers)),
        "lhc": np.zeros(len(eta_bin_centers)),
    }

    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        output = euclid_net(data.x, data.edge_index, data.edge_attr)
        cutoff = int(len(data.edge_index[1]) / 2.0)
        y, out = (
            data.y[:cutoff],
            torch.min(
                torch.stack([output[:cutoff], output[cutoff:]]), dim=0
            ).values.squeeze(),
        )

        # count hits per pid in each event, add indices to hits
        X, pids, idxs = data.x, data.pid, data.edge_index[:, :cutoff]
        pts, etas = data.pt, data.eta
        unique_pids = torch.unique(pids)
        pid_counts_map = {p.item(): torch.sum(pids == p).item() for p in unique_pids}
        n_particles = np.sum([counts >= 1 for counts in pid_counts_map.values()])
        pid_label_map = {p.item(): -5 for p in unique_pids}
        pid_pt_map = {pids[i].item(): pts[i].item() for i in range(len(pids))}
        pid_eta_map = {pids[i].item(): etas[i].item() for i in range(len(pids))}
        pid_found_map = {
            "perfect": {
                p.item(): False for p in unique_pids if pid_counts_map[p.item()] >= 1
            },
            "dm": {
                p.item(): False for p in unique_pids if pid_counts_map[p.item()] >= 1
            },
            "lhc": {
                p.item(): False for p in unique_pids if pid_counts_map[p.item()] >= 1
            },
        }
        hit_idx = torch.unsqueeze(torch.arange(X.shape[0]), dim=1)
        X = torch.cat((hit_idx.float(), X), dim=1)

        # separate segments into incoming and outgoing hit positions
        good_edges = out > thld
        idxs = idxs[:, good_edges]
        feats_o = X[idxs[0]]
        feats_i = X[idxs[1]]

        # geometric quantities => distance calculation
        r_o, phi_o, z_o = (
            1000 * feats_o[:, 1],
            np.pi * feats_o[:, 2],
            1000 * feats_o[:, 3],
        )
        eta_o = calc_eta(r_o, z_o)
        r_i, phi_i, z_i = (
            1000 * feats_i[:, 1],
            np.pi * feats_i[:, 2],
            1000 * feats_i[:, 3],
        )
        eta_i = calc_eta(r_i, z_i)
        dphi, deta = calc_dphi(phi_o, phi_i), eta_i - eta_o
        distances = torch.sqrt(dphi ** 2 + deta ** 2)
        dist_matrix = 100 * torch.ones(X.shape[0], X.shape[0])
        for i in range(dist_matrix.shape[0]):
            dist_matrix[i][i] = 0
        for h in range(len(feats_i)):
            dist_matrix[int(feats_o[h][0])][int(feats_i[h][0])] = distances[h]

        # run DBSCAN
        eps_dict = {
            "2.0": 0.4,
            "1.5": 0.4,
	    "1.3": 0.4,
            "1.0": 0.4,
            "0.9": 0.4,
            "0.8": 0.4,
            "0.7": 0.4,
            "0.6": 0.4,
        }
        eps, min_pts = eps_dict[pt_min], 1
        clustering = DBSCAN(eps=eps, min_samples=min_pts, metric="precomputed").fit(
            dist_matrix
        )
        labels = clustering.labels_

        # count reconstructed particles from hit clusters
        lhc_clusters, perfect_clusters, dm_clusters = 0, 0, 0
        for label in np.unique(labels):
            if label < 0:
                continue

            # get pids correspinding to hit cluster labels
            label_pids = pids[labels == label]
            selected_pid = np.bincount(label_pids).argmax()
            if pid_counts_map[selected_pid] < 1:
                continue

            # fraction of hits with the most common pid
            n_selected_pid = len(label_pids[label_pids == selected_pid])
            selected_pid_fraction = n_selected_pid / len(label_pids)

            # previously_found = pid_label_map[selected_pid] > -1
            pid_label_map[selected_pid] = label
            label_pt = pid_pt_map[selected_pid]

            if selected_pid_fraction > 0.75:
                lhc_clusters += 1  # all hits have the same pid
                pid_found_map["lhc"][selected_pid] = True

                if pid_counts_map[selected_pid] == len(label_pids):
                    perfect_clusters += 1  # all required hits for pid
                    pid_found_map["perfect"][selected_pid] = True

            if selected_pid_fraction > 0.5:
                true_counts = pid_counts_map[selected_pid]
                if n_selected_pid / true_counts > 0.5:
                    dm_clusters += 1
                    pid_found_map["dm"][selected_pid] = True

        for d, pid_found in pid_found_map.items():
            for key, val in pid_found.items():
                pid_pt = pid_pt_map[key]
                pid_eta = pid_eta_map[key]
                for j in range(len(pt_bins) - 1):
                    if (pid_pt < pt_bins[j + 1]) and (pid_pt > pt_bins[j]):
                        if val == True:
                            found_by_pt[d][j] += 1
                        else:
                            missed_by_pt[d][j] += 1
                for j in range(len(eta_bins) - 1):
                    if (pid_eta < eta_bins[j + 1]) and (pid_eta > eta_bins[j]):
                        if val == True:
                            found_by_eta[d][j] += 1
                        else:
                            missed_by_eta[d][j] += 1

            eff_by_pt = found_by_pt[d] / (found_by_pt[d] + missed_by_pt[d])
            eff_by_eta = found_by_eta[d] / (found_by_eta[d] + missed_by_eta[d])
            print(
                d,
                np.sum(found_by_pt[d])
                / (np.sum(found_by_pt[d]) + np.sum(missed_by_pt[d])),
            )
            print(
                d,
                np.sum(found_by_eta[d])
                / (np.sum(found_by_eta[d]) + np.sum(missed_by_eta[d])),
            )
            effs_by_pt[d].append(eff_by_pt)
            effs_by_eta[d].append(eff_by_eta)

        lhc_effs.append(lhc_clusters / n_particles)
        perfect_effs.append(perfect_clusters / n_particles)
        dm_effs.append(dm_clusters / n_particles)

        counter += 1

print("LHC Eff: {:.4f}+/-{:4f}".format(np.mean(lhc_effs), np.std(lhc_effs)))
print(
    "Perfect Eff: {:.4f}+/-{:.4f}".format(np.mean(perfect_effs), np.std(perfect_effs))
)
print("DM Eff: {:.4f}+/-{:.4f}".format(np.mean(dm_effs), np.std(dm_effs)))

pt_output = {}
for d, effs in effs_by_pt.items():
    print("Clustering Method:", d)
    effs = np.array(effs)
    mean_effs = np.nanmean(effs, axis=0)
    std_effs = np.nanstd(effs, axis=0)
    for i in range(len(pt_bin_centers)):
        print(pt_bin_centers[i], " GeV: ", mean_effs[i], " +- ", std_effs[i])

    pt_output[d] = {
        "bins": pt_bins,
        "bin_centers": pt_bin_centers,
        "effs_mean": mean_effs,
        "effs_std": std_effs,
    }


eta_output = {}
for d, effs in effs_by_eta.items():
    print("Clustering Method:", d)
    effs = np.array(effs)
    mean_effs = np.nanmean(effs, axis=0)
    std_effs = np.nanstd(effs, axis=0)
    for i in range(len(eta_bin_centers)):
        print(eta_bin_centers[i], " GeV: ", mean_effs[i], " +- ", std_effs[i])

    eta_output[d] = {
        "bins": eta_bins,
        "bin_centers": eta_bin_centers,
        "effs_mean": mean_effs,
        "effs_std": std_effs,
    }

output = {
    "pt": pt_output,
    "eta": eta_output,
    "lhc": lhc_effs,
    "perfect": perfect_effs,
    "dm": dm_effs,
}
np.save(f"track-stats/EQ-EN-train-L1-{pt_min}GeV", output)
