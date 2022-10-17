import os
import argparse
from time import time
from statistics import mean
from typing import List
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import yaml
from models.dataset import GraphDataset

warnings.filterwarnings("ignore")


def train(
    args: dict,
    model: torch.nn.Module,
    device: str,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        y, out = data.y, out.squeeze(1)
        loss = F.binary_cross_entropy(out, y, reduction="mean")
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.6f})]\tLoss: {loss.item():.6f}"
            )
        if args.dry_run == True:
            quit()
        losses.append(loss.item())
    print(f"...epoch time: {time() - epoch_t0}s")
    print(f"...epoch {epoch}: train loss = {mean(losses)}")
    return mean(losses)


def validate(
    model: torch.nn.Module, device: str, val_loader: DataLoader
) -> List[np.array]:
    model.eval()
    opt_thlds, losses, accs, aucs = [], [], [], []
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        y, out = data.y, out.squeeze(1)
        loss = F.binary_cross_entropy(out, y, reduction="mean").item()

        # define optimal threshold where TPR = TNR
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.001, 0.5, 0.001):
            TP = torch.sum((y == 1) & (out > thld)).item()
            TN = torch.sum((y == 0) & (out < thld)).item()
            FP = torch.sum((y == 0) & (out > thld)).item()
            FN = torch.sum((y == 1) & (out < thld)).item()
            acc = (TP + TN) / (TP + TN + FP + FN)
            TPR, TNR = TP / (TP + FN), TN / (TN + FP)
            delta = abs(TPR - TNR)
            if delta < diff:
                diff, opt_thld, opt_acc = delta, thld, acc

        auc = roc_auc_score(y.cpu(), (out > opt_thld).float().cpu())
        aucs.append(auc)
        opt_thlds.append(opt_thld)
        accs.append(acc)
        losses.append(loss)

    print(f"...validation accuracy = {np.mean(accs)}")
    print(f"...validation ROC AUC = {np.mean(aucs)}")
    return np.mean(opt_thlds), np.mean(losses), np.mean(aucs)


def test(
    model: torch.nn.Module, device: str, test_loader: DataLoader, thld: float = 0.5
) -> List[np.array]:
    model.eval()
    losses, accs, aucs, purity, effs = [], [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            TP = torch.sum((data.y == 1).squeeze() & (out > thld).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (out < thld).squeeze()).item()
            FP = torch.sum((data.y == 0).squeeze() & (out > thld).squeeze()).item()
            FN = torch.sum((data.y == 1).squeeze() & (out < thld).squeeze()).item()
            acc = (TP + TN) / (TP + TN + FP + FN)
            loss = F.binary_cross_entropy(
                out.squeeze(1), data.y, reduction="mean"
            ).item()
            auc = roc_auc_score(data.y.cpu(), (out > thld).float().cpu())
            aucs.append(auc)
            accs.append(acc)
            losses.append(loss)
            purity.append((TP / (TP + FN)))
            effs.append((TP / (TP + FP)))
    print(
        f"...test loss = {np.mean(losses)}\n...test accuracy = {np.mean(accs)}\n...test ROC AUC = {np.mean(aucs)}"
    )
    print(f"...test purity = {np.mean(purity)}\n...test efficiency = {np.mean(effs)}")
    return np.mean(losses), np.mean(accs), np.mean(aucs), np.mean(purity), np.mean(effs)


def main():
    print("In main")

    # Training arguments
    parser = argparse.ArgumentParser(
        description="Euclidean Equivariant Network"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="SO2",
        metavar="G",
        help="equivariance group (default: SO(2))",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For saving the current model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"use_cuda={use_cuda}")

    # open config file
    with open("./models/config.yaml", "r") as config_file:
        hparams = yaml.safe_load(config_file)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device}")
    train_kwargs = {"batch_size": hparams["batch_size"]}
    test_kwargs = {"batch_size": hparams["test_batch_size"]}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    indir = os.path.join(hparams["input_dir"], hparams["hitgraph"], hparams["pt"])
    print(indir)

    graph_files = np.array(os.listdir(indir))
    graph_files = np.array(
        [os.path.join(indir, graph_file) for graph_file in graph_files]
    )
    n_graphs = len(graph_files)

    IDs = np.arange(n_graphs)
    partition = {
        "train": graph_files[IDs[:1000]],
        "test": graph_files[IDs[1400:1500]],
        "val": graph_files[IDs[1000:1500]],
    }
    params = {"batch_size": 1, "shuffle": True, "num_workers": 6}

    train_set = GraphDataset(graph_files=partition["train"])
    train_loader = DataLoader(train_set, **params)
    print("...Successfully loaded train graphs")
    test_set = GraphDataset(graph_files=partition["test"])
    test_loader = DataLoader(test_set, **params)
    print("...Successfully loaded test graphs")
    val_set = GraphDataset(graph_files=partition["val"])
    val_loader = DataLoader(val_set, **params)
    print("...Successfully loaded val graphs")

    if args.group == "SO2":
        from models.euclidean_so2 import EuclidNet
    elif args.group == "SO3":
        from models.euclidean_so3 import EuclidNet
    else:
        raise NotImplementedError(f"Symmetry group {args.group} is not supported")

    model = EuclidNet(hparams).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_trainable_params}")
    optimizer = Adam(model.parameters(), lr=hparams["lr"])
    scheduler = StepLR(
        optimizer, step_size=hparams["step_size"], gamma=hparams["gamma"]
    )

    output = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "test_auc": [],
        "val_loss": [],
        "val_auc": [],
        "purity": [],
        "effs": [],
    }
    for epoch in range(1, hparams["epochs"] + 1):
        print(f"---- Epoch {epoch} ----")
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thld, val_loss, val_auc = validate(model, device, val_loader)
        print(f"...optimal threshold: {thld}")
        test_loss, test_acc, test_auc, purity, effs = test(
            model, device, test_loader, thld=thld
        )
        scheduler.step()

        output["train_loss"].append(train_loss)
        output["test_loss"].append(test_loss)
        output["test_acc"].append(test_acc)
        output["test_auc"].append(test_auc)
        output["val_loss"].append(val_loss)
        output["val_auc"].append(val_auc)
        output["purity"].append(purity)
        output["effs"].append(effs)

    np.save(
        f"train_output/so2/EN_{hparams['pt']}GeV_L{hparams['n_layers']}_hidden{hparams['n_hidden']}",
        output,
    )
    if args.save_model:
        torch.save(
            model.state_dict(),
            f"trained_models/so2/EN_{hparams['hitgraph']}_epoch{hparams['epochs']}_L{hparams['n_layers']}_h{hparams['n_hidden']}_{hparams['pt']}GeV.pt",
        )


if __name__ == "__main__":
    main()
