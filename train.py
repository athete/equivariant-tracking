import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from models.euclidean import EuclidNet
from models.dataset import GraphDataset


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time.time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        y, out = data.y, out.squeeze(1)
        loss = F.binary_cross_entropy(out, y, reduction="mean")
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.6f})]\tLoss: {loss.item():.6f}"
            )
        if args.dry_run:
            break
        losses.append(loss.item())
    print(f"...epoch time: {time.time() - epoch_t0}s")
    print(f"...epoch {epoch}: train loss = {np.mean(losses)}")
    return np.mean(losses)


def validate(model, device, val_loader):
    model.eval()
    opt_thlds, losses, accs = [], [], []
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        y, out = data.y, out.squeeze(1)
        loss = F.binary_cross_entropy(out, y, reduction="mean")

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

        opt_thlds.append(opt_thld)
        accs.append(acc)
        losses.append(loss.item())

    print(f"...validation accuracy = {np.mean(accs)}")
    return np.mean(opt_thlds), np.mean(losses)


def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            TP = torch.sum((data.y == 1).squeeze() & (out > thld).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (out < thld).squeeze()).item()
            FP = torch.sum((data.y == 0).squeeze() & (out > thld).squeeze()).item()
            FN = torch.sum((data.y == 1).squeeze() & (out < thld).squeeze()).item()
            acc = (TP + TN) / (TP + TN + FP + FN)
            loss = F.binary_cross_entropy(
                out.squeeze(1), data.y, reduction="mean"
            ).item()
            accs.append(acc)
            losses.append(loss)
    print(f"...test loss = {np.mean(losses)}\n...test accuracy = {np.mean(accs)}")
    return np.mean(losses), np.mean(accs)


def main():
    # Training argument
    parser = argparse.ArgumentParser(
        description="Euclidean Equivariant Network Implementation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--step-size", type=int, default=5, help="Learning rate step size"
    )
    parser.add_argument(
        "--pt", type=str, default="2", help="Cutoff pt value in GeV (default: 2)"
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
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status (default: 10)",
    )
    parser.add_argument(
        "--construction",
        type=str,
        default="geometric",
        help="graph construction method",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For saving the current Model",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=40, help="Number of hidden units per layer"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of repetitions of the equivariant block (default: 5)",
    )
    parser.add_argument(
        "--c-weight",
        type=float,
        default=1e-3,
        help="Weight hyperparameter for updates (default: 1e-3)",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"use_cuda={use_cuda}")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    home_dir = ""
    indir = f"hitgraphs/{args.construction}/{args.pt.replace('.', 'p')}/"
 
    graph_files = np.array(os.listdir(indir))
    graph_files = np.array(
        [os.path.join(indir, graph_file) for graph_file in graph_files]
    )
    graph_paths = [os.path.join(indir, filename) for filename in graph_files]
    n_graphs = len(graph_files)

    IDs = np.arange(n_graphs)
    np.random.shuffle(IDs)
    partition = {
        "train": graph_files[IDs[:1000]],
        "test": graph_files[IDs[1000:1400]],
        "val": graph_files[IDs[1400:1500]],
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

    model = EuclidNet(
        n_input=train_set.get(0).x.size()[1],
        n_hidden=args.hidden_size,
        n_layers=args.num_layers,
        n_output=1,
        c_weight=args.c_weight,
    )
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_trainable_params}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    output = {"train_loss": [], "test_loss": [], "test_acc": [], "val_loss": []}
    for epoch in range(1, args.epochs + 1):
        print(f"---- Epoch {epoch} ----")
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thld, val_loss = validate(model, device, val_loader)
        print(f"...optimal threshold: {thld}")
        test_loss, test_acc = test(model, device, test_loader, thld=thld)
        scheduler.step()

        if args.save_model:
            torch.save(
                model.state_dict(),
                f"trained_models/EN_{args.construction}_epoch{epoch}_{args.pt}GeV.pt",
            )

        output["train_loss"].append(train_loss)
        output["test_loss"].append(test_loss)
        output["test_acc"].append(test_acc)
        output["val_loss"].append(val_loss)

    np.save(f"train_output/EN_{args.construction}_{args.pt}GeV", output)
