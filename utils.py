from typing import List, Union
import torch
from torch import Tensor
from torch import nn


__all__ = [
    "unsorted_segment_sum",
    "unsorted_segment_mean",
    "euclidean_feats",
    "make_mlp",
]


def unsorted_segment_sum(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def euclidean_feats(
    edge_index: Tensor, x: Tensor, s: Union[Tensor, None]
) -> List[Tensor]:
    i, j = edge_index
    x_diff = x[i] - x[j]
    norms = norm(x_diff).unsqueeze(1)
    dots = dot(x[i], x[j]).unsqueeze(1)
    norms, dots = psi(norms), psi(dots)

    # Handle first GNN iteration
    s_cat = torch.cat([s[i], s[j]], dim=1) if s is not None else None

    return norms, dots, x_diff, s_cat


def norm(x: Tensor) -> Tensor:
    r"""Euclidean square norm
    `\|x\|^2 = x[0]^2+x[1]^2+x[2]^2`
    """
    x_sq = torch.pow(x, 2)
    return x_sq.sum(dim=-1)


def dot(x: Tensor, y: Tensor) -> Tensor:
    r"""Euclidean inner product
    `<x,y> = x[0]y[0]+x[1]y[1]+x[2]y[2]`
    """
    xy = x * y
    return xy.sum(dim=-1)


def psi(x: Tensor) -> Tensor:
    """`\psi(x) = sgn(x) \cdot \log(|x| + 1)`"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def make_mlp(
    input_size: int,
    sizes: List,
    hidden_activation: str = "SiLU",
    output_activation: str = None,
    layer_norm: bool = False,
    batch_norm: bool = False,
) -> nn.Sequential:
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False)
            )
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False)
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)
