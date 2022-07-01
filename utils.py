from typing import List
import torch
from torch import Tensor

__all__ = ["unsorted_segment_sum", "euclidean_feats"]


def unsorted_segment_sum(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def euclidean_feats(edge_index: Tensor, x: Tensor) -> List[Tensor]:
    i, j = edge_index
    x_diff = x[i] - x[j]
    norms = norm(x_diff).unsqueeze(1)
    dots = dot(x[i], x[j]).unsqueeze(1)
    norms, dots = psi(norms), psi(dots)
    return norms, dots, x_diff


def norm(x: Tensor) -> Tensor:
    r""" Euclidean square norm
         `\|x\|^2 = x[0]^2+x[1]^2+x[2]^2`
    """
    x_sq = torch.pow(x, 2)
    return x_sq.sum(dim=-1)


def dot(x: Tensor, y: Tensor) -> Tensor:
    r""" Euclidean inner product
         `<x,y> = x[0]y[0]+x[1]y[1]+x[2]y[2]`
    """
    xy = x * y
    return xy.sum(dim=-1)


def psi(x: Tensor) -> Tensor:
    """ `\psi(x) = sgn(x) \cdot \log(|x| + 1)`
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)
