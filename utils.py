import torch

__all__ = ["unsorted_segment_sum", "minkowski_feats"]


def unsorted_segment_sum(data, segment_ids, num_segments):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def minkowski_feats(edges, x):
    i, j = edges
    x_diff = x[i] - x[j]
    norms = normsq4(x_diff).unsqueeze(1)
    dots = dotsq4(x[i], x[j]).unsqueeze(1)
    norms, dots = psi(norms), psi(dots)
    return norms, dots, x_diff


def normsq4(p):
    r""" Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    """
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    r""" Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    """
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p):
    """ `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    """
    return torch.sign(p) * torch.log(torch.abs(p) + 1)
