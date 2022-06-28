import torch
import torch_geometric
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear

from utils import minkowski_feats, unsorted_segment_sum


class LGEB(MessagePassing):
    def __init__(self, n_input, n_output, n_hidden, last_layer=False):
        super(LGEB, self).__init__(aggr="add", flow="source_to_target")
        # TODO: add dims for edge attributes (delr, delphi, delz, delR)
        # TODO: add c_weight
        self.n_edge_attributes = 2  # dims for Minkowski norm & inner product

        self.phi_e = Sequential(
            Linear(n_input * 2 + self.n_edge_attributes, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.phi_h = Sequential(
            Linear(n_input + n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        self.phi_m = nn.Sequential(nn.Linear(self.hidden, 1), nn.Sigmoid())

        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def forward(self, x, h, edge_index, edge_attr):
        row, col = edge_index
        norms, dots, x_diff = minkowski_feats(edge_index, x)
        h_tilde, x_tilde = self.propagate(
            edgs_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            h=h,
            norms=norms,
            dots=dots,
            x_diff=x_diff,
            size=None,
        )
        return h_tilde, x_tilde

    def message(self, h_i, h_j, edge_attr, norms, dots):
        # h_i -> incoming
        # h_j -> outgoing
        m_ij = torch.cat([h_i, h_j, edge_attr, norms, dots], dim=1)
        m_ij = self.phi_e(m_ij)
        w = self.phi_m(m_ij)
        m_ij = m_ij * w
        return m_ij

    def aggregate(self, m_ij, h, x, edge_index, x_diff):
        row, col = edge_index
        trans = x_diff * self.phi_x(m_ij)
        h_agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0))
        x_agg = unsorted_segment_sum(trans, row, num_segments=x.size(0))
        return h_agg, x_agg

    def update(self, h_agg, x_agg, h, x):
        h = h + self.phi_h(h_agg)
        x = x + self.c_weight * x_agg
        return h, x

class LorentzNet(nn.Module):
    def __init__(self, n_scalar, n_hidden, n_layers):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([
            LGEB(self.n_hidden, self.n_hidden, self.n_hidden, last_layer=(i==n_layers-1))
            for i in range(self.n_layers)
        ])

        self.edge_mlp = Linear(2*self.n_hidden, 1)

    def forward(self, scalars, x, edge_index, edge_attr):
        h = self.embedding(scalars)

        for i in range(self.n_layers):
            h, x = self.LGEBs[i](h, x, edge_index, edge_attr)
        
        agg_out = torch.cat([x, h], dim=1)
        return torch.sigmoid(self.edge_mlp(agg_out))