import torch
import torch_geometric
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear

from utils import minkowski_feats, unsorted_segment_sum


class LGEB(MessagePassing):
    def __init__(self, n_input, n_hidden, c_weight = 1.0, last_layer=False):
        super(LGEB, self).__init__(aggr="add", flow="source_to_target")
        # TODO: Add support for scalar quantities
        # TODO: add dims for edge attributes (delr, delphi, delz, delR)
        self.n_edge_attributes = 6  # dims for norm & inner product + (delr, delphi, delz, delR)
        self.c_weight = c_weight

        self.phi_e = Sequential(
            Linear(n_input * 2 + self.n_edge_attributes, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())

        self.last_layer = last_layer

    def forward(self, x, edge_index, edge_attr):
        norms, dots, x_diff = minkowski_feats(edge_index, x)
        x_tilde = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            norms=norms,
            dots=dots,
            x_diff=x_diff,
            size=None,
        )
        return x_tilde

    def message(self, x_i, x_j, edge_attr, norms, dots):
        # x_i -> incoming
        # x_j -> outgoing
        m_ij = torch.cat([x_i, x_j, edge_attr, norms, dots], dim=1)
        m_ij = self.phi_e(m_ij)
        w = self.phi_m(m_ij)
        m_ij = m_ij * w
        return m_ij

    def aggregate(self, m_ij, x, edge_index, x_diff):
        row, col = edge_index
        trans = x_diff * self.phi_x(m_ij)
        trans = torch.clamp(trans, min=-100, max=100)
        x_agg = unsorted_segment_sum(trans, row, num_segments=x.size(0))
        return x_agg

    def update(self, x_agg, x):
        x = x + self.c_weight * x_agg
        return x


class LorentzNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_edges):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_edges = n_edges
        self.n_input = n_input
        self.LGEBs = nn.ModuleList(
            [
                LGEB(
                    n_input=self.n_input,
                    n_hidden=self.n_hidden,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(self.n_layers)
            ]
        )

        self.edge_mlp = Linear(self.n_hidden, self.n_edges)

    def forward(self, x, edge_index, edge_attr):

        for i in range(self.n_layers):
            print(i)
            x = self.LGEBs[i](x, edge_index, edge_attr)

        return torch.sigmoid(self.edge_mlp(x))
