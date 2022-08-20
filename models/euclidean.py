import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear

from utils import euclidean_feats, unsorted_segment_mean


class EB(nn.Module):
    def __init__(self, n_hidden: int, c_weight: float = 1.0) -> None:
        super(EB, self).__init__()
        # dims for norm & inner product
        self.n_edge_attributes = 2
        # Controls the scale of x during updates
        self.c_weight = c_weight

        # MLP to create the message
        self.phi_e = Sequential(
            Linear(self.n_edge_attributes, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        layer = Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # MLP to generate attention weights
        self.phi_x = Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        # MLP to generate weights for the messages
        self.phi_m = Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())

    def message(self, norms, dots):
        m_ij = torch.cat([norms, dots], dim=1)
        m_ij = self.phi_e(m_ij)
        w = self.phi_m(m_ij)
        m_ij = m_ij * w
        return m_ij

    def x_model(self, x, edge_index, x_diff, m):
        i, j = edge_index
        update_val = x_diff * self.phi_x(m)
        # LorentzNet authors clamp the update tensor as a precautionary measure
        update_val = torch.clamp(update_val, min=-100, max=100)
        x_agg = unsorted_segment_mean(update_val, i, num_segments=x.size(0))
        x = x + x_agg * self.c_weight
        return x


    def forward(self, x, edge_index):
        norms, dots, x_diff = euclidean_feats(edge_index, x)
        m = self.message(norms, dots)
        x_tilde = self.x_model(x, edge_index, x_diff, m)
        return x_tilde


class EuclidNet(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_layers: int,
        n_output: int,
        c_weight: float = 1e-3,
    ) -> None:
        super(EuclidNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_output = n_output

        self.EBs = nn.ModuleList(
            [
                EB(n_hidden=self.n_hidden, c_weight=c_weight)
                for i in range(self.n_layers)
            ]
        )

        # MLP to produce edge weights
        self.edge_mlp = nn.Sequential(
            Linear(2 * n_input, self.n_hidden),
            nn.ReLU(),
            Linear(self.n_hidden, n_hidden),
            nn.ReLU(),
            Linear(self.n_hidden, self.n_output),
        )

    def forward(self, x, edge_index):
        for i in range(self.n_layers):

            x = self.EBs[i](x, edge_index)
        m = torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=1)
        return torch.sigmoid(self.edge_mlp(m))
