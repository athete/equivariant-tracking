import torch
import torch.nn as nn
from torch_scatter import scatter_add
from utils import euclidean_feats, make_mlp


class EB(nn.Module):
    def __init__(
        self,
        n_edge_attributes: int = 2,
        n_scalar_attributes: int = 32,
        n_hidden: int = 32,
        nb_node_layer: int = 2,
        c_weight: float = 1.0,
    ) -> None:
        super(EB, self).__init__()
        # Controls the scale of x during updates
        self.c_weight = c_weight

        self.phi_e = make_mlp(
            n_edge_attributes,
            [n_hidden] * nb_node_layer,
            layer_norm=True,
        )

        # MLP to generate attention weights
        self.phi_x = make_mlp(
            n_hidden,
            [n_hidden] * nb_node_layer + [1],
            layer_norm=True,
        )

        # MLP to generate weights for the messages
        self.phi_m = make_mlp(
            n_hidden,
            [n_hidden] * nb_node_layer,
            output_activation="Sigmoid",
            layer_norm=True,
        )

        self.phi_s = make_mlp(
            n_scalar_attributes,
            [n_hidden] * nb_node_layer,
            layer_norm=True,
        )

    def message(self, norms, dots, s_cat=None, e=None):
        if s_cat is not None and e is not None:
            e_ij = torch.cat([norms, dots, s_cat, e], dim=1)
        elif s_cat is not None:
            e_ij = torch.cat([norms, dots, s_cat], dim=1)
        else:
            e_ij = torch.cat([norms, dots], dim=1)
        e_ij = self.phi_e(e_ij)  # The edge features
        m_ij = self.phi_m(e_ij)
        return m_ij, e_ij

    def x_model(self, x, edge_index, x_diff, m):
        i, j = edge_index
        update_val = x_diff * self.phi_x(m)
        # LorentzNet authors clamp the update tensor as a precautionary measure
        update_val = torch.clamp(update_val, min=-100, max=100)
        x_agg = scatter_add(update_val, i, dim=0, dim_size=x.size(0))
        x = x + x_agg * self.c_weight
        return x

    def s_model(self, s, v, edge_index, m):
        i, j = edge_index
        s_agg = scatter_add(m, i, dim=0, dim_size=v.size(0))
        if s is not None:
            s_agg = self.phi_s(torch.cat([s, s_agg], dim=1))
        else:
            s_agg = self.phi_s(s_agg)
        s = s + s_agg * self.c_weight
        return s

    def forward(self, v, edge_index, s=None, e=None):
        norms, dots, v_diff, s_cat = euclidean_feats(edge_index, v, s)
        m, e = self.message(norms, dots, s_cat, e)
        v_tilde = self.x_model(v, edge_index, v_diff, m)
        s_tilde = self.s_model(s, v, edge_index, m)
        return v_tilde, s_tilde, e


class EuclidNet(nn.Module):
    def __init__(self, hparams) -> None:
        super(EuclidNet, self).__init__()

        self.hparams = hparams
        self.n_input = hparams["n_input"]
        self.n_hidden = hparams["n_hidden"]
        self.n_layers = hparams["n_layers"]
        self.n_graph_iters = hparams["n_graph_iters"]
        self.c_weight = hparams["c_weight"]
        if hparams["equi_output"]:
            self.n_output = 3 * self.n_hidden
        else:
            self.n_output = self.n_hidden + 2 * hparams["vector_dim"]

        edge_attributes, scalar_attributes = 2 + 3 * self.n_hidden, 2 * self.n_hidden

        # Setup input network
        self.node_encoder = make_mlp(
            1,
            [hparams["n_hidden"]] * hparams["n_layers"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        self.edge_encoder = make_mlp(
            2 * hparams["n_hidden"],
            [hparams["n_hidden"]] * hparams["n_layers"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        self.EB = EB(
            edge_attributes,
            scalar_attributes,
            n_hidden=self.n_hidden,
            nb_node_layer=self.n_layers,
            c_weight=self.c_weight,
        )

        # MLP to produce edge weights
        self.edge_mlp = make_mlp(
            self.n_output,
            [self.n_hidden] * self.n_layers + [1],
            layer_norm=True,
        )

    def forward(self, x, edge_index):

        v = x[:, :2]
        s = x[:, 2].unsqueeze(-1)

        s = self.node_encoder(s)
        e = self.edge_encoder(torch.cat([s[edge_index[0]], s[edge_index[1]]], dim=1))

        for _ in range(self.n_graph_iters):
            v, s, e = self.EB(v, edge_index, s, e)

        if self.hparams["equi_output"]:
            m = torch.cat([s[edge_index[1]], s[edge_index[0]], e], dim=1)
        else:
            m = torch.cat([v[edge_index[1]], v[edge_index[0]], e], dim=1)

        return torch.sigmoid(self.edge_mlp(m))
