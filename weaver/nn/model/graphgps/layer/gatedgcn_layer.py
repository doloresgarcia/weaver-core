import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_scatter import scatter
import dgl
import torch_geometric
from torch_geometric.data import Data
import numpy as np


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        residual,
        act="relu",
        equivstable_pe=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.activation = register.act_dict[act]
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid(),
            )
        weird_batch_norm = False
        self.weird_batch_norm = weird_batch_norm

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None

        self.update_graph = True
        if self.update_graph:
            self.out_dim = out_dim
            hidden_nf = out_dim * 2
            act_fn = nn.ReLU()
            act_fn2 = nn.Sigmoid()
            self.edge_mlp = nn.Sequential(
                nn.Linear(out_dim * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1),
                act_fn2,
            )
        # elif self.update_graph_knn:
        #     self.out_dim = out_dim
        #     hidden_nf = out_dim * 2
        #     act_fn = nn.SiLU()
        #     self.node_mlp = nn.Sequential(
        #         nn.Linear(out_dim, hidden_nf),
        #         act_fn,
        #         nn.Linear(hidden_nf, 3),
        #     )

    def forward(self, inputs):
        batch = inputs[0]
        turn_vicosity_on = inputs[1]
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e
        # print("shaaaaaaaaaapes")
        # print(x.shape)
        # print(e.shape)

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)
        # print(Dx.shape, Ce.shape, Ex.shape)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(
            edge_index.to(torch.int64),
            Bx=Bx,
            Dx=Dx,
            Ex=Ex,
            Ce=Ce,
            e=e,
            Ax=Ax,
            PE=pe_LapPE,
        )
        if self.weird_batch_norm:
            x = self.bn_node_x([x, turn_vicosity_on])
            e = self.bn_edge_e([e, turn_vicosity_on])
        else:
            # if turn_vicosity_on:
            #     if self.bn_node_x.momentum - 0.00001 >= 0:
            #         print(
            #             "updating momentum bn_node_x", self.bn_node_x.momentum - 0.00001
            #         )
            #         self.bn_node_x.momentum = self.bn_node_x.momentum - 0.00001
            #     if self.bn_edge_e.momentum - 0.00001 >= 0:
            #         print(
            #             "updating momentum bn_edge_e", self.bn_edge_e.momentum - 0.00001
            #         )
            #         self.bn_edge_e.momentum = self.bn_edge_e.momentum - 0.00001
            x = self.bn_node_x(x)
            e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e
        if self.update_graph:
            # # take x and e and create a new graph
            i, j = batch.edge_index
            len_original = batch.edge_index.shape[1]
            # print("grap updated", len_original, batch.edge_index.shape)
            if len_original > 0:

                g = dgl.graph((i.long(), j.long()), num_nodes=x.shape[0])
                i, j = g.edges()
                g.ndata["h"] = x
                g.edata["h"] = e
                g.apply_edges(src_dot_dst("h", "h", "edge_feature"))  # , edges)
                edge_resulting = self.edge_mlp(g.edata["edge_feature"])
                batch = percentage_keep_per_graph(batch, edge_resulting)
            #! this code does this overall but then it doesnt take into account that some graphs are left without edges
            #     mask_edges_keep = edge_resulting > 0.5
            #     print("keeeping", torch.sum(mask_edges_keep) / len_original)
            #     percentage_keep = torch.sum(mask_edges_keep) / len_original
            #     if percentage_keep > 0.05:
            #         batch.edge_index = batch.edge_index[:, mask_edges_keep.view(-1)]
            #         batch.edge_attr = e[mask_edges_keep.view(-1)]

        # elif self.update_graph_knn:
        #     s_l_coordinates = self.node_mlp(x)
        #     batch = update_knn(batch, s_l_coordinates)
        #     # problem. need to recalculate the edge attributes for the new graph

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        # print("SIZE OF PROBLEM")
        # print(Dx_i.shape, Ex_j.shape, e_ij.shape)
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


@register_layer("gatedgcnconv")
class GatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(
            in_dim=layer_config.dim_in,
            out_dim=layer_config.dim_out,
            dropout=0.0,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
            residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
            act=layer_config.act,
            **kwargs
        )

    def forward(self, batch):
        return self.model(batch)


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {
            out_field: torch.cat(
                (edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1
            )
        }

    return func


# def update_knn(batch, coordinates):
#     batch
#     number_batch = batch.batch[-1]
#     graphs = []
#     features_all = torch_geometric.utils.unbatch(batch.x, batch.batch)
#     edge_index_all = torch_geometric.utils.unbatch_edge_index(
#         batch.edge_index, batch.batch
#     )
#     for index in range(0, number_batch):
#         mask_batch = batch == index
#         coordinates_of_graph = coordinates[mask_batch]
#         features = features_all[index]
#         data_graph = Data(
#             x=features,
#             pos=coordinates_of_graph,
#             edge_index=edge_index_all[index],
#         )
#         if features.shape[0] > 11:
#             k = 11
#         else:
#             k = features.shape[0] - 1
#         knn_transform = torch_geometric.transforms.KNNGraph(k)
#         data = knn_transform(data_graph)
#         graphs.append(data)
#     bg = torch_geometric.data.Batch.from_data_list(graphs)
#     return bg


def percentage_keep_per_graph(batch, edge_resulting):
    batch_numbers = batch.batch.batch
    number_batch = batch_numbers[-1]
    graphs = []
    batch_numbers = batch.batch.batch.long()
    features_all = torch_geometric.utils.unbatch(batch.x, batch_numbers)
    edge_index_all = torch_geometric.utils.unbatch_edge_index(
        batch.edge_index.long(), batch_numbers
    )

    attribute_all = batch.edge_attr
    counter = 0
    count_nodes = 0
    for index in range(0, number_batch + 1):
        batch.edge_attr
        number_of_edges_in_graph = edge_index_all[index].shape[1]
        edge_sigmoid_graph = edge_resulting[
            counter : counter + number_of_edges_in_graph
        ]
        edge_att_graph = attribute_all[counter : counter + number_of_edges_in_graph]
        mask_ = edge_sigmoid_graph > 0.5
        print(
            "keeping this percentage of the graph",
            torch.sum(mask_.view(-1)) / len(mask_.view(-1)),
        )
        if torch.sum(mask_.view(-1)) / len(mask_.view(-1)) < 0.1:
            ordered_sigmoid = torch.flip(
                torch.argsort(edge_sigmoid_graph.view(-1)), dims=[0]
            )
            number_keep = np.ceil(len(mask_.view(-1)) * 0.1).astype(int)
            mask_ = ordered_sigmoid[0:number_keep]
            # keep at least two edges per graph
        new_edge_index = edge_index_all[index][:, mask_.view(-1)]
        features = features_all[index]
        e = edge_att_graph[mask_.view(-1)]
        count_nodes = count_nodes + features.shape[0]
        data_graph = Data(x=features, edge_index=new_edge_index, edge_attr=e)

        graphs.append(data_graph)
        counter = counter + number_of_edges_in_graph
    bg = torch_geometric.data.Batch.from_data_list(graphs)
    return bg
