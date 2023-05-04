import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import sys
from functools import partial
import os.path as osp
import time
import numpy as np
from weaver.nn.model.layers.mlp_readout_layer import MLPReadout
from weaver.nn.model.layers.graph import create_graph_knn, create_graph_knn2

"""
    Graph Transformer
    
"""
from weaver.nn.model.layers.graph_transformer_layer import GraphTransformerLayer
from weaver.nn.model.layers.mlp_readout_layer import MLPReadout
from weaver.nn.model.layers.caley_gen import build_cayley_bank


class GraphTransformerNet(nn.Module):
    def __init__(self, dev):
        super().__init__()

        in_dim_node = 35  # node_dim (feat is an integer)
        hidden_dim = 128  # before 80
        out_dim = 128
        n_classes = 5
        num_heads = 8
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 10
        self.n_layers = n_layers
        self.readout = "mean"
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = dev
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = 10
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer

        # self.embedding_e = nn.Linear(1, hidden_dim) # node feat is an integer

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim,
                out_dim,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
            )
        )
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        cayley_bank = build_cayley_bank()
        self.cayley_bank = cayley_bank

    def forward(self, points, features, lorentz_vectors, mask):
        ####################################### Convert to graphs ##############################
        batch_size = points.shape[0]

        graphs = []
        graphs_expander = []
        for i in range(0, batch_size):
            pf_points = points[i]
            pf_features = features[i]
            pf_vectors = lorentz_vectors[i]
            mask_ = mask[i]
            seq_len = np.int(torch.sum(mask_ * 1))
            if seq_len < 5:
                seq_len = 5
            g = create_graph_knn(
                pf_points,
                pf_features,
                pf_vectors,
                mask_,
                seq_len=seq_len,
                device=self.device,
            )
            # g = create_graph_expander(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len,
            #         device = self.device, cayley_bank= self.cayley_bank)
            # g_exp = graph_expander(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len,
            #         device = self.device, cayley_bank= self.cayley_bank)
            graphs.append(g)
            # graphs_expander.append(g_exp)

        batched_graph = dgl.batch(graphs)
        # batched_graph_exp = dgl.batch(graphs_expander)
        g_base = batched_graph
        # g_expander = batched_graph_exp
        ############################## Embeddings #############################################
        h = g_base.ndata["h"]
        # e = g_base.edata['h']

        # input embedding
        h = self.embedding_h(h)

        # if self.lap_pos_enc:
        #    h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        #    h = h + h_lap_pos_enc
        # if self.wl_pos_enc:
        #    h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
        #    h = h + h_wl_pos_enc
        # h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        it = 1
        for conv in self.layers:
            # if it %2 == 0:
            #    g = g_expander
            # else:
            g = g_base
            h = conv(g, h)

            # if it == self.n_layers-1:
            # this amis to mimic the FA layer and to see if the bottelneck is really there in our case
            # graphs = []
            # for i in range(0, batch_size):
            # mask_ = mask[i]
            # seq_len = np.int(torch.sum(mask_*1))
            # if seq_len <5:
            # seq_len = 5
            # u, v = torch.tril_indices(seq_len, seq_len, offset=-1)
            # graph_fully_connected  = dgl.graph((u, v))
            # graph_fully_connected = dgl.to_bidirected(graph_fully_connected)
            # graphs.append(graph_fully_connected)
            # batched_graph = dgl.batch(graphs)
            # g_new = batched_graph
            # g = g_new

        # output
        g.ndata["h"] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        else:
            hg = dgl.mean_nodes(g, "h")  # default readout is mean node

        h_out = self.MLP_layer(hg)

        return h_out

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
