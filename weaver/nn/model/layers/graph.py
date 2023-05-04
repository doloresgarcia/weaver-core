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
from weaver.utils.nn.interactions_ee import pairwise_lv_fts

# from torch_geometric.data import Data
from weaver.nn.model.layers.caley_gen import _CAYLEY_BOUNDS
from scipy import sparse as sp
import networkx as nx
from dgl import RemoveSelfLoop
from weaver.nn.model.layers.helper_graph_function import (
    positional_encoding,
    reduced_pf,
    dif_points,
)
from weaver.nn.model.layers.helper_graph_function import (
    correct_momentum_vect,
    create_dij_interactions,
)


def create_graph_knn(pf_points, pf_features, pf_vectors, mask, seq_len, device):
    # print(example[0].keys())

    pf_points, pf_features, pf_vectors = reduced_pf(
        pf_points, pf_features, pf_vectors, mask, seq_len, device
    )
    g = dgl.knn_graph(pf_points, 16)  # , exclude_self= True
    i = g.edges()[0]
    j = g.edges()[1]
    # pf_vectors = correct_momentum_vect(pf_vectors, pf_points)
    # kt, delta = create_dij_interactions(i, j, pf_vectors, seq_len)
    x = pf_features
    ## w = torch.reshape(pf_vectors[:,3]/torch.sum(pf_vectors[:,3]),(-1,1))
    # edge_attr = torch.cat((kt, delta), dim=1)
    g = g.to(device)
    g.ndata["h"] = x.to(device)
    # g.edata["h"] = edge_attr.to(device)
    # g.ndata['w'] = w.to(device)
    return g


def create_graph_knn2(pf_points, pf_features, pf_vectors, mask, seq_len, device):
    # print(example[0].keys())
    pf_points, pf_features, pf_vectors = reduced_pf(
        pf_points, pf_features, pf_vectors, mask, seq_len, device
    )
    g = dgl.knn_graph(pf_points, 3, exclude_self=True)
    i = g.edges()[0]
    j = g.edges()[1]

    pf_vectors = correct_momentum_vect(pf_vectors, pf_points)
    kt, delta = create_dij_interactions(i, j, pf_vectors, seq_len)

    x_interactions2 = pf_features
    x_interactions2 = torch.reshape(x_interactions2, [seq_len, 1, 34])
    x_interactions2 = x_interactions2.repeat(1, seq_len, 1)

    xi2 = x_interactions2[i, j, :]  # (batch, dim, seq_len*(seq_len+1)/2)
    xj2 = x_interactions2[j, i, :]
    x_interactions_m2 = xi2 - xj2
    # edges = torch.zeros(seq_len,seq_len,4)
    # edges[i,j,:] = x_interactions_m
    # edges[j,i,:] = x_interactions_m
    edge_index = torch.zeros(2, len(i))
    edge_index[0, :] = i.long()
    edge_index[1, :] = j.long()
    # edge_attr = torch.cat((x_interactions_m,x_interactions_m2), dim = 1)
    x = pf_features
    edge_attr = torch.cat((kt, delta), dim=1)
    g = g.to(device)
    g.ndata["h"] = x.to(device)
    g.edata["h"] = edge_attr.to(device)
    g.ndata["pos"] = pf_points
    return g


def create_graph_knn_diffs(pf_points, pf_features, pf_vectors, mask, seq_len, device):
    # print(example[0].keys())
    pf_points, pf_features, pf_vectors = reduced_pf(
        pf_points, pf_features, pf_vectors, mask, seq_len, device
    )
    g = dgl.knn_graph(pf_points, 16)  # , exclude_self= True
    xtransform = RemoveSelfLoop()
    g = xtransform(g)
    # g = positional_encoding(g, 2, device)

    i = g.edges()[0]
    j = g.edges()[1]
    pf_vectors = correct_momentum_vect(pf_vectors, pf_points)
    kt, delta = create_dij_interactions(i, j, pf_vectors, seq_len)
    x_interactions_m = dif_points(pf_points, seq_len, i, j)
    x = pf_features
    edge_attr = torch.cat((kt, delta, x_interactions_m), dim=1)  # currently dim 4

    g = g.to(device)
    g.ndata["h"] = x.to(device)
    g.edata["h"] = edge_attr.to(device)
    # g.ndata['w'] = w.to(device)
    return g


def make_full_graph(g):
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Here we copy over the node feature data and laplace encodings
    full_g.ndata["feat"] = g.ndata["feat"]

    try:
        full_g.ndata["EigVecs"] = g.ndata["EigVecs"]
        full_g.ndata["EigVals"] = g.ndata["EigVals"]
    except:
        pass

    # Populate edge features w/ 0s
    full_g.edata["feat"] = torch.zeros(full_g.number_of_edges(), 1)
    full_g.edata["real"] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over
    full_g.edges[g.edges(form="uv")[0].tolist(), g.edges(form="uv")[1].tolist()].data[
        "feat"
    ] = g.edata["feat"]
    full_g.edges[g.edges(form="uv")[0].tolist(), g.edges(form="uv")[1].tolist()].data[
        "real"
    ] = torch.ones(g.edata["feat"].shape[0], dtype=torch.long)

    return full_g


def graph_return_knnij(output, y):
    number_p = np.int32(np.sum(output["pf_mask"]))
    pos = torch.permute(torch.tensor(output["pf_points"][:, 0:number_p]), (1, 0))
    number_features = len(output["pf_features"]) - 1
    pf_features = torch.permute(
        torch.tensor(output["pf_features"][:, 0:number_p]), (1, 0)
    )  # take out the jet number from the features

    pf_vectors = torch.permute(
        torch.tensor(output["pf_vectors"][:, 0:number_p]), (1, 0)
    )  # take out the jet number from the features
    seq_len = number_p
    g = dgl.knn_graph(pos, 5)  # , exclude_self= True

    i = g.edges()[0]
    j = g.edges()[1]

    pf_vectors = correct_momentum_vect(pf_vectors, pos)

    x_interactions = pf_vectors
    x_interactions = torch.reshape(x_interactions, [seq_len, 1, 4])
    x_interactions = x_interactions.repeat(1, seq_len, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    pairwise_lv_fts_p = partial(pairwise_lv_fts)
    x_interactions_m = pairwise_lv_fts_p(xi, xj, num_outputs=1)
    x = pf_features
    edge_attr = x_interactions_m

    g.ndata["feat"] = x
    g.edata["feat"] = edge_attr
    xtransform = RemoveSelfLoop()
    g = xtransform(g)
    g = positional_encoding(g, 20)
    g = make_full_graph(g)

    return g
