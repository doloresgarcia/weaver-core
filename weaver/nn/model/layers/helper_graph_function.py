import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from scipy import sparse as sp
import numpy as np
from functools import partial


def positional_encoding(g, pos_enc_dim, device):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(
        L, k=pos_enc_dim + 1, which="SR", tol=1e-2
    )  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata["pos_enc"] = (
        torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float().to(device)
    )

    return g


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs__ = [el[0] for el in list_graphs]
    y = [el[1] for el in list_graphs]
    bg = dgl.batch(list_graphs__)
    # bg = dgl.batch(list_graphs_g)
    # bg_exp = dgl.batch(list_graphs_gexp)
    return bg, y


def create_graph_expander(
    pf_points, pf_features, pf_vectors, mask, seq_len, device, cayley_bank
):
    # print(example[0].keys())
    pf_points = pf_points[:, 0:seq_len]
    pf_features = pf_features[:, 0:seq_len]
    pf_vectors = pf_vectors[:, 0:seq_len]
    coord_shift = (mask == 0)[:, 0:seq_len] * 1e9
    pf_points = pf_points + coord_shift
    pf_points = torch.permute(torch.tensor(pf_points), (1, 0))
    pf_features = torch.permute(torch.tensor(pf_features), (1, 0))
    pf_vectors = torch.permute(torch.tensor(pf_vectors), (1, 0))

    g = 0
    senders = []
    receivers = []
    node_lims = [seq_len]
    p = 2
    chosen_i = -1
    for i in range(len(_CAYLEY_BOUNDS)):
        sz, p = _CAYLEY_BOUNDS[i]
        if sz >= node_lims[g]:
            chosen_i = i
            break

    assert chosen_i >= 0
    _p, edge_pack = cayley_bank[chosen_i]
    assert p == _p

    for v, w in zip(*edge_pack):
        if v < node_lims[g] and w < node_lims[g]:
            senders.append(v)
            receivers.append(w)

    graph_expander = dgl.graph((senders, receivers))
    graph_expander = dgl.to_simple(graph_expander)

    i = graph_expander.edges()[0]
    j = graph_expander.edges()[1]

    # before calculating the esge attributed fix momentum
    energy = pf_vectors[:, 0].clone()
    momentum = pf_vectors[:, 1].clone()
    pf_vectors[:, 0] = (
        momentum * torch.sin(pf_features[:, 1]) * torch.cos(pf_features[:, 2])
    )
    pf_vectors[:, 1] = (
        momentum * torch.sin(pf_features[:, 1]) * torch.sin(pf_features[:, 2])
    )
    pf_vectors[:, 2] = momentum * torch.cos(pf_features[:, 2])
    pf_vectors[:, 3] = energy

    x_interactions = pf_vectors
    x_interactions = torch.reshape(x_interactions, [seq_len, 1, 4])
    x_interactions = x_interactions.repeat(1, seq_len, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    pairwise_lv_fts_p = partial(pairwise_lv_fts)
    x_interactions_m = pairwise_lv_fts_p(xi, xj, num_outputs=1)
    x = pf_features
    w = torch.reshape(pf_vectors[:, 3] / torch.sum(pf_vectors[:, 3]), (-1, 1))
    edge_attr = x_interactions_m
    graph_expander = graph_expander.to(device)
    graph_expander.ndata["h"] = x.to(device)
    graph_expander.edata["h"] = edge_attr.to(device)

    return graph_expander


def graph_expander(
    pf_points, pf_features, pf_vectors, mask, seq_len, device, cayley_bank
):
    # print(example[0].keys())
    pf_points = pf_points[:, 0:seq_len]
    pf_features = pf_features[:, 0:seq_len]
    pf_vectors = pf_vectors[:, 0:seq_len]
    coord_shift = (mask == 0)[:, 0:seq_len] * 1e9
    pf_points = pf_points + coord_shift
    pf_points = torch.permute(torch.tensor(pf_points), (1, 0))
    pf_features = torch.permute(torch.tensor(pf_features), (1, 0))
    pf_vectors = torch.permute(torch.tensor(pf_vectors), (1, 0))

    g = 0
    senders = []
    receivers = []
    node_lims = [seq_len]
    p = 2
    chosen_i = -1
    for i in range(len(_CAYLEY_BOUNDS)):
        sz, p = _CAYLEY_BOUNDS[i]
        if sz >= node_lims[g]:
            chosen_i = i
            break

    assert chosen_i >= 0
    _p, edge_pack = cayley_bank[chosen_i]
    assert p == _p

    for v, w in zip(*edge_pack):
        if v < node_lims[g] and w < node_lims[g]:
            senders.append(v)
            receivers.append(w)

    graph_expander = dgl.graph((senders, receivers))
    graph_expander = dgl.to_simple(graph_expander)

    i = graph_expander.edges()[0]
    j = graph_expander.edges()[1]

    graph_expander = graph_expander.to(device)

    return graph_expander


def reduced_pf(pf_points, pf_features, pf_vectors, mask, seq_len, device):
    pf_points = pf_points[:, 0:seq_len]
    pf_features = pf_features[:, 0:seq_len]
    pf_vectors = pf_vectors[:, 0:seq_len]
    coord_shift = (mask == 0)[:, 0:seq_len] * 1e9
    pf_points = pf_points + coord_shift
    pf_points = torch.permute(torch.tensor(pf_points), (1, 0))
    pf_features = torch.permute(torch.tensor(pf_features), (1, 0))
    pf_vectors = torch.permute(torch.tensor(pf_vectors), (1, 0))
    return pf_points, pf_features, pf_vectors


def dif_points(pf_points, seq_len, i, j):
    x_interactions = pf_points
    xi = x_interactions[i]
    xj = x_interactions[j]
    x_interactions_m = torch.abs(xi - xj)
    return x_interactions_m


def pairwise_lv_distance(xi, xj, eps=1e-8):
    dot_product = torch.diagonal(torch.matmul(xi, torch.permute(xj, (1, 0))))
    dot_product = dot_product / (
        xi.norm(dim=1).clamp(min=eps) * xj.norm(dim=1).clamp(min=eps)
    )
    dot_product = dot_product.clamp(min=-1 + eps, max=1 - eps)
    thetaij = torch.acos(dot_product)
    return thetaij


def correct_momentum_vect(pf_vectors, pf_points):
    energy = pf_vectors[:, 0].clone()
    momentum = pf_vectors[:, 1].clone()
    pf_vectors[:, 0] = (
        momentum * torch.sin(pf_points[:, 0]) * torch.cos(pf_points[:, 1])
    )
    pf_vectors[:, 1] = (
        momentum * torch.sin(pf_points[:, 0]) * torch.sin(pf_points[:, 1])
    )
    pf_vectors[:, 2] = momentum * torch.cos(pf_points[:, 1])
    pf_vectors[:, 3] = energy
    return pf_vectors


def pairwise_lv_ftsdij(xi, xj, eps=1e-8):
    dot_product = torch.diagonal(
        torch.matmul(xi[:, 1:4], torch.permute(xj[:, 1:4], (1, 0)))
    )
    dot_product = dot_product / (
        xi[:, 1:4].norm(dim=1).clamp(min=eps) * xj[:, 1:4].norm(dim=1).clamp(min=eps)
    )
    dot_product = dot_product.clamp(min=-1 + eps, max=1 - eps)
    costhetaij = dot_product
    dij = 2 * torch.min(xi[:, 0], xj[:, 0]) * (1 - costhetaij)
    delta = 1 - costhetaij
    return dij, delta


def create_dij_interactions(i, j, features_massobs, number_p):
    x_interactions = features_massobs[:, 0:4]
    x_interactions = torch.reshape(x_interactions, [number_p, 1, 4])
    x_interactions = x_interactions.repeat(1, number_p, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    xinterations, delta = pairwise_lv_ftsdij(xi, xj)
    kt = torch.reshape(xinterations, [-1, 1])
    delta = torch.reshape(delta, [-1, 1])
    return kt, delta
