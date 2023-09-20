import torch
import sys
from functools import partial
from torch_geometric.data import Data
import os.path as osp
import time
import numpy as np
import torch_geometric
from copy import deepcopy
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    to_dense_adj,
    scatter,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
import dgl


def create_graph(example):
    # print(example[0].keys())
    seq_len = np.int32(np.sum(example[0]["pf_mask"]))
    pf_points = torch.permute(
        torch.tensor(example[0]["pf_points"][:, 0:seq_len]), (1, 0)
    )
    pf_features = torch.permute(
        torch.tensor(example[0]["pf_features"][:, 0:seq_len]), (1, 0)
    )
    pf_vectors = torch.permute(
        torch.tensor(example[0]["pf_vectors"][:, 0:seq_len]), (1, 0)
    )
    pf_mask = torch.permute(torch.tensor(example[0]["pf_mask"][:, 0:seq_len]), (1, 0))
    # x = torch.cat((pf_points, pf_features, pf_vectors, pf_mask), dim=1)
    x = pf_features
    y = torch.tensor(example[1]["_label_"])
    # if seq_len > 7:
    #     data = Data(x=x, pos=pf_points, y=y)
    #     knn_transform = torch_geometric.transforms.KNNGraph(7)
    #     data = knn_transform(data)
    # else:
    #     i, j = torch.tril_indices(seq_len, seq_len, offset=-1)
    #     edge_index = torch.zeros(2, len(i))
    #     edge_index[0, :] = i.long()
    #     edge_index[1, :] = j.long()
    #     data = Data(x=x, edge_index=edge_index, pos=pf_points, y=y)

    i, j = torch.tril_indices(seq_len, seq_len, offset=-1)
    g = dgl.graph((i, j))  # create fully connected graph
    g = dgl.to_simple(g)  # remove repated edges
    g = dgl.to_bidirected(g)
    i, j = g.edges()
    edge_index = torch.zeros(2, len(i))
    edge_index[0, :] = i.long()
    edge_index[1, :] = j.long()
    data = Data(x=x, edge_index=edge_index, pos=pf_points, y=y)

    # data = get_position_encodings(data)
    return data, y.view(-1)

    # import networkx as nx
    # import torch_geometric
    # g = torch_geometric.utils.to_networkx(data)
    # nx.draw(g)


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch pytorch geometric: pytorch geometric batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    list_y = torch.cat([el[1] for el in list_graphs], dim=0)
    bg = torch_geometric.data.Batch.from_data_list(list_graphs_g)
    # bg = dgl.batch(list_graphs_g)
    # bg_exp = dgl.batch(list_graphs_gexp)
    return bg, list_y


def get_position_encodings(data):
    # Basic preprocessing of the input graph.
    if hasattr(data, "num_nodes"):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = "none"
    if laplacian_norm_type == "none":
        laplacian_norm_type = None
    is_undirected = True
    if is_undirected:
        undir_edge_index = data.edge_index.to(torch.int64)
    else:
        undir_edge_index = to_undirected(data.edge_index)
    # Eigen values and vectors.
    evals, evects = None, None
    pe_types = "LapPE"
    if "LapPE" == pe_types or "EquivStableLapPE" == pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(
                undir_edge_index, normalization=laplacian_norm_type, num_nodes=N
            )
        )
        evals, evects = np.linalg.eigh(L.toarray())

        if "LapPE" == pe_types:
            max_freqs = 16
            eigvec_norm = "L2"
        elif "EquivStableLapPE" == pe_types:
            max_freqs = cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm = cfg.posenc_EquivStableLapPE.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm
        )

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float("nan"))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float("nan")).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = (
            torch.max(EigVecs.abs(), dim=0, keepdim=True)
            .values.clamp_min(eps)
            .expand_as(EigVecs)
        )
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(
            dim=0, keepdim=True
        )
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs
