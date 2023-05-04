import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from weaver.nn.model.layers.graph import create_graph, create_graph_knn
from weaver.nn.model.layers.gmm_layer import GMMLayer
from weaver.nn.model.layers.mlp_readout_layer import MLPReadout

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

class MoNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        num_node_type = 34
        hidden_dim = 70
        out_dim = 70
        n_classes = 5
        kernel = 3                      # for MoNet
        dim = 2                # for MoNet
        dropout = 0.0
        n_layers = 4
        self.readout = "mean",     
        batch_norm = True
        residual = True 
        self.pos_enc = False
        self.device = dev
        self.embedding_h = nn.Linear(num_node_type, hidden_dim)
        
        aggr_type = "mean"                                    # default for MoNet
        
        
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(n_layers-1):
            self.layers.append(GMMLayer(hidden_dim, hidden_dim, dim, kernel, aggr_type,
                                        dropout, batch_norm, residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
            
        # Output layer
        self.layers.append(GMMLayer(hidden_dim, out_dim, dim, kernel, aggr_type,
                                    dropout, batch_norm, residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes) 
        

    def forward(self, points, features, lorentz_vectors, mask):
        ####################################### Convert to graphs ##############################
        batch_size = points.shape[0]

        graphs = []
        for i in range(0, batch_size):
            pf_points = points[i]
            pf_features = features[i]
            pf_vectors = lorentz_vectors[i]
            mask_ = mask[i]
            seq_len = np.int(torch.sum(mask_*1))
            if seq_len <5:
                seq_len = 5
            g = create_graph_knn(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len, device = self.device)
            graphs.append(g)
    
    
        batched_graph = dgl.batch(graphs)
        g = batched_graph
        ############################## Embeddings #############################################
        h = g.ndata['h']
        e = g.edata['h']

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h)
        
        # computing the 'pseudo' named tensor which depends on node degrees
        g.ndata['deg'] = g.in_degrees()
        g.apply_edges(self.compute_pseudo)
        pseudo = g.edata['pseudo'].to(self.device).float()
        
        for i in range(len(self.layers)):
            h = self.layers[i](g, h, self.pseudo_proj[i](pseudo))
        g.ndata['h'] = h
            
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        output = self.MLP_layer(hg)
        return output
    
    def compute_pseudo(self, edges):
        # compute pseudo edge features for MoNet
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        srcs = 1/torch.sqrt(edges.src['deg'].float()+1)
        dsts = 1/torch.sqrt(edges.dst['deg'].float()+1)
        pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)
        return {'pseudo': pseudo}

    