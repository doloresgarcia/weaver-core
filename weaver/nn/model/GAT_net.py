import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from weaver.nn.model.layers.graph import create_graph, create_graph_knn
from weaver.nn.model.layers.gat_layer import GATLayer
from weaver.nn.model.layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self,dev):
        super().__init__()
        num_node_type = 34
        num_edge_type = 4
        hidden_dim = 70
        num_heads = 4
        out_dim = 70
        n_classes = 5
        in_feat_dropout = 0.1
        dropout = 0.0
        n_layers = 9
        self.readout = "mean"
        self.batch_norm = True
        self.residual =True
        self.dropout = 0.0
        self.device = dev
        
   
        self.embedding_h = nn.Linear(num_node_type, hidden_dim*num_heads)
            
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1,
                                    dropout, self.batch_norm, self.residual))
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
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
    
 