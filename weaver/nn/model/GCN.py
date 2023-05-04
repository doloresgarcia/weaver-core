import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as np
import dgl
import dgl.function as fn
import sys
from functools import partial
import os.path as osp
import time
import numpy as np
from weaver.utils.interactions import pairwise_lv_fts
"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from weaver.nn.model.layers.gcn_layer import GCNLayer
from weaver.nn.model.layers.mlp_readout_layer import MLPReadout
from weaver.nn.model.layers.graph import create_graph, create_graph_knn, create_graph_knn2



class GCNNe_close_to_ParticleNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        num_node_type = 34
        num_edge_type = 4
        hidden_dim1 = 64 #net_params['hidden_dim']
        hidden_dim2 = hidden_dim1*2
        hidden_dim3 = hidden_dim2*2
        out_dim = hidden_dim3 #net_params['out_dim']
        n_classes = 5 #net_params['n_classes']
        in_feat_dropout = 0.1 #net_params['in_feat_dropout']
        dropout = 0.0 #net_params['dropout']
        n_layers = 4 #net_params['L']
        self.device = dev
        self.readout = "mean" #net_params['readout']
        self.batch_norm = False #net_params['batch_norm']
        self.residual = True  #net_params['residual']
        self.edge_feat = True #net_params['edge_feat']
        self.pos_enc = False #net_params['pos_enc']ç
        self.activation = False
        
        in_dim = num_node_type
        #self.embedding_h = nn.Linear(in_dim, hidden_dim1)
        
                
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        #self.embedding_h = nn.Embedding(num_node_type, hidden_dim)
        self.bn_fts = nn.BatchNorm1d(in_dim)
                
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(in_dim))

                
        self.acts = nn.ModuleList()
        self.acts.append(nn.ReLU())

        self.layers = nn.ModuleList([GCNLayer(in_dim, hidden_dim1, self.activation,
                                            dropout,  self.batch_norm, self.residual)])
        #self.layers = nn.ModuleList([GCNLayer(hidden_dim1, hidden_dim1, F.relu,
        #                                      dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        for i in range(n_layers-1):
            self.bns.append(nn.BatchNorm1d(hidden_dim1))
            self.acts.append(nn.ReLU())
            self.layers.append(GCNLayer(hidden_dim1, hidden_dim1, self.activation,
                                    dropout, self.batch_norm, self.residual))
        
        self.bns.append(nn.BatchNorm1d(hidden_dim1))
        self.acts.append(nn.ReLU())
        self.layers.append(GCNLayer(hidden_dim1, hidden_dim2, self.activation,
                                    dropout, self.batch_norm, self.residual))
        for i in range(n_layers-1):
            self.bns.append(nn.BatchNorm1d(hidden_dim2))
            self.acts.append(nn.ReLU())
            self.layers.append(GCNLayer(hidden_dim2, hidden_dim2,self.activation,
                                    dropout, self.batch_norm, self.residual))

        self.bns.append(nn.BatchNorm1d(hidden_dim2))
        self.acts.append(nn.ReLU())
        self.layers.append(GCNLayer(hidden_dim2, hidden_dim3, self.activation,
                                    dropout, self.batch_norm, self.residual))
                    
        for i in range(n_layers-1):
            self.bns.append(nn.BatchNorm1d(hidden_dim3))
            self.acts.append(nn.ReLU())
            self.layers.append(GCNLayer(hidden_dim3, hidden_dim3, self.activation,
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
            seq_len = torch.sum(mask_*1).cpu().numpy().astype(int)
            if seq_len <5:
                seq_len = 5
            g = create_graph_knn(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len, device = self.device)
            graphs.append(g)
    
    
        batched_graph = dgl.batch(graphs)
        g = batched_graph
        ############################## Embeddings #############################################
        h = g.ndata['h']
        e = g.edata['h']
        #if self.pos_enc:
        #    h = self.embedding_pos_enc(pos_enc) 
        #else:
        #    h = self.embedding_h(h)
        #h = self.in_feat_dropout(h)
        h = self.bn_fts(h)
        for conv, bn, act in zip(self.layers, self.bns, self.acts):
            h = bn(h)
            h = act(h)
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
    

class GCNNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        num_node_type = 34
        num_edge_type = 4
        hidden_dim1 = 64 #net_params['hidden_dim']
        hidden_dim2 = hidden_dim1*2
        hidden_dim3 = hidden_dim2*2
        out_dim = hidden_dim3 #net_params['out_dim']
        n_classes = 5 #net_params['n_classes']
        in_feat_dropout = 0.1 #net_params['in_feat_dropout']
        dropout = 0.0 #net_params['dropout']
        n_layers = 4 #net_params['L']
        self.device = dev
        self.readout = "mean" #net_params['readout']
        self.batch_norm = True #net_params['batch_norm']
        self.residual = True  #net_params['residual']
        self.edge_feat = True #net_params['edge_feat']
        self.pos_enc = False #net_params['pos_enc']ç
        
        in_dim = num_node_type
        #self.embedding_h = nn.Linear(in_dim, hidden_dim1)
        
                
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        #self.embedding_h = nn.Embedding(num_node_type, hidden_dim)
        self.bn_fts = nn.BatchNorm1d(in_dim)
        self.layers = nn.ModuleList([GCNLayer(in_dim, hidden_dim1, F.relu,
                                              dropout, self.batch_norm, self.residual)])
        #self.layers = nn.ModuleList([GCNLayer(hidden_dim1, hidden_dim1, F.relu,
        #                                      dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        for i in range(n_layers-1):
            self.layers.append(GCNLayer(hidden_dim1, hidden_dim1, F.relu,
                                    dropout, self.batch_norm, self.residual))
        
        self.layers.append(GCNLayer(hidden_dim1, hidden_dim2, F.relu,
                                    dropout, self.batch_norm, self.residual))
        for i in range(n_layers-1):
            self.layers.append(GCNLayer(hidden_dim2, hidden_dim2, F.relu,
                                    dropout, self.batch_norm, self.residual))

        self.layers.append(GCNLayer(hidden_dim2, hidden_dim3, F.relu,
                                    dropout, self.batch_norm, self.residual))
        for i in range(n_layers-1):
            self.layers.append(GCNLayer(hidden_dim3, hidden_dim3, F.relu,
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
            seq_len = torch.sum(mask_*1).cpu().numpy().astype(int)
            if seq_len <5:
                seq_len = 5
            g = create_graph_knn(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len, device = self.device)
            graphs.append(g)
    
    
        batched_graph = dgl.batch(graphs)
        g = batched_graph
        ############################## Embeddings #############################################
        h = g.ndata['h']
        e = g.edata['h']
        #if self.pos_enc:
        #    h = self.embedding_pos_enc(pos_enc) 
        #else:
        #    h = self.embedding_h(h)
        #h = self.in_feat_dropout(h)
        h = self.bn_fts(h)

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
    
