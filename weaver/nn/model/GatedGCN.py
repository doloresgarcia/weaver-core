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
from weaver.nn.model.layers.graph import create_graph, create_graph_knn, create_graph_knn2

class GatedGCNNet2(nn.Module):
    def __init__(self, dev):
        super().__init__()
        num_node_type = 34
        num_edge_type = 4
        hidden_dim = 70 #net_params['hidden_dim']
        out_dim = 70 #net_params['out_dim']
        n_classes = 5 #net_params['n_classes']
        in_feat_dropout = 0.0 #net_params['in_feat_dropout']
        dropout = 0.0 #net_params['dropout']
        n_layers = 4 #net_params['L']
        self.readout = "mean" #net_params['readout']
        self.batch_norm = True #net_params['batch_norm']
        self.residual = False  #net_params['residual']
        self.edge_feat = True #net_params['edge_feat']
        self.device = dev
        self.pos_enc = False #net_params['pos_enc']
        #if self.pos_enc:
        #    pos_enc_dim = net_params['pos_enc_dim']
        #    self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        #else:
        in_dim = 34
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.embedding_e = nn.Linear(num_edge_type, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
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
        
        #print(h.device, self.embedding_h.weight.device)
        h = self.embedding_h(h)
        #if self.pos_enc:
            #h_pos_enc = self.embedding_pos_enc(pos_enc) 
            #h = h + h_pos_enc

        #h = self.in_feat_dropout(h)
        e = self.embedding_e(e)   
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "energy_weight":
            hg = dgl.mean_nodes(g,'h','w')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        
        output = self.MLP_layer(hg)
        return output

class GatedGCNNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.device = dev
        num_node_type = 34
        num_edge_type = 4
        hidden_dim = 70 #net_params['hidden_dim']
        out_dim = 70 #net_params['out_dim']
        n_classes = 5 #net_params['n_classes']
        in_feat_dropout = 0.0 #net_params['in_feat_dropout']
        dropout = 0.0 #net_params['dropout']
        n_layers = 4 #net_params['L']
        self.readout = "sum" #net_params['readout']
        self.batch_norm = True #net_params['batch_norm']
        self.residual = False  #net_params['residual']
        self.edge_feat = True #net_params['edge_feat']
 
        self.pos_enc = False #net_params['pos_enc']
        #if self.pos_enc:
        #    pos_enc_dim = net_params['pos_enc_dim']
        #    self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        #else:
        in_dim = 34
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.embedding_e = nn.Linear(num_edge_type, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
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
            g = create_graph_knn2(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len, device = self.device)
            graphs.append(g)
    
    
        batched_graph = dgl.batch(graphs)
        print(batched_graph.num_nodes())
        g = batched_graph
        ############################## Embeddings #############################################
        h = g.ndata['h']
        e = g.edata['h']
        #print(h.device, self.embedding_h.weight.device)
        h = self.embedding_h(h)
        #if self.pos_enc:
            #h_pos_enc = self.embedding_pos_enc(pos_enc) 
            #h = h + h_pos_enc

        #h = self.in_feat_dropout(h)
        e = self.embedding_e(e.to('cuda'))   
        # convnets
        for conv in self.layers:
            h, e = conv(g.to('cuda'), h.to('cuda'), e.to('cuda'))
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
        



"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        #h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj 
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention       
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)




class GatedGCNNet3(nn.Module):
    def __init__(self, dev):
        super().__init__()
        num_node_type = 34
        num_edge_type = 4
        hidden_dim = 128 #net_params['hidden_dim']
        out_dim = 128 #net_params['out_dim']
        n_classes = 5 #net_params['n_classes']
        in_feat_dropout = 0.1 #net_params['in_feat_dropout']
        dropout = 0.0 #net_params['dropout']
        n_layers = 4*3 #net_params['L']
        self.readout = "mean" #net_params['readout']
        self.batch_norm = True #net_params['batch_norm']
        self.residual = True  #net_params['residual']
        self.edge_feat = True #net_params['edge_feat']
        self.device = dev
        self.pos_enc = False #net_params['pos_enc']
        #if self.pos_enc:
        #    pos_enc_dim = net_params['pos_enc_dim']
        #    self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        #else:
        in_dim = 34
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.embedding_e = nn.Linear(num_edge_type, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
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
        print(batched_graph.num_nodes())
        g = batched_graph
        ############################## Embeddings #############################################
        h = g.ndata['h']
        e = g.edata['h']
        #print(h.device, self.embedding_h.weight.device)
        h = self.embedding_h(h)
        #if self.pos_enc:
            #h_pos_enc = self.embedding_pos_enc(pos_enc) 
            #h = h + h_pos_enc

        #h = self.in_feat_dropout(h)
        e = self.embedding_e(e)   
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
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