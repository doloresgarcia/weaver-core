import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""
from weaver.nn.model.layers.graph import create_graph, create_graph_knn, create_graph_knn2, create_graph_expander
from weaver.nn.model.layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from weaver.nn.model.layers.caley_gen import build_cayley_bank

class GINNet(nn.Module):
    
    def __init__(self, dev):
        super().__init__()
        self.device = dev
        num_node_type = 34
        hidden_dim = 80
        n_classes = 5
        dropout = 0.0
        self.n_layers = 10
        n_mlp_layers = 2              # GIN
        learn_eps = True          # GIN
        neighbor_aggr_type = "sum" # GIN
        readout = "mean"                      # this is graph_pooling_type
        batch_norm = True
        residual = True

        cayley_bank = build_cayley_bank()
        self.cayley_bank = cayley_bank

        self.pos_enc = False
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = 34
            self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        
        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        
    def forward(self, points, features, lorentz_vectors, mask):
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
            #g = create_graph_knn(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len, device = self.device)
            g = create_graph_expander(pf_points,pf_features,pf_vectors, mask_, seq_len=seq_len,
                     device = self.device, cayley_bank= self.cayley_bank)
            graphs.append(g)
    
        batched_graph = dgl.batch(graphs)
        g = batched_graph
        ############################## Embeddings #############################################
        h = g.ndata['h']
        e = g.edata['h']

        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h)
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.linears_prediction[i](pooled_h)

        return score_over_layer
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss