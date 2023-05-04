def create_graph(pf_points,pf_features,pf_vectors, mask, seq_len, device):
    #print(example[0].keys())
    pf_points = pf_points[:,0:seq_len]
    pf_features = pf_features[:,0:seq_len]
    pf_vectors = pf_vectors[:,0:seq_len]
    coord_shift = (mask == 0)[:,0:seq_len] * 1e9
    pf_points = pf_points + coord_shift
    pf_points = torch.permute(torch.tensor(pf_points),(1,0))
    pf_features = torch.permute(torch.tensor(pf_features),(1,0))
    pf_vectors = torch.permute(torch.tensor(pf_vectors),(1,0))
    
    #x = torch.cat((pf_points,pf_features,pf_vectors), dim=1)
    x = pf_features
    edge_index, edge_attr = create_edges_attributes(pf_vectors, seq_len=seq_len)
    #print(x.shape, edge_index.shape, edge_attr.shape)
    #print(edge_index[0],edge_index[1])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pf_points)
    #g = dgl.graph((edge_index[0], edge_index[1]))
    #g = g.to('cuda')
    #g.ndata['h'] = x.to('cuda')
    #g.edata['h'] = edge_attr.to('cuda')
    return data.to(device)

def create_edges_attributes2(pf_vectors, pf_features, seq_len):


    # this one includes also the pf_feature differences 
    pairwise_lv_fts_p = partial(pairwise_lv_fts)
    i, j = torch.tril_indices(seq_len, seq_len, offset=-1)
    x_interactions = pf_vectors
    x_interactions2 = pf_features

    x_interactions = torch.reshape(x_interactions,[seq_len,1,4])
    x_interactiox_interactions2ns = torch.reshape(x_interactions2,[seq_len,1,4])

    x_interactions = x_interactions.repeat( 1,seq_len,1)
    x_interactions2 = x_interactions2.repeat( 1,seq_len,1)

    xi = x_interactions[i, j,:]  # (batch, dim, seq_len*(seq_len+1)/2)
    xj = x_interactions[j, i,:]

    xi2 = x_interactions2[i, j,:]  # (batch, dim, seq_len*(seq_len+1)/2)
    xj2 = x_interactions2[j, i,:]

    x_interactions_m = pairwise_lv_fts_p(xi, xj)
    x_interactions_m2 = xi2 - xj2 
    #edges = torch.zeros(seq_len,seq_len,4)
    #edges[i,j,:] = x_interactions_m
    #edges[j,i,:] = x_interactions_m
    edge_index = torch.zeros(2,len(i))
    edge_index[0,:] = i.long()
    edge_index[1,:] = j.long()
    edge_attr = torch.cat((x_interactions_m,x_interactions_m2), dim = 1)
    return edge_index.long(), edge_attr


def create_edges_attributes(pf_vectors, seq_len):
    pairwise_lv_fts_p = partial(pairwise_lv_fts)
    i, j = torch.tril_indices(seq_len, seq_len, offset=-1)
    x_interactions = pf_vectors
    x_interactions = torch.reshape(x_interactions,[seq_len,1,4])
    x_interactions = x_interactions.repeat( 1,seq_len,1)
    xi = x_interactions[i, j,:]  # (batch, dim, seq_len*(seq_len+1)/2)
    xj = x_interactions[j, i,:]
    x_interactions_m = pairwise_lv_fts_p(xi, xj)
    #edges = torch.zeros(seq_len,seq_len,4)
    #edges[i,j,:] = x_interactions_m
    #edges[j,i,:] = x_interactions_m
    edge_index = torch.zeros(2,len(i))
    edge_index[0,:] = i.long()
    edge_index[1,:] = j.long()
    edge_attr = x_interactions_m
    return edge_index.long(), edge_attr

def create_edges_attributes_fixedmomentum(pf_vectors, pf_features, seq_len):
    pf_vectors[:,1] = pf_vectors[:,0]*torch.sin(pf_features[:,1])*torch.cos(pf_features[:,2]) # TODO check it makes sense to use the energy
    pf_vectors[:,2] = pf_vectors[:,0]*torch.sin(pf_features[:,1])*torch.sin(pf_features[:,2])
    pf_vectors[:,3] = pf_vectors[:,0]*torch.cos(pf_features[:,2])
    # this one includes also the pf_feature differences 
    pairwise_lv_fts_p = partial(pairwise_lv_fts)
    i, j = torch.tril_indices(seq_len, seq_len, offset=-1)
    x_interactions = pf_vectors
    x_interactions2 = pf_features

    x_interactions = torch.reshape(x_interactions,[seq_len,1,4])
    x_interactiox_interactions2ns = torch.reshape(x_interactions2,[seq_len,1,4])

    x_interactions = x_interactions.repeat( 1,seq_len,1)
    x_interactions2 = x_interactions2.repeat( 1,seq_len,1)

    xi = x_interactions[i, j,:]  # (batch, dim, seq_len*(seq_len+1)/2)
    xj = x_interactions[j, i,:]

    xi2 = x_interactions2[i, j,:]  # (batch, dim, seq_len*(seq_len+1)/2)
    xj2 = x_interactions2[j, i,:]

    x_interactions_m = pairwise_lv_fts_p(xi, xj)
    x_interactions_m2 = xi2 - xj2 
    #edges = torch.zeros(seq_len,seq_len,4)
    #edges[i,j,:] = x_interactions_m
    #edges[j,i,:] = x_interactions_m
    edge_index = torch.zeros(2,len(i))
    edge_index[0,:] = i.long()
    edge_index[1,:] = j.long()
    edge_attr = torch.cat((x_interactions_m,x_interactions_m2), dim = 1)
    return edge_index.long(), edge_attr