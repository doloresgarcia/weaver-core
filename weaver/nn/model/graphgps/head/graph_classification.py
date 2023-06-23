import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head
import torch_geometric.graphgym.register as register


@register_head("graph_class")
class GNNClassHead(nn.Module):
    """
    GNN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNClassHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(
                dim_in,
                dim_out,
                cfg.gnn.layers_post_mp,
                has_act=False,
                has_bias=True,
                cfg=cfg,
            )
        )
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
