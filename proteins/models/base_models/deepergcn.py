import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv


class DeeperGCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.node_encoder = Linear(config.d_node_in, config.d_embed)
        self.edge_encoder = Linear(config.d_edge_in, config.d_embed)

        self.layers = torch.nn.ModuleList()
        for i in range(1, config.n_layers + 1):
            conv = GENConv(
                config.d_embed,
                config.d_embed,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )
            norm = LayerNorm(config.d_embed, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv,
                norm,
                act,
                block="res+",
                dropout=config.attention_dropout,
                ckpt_grad=bool(i % 3),
            )
            self.layers.append(layer)

        self.lin = Linear(config.d_embed, self.d_node_out)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)
