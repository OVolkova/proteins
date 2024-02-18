from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from proteins.models.config import GraphTransformerConfig
from proteins.models.graph_sparse.attention3n import (
    MultiHeadGraphAttention,
    get_indices_to_reduce_from_n3_shape,
)
from proteins.models.graph_sparse.attention2n import MultiHeadSimpleGraphAttention


class GraphAttentionLayer(nn.Module):
    """
    Layer block:
    It consists of a layer and a layer normalization.
    Layer normalization is applied before or after the layer.
    """

    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        if config.simple_attention:
            self.attention = MultiHeadSimpleGraphAttention(config)
        else:
            self.attention = MultiHeadGraphAttention(config)

        self.layer_norm_x = nn.LayerNorm(config.d_embed, config.layer_norm_eps)
        self.layer_norm_e = nn.LayerNorm(config.d_e_embed, config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        indices_to_reduce_from_n3_shape: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ):
        out_x, out_e = self.attention(
            self.layer_norm_x(x),
            edge_index,
            self.layer_norm_e(edge_features),
            indices_to_reduce_from_n3_shape,
        )
        out_x = out_x + x
        out_e = out_e + edge_features
        return out_x, out_e


class FeedForward(nn.Module):
    """
    Feed-forward layer
    """

    def __init__(self, d_embed, d_ff, has_bias=True, dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_with_relu",
                        nn.Linear(d_embed, d_ff, bias=has_bias),
                    ),
                    ("relu", nn.GELU()),
                    (
                        "linear",
                        nn.Linear(d_ff, d_embed, bias=has_bias),
                    ),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )
        self.layer_norm = nn.LayerNorm(d_embed, layer_norm_eps)
        self.apply(self.init_weights)

    def forward(self, x):
        return self.feed_forward(self.layer_norm(x)) + x

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            # Initialize the weights of the feed-forward layer
            # linear layer followed by ReLU is initialized with gain calculated for ReLU
            torch.nn.init.xavier_uniform_(
                module.linear_with_relu.weight,
                gain=torch.nn.init.calculate_gain("relu"),
            )
            # output linear layer is initialized with default gain=1.
            torch.nn.init.xavier_uniform_(module.linear.weight)


class ModelBlock(nn.Module):
    """
    Model block: It consists of a number of layers.
    """

    def __init__(self, config: GraphTransformerConfig, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer(config) for _ in range(config.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        indices_to_reduce_from_n3_shape: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ):
        for layer in self.layers:
            x, edge_features = layer(
                x, edge_index, edge_features, indices_to_reduce_from_n3_shape
            )
        return x, edge_features


class GraphTransformerLayer(nn.Module):
    """
    Graph model layer:
    It consists of a self-attention layer, and a feed-forward layer for nodes features and for edge features.
    """

    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.self_attention = GraphAttentionLayer(config)
        self.feed_forward_x = FeedForward(
            config.d_embed,
            config.d_ff,
            config.bias,
            config.ff_dropout,
            config.layer_norm_eps,
        )
        self.feed_forward_e = FeedForward(
            config.d_e_embed,
            config.d_e_ff,
            config.bias,
            config.e_ff_dropout,
            config.layer_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        indices_to_reduce_from_n3_shape: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ):
        x, edge_features = self.self_attention(
            x, edge_index, edge_features, indices_to_reduce_from_n3_shape
        )
        x = self.feed_forward_x(x)
        edge_features = self.feed_forward_e(edge_features)
        return x, edge_features


class GraphTransformer(nn.Module):
    """
    Transformer model for Graphs
    """

    def __init__(
        self,
        config: GraphTransformerConfig,
    ):
        super().__init__()
        self.is_simple_attention = config.simple_attention

        self.node_embedding = nn.Linear(
            config.d_node_in, config.d_embed, bias=config.bias_embed
        )
        self.edge_embedding = nn.Linear(
            config.d_edge_in, config.d_e_embed, bias=config.bias_embed
        )
        self.encoder = ModelBlock(config, GraphTransformerLayer)

        self.layer_norm_modes = nn.LayerNorm(config.d_embed, config.layer_norm_eps)
        self.layer_norm_edges = nn.LayerNorm(config.d_e_embed, config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        device: str = None,
        return_edge_features: bool = False,
    ):
        if self.is_simple_attention:
            indices_to_reduce_from_n3_shape = None
        else:
            indices_to_reduce_from_n3_shape = get_indices_to_reduce_from_n3_shape(
                edge_index, device=device
            )

        x = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_features)
        x, edge_features = self.encoder(
            x, edge_index, edge_features, indices_to_reduce_from_n3_shape
        )
        x = self.layer_norm_modes(x)
        edge_features = self.layer_norm_edges(edge_features)
        return x, edge_features


if __name__ == "__main__":
    config_ = GraphTransformerConfig(simple_attention=False)
    model = GraphTransformer(config_)
    print(model)
