from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax

from proteins.models.config import GraphTransformerConfig


class MultiHeadSimpleGraphAttention(nn.Module):
    """
    Multi-Head Attention Graph Layer
    It is a graph layer that uses multi-head attention
     to compute the new node embeddings and edges embeddings.

    The node embeddings are computed as follows:
    1. compute  scaled_QK = Q*K/sqrt(d_k) for each head
    2. compute edge embeddings with attention for edges features as follows:
        2.0. Q = E, K = scaled_QK, V = E
        2.1. compute attention with softmax as usual
        2.4. apply linear layer and dropout
    3. compute attention using edge embeddings as scaled_QK
    6. apply linear layer and dropout

    The result will be new node embeddings and new edges embeddings.

    TO THINK about:
        - if adj matrix is not symmetrical, would output be different?
    """

    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        assert config.d_embed % config.n_heads == 0
        self.n_heads = config.n_heads
        self.values = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.keys = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.queries = nn.Linear(config.d_embed, config.d_embed * config.n_heads)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.linear = nn.Linear(config.n_heads * config.d_embed, config.d_embed)
        self.linear_dropout = nn.Dropout(config.linear_dropout)

        # edges features
        self.edge_keys = nn.Linear(config.d_e_embed, config.d_embed * config.n_heads)
        self.edge_linear_embedding = nn.Linear(
            config.d_embed * config.n_heads, config.d_e_embed
        )

        self.apply(self.init_weights)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        indices_to_reduce_from_n3_shape: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nq = self.queries(x)
        nv = self.values(x)
        nk = self.keys(x)
        ek = self.edge_keys(edge_features)

        n_nodes = x.shape[0]
        n_edges = edge_index.shape[1]
        n_dim = nq.shape[1] // self.n_heads

        # (n_nodes, n_heads * hidden_size) -> (n_nodes, n_heads, hidden_size) -> (n_heads, n_nodes, hidden_size)
        nv = nv.view(-1, self.n_heads, n_dim).transpose(0, 1)
        nq = nq.view(-1, self.n_heads, n_dim).transpose(0, 1)
        nk = nk.view(-1, self.n_heads, n_dim).transpose(0, 1)
        ek = ek.view(-1, self.n_heads, n_dim).transpose(0, 1)

        # dense: (n_heads, n_nodes, hidden_size) * (n_heads, hidden_size, n_nodes) = (n_heads, n_nodes, n_nodes)
        # sparse: -> (n_heads, n_edges)
        scaled_qk = (nq[:, edge_index[0], :] * (nk[:, edge_index[1], :] + ek)).sum(
            -1
        ) / torch.sqrt(torch.tensor(nk.size(-1)))

        # softmax
        # print(5, scaled_qk.shape, edge_index[0].shape, n_nodes, edge_embeddings.shape)
        scaled_qk = softmax(
            src=scaled_qk.T,
            index=edge_index[0],
            num_nodes=n_nodes,
        )

        # dense:
        # 1. matmul: (n_heads, n_nodes, n_nodes) * (n_heads, n_nodes, hidden_size) = (n_heads, n_nodes, hidden_size)
        # 1. transpose: (n_heads, n_nodes, hidden_size) -> (n_nodes, n_heads, hidden_size)
        node_embeddings = scatter(
            src=scaled_qk.unsqueeze(-1) * nv.transpose(0, 1)[edge_index[1], :, :],
            index=edge_index[0],
            dim=0,
            dim_size=n_nodes,
            reduce="sum",
        )

        # (n_nodes, n_heads, hidden_size) -> (n_nodes, n_heads * hidden_size)
        node_embeddings = node_embeddings.view(n_nodes, self.n_heads * n_dim)
        node_embeddings = self.linear(node_embeddings)
        node_embeddings = self.linear_dropout(node_embeddings)

        edge_embeddings = self.edge_linear_embedding(
            ek.transpose(0, 1).view(n_edges, self.n_heads * n_dim)
        )
        edge_embeddings = self.linear_dropout(edge_embeddings)

        return node_embeddings, edge_embeddings

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def sample_graph(n_dim=4, e_dim=8, n_nodes=10, n_edges_sample=30):
    nodes = torch.randn(n_nodes, n_dim)

    edges = torch.randint(low=0, high=n_nodes, size=(2, n_edges_sample))

    order = torch.sort(edges[1]).indices
    edges = edges[:, order]
    order = torch.sort(edges[0]).indices
    edges = edges[:, order]
    edges = torch.unique(edges, dim=1)
    edges = edges[:, edges[0, :] != edges[1, :]]

    n_edges = edges.shape[1]

    edge_features = torch.randn(n_edges, e_dim)

    return nodes, edges, edge_features


if __name__ == "__main__":
    nodes_, edges_, edge_features_ = sample_graph()

    model = MultiHeadSimpleGraphAttention(
        GraphTransformerConfig(
            bias_embed=True,
            d_node_in=10,
            d_edge_in=20,
            d_node_out=5,
            d_edge_out=7,
            d_embed=4,
            n_heads=4,
            e_heads=2,
            d_e_embed=8,
        )
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)
    new_nodes, new_edges = model(nodes_, edges_, edge_features_)
    print(new_nodes.shape, new_edges.shape)
    # print(nodes_, edge_features_)
    # print(new_nodes, new_edges)
