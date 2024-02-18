from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax

from proteins.models.config import GraphTransformerConfig


def get_indices_to_reduce_from_n3_shape(
    edge_index: torch.Tensor, device: Optional[torch.device] = None
) -> (torch.Tensor, torch.Tensor):
    """
    Function to prepare indices for sparse matrix multiplication on n*n*n.

    This function could be run in advance to prepare and store indices, and use it in every attention layer.

    :param edge_index:
    :return: indices from edge_index[1] on dim 1, indices from edge_index[1] on dim 2
        could be used in sparse matrix construction as following:
        torch.stack(
            [
                indices of nodes on dim 0 (torch.repeat_interleave(nodes, counts**2)),
                edge_index[1, indices from edge_index[1] on dim 1],
                edge_index[1, indices from edge_index[1] on dim 2],
            ]

        but indices from dim 1 and 2 could be also applied directly to edge_features
    """
    nodes, counts = torch.unique(edge_index[0, :], return_counts=True)
    cum_sum = torch.cumsum(counts, dim=0)
    cum_sum = torch.stack(
        [
            torch.hstack(
                [torch.tensor([0], dtype=torch.int64, device=device), cum_sum]
            )[:-1],
            cum_sum,
            counts,
        ]
    )
    eik = torch.cat(
        [
            torch.arange(s, e, dtype=torch.int64, device=device).repeat(r)
            for s, e, r in cum_sum.T
        ]
    )
    eiq = torch.cat(
        [
            torch.arange(s, e, dtype=torch.int64, device=device)
            .repeat((r, 1))
            .transpose(0, 1)
            .reshape(r**2)
            for s, e, r in cum_sum.T
        ]
    )

    return eiq, eik


class MultiHeadGraphAttention(nn.Module):
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
        self.e_heads = config.e_heads
        self.values = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.keys = nn.Linear(config.d_embed, config.d_embed * config.n_heads)
        self.queries = nn.Linear(config.d_embed, config.d_embed * config.n_heads)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.linear = nn.Linear(config.n_heads * config.d_embed, config.d_embed)
        self.linear_dropout = nn.Dropout(config.linear_dropout)

        # edges features
        self.edge_queries = nn.Linear(
            config.d_e_embed, config.d_e_embed * config.e_heads
        )
        self.edge_values = nn.Linear(
            config.d_e_embed, config.d_e_embed * config.e_heads
        )
        self.edge_nodes_keys = nn.Linear(
            config.n_heads, config.d_e_embed * config.e_heads
        )
        #  to do: edge embeddings linear layer for matching size on dif.blocks
        self.edge_attention_dropout = nn.Dropout(config.edge_attention_dropout)
        self.edge_linear = nn.Linear(config.e_heads * config.d_e_embed, config.n_heads)
        self.edge_linear_embedding = nn.Linear(config.n_heads, config.d_e_embed)
        self.edge_linear_dropout = nn.Dropout(config.edge_linear_dropout)

        self.apply(self.init_weights)

    def edge_attention(
        self,
        scaled_qk: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        indices_to_reduce_from_n3_shape: torch.Tensor,
        n_edges: int,
    ) -> torch.Tensor:
        # prepare indices for sparse matrix multiplication on n*n*n
        if indices_to_reduce_from_n3_shape is None:
            eiq, eik = get_indices_to_reduce_from_n3_shape(edge_index)
        else:
            eiq, eik = indices_to_reduce_from_n3_shape

        # do the logic

        eq = self.edge_queries(edge_features)
        ev = self.edge_values(edge_features)
        ek = self.edge_nodes_keys(scaled_qk.transpose(0, 1))

        e_dim = eq.shape[1] // self.e_heads

        # dense:(nodes, n_nodes, n_heads * hidden_size) -> (nodes, n_nodes, n_heads, hidden_size) -> (nodes, n_heads, n_nodes, hidden_size)
        # sparse: (n_edges, n_heads, hidden_size)
        ev = ev.view(-1, self.e_heads, e_dim)
        eq = eq.view(-1, self.e_heads, e_dim)
        ek = ek.view(-1, self.e_heads, e_dim)

        # dense: (n_heads, n_nodes, n_nodes, hidden_size) * (n_heads,  n_nodes, hidden_size, n_nodes)
        #   = (n_heads, n_nodes, n_nodes, n_nodes)
        # sparse: -> (n_edges_for_3n_operations, n_heads)
        scaled_eqk = (eq[eiq, :, :] * ek[eik, :, :]).sum(-1) / torch.sqrt(
            torch.tensor(ek.size(-1))
        )

        # softmax
        scaled_eqk = softmax(
            src=scaled_eqk,
            index=eiq,
            num_nodes=n_edges,
        )
        scaled_eqk = self.edge_attention_dropout(scaled_eqk)

        # matmul
        # dense: (n_heads, n_nodes, n_nodes, n_nodes) * (n_heads, n_nodes, n_nodes, hidden_size)
        #   =  (n_heads, n_nodes, n_nodes, hidden_size)
        # sparse: -> (n_edges, n_heads, hidden_size)
        output = scatter(
            src=scaled_eqk.unsqueeze(-1) * ev[eik, :, :],
            index=eiq,
            dim=0,
            dim_size=n_edges,
            reduce="sum",
        )

        # sparse: -> (n_edges, output_dim)
        output = self.edge_linear(output.view(-1, self.e_heads * e_dim))
        output = self.edge_linear_dropout(output)
        return output

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

        n_nodes = x.shape[0]
        n_edges = edge_index.shape[1]
        n_dim = nq.shape[1] // self.n_heads

        # (n_nodes, n_heads * hidden_size) -> (n_nodes, n_heads, hidden_size) -> (n_heads, n_nodes, hidden_size)
        nv = nv.view(-1, self.n_heads, n_dim).transpose(0, 1)
        nq = nq.view(-1, self.n_heads, n_dim).transpose(0, 1)
        nk = nk.view(-1, self.n_heads, n_dim).transpose(0, 1)

        # dense: (n_heads, n_nodes, hidden_size) * (n_heads, hidden_size, n_nodes) = (n_heads, n_nodes, n_nodes)
        # sparse: -> (n_heads, n_edges)
        scaled_qk = (nq[:, edge_index[0], :] * nk[:, edge_index[1], :]).sum(
            -1
        ) / torch.sqrt(torch.tensor(nk.size(-1)))

        # edges features
        edge_embeddings = self.edge_attention(
            scaled_qk=scaled_qk,
            edge_index=edge_index,
            edge_features=edge_features,
            indices_to_reduce_from_n3_shape=indices_to_reduce_from_n3_shape,
            n_edges=n_edges,
        )

        # softmax
        # print(5, scaled_qk.shape, edge_index[0].shape, n_nodes, edge_embeddings.shape)
        scaled_qk = softmax(
            src=edge_embeddings + scaled_qk.T,
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

        edge_embeddings = self.edge_linear_embedding(edge_embeddings)
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

    model = MultiHeadGraphAttention(
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
