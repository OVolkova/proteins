import torch
import torch.nn as nn


class FeaturesPredictor(nn.Module):
    """
    1 layer decoder model for features from embeddings
    """

    def __init__(self, d_embed, d_out, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_embed, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.linear(self.dropout(x))


class LinkPredictor(nn.Module):
    """
    1 layer decoder model for link prediction from node embeddings
    """

    def __init__(self, d_embed, d_out, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_embed, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        src, dst = x[edge_index[0]], x[edge_index[1]]
        return self.linear(self.dropout(src * dst))
