"""Shared GNN building blocks used by the deep-learning models.

Ported verbatim from OASIS2_Colab.py (lines 1204-1260): `SinusoidalTimeEncoding`,
`GraphConvLayer`, `GATLayer`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous time values."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) or scalar
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GraphConvLayer(nn.Module):
    """Simple graph convolution: H' = σ(A_hat @ H @ W)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, adj):
        # x: (B, N, D), adj: (1, N, N) or (N, N)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        out = torch.bmm(adj.expand(x.size(0), -1, -1), x)
        return F.elu(self.W(out))


class GATLayer(nn.Module):
    """Graph Attention layer (single-head for speed)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        # x: (B, N, D), adj: (1, N, N)
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out)
        # Pairwise attention
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, out)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, out)
        e = self.leaky(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (B, N, N)
        # Mask by adjacency
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        mask = adj.expand(B, -1, -1)
        e = e.masked_fill(mask < 1e-6, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = torch.bmm(alpha, h)
        return F.elu(out)
