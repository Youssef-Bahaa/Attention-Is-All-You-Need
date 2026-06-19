import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        B, T_q, D = q.shape
        _, T_k, _ = k.shape

        Q = self.query(q)  # (B, T_q, D)
        K = self.key(k)  # (B, T_k, D)
        V = self.value(v)  # (B, T_k, D)

        # Reshape for multi-head split: (B, heads, T, head_dim)
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2))  # (B, heads, T_q, T_k)
        scores = scores / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # (B, heads, T_q, head_dim)

        # Concatenate heads back together
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)

        return self.out(out)