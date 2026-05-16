import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key   = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)


    def forward(self, x):
        Q = self.query(x)   # (B, T, D)
        K = self.key(x)     # (B, T, D)
        V = self.value(x)   # (B, T, D)

        scores = torch.matmul(Q, K.transpose(1,2)) # (B, T, T)
        scores = scores / (self.embed_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return out

