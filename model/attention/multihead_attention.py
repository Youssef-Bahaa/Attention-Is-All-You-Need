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
        self.key   = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        B, T, D = x.shape

        Q = self.query(x)   # (B, T, D)
        K = self.key(x)     # (B, T, D)
        V = self.value(x)   # (B, T, D)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, T, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, T, head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, T, head_dim)


        scores = torch.matmul(Q, K.transpose(-1, -2)) # (B, heads, T, T)
        scores = scores / (self.head_dim ** 0.5)

        mask = self.mask(T).to(x.device)   # (1,1,T,T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out(out)