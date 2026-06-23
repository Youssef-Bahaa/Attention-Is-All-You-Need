import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model.attention.multihead_attention import MultiHeadAttention
from feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_out = self.attn(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ff(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
