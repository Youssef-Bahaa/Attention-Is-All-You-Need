import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model.attention.multihead_attention import MultiHeadAttention
from feedforward import FeedForward
from EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.norm(x)

