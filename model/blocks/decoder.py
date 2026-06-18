import torch
import torch.nn as nn
from DecoderLayer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_output, x):

        for layer in self.layers:
            x = layer(x, encoder_output)

        return self.norm(x)