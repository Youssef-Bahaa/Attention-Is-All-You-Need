import torch
import torch.nn as nn
from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_output, x, src_mask=None, tgt_mask=None):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)

        return self.norm(x)