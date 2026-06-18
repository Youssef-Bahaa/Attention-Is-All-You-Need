from model.attention.cross_attention import CrossAttention
from utils.masking import CausalMask
from model.attention.multihead_attention import MultiHeadAttention
from feedforward import FeedForward
import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.self_attn  = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, encoder_output, decoder_x):
        x = self.self_attn(decoder_x)
        decoder_x = self.norm1(
            decoder_x + x
        )

        x = self.cross_attention(decoder_x, encoder_output)

        decoder_x = self.norm2(x + decoder_x)

        x = self.ffn(decoder_x)

        decoder_x = self.norm3(x + decoder_x)

        return decoder_x



