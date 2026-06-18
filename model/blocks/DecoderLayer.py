import torch
import torch.nn as nn

from model.attention.multihead_attention import MultiHeadAttention
from model.attention.cross_attention import CrossAttention
from feedforward import FeedForward
from utils.masking import CausalMask

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, decoder_x, encoder_output):
        B, T, _ = decoder_x.shape
        causal_mask = CausalMask.create(T, decoder_x.device)

        x = self.self_attn(decoder_x, mask=causal_mask)
        decoder_x = self.norm1(decoder_x + x)

        x = self.cross_attention(decoder_x, encoder_output)
        decoder_x = self.norm2(decoder_x + x)

        x = self.ffn(decoder_x)
        decoder_x = self.norm3(decoder_x + x)

        return decoder_x