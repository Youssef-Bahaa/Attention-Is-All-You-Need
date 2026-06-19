import torch
import torch.nn as nn

from model.attention.multihead_attention import MultiHeadAttention
from feedforward import FeedForward
from utils.masking import CausalMask

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, decoder_x, encoder_output):
        B, T, _ = decoder_x.shape
        causal_mask = CausalMask.create(T, decoder_x.device)

        # 1. Masked Multi-Head Self-Attention (Q=decoder, K=decoder, V=decoder)
        x = self.self_attn(q=decoder_x, k=decoder_x, v=decoder_x, mask=causal_mask)
        decoder_x = self.norm1(decoder_x + self.dropout(x))

        # 2. Multi-Head Cross-Attention (Q=decoder, K=encoder, V=encoder)
        x = self.cross_attention(q=decoder_x, k=encoder_output, v=encoder_output, mask=None)
        decoder_x = self.norm2(decoder_x + self.dropout(x))

        x = self.ffn(decoder_x)
        decoder_x = self.norm3(decoder_x + self.dropout(x))

        return decoder_x