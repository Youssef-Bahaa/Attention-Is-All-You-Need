import torch
import torch.nn as nn

from blocks.encoder import Encoder
from blocks.decoder import Decoder
from embeddings.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_encoder_layers,
        num_decoder_layers,
        max_len=512,
        drop_out = 0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoding = PositionalEncoding(embed_dim, max_len)

        self.encoder = Encoder(
        num_encoder_layers, embed_dim, num_heads, ff_dim
        )

        self.dropout = nn.Dropout(drop_out)

        self.decoder = Decoder(
            num_decoder_layers, embed_dim, num_heads, ff_dim
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):

        src = self.token_embedding(src)  # (B, S, D)
        tgt = self.token_embedding(tgt)  # (B, T, D)

        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        encoder_out = self.encoder(src)


        out = self.decoder(
            encoder_output=encoder_out,
            x=tgt,
        )

        logits = self.fc_out(out)  # (B, T, vocab_size)

        return logits



