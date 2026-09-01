import torch.nn as nn

from model.blocks.encoder import Encoder
from model.blocks.decoder import Decoder
from model.embeddings.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_encoder_layers,
        num_decoder_layers,
        max_len=512,
        dropout = 0.1
    ):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)

        self.pos_encoding = PositionalEncoding(embed_dim, max_len)

        self.encoder = Encoder(
        num_encoder_layers, embed_dim, num_heads, ff_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.decoder = Decoder(
            num_decoder_layers, embed_dim, num_heads, ff_dim
        )

        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        src = self.src_embedding(src)  # (B, S, D)
        tgt = self.tgt_embedding(tgt)  # (B, T, D)

        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        encoder_out = self.encoder(src, mask=src_mask)
        out = self.decoder(encoder_output=encoder_out, x=tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        logits = self.fc_out(out)  # (B, T, vocab_size)

        return logits

    def encode(self, src, src_mask=None):
        src = self.src_embedding(src)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        return self.encoder(src, mask=src_mask)

    def decode(self, tgt, encoder_out, src_mask=None, tgt_mask=None):
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        out = self.decoder(encoder_output=encoder_out, x=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(out)

