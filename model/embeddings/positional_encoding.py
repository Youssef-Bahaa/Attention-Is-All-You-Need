import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, len):
        super().__init__()

        position = torch.arange(
            0,len,
            dtype=torch.float32
        ).unsqueeze(1)

        pe = torch.zeros(len, d_model)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        '''
        x shape:
        (batch, len, d_model)
        '''

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x