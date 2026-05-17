import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key   = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        self.scale = embed_dim ** 0.5

    def forward(self, input, output):

        Q = self.query(output) # [30, 128]
        K = self.key(input)    # [20, 128]
        V = self.value(input)  # [20, 128]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim = -1)
        attended = torch.matmul(weights, V)

        return attended