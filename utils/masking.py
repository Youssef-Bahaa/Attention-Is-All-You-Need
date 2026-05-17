import torch
import torch.nn as nn

class CausalMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq_length):
        '''
         example for seq_length = 4
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
        '''

        mask = torch.tril(
            torch.ones(seq_length, seq_length)
        )

        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask
