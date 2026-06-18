import torch

class CausalMask:
    @staticmethod
    def create(seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)