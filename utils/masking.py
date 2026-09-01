import torch

class CausalMask:
    @staticmethod
    def create(seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)


def make_padding_mask(seq, pad_idx=0):
    # seq: (B, S)
    # returns (B, 1, 1, S) — True where token is real, False where pad
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)



def make_src_mask(src, pad_idx=0):
    return make_padding_mask(src, pad_idx)


def make_tgt_mask(tgt, pad_idx=0):
    pad_mask = make_padding_mask(tgt, pad_idx)               # (B,1,1,T)
    seq_len = tgt.size(1)
    causal = CausalMask.create(seq_len, tgt.device).bool()   # (1,1,T,T)
    return pad_mask & causal