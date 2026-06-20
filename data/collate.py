import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    """
    Pads a batch of (src, tgt) pairs.
    Returns:
        src:   (B, S)  source token ids
        tgt:   (B, T)  decoder input  — tgt without last token
        tgt_y: (B, T)  decoder labels — tgt without first <sos>
    """

    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)

    return src_padded, tgt_padded[:, :-1], tgt_padded[:, 1:]