import torch
from torch.utils.data import Dataset

class Multi30kDataset(Dataset):
    def __init__(self, hf_split, src_tokenizer, tgt_tokenizer, max_len=128):
        self.data = hf_split
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.src_tok.encode(self.data[idx]["en"])[:self.max_len]
        tgt = self.tgt_tok.encode(self.data[idx]["de"])[:self.max_len]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)