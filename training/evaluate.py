import torch
import sacrebleu
from utils.masking import make_src_mask, make_tgt_mask
import pickle
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset
from data.dataset import Multi30kDataset
from data.collate import collate_fn
from model.transformer import Transformer

@torch.no_grad()
def greedy_decode_batch(model, src, src_mask, tgt_vocab, max_len, device):
    sos_idx, eos_idx = tgt_vocab.SOS_IDX, tgt_vocab.EOS_IDX
    batch_size = src.size(0)
    encoder_out = model.encode(src, src_mask)

    ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)
        out = model.decode(ys, encoder_out, src_mask, tgt_mask)
        next_tok = out[:, -1, :].argmax(dim=-1)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= (next_tok == eos_idx)
        if finished.all():
            break
    return ys




def ids_to_text(ids, vocab, eos_idx):
    tokens = []
    for i in ids:
        if i == eos_idx:
            break
        tokens.append(vocab.idx2token[i])
    return " ".join(t for t in tokens if t not in ("<pad>", "<sos>"))


def evaluate_bleu(model, dataloader, src_vocab, tgt_vocab, device, max_len=128):
    model.eval()
    hyps, refs = [], []
    eos_idx = tgt_vocab.EOS_IDX

    for src_b, tgt, tgt_y in dataloader:
        src_b = src_b.to(device)
        full_tgt = torch.cat([tgt[:, :1], tgt_y], dim=1)
        src_mask = make_src_mask(src_b).to(device)
        pred_ids = greedy_decode_batch(model, src_b, src_mask, tgt_vocab, max_len, device)

        for pred_row, tgt_row in zip(pred_ids, full_tgt):
            hyps.append(ids_to_text(pred_row.tolist(), tgt_vocab, eos_idx))
            refs.append(ids_to_text(tgt_row.tolist(), tgt_vocab, eos_idx))

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score, hyps, refs


def build_eval_loader(split="test", batch_size=128, max_len=128, pad_idx=0):
    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    ds = load_dataset("bentrevett/multi30k")
    _collate = partial(collate_fn, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
    loader = DataLoader(
        Multi30kDataset(ds[split], src_vocab, tgt_vocab, max_len),
        batch_size=batch_size, shuffle=False, collate_fn=_collate,
    )
    return loader, src_vocab, tgt_vocab


def load_model(checkpoint, src_vocab, tgt_vocab, device,
                embed_dim=512, num_heads=8, ff_dim=2048, num_layers=6, max_len=128):
    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
        num_encoder_layers=num_layers, num_decoder_layers=num_layers, max_len=max_len,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model