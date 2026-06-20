from functools import partial

from data.tokenizer import build_vocab
from data.dataset import Multi30kDataset
from data.collate import collate_fn
from model.transformer import Transformer

from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os

# Hyperparameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 3
DROPOUT = 0.1
MAX_LEN = 128
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
PAD_IDX = 0



def train_epoch(model, loader, optimizer, criterion):
    pass


def evaluate(model, loader, criterion):
    pass


def main():
    ds = load_dataset('bentrevett/multi30k')
    src_vocab, tgt_vocab = build_vocab(ds)

    print(f"src vocab: {len(src_vocab):,}  |  tgt vocab: {len(tgt_vocab):,}")

    _collate = partial(collate_fn, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)

    train_loader = DataLoader(Multi30kDataset(ds["train"], src_vocab, tgt_vocab, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True,  collate_fn=_collate)

    val_loader = DataLoader(
        Multi30kDataset(ds["validation"], src_vocab, tgt_vocab, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate
    )



    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            saved = "saved"

        print(f"Epoch {epoch:02d} | train {train_loss:.3f} | val {val_loss:.3f}{saved}")



