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
import pickle
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hyperparameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
DROPOUT = 0.1
MAX_LEN = 128
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
CLIP = 1.0
PAD_IDX = 0



def lr_lambda(step):
    step = max(step, 1)
    warmup = 4000
    return (512 ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)



def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for src, tgt, tgt_y in loader:
        src, tgt, tgt_y = src.to(DEVICE), tgt.to(DEVICE), tgt_y.to(DEVICE)

        optimizer.zero_grad()
        out = model(src, tgt)  # (B, T, tgt_vocab)
        out = out.reshape(-1, out.size(-1))  # (B*T, tgt_vocab)
        tgt_y = tgt_y.reshape(-1)  # (B*T,)

        loss = criterion(out, tgt_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)



def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for src, tgt, tgt_y in loader:
        src, tgt, tgt_y = src.to(DEVICE), tgt.to(DEVICE), tgt_y.to(DEVICE)
        out = model(src, tgt)
        out = out.reshape(-1, out.size(-1))
        tgt_y = tgt_y.reshape(-1)
        total_loss += criterion(out, tgt_y).item()
    return total_loss / len(loader)


def main():
    ds = load_dataset('bentrevett/multi30k')
    src_vocab, tgt_vocab = build_vocab(ds)

    with open("src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open("tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)

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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val = float("inf")

    logger.info(f"src vocab: {len(src_vocab):,} | tgt vocab: {len(tgt_vocab):,}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        val_loss = evaluate(model, val_loader, criterion)

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            saved = "saved"
            logger.info(f"Epoch {epoch:02d} | train {train_loss:.3f} | val {val_loss:.3f}{saved}")
        else:
            logger.info(f"Epoch {epoch:02d} | train {train_loss:.3f} | val {val_loss:.3f}")

        print(f"Epoch {epoch:02d} | train {train_loss:.3f} | val {val_loss:.3f}{saved}")

if __name__ == "__main__":
    main()

