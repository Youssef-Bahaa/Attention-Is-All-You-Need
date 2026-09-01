import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

from data.tokenizer import build_vocab
from data.dataset import Multi30kDataset
from data.collate import collate_fn
from model.transformer import Transformer

from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import logging
from utils.masking import make_padding_mask
from training.evaluate import evaluate_bleu

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
DROPOUT = 0.2
MAX_LEN = 128
BATCH_SIZE = 128
NUM_EPOCHS = 60
LEARNING_RATE = 1
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
CLIP = 1.0
PAD_IDX = 0
BLEU_EVERY = 5
PATIENCE = 10


def lr_lambda(step):
    step = max(step, 1)
    warmup = 4000
    return (512 ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for src, tgt, tgt_y in loader:
        src, tgt, tgt_y = src.to(DEVICE), tgt.to(DEVICE), tgt_y.to(DEVICE)
        src_mask = make_padding_mask(src, PAD_IDX)
        tgt_mask = make_padding_mask(tgt, PAD_IDX)

        optimizer.zero_grad()
        out = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        out = out.reshape(-1, out.size(-1))
        tgt_y = tgt_y.reshape(-1)

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

        src_mask = make_padding_mask(src, PAD_IDX)
        tgt_mask = make_padding_mask(tgt, PAD_IDX)

        out = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
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
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING)

    best_val = float("inf")
    best_bleu = 0.0
    epochs_no_improve = 0

    logger.info(f"src vocab: {len(src_vocab):,} | tgt vocab: {len(tgt_vocab):,}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        val_loss = evaluate(model, val_loader, criterion)

        bleu_str = ""
        if epoch % BLEU_EVERY == 0 or epoch == NUM_EPOCHS:
            bleu_score, _, _ = evaluate_bleu(model, val_loader, src_vocab, tgt_vocab, DEVICE, max_len=MAX_LEN)
            bleu_str = f" | BLEU {bleu_score:.2f}"
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                torch.save(model.state_dict(), "best_bleu_model.pt")
                bleu_str += " (best BLEU saved)"

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            saved = " (saved)"
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1

        logger.info(f"Epoch {epoch:02d} | train {train_loss:.3f} | val {val_loss:.3f}{bleu_str}{saved}")
        print(f"Epoch {epoch:02d} | train {train_loss:.3f} | val {val_loss:.3f}{bleu_str}{saved}")

        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            print(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Best val loss: {best_val:.3f} | Best BLEU: {best_bleu:.2f}")
    print(f"Best val loss: {best_val:.3f} | Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()