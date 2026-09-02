import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
from model.transformer import Transformer
from utils.masking import make_src_mask, make_tgt_mask

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
DROPOUT = 0.2
MAX_LEN = 128


def load_model_vocabs(checkpoint=None):
    if checkpoint is None:
        checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "best_bleu_model.pt")

    with open(os.path.join(PROJECT_ROOT, "checkpoints", "src_vocab.pkl"), "rb") as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, "checkpoints", "tgt_vocab.pkl"), "rb") as f:
        tgt_vocab = pickle.load(f)

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

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model, src_vocab, tgt_vocab


@torch.no_grad()
def translate(sentence, model, src_vocab, tgt_vocab, max_len=50):
    src_ids = src_vocab.encode(sentence)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
    src_mask = make_src_mask(src_tensor).to(DEVICE)

    encoder_out = model.encode(src_tensor, src_mask)

    decoded = [tgt_vocab.SOS_IDX]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(decoded).unsqueeze(0).to(DEVICE)
        tgt_mask = make_tgt_mask(tgt_tensor).to(DEVICE)

        out = model.decode(tgt_tensor, encoder_out, src_mask, tgt_mask)

        next_id = out[0, -1, :].argmax(-1).item()
        if next_id == tgt_vocab.EOS_IDX:
            break
        decoded.append(next_id)

    return tgt_vocab.decode(decoded)


if __name__ == "__main__":
    model, src_vocab, tgt_vocab = load_model_vocabs()
    sentence = "a dog is running at the park"
    print(f"EN: {sentence}")
    print(f"DE: {translate(sentence, model, src_vocab, tgt_vocab)}")