import torch
import sys, os
from model.transformer import Transformer




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 3
MAX_LEN = 128


def load_model_vocabs(checkpoint="best_model.pt"):
    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("tgt_vocab.pkl", "rb") as f:
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
    ).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model, src_vocab, tgt_vocab




def translate(sentence, model, src_vocab, tgt_vocab, max_len=50):
    pass


if __name__ == "__main__":
    model, src_vocab, tgt_vocab = load_model_vocabs()
    sentence = "A dog is running in the park."
    print(f"EN: {sentence}")
    print(f"DE: {translate(sentence, model, src_vocab, tgt_vocab)}")