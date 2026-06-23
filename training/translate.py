import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import sys, os, pickle
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
    src_ids = src_vocab.encode(sentence)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        src_emb = model.dropout(model.pos_encoding(model.src_embedding(src_tensor)))
        enc_out = model.encoder(src_emb)

        decoded = [tgt_vocab.SOS_IDX]
        with torch.no_grad():
            for _ in range(max_len):
                tgt_tensor = torch.tensor(decoded).unsqueeze(0).to(DEVICE)
                tgt_emb = model.dropout(model.pos_encoding(model.tgt_embedding(tgt_tensor)))

                out = model.decoder(
                    encoder_output=enc_out,
                    x=tgt_emb,
                    src_mask=None,
                    tgt_mask=None,
                )

                next_id = model.fc_out(out)[0, -1, :].argmax(-1).item()
                if next_id == tgt_vocab.EOS_IDX:
                    break
                decoded.append(next_id)

    # 4. Decode ids -> string (Vocab.decode skips <sos>/<eos>/<pad>)
    return tgt_vocab.decode(decoded)



if __name__ == "__main__":
    model, src_vocab, tgt_vocab = load_model_vocabs()
    sentence = "A dog is running in the park."
    print(f"EN: {sentence}")
    print(f"DE: {translate(sentence, model, src_vocab, tgt_vocab)}")