from collections import Counter

class Vocab:
    def __init__(self, counter, min_freq=2):
        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.SOS = '<sos>'
        self.EOS = '<eos>'


        self.SPECIALS = [self.PAD, self.UNK, self.SOS, self.EOS]

        self.token2idx = {tok : i for i,tok in enumerate(self.SPECIALS)}
        self.idx2token = list(self.SPECIALS)

        for token, freq in counter.items():
            if token not in self.token2idx and freq >= min_freq:
                self.token2idx[token] = len(self.idx2token)
                self.idx2token.append(token)

        self.PAD_IDX = self.token2idx[self.PAD]
        self.UNK_IDX = self.token2idx[self.UNK]
        self.SOS_IDX = self.token2idx[self.SOS]
        self.EOS_IDX = self.token2idx[self.EOS]


    def __len__(self):
        return len(self.token2idx)

    def encode(self, sentence: str):
        tokens = [self.SOS] + sentence.lower().split() + [self.EOS]
        indices = [self.token2idx.get(t, self.UNK_IDX)  for t in tokens]
        return indices

    def decode(self, indices):
        tokens = [self.idx2token[i] for i in indices
                  if i not in (self.PAD_IDX, self.SOS_IDX, self.EOS_IDX)]

        return " ".join(tokens)


def build_vocab(hf_dataset, min_freq=2):
        en_counter, de_counter = Counter(), Counter()
        for row in hf_dataset['train']:
            for tok in row['en'].lower().split():
                en_counter[tok] += 1
            for tok in row['de'].lower().split():
                de_counter[tok] += 1

        return Vocab(en_counter, min_freq), Vocab(de_counter, min_freq)


