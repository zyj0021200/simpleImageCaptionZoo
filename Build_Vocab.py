from collections import Counter
from DatasetClass import CaptionData

class Caption_Vocabulary(object):
    def __init__(self):
        self.word2ix = {}
        self.ix2word = {}
        self.idx = 0

    def add_word(self, new_word):
        if not new_word in self.word2ix:
            self.word2ix[new_word] = self.idx
            self.ix2word[self.idx] = new_word
            self.idx += 1

    def __len__(self):
        return len(self.word2ix)

    def __call__(self, word):
        if not word in self.word2ix:
            return self.word2ix['<unk>']
        return self.word2ix[word]

def build_vocab(json, threshold):
    capdata = CaptionData(json)
    ids = capdata.anns.keys()
    counter = Counter()

    for i, id in enumerate(ids):
        tokens = capdata.anns[id]['tokens']
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." % (i, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Caption_Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<sta>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab
