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