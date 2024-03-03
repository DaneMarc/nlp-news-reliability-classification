import numpy as np
import gensim
import gensim.downloader as api

class Embedding:
    def __init__(self, type='word2vec', train=False, max_seq_len=630, path=None):
        self.max_seq_len = max_seq_len

        if train:
            if type == 'word2vec':
                self.model = gensim.models.Word2Vec.load(path)
            elif type == 'fasttext':
                self.model = gensim.models.FastText.load(path)
            else:
                print('Invalid type')
        else:
            if type == 'word2vec':
                self.model = api.load('word2vec-google-news-300')
            elif type == 'glove':
                self.model = api.load('glove-wiki-gigaword-300')
            else:
                self.model = api.load('fasttext-wiki-news-subwords-300')

    # input a list of tokens
    def get_embedding(self, doc):
        doc_embedding = []
        for token in doc:
            if token in self.model:
                doc_embedding.append(self.model[token])
            else:
                doc_embedding.append(np.zeros(300))

        # pad or clip the sequence
        if len(doc_embedding) < self.max_seq_len:
            doc_embedding += [np.zeros(300)] * (self.max_seq_len - len(doc_embedding))
        else:
            doc_embedding = doc_embedding[:self.max_seq_len]

        return doc_embedding
    