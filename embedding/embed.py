import os
import numpy as np
import gensim.downloader as api

from gensim.models import Word2Vec, FastText, doc2vec

class Embedding:
    def __init__(self, type='word2vec', train=False, max_seq_len=630, path=None, docs=[]):
        self.max_seq_len = max_seq_len
        self.type = type

        if train:
            if type == 'word2vec':
                if path:
                    model = Word2Vec.load(path)
                    model.train(docs, total_examples=len(docs), epochs=model.epochs)
                else:
                    if docs:
                        model = Word2Vec(sentences=docs, vector_size=300)
            elif type == 'fasttext':
                if path:
                    model = FastText.load(path)
                    model.train(docs, total_examples=len(docs), epochs=model.epochs)
                else:
                    if docs:
                        model = FastText(sentences=docs, vector_size=300)
            elif type == 'doc2vec':
                if path:
                    model = doc2vec.Doc2Vec.load(path)
                    model.train(docs, total_examples=len(docs), epochs=model.epochs)
                else:
                    docs = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
                    model = doc2vec.Doc2Vec(documents=docs, vector_size=300, dm=1)
            else:
                print('Invalid type')

            model.save(fname=os.path.join('embedding', 'models', type + '.model'))
            self.model = model
        else:
            if type == 'word2vec':
                self.model = api.load('word2vec-google-news-300') if not path else Word2Vec.load(path).wv
            elif type == 'glove':
                self.model = api.load('glove-wiki-gigaword-300')
            elif type == 'fasttext':
                self.model = api.load('fasttext-wiki-news-subwords-300') if not path else FastText.load(path).wv
            else:
                print('Invalid type')

    # input a list of tokens
    def get_embedding(self, doc):
        if self.type == 'doc2vec':
            return self.model.infer_vector(doc)
        else:
            doc_embedding = []
            for token in doc:
                if token in self.model:
                    doc_embedding.append(self.model.wv[token])
                else:
                    doc_embedding.append(np.zeros(300))

            # pad or clip the sequence
            if len(doc_embedding) < self.max_seq_len:
                doc_embedding += [np.zeros(300)] * (self.max_seq_len - len(doc_embedding))
            else:
                # TODO: use PCA to reduce the dimension
                doc_embedding = doc_embedding[:self.max_seq_len]

            return doc_embedding
    