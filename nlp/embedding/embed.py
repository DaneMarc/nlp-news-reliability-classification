import os
import numpy as np
import gensim.downloader as api

from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedding:
    def __init__(self, type='word2vec', max_seq_len=500, path=None, docs=[]):
        self.max_seq_len = max_seq_len
        self.type = type

        if type == 'word2vec':
            self.model = api.load('word2vec-google-news-300') if not path else Word2Vec.load(path).wv
        elif type == 'glove':
            self.model = api.load('glove-wiki-gigaword-300')
        elif type == 'fasttext':
            self.model = api.load('fasttext-wiki-news-subwords-300') if not path else FastText.load(path).wv
        else:
            print('Invalid type')

    # input a list of tokens
    def get_embedding(self, docs, doc_embed=True, tfidf=False, sentiment=False):
        docs = []

        if tfidf:
            vectorizer = TfidfVectorizer(min_df=0, max_df=9999999)
            tfidf_scores = vectorizer.fit_transform(docs)

        if doc_embed:
            for doc in docs:
                doc_embedding = []

                if tfidf:
                    weights = []
                    for i, token in enumerate(doc):
                        if token in self.model and token in vectorizer.vocabulary_:
                            doc_embedding.append(self.model[token])
                            weights.append(tfidf_scores[i, vectorizer.vocabulary_[token]])
                else:
                    for token in doc:
                        if token in self.model:
                            doc_embedding.append(self.model[token])

                if len(doc_embedding) == 0:
                    doc_embedding = np.zeros((300,))
                else:
                    if tfidf:
                        doc_embedding = np.average(doc_embedding, axis=0, weights=weights)
                    else:
                        doc_embedding = np.mean(doc_embedding, axis=0)

                docs.append(doc_embedding)
        else:
            for doc in docs:
                doc_embedding = []
                weights = []

                if tfidf:
                    for i, token in enumerate(doc):
                        if token in self.model and token in vectorizer.vocabulary_:
                            tfidf_score = tfidf_scores[i, vectorizer.vocabulary_[token]]
                            doc_embedding.append(self.model[token] * tfidf_score)
                            weights.append(tfidf_score)
                else:
                    for token in doc:
                        if token in self.model:
                            doc_embedding.append(self.model[token])

                if len(doc_embedding) > 0:
                    doc_embedding = self.pad(doc_embedding, weights)
                else:
                    doc_embedding = np.zeros((self.max_seq_len * 300))

                docs.append(doc_embedding)
        
        length = 300 if doc_embed else 300 * self.max_seq_len

        return docs, length
    

    def pad(self, doc, weights):
        if len(doc) < self.max_seq_len:
            doc = np.array(doc)
            mean = np.mean(doc, axis=0)
            zeros = np.repeat([mean], self.max_seq_len - len(doc), axis=0)
            doc = np.vstack((doc, zeros))
            return doc.flatten()
        elif len(doc) > self.max_seq_len:
            # clip
            #return np.array(self.tfidf_clip(doc, weights)).flatten()
            return np.array(doc[:self.max_seq_len]).flatten()

            # pca
            #doc = np.array(doc).T
            #doc = self.pca.fit_transform(doc)
            #return doc.T.flatten()
        else:
            return np.array(doc).flatten()
        
    def tfidf_clip(self, doc, weights):
        new_doc = []
        sorted = weights.copy()
        sorted.sort(reverse=True)
        min_weight = sorted[self.max_seq_len-1]
        
        for i in range(len(doc)):
            if weights[i] >= min_weight and len(new_doc) < self.max_seq_len:
                new_doc.append(doc[i])
            
        return new_doc
    