import numpy as np
import gensim.downloader as api

from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedding:
    def __init__(self, type='fasttext', max_seq_len=270, path=None):
        self.max_seq_len = max_seq_len
        self.type = type
        self.dim_size = 300

        if type == 'word2vec':
            self.model = api.load('word2vec-google-news-300') if not path else Word2Vec.load(path).wv
        elif type == 'glove':
            self.model = api.load('glove-wiki-gigaword-300')
        elif type == 'fasttext':
            self.model = api.load('fasttext-wiki-news-subwords-300') if not path else FastText.load(path).wv
        else:
            print('Invalid type')

    # input a list of tokens
    def get_embedding(self, docs, doc_embed=True, tfidf=False, sentiment=False, flatten=True):
        embedded_doc = []

        if tfidf:
            vectorizer = TfidfVectorizer(min_df=0, max_df=9999999)
            tfidf_scores = vectorizer.fit_transform(docs)

        if doc_embed:
            for i, doc in enumerate(docs):
                doc_embedding = []

                if tfidf:
                    weights = []
                    for token in doc:
                        if token in self.model and token in vectorizer.vocabulary_:
                            doc_embedding.append(self.model[token])
                            weights.append(tfidf_scores[i, vectorizer.vocabulary_[token]])
                else:
                    for token in doc:
                        if token in self.model:
                            doc_embedding.append(self.model[token])

                if len(doc_embedding) == 0:
                    doc_embedding = np.zeros((self.dim_size,))
                else:
                    if tfidf:
                        doc_embedding = np.average(doc_embedding, axis=0, weights=weights)
                    else:
                        doc_embedding = np.mean(doc_embedding, axis=0)

                embedded_doc.append(doc_embedding)
        else:
            for i, doc in enumerate(docs):
                doc_embedding = []
                weights = []

                if tfidf:
                    for token in doc:
                        if token in self.model and token in vectorizer.vocabulary_:
                            tfidf_score = tfidf_scores[i, vectorizer.vocabulary_[token]]
                            doc_embedding.append(self.model[token] * tfidf_score)
                            weights.append(tfidf_score)
                else:
                    for token in doc:
                        if token in self.model:
                            doc_embedding.append(self.model[token])

                if len(doc_embedding) > 0:
                    doc_embedding = self.pad(doc_embedding, weights, flatten)
                else:
                    doc_embedding = np.zeros((self.max_seq_len * self.dim_size))

                embedded_doc.append(doc_embedding)
        
        length = self.dim_size if doc_embed else self.dim_size * self.max_seq_len

        return embedded_doc, length
    

    def pad(self, doc, weights, flatten):
        if len(doc) < self.max_seq_len:
            doc = np.array(doc)
            mean = np.mean(doc, axis=0)
            zeros = np.repeat([mean], self.max_seq_len - len(doc), axis=0)
            #zeros = np.zeros((self.max_seq_len - len(doc), self.dim_size))
            doc = np.vstack((doc, zeros))
            return doc.flatten() if flatten else doc
        elif len(doc) > self.max_seq_len:
            # clip
            #return np.array(self.tfidf_clip(doc, weights)).flatten()
            doc = np.array(doc[:self.max_seq_len])
            return doc.flatten() if flatten else doc

            # pca
            #doc = np.array(doc).T
            #doc = self.pca.fit_transform(doc)
            #return doc.T.flatten()
        else:
            return np.array(doc).flatten() if flatten else np.array(doc)
        
    def tfidf_clip(self, doc, weights):
        new_doc = []
        sorted = weights.copy()
        sorted.sort(reverse=True)
        min_weight = sorted[self.max_seq_len-1]
        
        for i in range(len(doc)):
            if weights[i] >= min_weight and len(new_doc) < self.max_seq_len:
                new_doc.append(doc[i])
            
        return new_doc
    