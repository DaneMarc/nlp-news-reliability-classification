import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

class Embedding:
    def __init__(self, doc_embed=True, max_seq_len=20):
        self.doc_embed = doc_embed
        self.max_seq_len = max_seq_len
        self.pca = PCA(n_components=self.max_seq_len)

        if self.doc_embed:
            self.dim_size = 300
            self.model = KeyedVectors.load('nlp/embedding/cc.en.300.kv')
        else:
            self.dim_size = 100
            self.model = KeyedVectors.load('nlp/embedding/cc.en.100.kv')

    '''
    Flags
    ------
    tfidf: Tf-Idf weighted word embeddings
    sentiment: Valence score weighted word embeddings
    tfidf_clip: Only used for concatenated embeddings (doc_embed=False). Clips doc embeddings by using only the top n=max_seq_len words.
    pca: Only used for concatenated embeddings (doc_embed=False). Uses PCA to reduce dimensionality along word axis.
    flatten: Only used for concatenated embeddings (doc_embed=False). Flattens embedding to 1D.
    '''
    def get_embedding(self, docs, tfidf=False, sentiment=False, tfidf_clip=False, pca=False, flatten=False):
        embedded_docs = []
        flattened = [[j for sub in doc for j in sub] for doc in docs.iloc[:,-1]]

        if tfidf:
            vectorizer = TfidfVectorizer(min_df=0, max_df=9999999, lowercase=False, token_pattern=r'(?u)(?<!\S)\S+(?!\S)')
            tfidf_scores = vectorizer.fit_transform([' '.join(arr) for arr in flattened])

        if sentiment:
            vader = SentimentIntensityAnalyzer()

        for j, doc in tqdm(docs.iterrows(), total=docs.shape[0]):
            doc_embedding = []
            tfidf_weights = []
            sentiment_weights = []
            tokens = flattened[j]

            if sentiment:
                sentences = doc.iloc[-2]
                sent_tokens = doc.iloc[-1]
                for i, sentence in enumerate(sentences):
                    score = vader.polarity_scores(sentence)['compound']
                    if score == 0:
                        score = 0.1 # smooth neutral sentences
                    if self.doc_embed:
                        score = abs(score) # abs to make weight positive and prevent 0 sum when getting mean
                    if tfidf:
                        sentiment_weights += [score] * len([tok for tok in sent_tokens[i] if tok in self.model and tok in vectorizer.vocabulary_])
                    else:
                        sentiment_weights += [score] * len([tok for tok in sent_tokens[i] if tok in self.model])

            if tfidf:
                for i, token in enumerate(tokens):
                    if token in self.model and token in vectorizer.vocabulary_:
                        tfidf_weights.append(tfidf_scores[j, vectorizer.vocabulary_[token]])
                        
            if tfidf or sentiment:
                if tfidf and not sentiment:
                    weights = tfidf_weights
                elif sentiment and not tfidf:
                    weights = sentiment_weights
                else: # tfidf and sentiment
                    weights = [a*b for a,b in zip(tfidf_weights, sentiment_weights)]
                total_weight = sum(weights)

            if self.doc_embed:
                for token in tokens:
                    if token in self.model:
                        doc_embedding.append(self.model[token])

                if len(doc_embedding) == 0:
                    doc_embedding = np.zeros((self.dim_size,))
                else:
                    if tfidf or sentiment:
                        doc_embedding = np.average(doc_embedding, axis=0, weights=weights)
                    else:
                        doc_embedding = np.mean(doc_embedding, axis=0)
            else:
                if tfidf or sentiment:
                    k = 0
                    weights = [w / total_weight for w in weights]
                    if tfidf:
                        for i, token in enumerate(tokens):
                            if token in self.model and token in vectorizer.vocabulary_:
                                doc_embedding.append(self.model[token] * weights[k])
                                k += 1
                    else:
                        for i, token in enumerate(tokens):
                            if token in self.model:
                                doc_embedding.append(self.model[token] * weights[k])
                                k += 1
                else:
                    for token in tokens:
                        if token in self.model:
                            doc_embedding.append(self.model[token])

                if len(doc_embedding) == 0:
                    if flatten:
                        doc_embedding = np.zeros((self.max_seq_len * self.dim_size))
                    else:
                        doc_embedding = np.zeros((self.max_seq_len, self.dim_size))
                else:
                    doc_embedding = self.pad(doc_embedding, tfidf_weights, tfidf_clip, pca, flatten)

            embedded_docs.append(doc_embedding)
        
        return embedded_docs
    

    def pad(self, doc, weights, tfidf_clip=False, pca=False, flatten=True):
        if len(doc) < self.max_seq_len:
            doc = np.array(doc)
            zeros = np.zeros((self.max_seq_len - len(doc), self.dim_size))
            doc = np.vstack((doc, zeros))
        elif len(doc) > self.max_seq_len:
            if tfidf_clip and len(weights) > 0:
                #print('tfidf clipping')
                doc = np.array(self.tfidf_clip(doc, weights))
            elif pca:
                doc = np.array(doc).T
                doc = self.pca.fit_transform(doc)
                #print('pca clipping')
                doc = doc.T
            else:
                #print('normal clipping')
                doc = np.array(doc[:self.max_seq_len])
        else:
            doc = np.array(doc)
        
        return doc.flatten() if flatten else doc
        

    def tfidf_clip(self, doc, weights):
        new_doc = []
        sorted = weights.copy()
        sorted.sort(reverse=True)
        min_weight = sorted[self.max_seq_len-1]
        
        for i in range(len(doc)):
            if weights[i] >= min_weight and len(new_doc) < self.max_seq_len:
                new_doc.append(doc[i])
            
        return new_doc
    

    def save_embedding(self, docs, tfidf=False, sentiment=False, tfidf_clip=False, pca=False, flatten=False, path=None):
        embedded_docs = self.get_embedding(docs, tfidf, sentiment, tfidf_clip, pca, flatten)
        new_docs = docs.drop(docs.columns[-len(docs.columns)+1:], axis=1, inplace=False)
        new_docs['embeddings'] = embedded_docs

        if path:
            new_docs.to_pickle(path)
        else:
            string = f'{"doc" if self.doc_embed else "concat"}{"_tfidf" if tfidf else ""}{"_sent" if sentiment else ""}{"_tfidfClip" if tfidf_clip else ""}{"_pca" if pca else ""}{"_flat" if flatten else ""}.pkl'
            new_docs.to_pickle(string)
    