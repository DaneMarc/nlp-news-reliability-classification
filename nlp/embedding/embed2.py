import numpy as np
#import fasttext
#import fasttext.util

from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

class Embedding:
    def __init__(self, max_seq_len=163, path=None):
        self.max_seq_len = max_seq_len
        self.type = type
        self.dim_size = 300
        self.pca = PCA(n_components=self.max_seq_len)

        #fasttext.util.download_model('en', if_exists='ignore')
        #self.model = fasttext.load_model('../../wiki.simple.100.bin')
        #self.model = fasttext.load_model('cc.en.300.bin')
        if path:
            self.model = KeyedVectors.load(path)
        else:
            self.model = KeyedVectors.load('cc.en.300.kv')


    # input a list of tokens
    def get_embedding(self, docs, doc_embed=True, tfidf=False, sentiment=False, tfidf_clip=False, pca=False, flatten=False):
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
                sentences = doc.iloc[:,-2]
                sent_tokens = doc.iloc[:,-1]
                for i, sentence in enumerate(sentences):
                    score = vader.polarity_scores(sentence)['compound']
                    if score == 0:
                        score = 0.1
                    if doc_embed:
                        score = abs(score)
                    sentiment_weights += [score] * len([tok for tok in sent_tokens[i] if tok in self.model]) # abs to make weight positive
                print(sentiment_weights)

            if tfidf:
                for i, token in enumerate(tokens):
                    if token in self.model and token in vectorizer.vocabulary_:
                        tfidf_weights.append(tfidf_scores[j, vectorizer.vocabulary_[token]])
                print(tfidf_weights)
                        
            if tfidf or sentiment:
                k = 0
                weights = []

                if tfidf and not sentiment:
                    weights = tfidf_weights
                elif sentiment and not tfidf:
                    weights = sentiment_weights
                else:
                    weights = [a*b for a,b in zip(tfidf_weights, sentiment_weights)]

                if tfidf:
                    for token in tokens:
                        if token in self.model and token in vectorizer.vocabulary_:
                            doc_embedding.append(self.model[token] * weights[k])
                            k += 1
                else:
                    for token in tokens:
                        if token in self.model:
                            doc_embedding.append(self.model[token] * weights[k])
                            k += 1
            else:
                for token in tokens:
                    if token in self.model:
                        doc_embedding.append(self.model[token])

            if doc_embed:
                if len(doc_embedding) == 0:
                    doc_embedding = np.zeros((self.dim_size,))
                else:
                    doc_embedding = np.mean(doc_embedding, axis=0)
            else:
                if len(doc_embedding) == 0:
                    if flatten:
                        doc_embedding = np.zeros((self.max_seq_len * self.dim_size))
                    else:
                        doc_embedding = np.zeros((self.max_seq_len, self.dim_size))
                else:
                    doc_embedding = self.pad(doc_embedding, tfidf_weights, tfidf_clip, pca, flatten)

            #print(doc_embedding)
            embedded_docs.append(doc_embedding)
        
        length = self.dim_size if doc_embed else self.dim_size * self.max_seq_len

        return embedded_docs, length
    

    def pad(self, doc, weights, tfidf_clip=False, pca=False, flatten=True):
        if len(doc) < self.max_seq_len:
            doc = np.array(doc)
            print(doc)
            zeros = np.zeros((self.max_seq_len - len(doc), self.dim_size))
            doc = np.vstack((doc, zeros))
            print(doc)
        elif len(doc) > self.max_seq_len:
            if tfidf_clip and len(weights) > 0:
                print('tfidf clipping')
                doc = np.array(self.tfidf_clip(doc, weights))
            elif pca:
                doc = np.array(doc).T
                print(doc.shape)
                doc = self.pca.fit_transform(doc)
                print(doc.shape)
                print('pca clipping')
                doc = doc.T
            else:
                print('normal clipping')
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
    
    def save_embedding(self, docs, doc_embed=True, tfidf=False, sentiment=False, tfidf_clip=False, pca=False, flatten=False, path=None):
        embedded_docs, _ = self.get_embedding(docs, doc_embed, tfidf, sentiment, tfidf_clip, pca, flatten)
        new_docs = docs.drop(docs.columns[-2:], axis=1, inplace=False)
        new_docs['embeddings'] = embedded_docs

        if path:
            new_docs.to_pickle(path)
        else:
            string = f'data/{"doc" if doc_embed else "concat"}{"_tfidf" if tfidf else ""}{"_sent" if sentiment else ""}{"_tfidfClip" if tfidf_clip else ""}{"_pca" if pca else ""}{"_flat" if flatten else ""}.pkl'
            new_docs.to_pickle(string)
    