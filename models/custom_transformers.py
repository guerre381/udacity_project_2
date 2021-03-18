# import libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import gensim
import re



def identity(x):
    """Identity function

    :param x: any input
    :return: input
    """
    return x



class TokenizeTransform(BaseEstimator, TransformerMixin):
    """Custom transformer for tokenizing and lemmatizing words - Sklearn extension"""


    def __init__(self):
        """TokenizeTransform constructor"""

        # define regular expression to find url in text
        self.url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


    def fit(self, X, y=None):
        """Training transformer on data but not doing anything here

        :param X: train input
        :param y: train output
        :return: object itself by convention
        """
        return self


    def tokenize(self, text):
        """Tokenize and Lemmatize paragraphs

        :param text: paragraph
        :return: list of lemmatized words
        """

        # find urls in text
        detected_urls = re.findall(self.url_regex, text)

        # remove urls
        for url in detected_urls:
            text = text.replace(url, "")

        # tokenize text
        tokens = word_tokenize(text)

        # lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        clean_tokens = [lemmatizer.lemmatize(tok).lower().strip()
                        for tok in tokens
                        if tok.isalpha()]

        return clean_tokens


    def transform(self, X):
        """tokenize and lemmatize every paragraphs input entries

        :param X: list of paragraphs
        :return: list of list of lemmatized words
        """

        return pd.Series(X).apply(self.tokenize).values




class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Custom transformer for calculating paragraphs embedding - Sklearn extension"""


    def __init__(self, size=300, iter=100, window=3, min_count=5):
        """TfidfEmbeddingVectorizer constructor

        :param size: dimensionality of the word vectors
        :param iter: word embedding epochs
        :param window: max distance between neigboor words for embedding
        :param min_count: min occurrence of words to be considered
        :param sg: paragraph
        """

        self.word2vec = None
        self.word2weight = None
        self.vocab = None

        # parameters for word embedding model word2vec
        self.size = size
        self.iter = iter
        self.window = window
        self.min_count = min_count

        # parameters for tfidf
        self.min_df = min_count


    def fit(self, X, y=None):
        """Training transformer on data but not doing anything here

        :param X: train input
        :param y: train output
        :return: object itself by convention
        """

        # create word embedding model with gensim
        embedding = gensim.models.Word2Vec(sentences=X, size=self.size, iter=self.iter,
                                           window=self.window, min_count=self.min_count)
        # keep word to vector conversion in dictionary
        self.word2vec = {w:embedding.wv[w] for w in embedding.wv.vocab}

        # create tf-idf mdodel
        tfidf = TfidfVectorizer(analyzer=identity, min_df=self.min_df)
        tfidf.fit(X)

        # keep word to weight conversion in dictionary
        self.word2weight = {w:tfidf._tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()}

        # create vocab set common to both models
        self.vocab = set(tfidf.vocabulary_.keys()).intersection(set(embedding.wv.vocab))

        return self


    def transform(self, X):
        """Calculate mean word embedding weighted with tfidf coefficients for each paragraphs

        :param X: list of paragraphs
        :return: list of paragraphs embedding
        """

        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in text if w in self.vocab] or
                    [np.zeros(self.size)], axis=0)
            for text in X
        ])

