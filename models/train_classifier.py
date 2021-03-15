import sys
import logging
import pandas as pd
import numpy as np
import re


import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from collections import defaultdict
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import gensim
from gensim.models import Word2Vec

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#set logging configuration
logging.basicConfig(encoding='utf-8',
                    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """load clean data from database

    :param database_filepath: path to database file containing data
    :return X: text input dataframe for machine learning
    :return Y: categories output dataframe for machine learning
    :return columns: output category names
    """
    logging.info("run load_data")

    # create engine and connect to file based-database
    engine = create_engine(f"sqlite:///{database_filepath}")

    #load data in dataframe
    df = pd.read_sql_table('data', engine)
    logging.info(f"data retrieved from db file: {database_filepath}")

    # remove 'related' entries equal to 2
    df = df[df['related'] != 2]

    #split data between input and outputs
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    """tokenize text into a list of lemmatized words

    :param text: text paragraph to be tokenized
    :return: clean tokens
    """

    logging.debug("run tokenize")

    # remove URLs from text
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    logging.debug("URLs removed")

    #tokenize text entry
    tokens = word_tokenize(text)
    logging.debug("text tokenized")

    # lemmatize each words in tokens list
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    logging.debug("tokens lemmatized")

    return clean_tokens


class TfidfWeightedEmbedding(BaseEstimator, TransformerMixin):

    def __init__(self, original_corpus):
        self.word2vec = None
        self.corpus = original_corpus

    def fit(self, X, y=None):
        self.word2vec = gensim.models.Word2Vec(sentences=sentences)


    def transform(self, X):
        print(X)

def build_model(sentences):
    """build machine learning pipeline

    :return: model machine learning pipeline
    """

    logging.info("run build_model")

    #prepare word embedding
    w2c = gensim.models.Word2Vec(sentences=sentences)
    # load pre-trained word embedding

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('w2v', TfidfWeightedEmbedding(w2c)),
        #('clf', MultiOutputClassifier(SVC()))
    ])

    logging.info("model built")

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model based on test data

    :param model: Machine learning model
    :param X_test: input test data
    :param Y_test: output test data
    :param category_names: path to csv file containing category data
    """

    logging.info("run evaluate_model")

    Y_pred = model.transform(X_test)
    print(classification_report(Y_test, Y_pred))



def save_model(model, model_filepath):
    """Extract data form csv files and merge them into dataframe

    :param messages_filepath: path to csv file containing message data
    :param categories_filepath: path to csv file containing category data
    :return: merge dataframe
    """
    logging.info("run save_model")



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        logging.info("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        logging.info("'Building model...")
        model = build_model()
        
        logging.info("Training model...")
        model.fit(X_train, Y_train)
        
        logging.info("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        logging.info("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    # train_classifier.py /data/database.db model
    main()