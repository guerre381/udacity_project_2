# import libraries
import sys
import os
import logging
import pandas as pd
import joblib
import pathlib

from sqlalchemy import create_engine

from custom_transformers import TokenizeTransform, TfidfEmbeddingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns




# set logging configuration
logging.basicConfig(encoding='utf-8',
                    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)




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

    # load data in dataframe
    df = pd.read_sql_table('data', engine)
    logging.info(f"data retrieved from db file: {database_filepath}")

    #  the few inputs equal to 2 are removed and data are shuffled
    df.loc[df['related']==2, 'related'] = 0
    df = df.sample(frac=1, random_state=42)

    # split data between input and outputs
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, Y.columns



def build_model(search = False):
    """build machine learning pipeline

    :param search: enable parameter optimization
    :return: model machine learning pipeline
    """

    logging.info("run build_model")

    # pipeline definition
    pipeline = Pipeline([
        ('tokenize', TokenizeTransform()), # split text into lemmatized words
        ('tfidf_emb', TfidfEmbeddingVectorizer()),
        ('clf', MLPClassifier())
    ], verbose=True)

    # set pipeline parameters
    pipeline.set_params(**{
                        'tfidf_emb__size':300,
                        'tfidf_emb__iter':200,
                        'tfidf_emb__min_count': 3,

                        'clf__alpha': 1e-03,
                        'clf__max_iter':600,
                        'clf__learning_rate':'adaptive',
                        'clf__hidden_layer_sizes':(300, 300, 300, 36,),
                        'clf__random_state':1,
                        'clf__early_stopping':True,
                        'clf__solver':'adam'
                         })

    if search == True:
        parameters = {
            'tfidf_emb__size': (200, 300),
            'tfidf_emb__iter': (100, 200),
            'tfidf_emb__min_count': (3, 5),

            'clf__alpha': (1e-03, 1e-04),
            'clf__hidden_layer_sizes':[(300, 200, 100, 36,), (300, 100, 36,)],
            'clf__early_stopping': (True, False),
            'clf__solver': ('adam','sgd')
        }

        pipeline = GridSearchCV(pipeline, parameters)

    return pipeline




def plot_confusion_matrix(conf_matrix_list, labels, cm_file_path):
    """Plot confusion matrix for each categories

    :param conf_matrix_list: confusion matrices' list of all categories
    :param labels: label names' list of all categorires
    :param cm_file_path: path to file where to save confusion matrix
    """

    logging.info("run plot_confusion_matrix")

    # construct plot figure with 36 subplots in a square grid
    fig, ax = plt.subplots(6, 6, figsize=(12, 7))

    # for each categories' name and confusion matrices
    for axes, cm, label in zip(ax.flatten(), conf_matrix_list, labels):

        #plot heatmap of single confusion matrix in list
        sns.heatmap(cm, annot=True, fmt='.2f', cbar=False, ax=axes)

        # label axis
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')

        # set title
        axes.set_title(label)

    # save plots in file
    fig.tight_layout()
    fig.savefig(cm_file_path)




def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model based on test data

    :param model: Machine learning model
    :param X_test: input test data
    :param Y_test: output test data
    :param category_names: path to csv file containing category data
    """

    logging.info("run evaluate_model")

    # find current foler path for savings
    folder_path = os.path.dirname(__file__)

    # predict outputs on test data
    Y_pred = model.predict(X_test)

    # create classification report with precision, recall, and F1 score for each categories
    clf_report_df = pd.DataFrame(classification_report(Y_test, Y_pred,
                                        target_names=category_names, output_dict=True)).T
    clf_report_df.to_markdown(buf=os.path.join(folder_path,'test','classification_report.md'), mode='w')

    # calculate confusion matrix for each categories and save corresponding heatmap plots
    conf_matrix_df = multilabel_confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(conf_matrix_df, category_names,
                          os.path.join(folder_path,'test','confusion_matrix.png'))




def save_model(model, model_filepath):
    """Save model after learning

    :param model: machine learning model
    :param model_filepath: file path where to save model
    """

    logging.info("run save_model")

    # save model with jolib library
    joblib.dump(model, model_filepath)




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        logging.info("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
        
        logging.info("'Building model...")
        model = build_model()

        logging.info("Training model...")
        Y_train.iloc[0,:] = 1
        model.fit(X_train, Y_train)
        
        logging.info("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        logging.info("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        logging.info('Trained model saved!')

    else:
        logging.info('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')




if __name__ == '__main__':
    # python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    main()