# import libraries
import sys
import json
import plotly
import joblib
import random
import pandas as pd
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from plotly.graph_objs import Scatter
from sqlalchemy import create_engine
from sklearn.manifold import TSNE
from collections import Counter

sys.path.append('../models')
from custom_transformers import TokenizeTransform, TfidfEmbeddingVectorizer


app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("../models/model.pkl")
print(model.get_params())


@app.route('/')
@app.route('/index')
def index():
    """Display visuals from data"""

    ### create figure one
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_1 = []
    graph_1.append(Bar(x=genre_names, y=genre_counts))

    layout_1 = dict(title = 'Distribution of Message Genres',
                      xaxis = dict(title = 'Genre'),
                      yaxis = dict(title = 'Count')
                      )

    ### create figure two
    categories_percentage = df.iloc[:,4:].mean().sort_values(ascending=False)
    categories_names = list(categories_percentage.index)

    graph_2 = []
    graph_2.append(Bar(x=categories_names, y=categories_percentage))

    layout_2 = dict(title = 'Distribution of Categories',
                      xaxis = dict(title = 'categories', tickangle=-45),
                      yaxis = dict(title = 'percentage')
                      )

    ### create figure three
    graph_3 = []
    for col in df.columns[4:]:
        nb_words = df.loc[df[col]==1,'message'].str.split().apply(len).value_counts()
        graph_3.append(Histogram(x=nb_words, nbinsx=100, name=col))

    layout_3 = dict(title = 'Distribution of Messages length',
                      xaxis = dict(title = 'distribution'),
                      yaxis = dict(title = 'nb of words')
                      )

    ### create figure four

    # number of words to consider in plot
    NB_WORDS = 50

    # select TfidfEmbeddingVectorizer from pipeline
    tfidf_emb_model = model.named_steps['tfidf_emb']

    # identify the most relevant words (highest tfidf score)
    relevant_words = Counter(tfidf_emb_model.word2weight).most_common(NB_WORDS)

    # retrieve embedding from most relevant words
    relevant_emb = {w:tfidf_emb_model.word2vec[w] for w, _coeff in relevant_words if w in tfidf_emb_model.vocab}
    relevant_em_df = pd.DataFrame().from_dict(relevant_emb, orient='index')

    # principal component analysis for 2D visualzation
    relevant_em_df[['PC1', 'PC2']] = TSNE(n_components=2).fit_transform(relevant_em_df)
    relevant_em_df['size'] = relevant_em_df.index.map(lambda x: tfidf_emb_model.word2weight[x])
    relevant_em_df['color'] = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(NB_WORDS)]

    graph_4 = []
    graph_4.append(Scatter(x=relevant_em_df['PC1'],
                           y=relevant_em_df['PC2'],
                           mode='text',
                           text=relevant_em_df.index,
                           marker={'opacity': 0.3},
                           textfont={'size': relevant_em_df['size'],
                                     'color': relevant_em_df['color']}
                           )
                   )

    layout_4 = dict(title = 't-SNE of words with the strongest tfidf coefficients',
                      xaxis = dict(title = 'PC1'),
                      yaxis = dict(title = 'PC2')
                      )


    # combine all graphs with corresponding layout
    figures = []
    figures.append(dict(data=graph_1, layout=layout_1))
    figures.append(dict(data=graph_2, layout=layout_2))
    figures.append(dict(data=graph_3, layout=layout_3))
    figures.append(dict(data=graph_4, layout=layout_4))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template(
        'master.html',
        ids=ids,
        graphJSON=graphJSON)



@app.route('/go')
def go():
    """Receive user inputs and display model classification results"""

    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()