import math
import re
import string

import numpy as np
import pandas as pd
import sklearn as sk
import nltk
from nltk import word_tokenize
# from nltk.corpus import stopwords

from gensim.models import Word2Vec
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import strip_non_alphanum, strip_numeric
from gensim.test.utils import get_tmpfile
import gensim.downloader as api

from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocessor(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text

def main():

    df = pd.read_csv("imdb.csv")

    df['review'] = df['review'].apply(preprocessor)

    df['sentiment'] = LabelEncoder().fit_transform(df['sentiment'])
    
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], random_state=0, test_size=0.5)

    # Marginally better results with TfidfVectorizer
    # CountVectorizer tokenizes strings
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1,2), max_features=100000)),
        ('classifier', LogisticRegression(max_iter=50))
        ])

    # vectorizer = pipeline['vectorizer']
    # classifier = pipeline['classifier']

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

    # print(len(vectorizer.vocabulary_))

if __name__ == "__main__":
    main()
