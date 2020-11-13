import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import string
import seaborn as sns
import imblearn
import scipy
import math
import re
import nltk
import os
import glob

from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_non_alphanum


#nltk.download('punkt')

from tempfile import TemporaryDirectory

from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from numpy import dot
from numpy.linalg import norm

from scipy.cluster import hierarchy
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from itertools import combinations, permutations
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

from scipy.stats import chi2_contingency,spearmanr, pearsonr, entropy

def plot_kmeans(X, n, clf):
    labels = clf.labels_
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    colors = ['r','g','b','y']
    col_map=dict(zip(set(labels),colors))
    label_color = [col_map[l] for l in labels]

    fig, ax = plt.subplots()
    ax.scatter(coords[:,0], coords[:,1], c=label_color)
    ax.scatter(clf.cluster_centers_[:,0], clf.cluster_centers_[:,1], marker='x', s=50, c='#000000')

def kmeans_cluster(X, n):
    clf = KMeans(n_clusters=n, max_iter=100, init="k-means++", n_init=10, random_state=42)
    # X_labels = clf.fit_predict(X)
    clf.fit(X)
    print("Words shape:", X.shape)
    print("Number of labels:", clf.labels_.shape)

    return clf

def plot_elbow(X):
    wcss = []
    for i in range(1, 50):
        kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42, n_jobs=7)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        print(f"Done with loop {i}")
    plt.plot(range(1,50), wcss)
    plt.title("The elbow method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")

def doc2vec(sentences):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    model = Doc2Vec(documents, vector_size=200, window=2, min_count=1, workers=7)

    return model.docvecs.vectors_docs

def bag_of_words(sentences):
    flattened_sentences = [' '.join(i) for i in sentences]
    tfidf = TfidfVectorizer()
    bag_of_words = tfidf.fit_transform(flattened_sentences)

    return bag_of_words.toarray()

def preprocessing():
    with open("datasets/unsup_dataset.csv") as data:
        df = pd.read_csv(data, names=["text"])["text"]

    df = df.str.lower()
    df = df.str.replace(r're:\s*', '')
    df = df.str.replace(r'sv:\s*', '')
    df = df.str.replace(r'fwd:\s*', '')
    df = df.str.replace(r"\[.*\]",'')
    df = df.apply(strip_numeric)
    df = df.apply(strip_non_alphanum)
    df = df.str.replace(r'\s+', ' ')
    df = df.str.strip()
    df = df.drop_duplicates()

    return list(df)


def make_model():
    
    sentences = preprocessing()
    model = doc2vec(sentences)
    #model = bag_of_words(sentences)

    return (sentences, model)

def write_files(clf, docs, n):
    label_dict = {}
    for idx, label in enumerate(clf.labels_):
        doc = docs[idx] + "\n"
        
        if label in label_dict:
            label_dict[label].append(doc)
        else:
            label_dict[label] = [doc]
    
    files = glob.glob('clusters/*')
    for f in files:
        if f != ".gitkeep":
            os.remove(f)

    for i in range(n):
        with open("clusters/" + str(i), 'w') as out_file:
            docs = label_dict[i]
            out_file.writelines(docs)

def main():
    documents, model = make_model()

    X = model
    n = 2
    clf = kmeans_cluster(X, n)
    plot_kmeans(X, n, clf)

    write_files(clf, documents, n)
    


    plt.show()


if __name__ == "__main__":
    main()
