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
from preprocessing import preprocessing_init, tag_documents


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
    y = clf.fit(X)

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

def doc2vec(train, test, labels):
    print("Starting to tag documents..")
    tagged_train, tagged_test = tag_documents(train, test, labels)
    print(f"Done tagging documents. Length of tagged training documents is {len(tagged_train)}, length of tagged test documents is {len(tagged_test)}\n")
    print("Start training doc2vec model..")
    model = Doc2Vec(tagged_train, min_count=1, vector_size=250, sample=1e-4, negative=6 ,workers=7, epochs = 10)
    print("Done with doc2vec model \n")
    
    docvecs = []
    test_docvecs = []
    print("Start inferring from test-set..")
    for i in range(len(train)):
        docvecs.append(model.docvecs["id" + str(i)])
        test_docvecs.append(model.infer_vector(tagged_test[i]))
    print("Done inferring \n")
    return docvecs, test_docvecs

def bag_of_words(sentences):
    tfidf = TfidfVectorizer(max_features=5000)
    bag_of_words = tfidf.fit_transform(sentences)

    return bag_of_words.toarray()

def create_new_dataset():
    all_files_neg = glob.iglob("datasets/test/neg/*")
    all_files_pos = glob.iglob("datasets/test/pos/*")

    df_neg = pd.concat((pd.read_table(f, names = ["text"]) for f in all_files_neg))
    df_neg["label"] = 0

    df_pos = pd.concat((pd.read_table(f, names = ["text"]) for f in all_files_pos))
    df_pos["label"] = 1

    df = pd.concat([df_pos, df_neg])

    df.to_csv("datasets/test_dataset.csv")
    
    return df

def read_dataset(dataset_string):

    df = pd.read_csv(f"datasets/{dataset_string}_dataset.csv", names = ["text", "label"])
    df = df.dropna()

    df = df[1:].sample(frac=1).reset_index(drop=True)

    return list(df["text"]), list(df["label"])

def preprocessing(df):

    df = df.dropna()

    df = df[1:].sample(frac=1).reset_index(drop=True)

    return list(df["text"]), list(df["label"])


def make_model():
    print("Reading dataset...")
    train, train_label = read_dataset("train")
    test, test_label   = read_dataset("test")
    print("Done reading dataset\n")
    print("Starting pre-processing..")
    train = preprocessing_init(train)
    test  = preprocessing_init(test)
    print(f"Done with pre-processing. Length of train is {len(train)}, lenggth of test is {len(test)}\n")
    docvecs, test_docvecs = doc2vec(train, test, train_label)
    #model = bag_of_words(sentences)

    return (docvecs, test_docvecs, train_label, test_label)

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

def find_accuracy(clf_labels, test_label):
    correct = 0
    totals = len(clf_labels)

    print(clf_labels.shape)
    print(len(test_label))


    for idx, label in enumerate(clf_labels):
        if int(label) == int(test_label[idx]):
            correct += 1
    print(correct)
    total_correct = (float(correct)/float(totals)) * 100
    print(f"Accuracy is {total_correct} %")
    with open("results.txt", "w") as res:
        res.write(f"Percentage of accurate guesses is {total_correct} %")

def main():
    docvecs, test_docvecs, train_label, test_label = make_model()
    print("Starting clustering..")
    X = test_docvecs
    n = 2
    clf = kmeans_cluster(X, n)
    plot_kmeans(X, n, clf)
    print("Done with clustering \n")
    #write_files(clf, documents, n)
    print("Finding the accuracy..")
    find_accuracy(clf.labels_, test_label)
    print("Accuracy is calculated\n Done")


    plt.show()


if __name__ == "__main__":
    main()
