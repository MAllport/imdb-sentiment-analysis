import pandas as pd
import nltk
import json
import logging
import re
from gensim.models.doc2vec import TaggedDocument
from bs4 import BeautifulSoup


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def tag_documents(train, test, labels):
    tagged_train = []
    tagged_test  = []

    for idx, (train_doc, test_doc) in enumerate(zip(train, test)):
        if train_doc != []:
            token_list = []
            for sentence in train_doc:
                if sentence not in [None, ""]:
                    tokens = nltk.word_tokenize(sentence)
                    for token in tokens:
                        if token not in [None, ""]:
                            token_list.append(token)
            tagged_train.append(TaggedDocument(words = token_list, tags=["id" + str(idx)]))

        if test_doc != []:
            token_list = []
            for sentence in test_doc:
                if sentence not in [None, ""]:
                    tokens = nltk.word_tokenize(sentence)
                    for token in tokens:
                        if token not in [None, ""]:
                            token_list.append(token)
            tagged_test.append(token_list)
    return tagged_train, tagged_test



def preprocessing_init(train):
    #Read labeled and unlabeled training data
    
    #Choose tokenizer from nltk
    
    num_reviews = len(train)
    
    labeled = []
    
    #Clean labeled reviews
    for i in range(0, num_reviews):
        #The function review_to_sentences has been defined below
        labeled.append(review_to_sentences(train[i]))
    
    return labeled
# Here is the function review_to_sentences 

def review_to_sentences(review, removeStopwords=False, removeNumbers=False, removeSmileys=False):
    
    rawSentences = tokenizer.tokenize(review.strip())
    cleanedReview = []
    for idx, sentence in enumerate(rawSentences):
        if len(sentence) > 0:
            cleanedReview.append(review_to_words(sentence, removeStopwords, removeNumbers, removeSmileys))
              
    return cleanedReview

#The function review_to_words

def review_to_words(rawReview, removeStopwords=False, removeNumbers=False, removeSmileys=False):
    
    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    reviewText = BeautifulSoup(markup= rawReview, features="html.parser").get_text()

    # Emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replaces the pattern by the desired character/string
    
    if removeNumbers and removeSmileys:
        # any character that is not in a to z and A to Z (non text)
        reviewText = re.sub("[^a-zA-Z]", " ", reviewText)
    elif removeSmileys:
         # numbers are also included
        reviewText = re.sub("[^a-zA-Z0-9]", " ", reviewText)
    elif removeNumbers:
        reviewText = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", reviewText)
    else:
        reviewText = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", reviewText)
    
    return reviewText