# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:26:23 2020

@author: Pankaj Bhanushali
"""
#reading data
import pandas as pd
cc = pd.read_csv("movies_review.csv")

#cleaning data
#removing all special characters
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("",line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_clean = preprocess_reviews(cc.review)

#removing stopwords
#stopwards are not much useful in sentiment analysis
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
                ' '.join([word for word in review.split()
                        if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words = remove_stop_words(reviews_clean)

#lemmatization
#replacing all words with their root words
from nltk.stem.porter import PorterStemmer
def get_stemmed_text(corpus):
   
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in reviews.split()]) for reviews in corpus]

stemmed_reviews = get_stemmed_text(no_stop_words)

#vectorization
#assigning numbers to words present in reviews
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(stemmed_reviews)
X = cv.transform(stemmed_reviews)

cc["sentiment"].replace({"positive":1, "negative":-1}, inplace=True)

#build classifier
#building a model and checking accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

target = cc.sentiment
final_model = LogisticRegression(C=0.30)
final_model.fit(X, target)
print("Final Accuracy: %s" % accuracy_score(target, final_model.predict(X)))
