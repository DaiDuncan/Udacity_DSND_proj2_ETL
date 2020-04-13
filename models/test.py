import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

def tokenize(text):
    
    #delete the Punctuations
    text = re.sub('\W+',' ', text).replace("_",' ')   
    
    #delete the Stop words
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words('english')] 
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

pkl_filename = 'classifier.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

print(pickle_model.get_params())