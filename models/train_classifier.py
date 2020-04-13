import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    database_tablename = "disaster_cleaned"
    df = pd.read_sql_table(database_tablename,engine) 
    
    category_names = df.columns[~(df.columns.isin(['id', 'message', 'original', 'genre']))]
    Y = df[category_names].values
    
    #Guess: the values of 'genre' maybe important, put it together with the 'message'
    df['message_genre']=df['message']+' '+df['genre']
    X = df['message_genre'].values
    
    return X, Y, category_names
    
    
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


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for idx,category_name in enumerate(category_names):
        #if category_name=='child_alone':
            #target_names = ['child_alone_0']
        #else:
        target_names = [category_name+"_0", category_name+"_1"]
            
        print(classification_report(Y_test[:, idx], Y_pred[:, idx], target_names=target_names))
    
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()