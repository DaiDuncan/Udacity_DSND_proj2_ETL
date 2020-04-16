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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import pickle



def load_data(database_filepath):
    '''
    Load DataFrame from SQL-Table "ETL_pipeline_cleaned"

    INPUT:
        database_filepath:(str) path to SQL .db file

    OUTPUT:
        X:(DataFrame) model inputs
        Y:(np.array) model results
        category_names:(pd.Index) different classifications as outputs of model
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))

    database_tablename = "ETL_pipeline_cleaned"   #!!!user should confirm it
    df = pd.read_sql_table(database_tablename, engine)

    #Get values of the categories as output Y
    category_names = df.columns[~(df.columns.isin(['id', 'message', 'original', 'genre']))]
    Y = df[category_names].values

    #Get values of the messages as input X
    #!!!Guess: the values of 'genre' maybe important, put it together with the 'message'
    X = df[['message', 'genre']]

    return X, Y, category_names



def tokenize(text):
    '''
    Put the text into cleaned tokens.

    Tokenize the text into Bag-of-words.
    Wrangle the words with Normalization and Lemmatization.

    INPUT:
        text: a message
    OUTPUT:
        cleaned_tokens:(list)
    '''

    #delete the Punctuations
    text = re.sub('\W+',' ', text).replace("_",' ')

    #delete the Stop words
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words('english')]

    #Normalization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for tok in tokens:
        cleaned_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(cleaned_tok)

    return cleaned_tokens



def build_model(with_genre=True):
    '''
    Build a Pipeline for Machine Learning with GridSearchCV.
    Because of the burden of calculation, just search one hyperparameter.
    !!!Attention: there are more than 30 outputs of the classifier.

    INPUT:
        with_genre: (default True)if True, informaiton of 'genre' to be used; otherwise just the informaiton of 'message' to be used.

    OUTPUT:
        model: chosen model after GridSearchCV
    '''

    token_transformer = Pipeline(steps=[
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
        ])

    if with_genre==True:
        preprocessor = ColumnTransformer(transformers=[
            ('token_transformer', token_transformer, 'message'),
            ('genre_categories', OneHotEncoder(drop='first', dtype='int'),['genre'])  #keep: drop='first'
            ], remainder='drop')


    elif with_genre==False:
        preprocessor = token_transformer


    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('clf', MultiOutputClassifier(RandomForestClassifier()))
                         ])

    parameters = {
        'clf__estimator__min_samples_split': [2, 3]   #'clf__estimator__n_estimators': 100
    }

    model = GridSearchCV(clf, param_grid=parameters)
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Use classification_report to evaluate model.

    INPUT:
        model: optimized and trained model
        X_test: test data as input
        Y_test: original results for X_test
        category_names:(pd.Index) different classifications as outputs of model

    OUTPUT:
        report_output:(dict) classification_report
    '''

    Y_pred = model.predict(X_test)

    report_output = dict()
    for idx, category_name in enumerate(category_names):
        #if category_name=='child_alone':
            #target_names = ['child_alone_0']
        #else:
        target_names = [category_name+"_0", category_name+"_1"]
        result = classification_report(Y_test[:, idx], Y_pred[:, idx],
                        target_names=target_names, output_dict=True)
        print(result)
        report_output[category_name] = result

    print("\nBest Parameters:", model.best_params_)

    return report_output

def report_dict_2_df(report_output):
    '''Get the cleaned and tidy report as Dataframe.

    INPUT:
        report_output:(dict) report of the evaluation of the model.
    OUTPUT:
        report_overview_df:report with 'precision' 'recall' 'f1-score' 'support'
        report_accuracy_df:report with 'accuracy'
    '''
    report_overview = dict()
    report_accuracy = dict()
    for key, dict_of_key in report_output.items():
        for key_detail, detail in dict_of_key.items():
            if key_detail=='macro avg' or key_detail=='weighted avg':
                report_overview[key+'_'+key_detail] = detail
                continue
            if key_detail == 'accuracy':
                report_accuracy[key] = detail
                continue
        report_overview[key_detail] = detail

    report_overview_df = pd.DataFrame.from_dict(report_overview, orient='index')
    report_accuracy_df = pd.DataFrame.from_dict(report_accuracy, orient='index', columns=['accuracy'])

    return report_overview_df, report_accuracy_df



def save_model(model, model_filepath):
    '''Use pickle to save the model.
    '''

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 5: #'python' is one part.
        database_filepath, model_filepath, report_1_filepath, report_2_filepath = sys.argv[1:]

        #decide, whether use the infomation of 'genre' or not
        flag_with_genre = True
        #######
        '''
        while 1:
            flag = input("Do you want to use the infomation of 'genre'(True or False)(default True):")

            if flag=="True":
                flag_with_genre = True
                break
            elif flag=="False":
                flag_with_genre = False
                break
            else:
                print("Please enter 'True' or 'False'")
        '''
        #######


        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        if flag_with_genre == False:
            X_train = X_train['message']
            X_test = X_test['message']

        print('Building model...')
        model = build_model(with_genre=flag_with_genre)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        report = evaluate_model(model, X_test, Y_test, category_names)
        report_overview_df, report_accuracy_df = report_dict_2_df(report)

        print('Saving report_overview...\n    report_overview_df: {}'.format(report_1_filepath))
        report_overview_df.to_csv(report_1_filepath,index=True)

        print('Saving report_accuracy...\n    report_accuracy_df: {}'.format(report_2_filepath))
        report_accuracy_df.to_csv(report_2_filepath,index=True)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database'\
            'as the first argument; the filepath of the pickle file to '\
            'save the model to as the second argument; the filepath of two '\
            'csv files to save the two Dateframe of the report. \n\n'\
            'Example: python train_classifier.py ../data/DisasterResponse.db'\
            'classifier.pkl df_overview.csv df_accuracy.csv')



if __name__ == '__main__':
    main()
