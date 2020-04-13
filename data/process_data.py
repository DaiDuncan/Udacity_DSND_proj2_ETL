import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #use 'id' to merge the two DataFrame into df
    df = messages.merge(categories, on='id')
    
    return df
    
    
def clean_data(df):
    
    #(form of df)get the different categories as names for columns
    categories = df['categories'].str.split(';', expand=True)   
    row = categories.iloc[0]
    category_colnames = [text[:-2] for text in row]   #36 categories
    categories.columns = category_colnames
    
    #(content of df)change datetype: 0-1 string to numeric
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
        
    #discover: there are 3 diffrent values in 'related'(0 1 2), but just unique values in 'child_alone'(0)
    categories['related'][categories['related']==2] = 1
    
    df_cleaned = df.drop(['categories'], axis=1)
    df_cleaned = pd.concat([df_cleaned, categories], axis=1)
    
    #drop the duplicated records
    df_cleaned.drop_duplicates(inplace=True)
    
    return df_cleaned


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename)) 
    
    database_tablename = "disaster_cleaned"
    df.to_sql(database_tablename, engine, index=False)
    
    print("Save data to sql-Table {} successful".format(database_tablename))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()