import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load the data from .csv files.

    Combined the two datesets into one Dataframe

    INPUT:
        messages_filepath: the filepath of the data(messages)
        categories_filepath: the filepath of the data(categories)

    OUTPUT:
        df: merge the two DataFrame into df based on the common feature 'id'
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # use 'id' to merge the two DataFrame into df
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''Clean the DataFrame.

    INPUT:
        df: merged DataFrame

    OUTPUT:
        df_cleaned: cleaned DataFrame

    '''
    # (form of DataFrame)create a DataFrame with 36 categories
    categories = df['categories'].str.split(';', expand=True)

    # rename the columns of categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])   #
    categories.columns = category_colnames

    # (content of DataFrame)change string into 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1:])) #


    # !!!discover: there are 3 diffrent values in 'related'(0 1 2), but just a single value in 'child_alone'(0)
    # correct the error data
    df.related.replace(2,1,inplace=True) #

    categories.drop(['child_alone'], axis=1, inplace=True)  #can also be reserved

    # drop the original categories columns
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate df and categories
    df = pd.concat([df, categories], axis=1) #

    # drop duplicates
    df.drop_duplicates(inplace=True)   #273 duplicated records

    return df


def save_data(df, database_filename):
    '''Save the data from DataFrame to SQL-table

    !!!You should define the name of the SQL-table.
    '''

    # engine is a .db file
    engine = create_engine('sqlite:///{}'.format(database_filename))

    database_tablename = "ETL_pipeline_cleaned"
    df.to_sql(database_tablename, engine, index=False)

    print("\nSave data to SQL-Table {} successful\n".format(database_tablename))

    #test
    #engine.execute("SELECT * FROM InsertTableName").fetchall()


def main():

    # run python with .py message.csv categories.csv .db
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
