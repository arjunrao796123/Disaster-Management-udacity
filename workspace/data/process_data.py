# import libraries
import pandas as pd
import numpy as np

import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Takes in the 'messages_filepath' and 'categories_filepath' to load the messages and categories csv files.
    The id column is commonin these 2 csv files and a merge is performed on that id column.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

    pass


def clean_data(df):
    '''
    create a dataframe of the 36 individual category columns
    renamed the columns in this part by defining columns variable
    '''
    split_df = pd.DataFrame(df['categories'].str.split(";", expand = True).values,columns = df['categories'].str.split(";", expand = True).iloc[1,:].str[:-2].values ) 
    split_df_merged = pd.concat([df,split_df], axis=1).drop(['categories'],axis=1)
    for column in split_df_merged.iloc[:,4:]:
        # set each value to be the last character of the string
        split_df_merged[column]  = split_df_merged[column].apply(lambda x: 1 if ('1') in x else 0)
        # convert column from string to numeric
        split_df_merged[column] = split_df_merged[column].astype(int)
    
    split_df_merged = split_df_merged.drop_duplicates()
    return split_df_merged
    pass


def save_data(df, database_filename):
    '''
    save data base as 'database_filename' variable
    '''
    engine = create_engine('sqlite:///'+database_filename)

    df.to_sql(database_filename, engine, index=False) 
    pass  


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