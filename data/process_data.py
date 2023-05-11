import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        Handles the loading of all data.
        
        Parameters
        -----------
        messages_filepath: File path for the messages
        
        categories_filepath: File path for the categories
        Returns
        ------------
        df : loaded dataset.    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how = 'inner', on = 'id')
    return df



def clean_data(df):
    '''
        Handles the cleaning of the data.
        
        Parameters
        -------------
        df : dataframe that the cleaning steps are to be performed on.     
        Returns
        --------------
        df : dataframe after cleaning steps have been performed on it.
    '''
    categories = df['categories'].str.split(";",expand = True)

    row = categories.iloc[0]
    category_colnames = row.tolist()
    categories.columns = category_colnames

    column_map = {col : col[-1] for col in categories.columns}
    categories = categories.rename(columns = column_map)

    df.drop(columns = ['categories'], inplace = True)
    df = pd.concat([df,categories], axis = 1)
    
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    Saves the df parameter in the sql database under the database_filename
    
    Parameters
    -------------
        df: Dataframe to be saved.
        
        database_filename: Filename of the sql database that the user wants to store the df in.
    Returns
    --------------
    '''  
    engine = create_engine(database_filename)
    df.to_sql('myTable', engine, index=False)


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