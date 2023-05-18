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
    # Read in datafiles
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Combine messages and catefories on 'id' field.
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
    # Expand categories column to capture all different categories for each row.
    categories = df['categories'].str.split(";",expand = True)

    # We want to rename column headers in categories. Grab the first row.
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames # These are the new names

    # Replace each column name with the last digit of the string. 
    for column in categories:
        categories[column] = categories[column].apply(lambda x : int(x[-1]))


    # Drop old categories column from original df
    df.drop(columns = ['categories'], inplace = True)
    
    # Concatenate original dataframe and new categories dataframe.
    df = pd.concat([df,categories], axis = 1)
    
    # Make sure there are no duplicates.
    df.drop_duplicates(inplace = True)
    
    # Solving the edge case where 'related' has a value of 2
    df['related'] = df['related'].apply(lambda x : 0 if x == 2 else 0)

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
    # Create engine to store df in an sqllite database.
    engine = create_engine('sqlite:///' + database_filename)
    # Store df in database.
    df.to_sql('myTable', engine, index=False, if_exists = 'replace')


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