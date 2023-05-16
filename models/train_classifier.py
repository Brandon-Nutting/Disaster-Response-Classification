import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import GridSearchCV



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")

def load_data(database_filepath):
    '''
    This function handles all the laoding of the data saved during the ETL pipeline.
    
    Parameters
    ------------
    database_filepath: The filepath of the database we wish to load.
    Returns
    ------------
    x : The features.
    y : The target variables
    y.columns : The category names
    '''
    engine = create_engine(database_filepath)
    df = pd.read_sql_table("myTable", con = engine)
    df = df.reset_index(drop = True)
    X = df['message']
    y = df.iloc[:, 4:]
    return x,y,y.columns
    
def tokenize(text):
    '''
    Handles the tokenization of the data. Converts to lowercase, removes punctuation.
    
    Parameters
    --------------
    text :  The text to be tokenized.
    Returns
    ---------------
    tokens : The tokenized text.
    '''
     # tokenize set
    tokens = word_tokenize(text)
    
    # Lemmentinize, remove stop words.
    lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    clean_tokens = []
    for tok in tokens: 
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
    return tokens


def build_model():
    '''
    Handles the building and fitting of the model of the model.
    
    Parameters
    -------------
    
    Returns
    --------------
    cv : The gridsearch object.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 50, 100],
        'tfidf__use_idf': [True, False],
        'vect__max_features': [None, 5000, 10000],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3, n_jobs = -1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model. Prints out classification report for each category.
    
    Parameters
    ------------
    model : The model we wish to evaluate. 
    X_test : The testing set for feature variables.
    Y_test : The testing set for target variables.
    category_names : Names of the categories.
    
    Returns
    -------------
    
    '''


def save_model(model, model_filepath):
    pass


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