import sys
from sklearn.metrics import classification_report
import pickle
# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline,FeatureUnion
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def db_load(database_filepath):
    '''
    Load the data base and return the database that has been loaded
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    connection2 = engine.connect()
    df = pd.read_sql_table(database_filepath,engine) 
    return df



def unique_cols(df):
    '''
    check columns where all values are the same
    if all values are same in the column, True is returned else False
    We require the columns that are False,that is, the ones that do not have all values the same
    '''
    a = df.values
    return (a[0] == a).all(0)


#drop columns where all values are the same
def drop_cols(un,df):
    '''
    Input: un -> This is the list of False, True boolean values indicating if a column has unique values or not
    Based on the values returned from uique_cols, we drop the columns which 
    have all values same, that is, the one that is True in the unique_cols()
    Output: df -> Dataframe with columns having all unique values
    '''
    drop =[]

    for i in range(len(un)):
        if un[i] ==True:
                drop.append(df.columns[i])
    df = df.drop(drop,axis=1)
    return df


def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'url_place_holder_string')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = porter.stem(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
    pass

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Created a pipeline for the model
    Grid search params applied in the main function
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier())
    )
    ])
    return pipeline
    pass

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    calculate accuracy and give classification report for the best model from grid search
    '''
    y_pred = model.predict(X_test)
    overall_accuracy = (y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    pass


def save_model(model, model_filepath):
    '''
    save the "model" in the "model_filepath"
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    pass

def features(df):
    '''
    Only the message column is the feature
    Everything from column 4 is a lbael
    '''
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    return X,y

def load_data(database_filepath):
    '''
    Apply the db_load() to retieve the database
    Apply unique col() to check for columns with unique values
    Apply drop_cols() to drop the columns with same values
    Create features and labels using the features()
    '''
    df = db_load(database_filepath)
    un = unique_cols(df)
    df = drop_cols(un,df)
    X,y = features(df)
    categories = y.columns.values
    return X,y,categories
    pass



def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        #Apply Grid search CV
        parameters_grid = {'clf__estimator__n_estimators': [100,200],
              'clf__estimator__min_samples_leaf': [2,3]
                  }

        cv = GridSearchCV(model, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1,cv=2)
        

        print('Training model...')
        #model.fit(X_train, Y_train)
        cv.fit(X_train, Y_train) 
        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()