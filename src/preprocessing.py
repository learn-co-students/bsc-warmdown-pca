from .load_data import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess():
    # Create a dictionary to story
    # fit transformers
    transformers = {}
    
    # Load in the data
    data = load_data()
    
    # Isolate the target and predictors
    y = data['Stop Resolution']
    X = data.drop('Stop Resolution', axis=1)
    
    # LabelEncode the target
    target_encoder = LabelEncoder()
    target_encoder.fit(y)
    target_encoded = target_encoder.transform(y)
    transformers['label_encoder'] = target_encoder

    # Create a list of categorical feature names
    categoricals = ['Subject Age Group','Weapon Type', 
                    'Officer Gender', 'Officer Race', 
                    'Subject Perceived Race', 'Subject Perceived Gender',
                    'Precinct', 'Sector', 'Beat']

    # Initialize a OneHotEncoder
    # Will set handle_unknown to 'ignore' so
    # new categories in our testing data do not 
    # throw an error. We will also set sparse to `False`
    # to prevent the encoder from returning a sparse matrix.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Create a tuple with the encoder at the first index
    # and the list of categorical features at the second index
    encoder_step = (encoder, categoricals)

    # Pass the tuple into `make_column_transformer`
    # and set remainder to 'passthrough' to prevent
    # the features we did not OneHotEncode  
    # from being dropped.
    encoder = make_column_transformer(encoder_step, 
                                      remainder='passthrough')
    transformers['ohe'] = encoder
    
    X_train, X_test, y_train, y_test = train_test_split(X, target_encoded, 
                                                        random_state=2021, 
                                                        test_size=.5)
    encoder.fit(X_train)
    transformed = list(encoder.transformers_[0][1].get_feature_names(categoricals))
    untransformed = [x for x in X_train.columns  if x not in categoricals]
    columns = transformed + untransformed
    X_train_encoded = pd.DataFrame(encoder.transform(X_train),  columns=columns)
    
    X_test_encoded = pd.DataFrame(encoder.transform(X_test),  columns=columns)
    
    return X_train_encoded, X_test_encoded, y_train, y_test, transformers

    

