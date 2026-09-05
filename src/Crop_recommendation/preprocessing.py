import pandas as pd 
from sklearn.model_selection import train_test_split
from src.Crop_recommendation.model_io import save_encoded_label
from src.Crop_recommendation.config import load_config 
from pathlib import Path
config = load_config()

feature = ['N' , "P" , "K" , "temperature" , "humidity" , "ph" , "rainfall" ]
target = "label"

def encode(df):
    from sklearn.preprocessing import LabelEncoder

    # Encode Labels

    # This process turns columns .
    le = LabelEncoder()

    df['label'] = le.fit_transform(df['label'])
    path = Path(config["encoder"]["path"] )
    save_encoded_label(le , path )
    return df


def preprocess(df):
    X = df[feature]
    y = df[target]

    return X , y 

def split(X , y , test_size , random_state ):
    return train_test_split(X , y , test_size = test_size , random_state=random_state , stratify= y )

    


