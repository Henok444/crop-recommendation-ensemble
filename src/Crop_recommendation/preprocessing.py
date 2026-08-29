import pandas as pd 
from sklearn.model_selection import train_test_split

feature = ['N' , "P" , "K" , "temperature" , "humidity" , "ph" , "rainfall" ]
target = ["label"]



def preprocess(df):
    X = df[feature]
    y = df[target]

    return X , y 

def split(X , y , test_size , random_state ):
    return train_test_split(X , y , test_size = test_size , random_state=random_state , stratify= y )



